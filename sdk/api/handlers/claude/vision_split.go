// Package claude provides Claude-compatible HTTP handlers.
//
// vision_split.go implements an image-split passthrough for upstream
// OpenAI-compatible models that do not accept image input (e.g. some
// deepseek-free upstreams that reject image blocks with a 400). When a
// Claude /v1/messages request carries image content blocks and the routed
// model's upstream cannot accept images, the images are peeled out of the
// request, sent to a vision-capable model (grok-4.6) through the same proxy
// execution pipeline, and the returned text description replaces the image
// blocks. The original model still processes the request, but only sees text.
package claude

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gin-gonic/gin"
	. "github.com/router-for-me/CLIProxyAPI/v7/internal/constant"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// visionModel is the model used to analyze images when the requested model's
// upstream rejects image input. grok-4.6 is vision-capable and reaches the
// proxy through the xai provider (verified working end-to-end).
const visionModel = "grok-4.6"

// upstreamRejectsImages reports whether the given model's upstream is known to
// reject image input. This mirrors models routed to OpenAI-compatible upstreams
// like tokenharbor's deepseek-v4-flash:free that return
// "does not accept image input". Unknown models are treated leniently (false).
func upstreamRejectsImages(model string) bool {
	base := strings.ToLower(strings.TrimSpace(model))
	base = strings.TrimSuffix(base, "[1m]")
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	return base == "deepseek-v4-flash" || base == "deepseek-v4-flash:free" ||
		strings.HasPrefix(base, "deepseek-v4-") ||
		strings.HasPrefix(base, "deepseek-r")
}

// hasClaudeImageBlocks returns true when rawJSON is a Claude /v1/messages body
// that contains at least one image content block with base64 data, including
// images nested inside tool_result blocks (e.g. a Read tool returning a
// screenshot).
func hasClaudeImageBlocks(rawJSON []byte) bool {
	for _, m := range gjson.GetBytes(rawJSON, "messages").Array() {
		if contentHasImageBlocks(m.Get("content")) {
			return true
		}
	}
	return false
}

// contentHasImageBlocks reports whether a content array contains an image block
// with base64 data, recursing into tool_result blocks that carry their own
// content arrays.
func contentHasImageBlocks(content gjson.Result) bool {
	if !content.Exists() || !content.IsArray() {
		return false
	}
	for _, block := range content.Array() {
		switch block.Get("type").String() {
		case "image":
			src := block.Get("source")
			if src.Get("type").String() == "base64" && src.Get("data").String() != "" {
				return true
			}
		case "tool_result":
			if contentHasImageBlocks(block.Get("content")) {
				return true
			}
		}
	}
	return false
}

// splitImagesForUpstream inspects a Claude /v1/messages request. If the payload
// contains image blocks and the requested model's upstream rejects images, it
// asks the vision model to describe the images and rewrites the payload so the
// image blocks are replaced with the description text. It returns the (possibly
// rewritten) bytes and whether a split actually happened.
func splitImagesForUpstream(c *gin.Context, h *ClaudeCodeAPIHandler, rawJSON []byte, modelName string) ([]byte, bool) {
	if !upstreamRejectsImages(modelName) {
		return rawJSON, false
	}
	if !hasClaudeImageBlocks(rawJSON) {
		return rawJSON, false
	}

	log.Infof("claude vision-split: model %s cannot accept images; delegating to %s", modelName, visionModel)

	// Build an OpenAI chat payload containing only the images, described by a
	// prompt that asks for a factual, verbose description.
	content := buildVisionOpenAIContent(rawJSON)
	want := struct {
		Model     string `json:"model"`
		Messages  []any  `json:"messages"`
		MaxTokens int    `json:"max_tokens"`
	}{
		Model: visionModel,
		Messages: []any{
			map[string]any{
				"role":    "user",
				"content": content,
			},
		},
		MaxTokens: 1024,
	}
	payload, err := json.Marshal(want)
	if err != nil {
		log.Errorf("claude vision-split: marshal vision payload: %v", err)
		return rawJSON, false
	}

	alt := h.GetAlt(c)
	ctx, cancel := h.GetContextWithCancel(h, c, context.Background())
	defer cancel()

	resp, _, errMsg := h.ExecuteWithAuthManager(ctx, OpenAI, visionModel, payload, alt)
	if errMsg != nil {
		log.Errorf("claude vision-split: vision call failed: %v", errMsg.Error)
		return rawJSON, false
	}

	desc := extractOpenAIChatDescription(resp)
	if desc == "" {
		log.Warnf("claude vision-split: vision model returned empty description; leaving images untouched")
		return rawJSON, false
	}

	newJSON, err := replaceImageBlocksWithText(rawJSON, desc)
	if err != nil {
		log.Errorf("claude vision-split: rewrite failed: %v", err)
		return rawJSON, false
	}
	return newJSON, true
}

// buildVisionOpenAIContent converts Claude image blocks in rawJSON into an
// OpenAI-style content array (text prompt + image_url data URIs), suitable for
// sending to the vision model via the proxy's OpenAI-compatible executor.
// Images nested inside tool_result blocks are included too.
func buildVisionOpenAIContent(rawJSON []byte) []any {
	content := []any{
		map[string]any{
			"type": "text",
			"text": "Describe the following image(s) in detail for a text-only assistant. " +
				"Include any visible text (OCR), objects, layout, colors, and relationships. " +
				"Be thorough and factual. If there are multiple images, describe each one.",
		},
	}
	for _, m := range gjson.GetBytes(rawJSON, "messages").Array() {
		appendVisionImagesFromContent(&content, m.Get("content"))
	}
	// Fallback: if for any reason no images were appended, this payload would
	// be text-only anyway and the vision model simply describes nothing.
	return content
}

// appendVisionImagesFromContent appends OpenAI image_url parts for every base64
// image block found in a content array, recursing into tool_result blocks.
func appendVisionImagesFromContent(out *[]any, content gjson.Result) {
	if !content.Exists() || !content.IsArray() {
		return
	}
	for _, block := range content.Array() {
		switch block.Get("type").String() {
		case "image":
			src := block.Get("source")
			if src.Get("type").String() != "base64" {
				continue
			}
			media := src.Get("media_type").String()
			if media == "" {
				media = "image/png"
			}
			data := src.Get("data").String()
			if data == "" {
				continue
			}
			*out = append(*out, map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": fmt.Sprintf("data:%s;base64,%s", media, data),
				},
			})
		case "tool_result":
			appendVisionImagesFromContent(out, block.Get("content"))
		}
	}
}

// extractOpenAIChatDescription pulls the assistant text out of an OpenAI
// chat completions response.
func extractOpenAIChatDescription(raw []byte) string {
	text := gjson.GetBytes(raw, "choices.0.message.content").String()
	if text != "" {
		return strings.TrimSpace(text)
	}
	for _, c := range gjson.GetBytes(raw, "choices.0.message").Array() {
		if c.Get("type").String() == "text" {
			if t := strings.TrimSpace(c.Get("text").String()); t != "" {
				return t
			}
		}
	}
	return ""
}

// replaceImageBlocksWithText replaces every image content block in a Claude
// /v1/messages body with a single text block containing the description.
// Images nested inside tool_result blocks are handled too. Text blocks and
// other content are preserved.
func replaceImageBlocksWithText(rawJSON []byte, description string) ([]byte, error) {
	out := rawJSON
	used := false
	msgs := gjson.GetBytes(out, "messages")
	if !msgs.Exists() || !msgs.IsArray() {
		return out, nil
	}
	for msgIdx, m := range msgs.Array() {
		content := m.Get("content")
		if !content.Exists() || !content.IsArray() {
			continue
		}
		newContent, newUsed, err := rewriteContentArray(content, description, used)
		if err != nil {
			return rawJSON, err
		}
		used = newUsed
		path := fmt.Sprintf("messages.%d.content", msgIdx)
		updated, errSet := sjson.SetBytes(out, path, newContent)
		if errSet != nil {
			return rawJSON, errSet
		}
		out = updated
	}
	return out, nil
}

// rewriteContentArray returns a rebuilt content array with the first base64
// image block replaced by the description text and all remaining image blocks
// dropped. It recurses into tool_result blocks that carry their own content
// arrays. used reports whether the description has already been placed
// anywhere in the body.
func rewriteContentArray(content gjson.Result, description string, used bool) ([]any, bool, error) {
	if !content.Exists() || !content.IsArray() {
		return nil, used, nil
	}
	kept := make([]any, 0, len(content.Array()))
	for _, block := range content.Array() {
		switch block.Get("type").String() {
		case "image":
			src := block.Get("source")
			if src.Get("type").String() != "base64" || src.Get("data").String() == "" {
				kept = append(kept, block.Value())
				continue
			}
			if !used {
				// Insert the description once, as a text block, replacing this image.
				kept = append(kept, map[string]any{
					"type": "text",
					"text": "[image analyzed by vision model]\n\n" + description,
				})
				used = true
				continue
			}
			// The description is already placed; drop this image entirely.
		case "tool_result":
			inner := block.Get("content")
			if inner.Exists() && inner.IsArray() {
				rebuiltInner, newUsed, err := rewriteContentArray(inner, description, used)
				if err != nil {
					return nil, used, err
				}
				kept = append(kept, rebuildToolResultWithContent(block, rebuiltInner))
				used = newUsed
				continue
			}
			kept = append(kept, block.Value())
		default:
			kept = append(kept, block.Value())
		}
	}
	return kept, used, nil
}

// rebuildToolResultWithContent rebuilds a tool_result block whose nested content
// array was rewritten. It preserves the block's other fields (tool_use_id).
func rebuildToolResultWithContent(block gjson.Result, content []any) map[string]any {
	rebuilt := map[string]any{}
	block.ForEach(func(key, value gjson.Result) bool {
		if key.String() == "content" {
			return true
		}
		rebuilt[key.String()] = value.Value()
		return true
	})
	rebuilt["content"] = content
	return rebuilt
}
