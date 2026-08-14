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
// that contains at least one image content block with base64 data.
func hasClaudeImageBlocks(rawJSON []byte) bool {
	found := false
	for _, m := range gjson.GetBytes(rawJSON, "messages").Array() {
		content := m.Get("content")
		if !content.Exists() || !content.IsArray() {
			continue
		}
		for _, block := range content.Array() {
			if block.Get("type").String() == "image" {
				src := block.Get("source")
				if src.Get("type").String() == "base64" && src.Get("data").String() != "" {
					found = true
				}
			}
		}
	}
	return found
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
		msgContent := m.Get("content")
		if !msgContent.Exists() || !msgContent.IsArray() {
			continue
		}
		for _, block := range msgContent.Array() {
			if block.Get("type").String() != "image" {
				continue
			}
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
			content = append(content, map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": fmt.Sprintf("data:%s;base64,%s", media, data),
				},
			})
		}
	}
	// Fallback: if for any reason no images were appended, this payload would
	// be text-only anyway and the vision model simply describes nothing.
	return content
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
// Text blocks and other content are preserved.
func replaceImageBlocksWithText(rawJSON []byte, description string) ([]byte, error) {
	out := rawJSON
	msgIdx := 0
	used := false
	msgs := gjson.GetBytes(out, "messages")
	if !msgs.Exists() || !msgs.IsArray() {
		return out, nil
	}
	for _, m := range msgs.Array() {
		content := m.Get("content")
		if !content.Exists() || !content.IsArray() {
			msgIdx++
			continue
		}
		replaced := false
		blockIdx := -1
		for _, block := range content.Array() {
			blockIdx++
			if block.Get("type").String() != "image" {
				continue
			}
			src := block.Get("source")
			if src.Get("type").String() != "base64" || src.Get("data").String() == "" {
				continue
			}
			if !used {
				// Insert the description once, as a text block, replacing this image.
				path := fmt.Sprintf("messages.%d.content.%d", msgIdx, blockIdx)
				var err error
				out, err = sjson.SetBytes(out, path, map[string]any{
					"type": "text",
					"text": "[image analyzed by vision model]\n\n" + description,
				})
				if err != nil {
					return rawJSON, err
				}
				used = true
				replaced = true
				break
			}
		}
		if replaced {
			// Remove any remaining image blocks in this message so the upstream
			// never sees an image block after the split point.
			out = removeTrailingImageBlocks(out, msgIdx)
		}
		msgIdx++
	}
	return out, nil
}

// removeTrailingImageBlocks removes image blocks from the message at msgIdx
// (after the split already replaced the first image). This is a defensive
// cleanup so the rewritten body never contains a raw image block.
func removeTrailingImageBlocks(rawJSON []byte, msgIdx int) []byte {
	contentPath := fmt.Sprintf("messages.%d.content", msgIdx)
	content := gjson.GetBytes(rawJSON, contentPath)
	if !content.Exists() || !content.IsArray() {
		return rawJSON
	}
	kept := make([]any, 0, len(content.Array()))
	for _, block := range content.Array() {
		if block.Get("type").String() == "image" {
			continue
		}
		kept = append(kept, block.Value())
	}
	b, err := sjson.SetBytes(rawJSON, contentPath, kept)
	if err != nil {
		return rawJSON
	}
	return b
}
