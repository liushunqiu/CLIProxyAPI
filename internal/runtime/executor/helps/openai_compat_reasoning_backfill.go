package helps

import (
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// openAICompatReasoningUnavailable is the sentinel reasoning value backfilled
// onto tool-call-only assistant turns when no real reasoning can be derived.
// It mirrors Kimi's "[reasoning unavailable]" sentinel so the upstream's
// "every assistant turn must carry reasoning_content" contract is satisfied
// without fabricating misleading reasoning text.
const openAICompatReasoningUnavailable = "[reasoning unavailable]"

// EnsureOpenAICompatAssistantReasoning backfills reasoning_content on
// assistant messages that carry tool_calls but lack a non-empty
// reasoning_content field.
//
// Some OpenAI-compatible upstreams (e.g. DeepSeek) require every assistant
// message to carry reasoning_content while in thinking mode. Historical turns
// that the client emitted as a pure tool call (no preceding reasoning) have
// nothing to pass back, so the upstream rejects the whole request with HTTP
// 400. This mirrors the reasoning backfill in Kimi's
// normalizeKimiToolMessageLinks, but is gated by the caller on thinking mode
// being active so non-thinking requests are byte-for-byte unchanged.
//
// The fallback value prefers the most recent non-empty reasoning_content seen
// on an earlier assistant turn, then the message's own text content, and
// finally a sentinel string. It returns the (possibly patched) body and the
// number of messages patched.
func EnsureOpenAICompatAssistantReasoning(body []byte) ([]byte, int) {
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return body, 0
	}

	messages := util.GetGJSONBytesNoCopy(body, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return body, 0
	}

	msgs := messages.Array()
	type messagePatch struct {
		index int
		value string
	}
	patches := make([]messagePatch, 0)
	latestReasoning := ""
	hasLatestReasoning := false

	for msgIndex, msg := range msgs {
		if !openAICompatAssistantHasToolCalls(msg) {
			continue
		}

		reasoning := msg.Get("reasoning_content")
		if reasoning.Exists() && strings.TrimSpace(reasoning.String()) != "" {
			latestReasoning = reasoning.String()
			hasLatestReasoning = true
			continue
		}

		patches = append(patches, messagePatch{
			index: msgIndex,
			value: openAICompatFallbackReasoning(msg, hasLatestReasoning, latestReasoning),
		})
	}

	if len(patches) == 0 {
		return body, 0
	}

	// Single patch: avoid rebuilding the whole messages array.
	if len(patches) == 1 {
		patch := patches[0]
		path := fmt.Sprintf("messages.%d.reasoning_content", patch.index)
		updated, errSet := sjson.SetBytes(body, path, patch.value)
		if errSet != nil {
			return body, 0
		}
		return updated, 1
	}

	// Multiple patches: rebuild the messages array once, applying each patch
	// to the affected message's raw JSON in place.
	patchIndex := 0
	messageItems := make([]string, 0, len(msgs))
	for msgIndex, msg := range msgs {
		messageJSON := msg.Raw
		for patchIndex < len(patches) && patches[patchIndex].index == msgIndex {
			next, errSet := sjson.SetBytes([]byte(messageJSON), "reasoning_content", patches[patchIndex].value)
			if errSet != nil {
				return body, len(patches)
			}
			messageJSON = string(next)
			patchIndex++
		}
		messageItems = append(messageItems, messageJSON)
	}
	updated, errSet := sjson.SetRawBytes(body, "messages", JoinRawJSONStrings(messageItems))
	if errSet != nil {
		return body, 0
	}
	return updated, len(patches)
}

// openAICompatAssistantHasToolCalls reports whether the message is an
// assistant turn carrying at least one tool call.
func openAICompatAssistantHasToolCalls(msg gjson.Result) bool {
	if strings.TrimSpace(msg.Get("role").String()) != "assistant" {
		return false
	}
	toolCalls := msg.Get("tool_calls")
	return toolCalls.Exists() && toolCalls.IsArray() && len(toolCalls.Array()) > 0
}

// openAICompatFallbackReasoning derives a reasoning value for a tool-call-only
// assistant turn, preferring the most recent real reasoning, then the turn's
// own text content, and finally a sentinel. It mirrors Kimi's
// fallbackAssistantReasoning.
func openAICompatFallbackReasoning(msg gjson.Result, hasLatest bool, latest string) string {
	if hasLatest && strings.TrimSpace(latest) != "" {
		return latest
	}

	content := msg.Get("content")
	if content.Type == gjson.String {
		if text := strings.TrimSpace(content.String()); text != "" {
			return text
		}
	}
	if content.IsArray() {
		parts := make([]string, 0, len(content.Array()))
		for _, item := range content.Array() {
			text := strings.TrimSpace(item.Get("text").String())
			if text == "" {
				continue
			}
			parts = append(parts, text)
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}

	return openAICompatReasoningUnavailable
}
