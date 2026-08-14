package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestOpenAICompatExecutorUsesCompatibleClaudeTranslation(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"prior reasoning","signature":""},{"type":"tool_use","id":"call_1","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}

	assistant := gjson.GetBytes(upstreamBody, "messages.0")
	if got := assistant.Get("reasoning_content").String(); got != "prior reasoning" {
		t.Fatalf("reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
	}
	if !assistant.Get("tool_calls").Exists() {
		t.Fatalf("tool_calls missing from upstream request: %s", upstreamBody)
	}
}

// TestOpenAICompatExecutorBackfillsReasoningOnToolCallOnlyTurn verifies that a
// historical assistant turn carrying only tool_calls (no thinking block) gets
// a backfilled reasoning_content when the request is in thinking mode, so
// upstreams like DeepSeek no longer reject it with the
// "reasoning_content ... must be passed back" 400.
func TestOpenAICompatExecutorBackfillsReasoningOnToolCallOnlyTurn(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	// Claude-format payload: a tool-call-only assistant turn (no thinking block)
	// followed by its tool result. The model name carries a (max) suffix so the
	// translated OpenAI body enters thinking mode, exercising the gate.
	// UserDefined must be true so the thinking applier treats this as a
	// configured compat model and emits reasoning_effort (matching production
	// openai-compatibility.models[] registration).
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash(max)",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":[{"type":"text","text":"list files"}]},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"ls"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"a\nb"}]},{"role":"user","content":[{"type":"text","text":"what next"}]}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true, UserDefined: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}

	// Find the assistant turn with tool_calls in the translated upstream body.
	assistantIdx := findAssistantWithToolCalls(t, upstreamBody)
	assistant := gjson.GetBytes(upstreamBody, "messages."+strconv.Itoa(assistantIdx))
	if !assistant.Get("reasoning_content").Exists() {
		t.Fatalf("reasoning_content missing on tool-call-only turn: %s", upstreamBody)
	}
	if got := assistant.Get("reasoning_content").String(); strings.TrimSpace(got) == "" {
		t.Fatalf("reasoning_content is empty: %s", upstreamBody)
	}
}

// TestOpenAICompatExecutorDoesNotBackfillReasoningWhenThinkingDisabled proves
// the gate: with thinking off (no suffix, no reasoning_effort) and a
// tool-call-only turn, no spurious reasoning_content is added.
func TestOpenAICompatExecutorDoesNotBackfillReasoningWhenThinkingDisabled(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":[{"type":"text","text":"list files"}]},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"ls"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"a\nb"}]},{"role":"user","content":[{"type":"text","text":"what next"}]}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}

	assistantIdx := findAssistantWithToolCalls(t, upstreamBody)
	if gjson.GetBytes(upstreamBody, "messages."+strconv.Itoa(assistantIdx)+".reasoning_content").Exists() {
		t.Fatalf("reasoning_content unexpectedly added when thinking is off: %s", upstreamBody)
	}
}

// findAssistantWithToolCalls returns the index of the first assistant message
// carrying tool_calls, failing the test if none is found.
func findAssistantWithToolCalls(t *testing.T, body []byte) int {
	t.Helper()
	idx := -1
	messages := gjson.GetBytes(body, "messages")
	messages.ForEach(func(i, msg gjson.Result) bool {
		if msg.Get("role").String() == "assistant" && msg.Get("tool_calls").IsArray() {
			idx = int(i.Int())
			return false
		}
		return true
	})
	if idx == -1 {
		t.Fatalf("no assistant+tool_calls turn in upstream body: %s", body)
	}
	return idx
}
