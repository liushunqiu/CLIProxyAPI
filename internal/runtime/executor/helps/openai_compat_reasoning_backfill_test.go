package helps

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestEnsureOpenAICompatAssistantReasoningBackfillsSentinelWhenNoReasoning(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{}"}}]}]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 1 {
		t.Fatalf("patched = %d, want 1", patched)
	}
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != openAICompatReasoningUnavailable {
		t.Fatalf("reasoning_content = %q, want %q; body=%s", got, openAICompatReasoningUnavailable, out)
	}
}

func TestEnsureOpenAICompatAssistantReasoningReusesPriorReasoning(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"assistant","content":"","reasoning_content":"because the counts did not add up","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"ok"},
		{"role":"assistant","content":"","tool_calls":[{"id":"call_2","type":"function","function":{"name":"Read","arguments":"{}"}}]}
	]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 1 {
		t.Fatalf("patched = %d, want 1", patched)
	}
	if got := gjson.GetBytes(out, "messages.2.reasoning_content").String(); got != "because the counts did not add up" {
		t.Fatalf("reasoning_content = %q, want %q; body=%s", got, "because the counts did not add up", out)
	}
	// The earlier turn already carrying reasoning must be left unchanged.
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "because the counts did not add up" {
		t.Fatalf("prior reasoning_content was altered: %q", got)
	}
}

func TestEnsureOpenAICompatAssistantReasoningSkipsExistingReasoning(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","content":"","reasoning_content":"already here","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{}"}}]}]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 0 {
		t.Fatalf("patched = %d, want 0", patched)
	}
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "already here" {
		t.Fatalf("reasoning_content = %q, want %q", got, "already here")
	}
}

func TestEnsureOpenAICompatAssistantReasoningBackfillsEmptyString(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","content":"","reasoning_content":"   ","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{}"}}]}]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 1 {
		t.Fatalf("patched = %d, want 1", patched)
	}
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != openAICompatReasoningUnavailable {
		t.Fatalf("reasoning_content = %q, want %q", got, openAICompatReasoningUnavailable)
	}
}

func TestEnsureOpenAICompatAssistantReasoningUsesOwnContentWhenNoPriorReasoning(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","content":[{"type":"text","text":"Let me inspect the file."}],"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{}"}}]}]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 1 {
		t.Fatalf("patched = %d, want 1", patched)
	}
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "Let me inspect the file." {
		t.Fatalf("reasoning_content = %q, want %q; body=%s", got, "Let me inspect the file.", out)
	}
}

func TestEnsureOpenAICompatAssistantReasoningLeavesNonAssistantUntouched(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"system","content":"system prompt"},
		{"role":"user","content":"hello"},
		{"role":"tool","tool_call_id":"call_1","content":"result"}
	]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 0 {
		t.Fatalf("patched = %d, want 0", patched)
	}
	if !gjson.GetBytes(out, "messages.0.role").Exists() {
		t.Fatalf("body corrupted: %s", out)
	}
}

func TestEnsureOpenAICompatAssistantReasoningLeavesAssistantWithoutToolCallsUntouched(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","content":"plain text reply, no tool calls"}]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 0 {
		t.Fatalf("patched = %d, want 0", patched)
	}
	if gjson.GetBytes(out, "messages.0.reasoning_content").Exists() {
		t.Fatalf("unexpected reasoning_content added: %s", out)
	}
}

func TestEnsureOpenAICompatAssistantReasoningHandlesMultiplePatches(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"ok"},
		{"role":"assistant","content":"","tool_calls":[{"id":"call_2","type":"function","function":{"name":"Read","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_2","content":"ok"},
		{"role":"assistant","content":"","tool_calls":[{"id":"call_3","type":"function","function":{"name":"Read","arguments":"{}"}}]}
	]}`)

	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 3 {
		t.Fatalf("patched = %d, want 3", patched)
	}
	for i := 0; i < 3; i++ {
		msgIdx := i * 2
		if got := gjson.GetBytes(out, "messages."+itoa(msgIdx)+".reasoning_content").String(); got != openAICompatReasoningUnavailable {
			t.Fatalf("messages.%d.reasoning_content = %q, want %q", msgIdx, got, openAICompatReasoningUnavailable)
		}
	}
}

func TestEnsureOpenAICompatAssistantReasoningReturnsBodyOnInvalidJSON(t *testing.T) {
	body := []byte(`not json`)
	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 0 {
		t.Fatalf("patched = %d, want 0", patched)
	}
	if string(out) != "not json" {
		t.Fatalf("body mutated on invalid JSON")
	}
}

func TestEnsureOpenAICompatAssistantReasoningReturnsBodyWhenNoMessages(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4-flash"}`)
	out, patched := EnsureOpenAICompatAssistantReasoning(body)
	if patched != 0 {
		t.Fatalf("patched = %d, want 0", patched)
	}
	if !gjson.GetBytes(out, "model").Exists() {
		t.Fatalf("body corrupted: %s", out)
	}
}

// itoa keeps the test file dependency-free.
func itoa(i int) string {
	const digits = "0123456789"
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = digits[i%10]
		i /= 10
	}
	return string(buf[pos:])
}
