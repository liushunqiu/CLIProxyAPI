package claude

import (
	"encoding/json"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/tidwall/gjson"
)

func TestUpstreamRejectsImages(t *testing.T) {
	tests := []struct {
		name  string
		model string
		cfg   *config.VisionSplitConfig
		want  bool
	}{
		{
			name:  "nil config uses default deepseek-v4-flash exact",
			model: "deepseek-v4-flash",
			cfg:   nil,
			want:  true,
		},
		{
			name:  "nil config uses default prefix matching",
			model: "deepseek-v4-flash-lite",
			cfg:   nil,
			want:  true,
		},
		{
			name:  "nil config uses default deepseek-r prefix",
			model: "deepseek-r2",
			cfg:   nil,
			want:  true,
		},
		{
			name:  "nil config unknown model is lenient",
			model: "grok-4.6",
			cfg:   nil,
			want:  false,
		},
		{
			name:  "empty config list falls back to default",
			model: "deepseek-v4-flash",
			cfg:   &config.VisionSplitConfig{},
			want:  true,
		},
		{
			name:  "empty config list still lenient on others",
			model: "grok-4.6",
			cfg:   &config.VisionSplitConfig{},
			want:  false,
		},
		{
			name:  "configured list matches exactly",
			model: "custom-model",
			cfg: &config.VisionSplitConfig{
				ImageRejectingModels: []string{"custom-model", "another-model"},
			},
			want: true,
		},
		{
			name:  "configured list does not prefix match",
			model: "custom-model-pro",
			cfg: &config.VisionSplitConfig{
				ImageRejectingModels: []string{"custom-model"},
			},
			want: false,
		},
		{
			name:  "configured list overrides default",
			model: "deepseek-v4-flash",
			cfg: &config.VisionSplitConfig{
				ImageRejectingModels: []string{"custom-model"},
			},
			want: false,
		},
		{
			name:  "configured list strips path prefix and 1m suffix",
			model: "provider/deepseek-v4-flash:free [1m]",
			cfg: &config.VisionSplitConfig{
				ImageRejectingModels: []string{"deepseek-v4-flash:free"},
			},
			want: true,
		},
		{
			name:  "case insensitive match",
			model: "DeepSeek-V4-Flash",
			cfg: &config.VisionSplitConfig{
				ImageRejectingModels: []string{"deepseek-v4-flash"},
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := upstreamRejectsImages(tt.model, tt.cfg); got != tt.want {
				t.Fatalf("upstreamRejectsImages(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestHasClaudeImageBlocks(t *testing.T) {
	cases := []struct {
		name    string
		rawJSON string
		want    bool
	}{
		{
			name:    "no images",
			rawJSON: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			want:    false,
		},
		{
			name: "top-level image",
			rawJSON: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"},` +
				`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}]}`,
			want: true,
		},
		{
			name: "image inside tool_result",
			rawJSON: `{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x",` +
				`"content":[{"type":"text","text":"ok"},` +
				`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}]}]}`,
			want: true,
		},
		{
			name: "image inside tool_result deep",
			rawJSON: `{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":` +
				`[{"type":"tool_result","tool_use_id":"y","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}]}]}]}`,
			want: true,
		},
		{
			name:    "empty base64 data",
			rawJSON: `{"model":"m","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}]}]}`,
			want:    false,
		},
		{
			name:    "no messages array",
			rawJSON: `{"model":"m"}`,
			want:    false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := hasClaudeImageBlocks([]byte(tc.rawJSON))
			if got != tc.want {
				t.Fatalf("hasClaudeImageBlocks() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBuildVisionOpenAIContent(t *testing.T) {
	rawJSON := `{"model":"m","messages":[
		{"role":"user","content":[{"type":"text","text":"look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"topimg"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":[{"type":"text","text":"ok"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"toolimg"}}]}]}
	]}`

	content := buildVisionOpenAIContent([]byte(rawJSON))
	if len(content) < 2 {
		t.Fatalf("expected at least 2 content items, got %d", len(content))
	}
	// First item should be text prompt
	if m, ok := content[0].(map[string]any); !ok || m["type"] != "text" {
		t.Fatalf("first item should be text prompt")
	}
	// Second item should be image_url from top-level image
	if m, ok := content[1].(map[string]any); !ok {
		t.Fatalf("second item should be image_url")
	} else if m["type"] != "image_url" {
		t.Fatalf("second item type should be image_url, got %v", m["type"])
	} else {
		url := m["image_url"].(map[string]any)["url"].(string)
		if url != "data:image/png;base64,topimg" {
			t.Fatalf("second item url mismatch: %s", url)
		}
	}
	// Third item should be image_url from tool_result image
	if len(content) < 3 {
		t.Fatalf("expected 3 content items, got %d", len(content))
	}
	if m, ok := content[2].(map[string]any); !ok {
		t.Fatalf("third item should be image_url")
	} else if m["type"] != "image_url" {
		t.Fatalf("third item type should be image_url, got %v", m["type"])
	} else {
		url := m["image_url"].(map[string]any)["url"].(string)
		if url != "data:image/jpeg;base64,toolimg" {
			t.Fatalf("third item url mismatch: %s", url)
		}
	}
}

func TestReplaceImageBlocksWithText(t *testing.T) {
	const visionDesc = "This is a screenshot."
	const prefix = "[image analyzed by vision model]\n\n"
	expectedText := prefix + visionDesc

	rawJSON := []byte(`{"model":"m","messages":[
		{"role":"user","content":[{"type":"text","text":"look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"topimg"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":[{"type":"text","text":"ok"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"toolimg"}}]}]}
	]}`)

	out, err := replaceImageBlocksWithText(rawJSON, visionDesc)
	if err != nil {
		t.Fatalf("replaceImageBlocksWithText() error: %v", err)
	}

	// Verify no "image" type blocks remain anywhere
	imageCount := 0
	for _, m := range gjson.GetBytes(out, "messages").Array() {
		imageCount += countImageBlocksRecursive(m.Get("content"))
	}
	if imageCount != 0 {
		t.Fatalf("expected 0 image blocks after replacement, got %d", imageCount)
	}

	// Verify the description text (with prefix) appears exactly once
	count := 0
	for _, m := range gjson.GetBytes(out, "messages").Array() {
		for _, b := range m.Get("content").Array() {
			if b.Get("type").String() == "text" && b.Get("text").String() == expectedText {
				count++
			}
		}
	}
	if count != 1 {
		t.Fatalf("expected description text to appear exactly once, got %d", count)
	}

	// Verify tool_result block still has its tool_use_id
	toolUseID := gjson.GetBytes(out, "messages.1.content.0.tool_use_id").String()
	if toolUseID != "x" {
		t.Fatalf("expected tool_use_id to be preserved, got %q", toolUseID)
	}
}

func TestReplaceImageBlocksWithTextMultipleImagesInOneMessage(t *testing.T) {
	const visionDesc = "Description."
	const prefix = "[image analyzed by vision model]\n\n"
	expectedText := prefix + visionDesc

	rawJSON := []byte(`{"model":"m","messages":[
		{"role":"user","content":[
			{"type":"text","text":"a"},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"img1"}},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"img2"}},
			{"type":"text","text":"b"}
		]}
	]}`)

	out, err := replaceImageBlocksWithText(rawJSON, visionDesc)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Should have exactly 3 content items: text "a", description text, text "b"
	content := gjson.GetBytes(out, "messages.0.content")
	if !content.IsArray() {
		t.Fatal("content should be array")
	}
	items := content.Array()
	if len(items) != 3 {
		t.Fatalf("expected 3 content items, got %d", len(items))
	}
	if items[0].Get("text").String() != "a" {
		t.Fatalf("first item text mismatch: %s", items[0].Get("text").String())
	}
	if items[1].Get("type").String() != "text" || items[1].Get("text").String() != expectedText {
		t.Fatalf("second item should be description text, got type=%s text=%q", items[1].Get("type").String(), items[1].Get("text").String())
	}
	if items[2].Get("text").String() != "b" {
		t.Fatalf("third item text mismatch: %s", items[2].Get("text").String())
	}
}

func TestReplaceImageBlocksWithTextToolResultImagesAreDropped(t *testing.T) {
	const visionDesc = "Description."
	const prefix = "[image analyzed by vision model]\n\n"
	expectedText := prefix + visionDesc

	rawJSON := []byte(`{"model":"m","messages":[
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"img1"}},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"img2"}}
		]}]}
	]}`)

	out, err := replaceImageBlocksWithText(rawJSON, visionDesc)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Tool_result content should have exactly 1 item: the description
	content := gjson.GetBytes(out, "messages.0.content.0.content")
	if !content.IsArray() {
		t.Fatal("tool_result content should be array")
	}
	items := content.Array()
	if len(items) != 1 {
		t.Fatalf("expected 1 content item in tool_result, got %d", len(items))
	}
	if items[0].Get("type").String() != "text" || items[0].Get("text").String() != expectedText {
		t.Fatalf("expected description text, got type=%s text=%q", items[0].Get("type").String(), items[0].Get("text").String())
	}
}

func TestReplaceImageBlocksWithTextTopLevelAndToolResult(t *testing.T) {
	const visionDesc = "Description."

	rawJSON := []byte(`{"model":"m","messages":[
		{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"top"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"inner"}}
		]}]}
	]}`)

	out, err := replaceImageBlocksWithText(rawJSON, visionDesc)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Top-level image should be replaced with description
	msg0 := gjson.GetBytes(out, "messages.0.content")
	if !msg0.IsArray() || len(msg0.Array()) != 1 {
		t.Fatalf("msg0 should have 1 content item")
	}
	if msg0.Array()[0].Get("type").String() != "text" {
		t.Fatalf("msg0 item should be text")
	}

	// Tool_result image should be dropped (description already used)
	msg1Content := gjson.GetBytes(out, "messages.1.content.0.content")
	if !msg1Content.IsArray() {
		t.Fatal("msg1 tool_result content should be array")
	}
	if len(msg1Content.Array()) != 0 {
		t.Fatalf("msg1 tool_result should have 0 content items (image dropped), got %d", len(msg1Content.Array()))
	}
}

// countImageBlocksRecursive recursively counts image blocks in a content array.
func countImageBlocksRecursive(content gjson.Result) int {
	if !content.Exists() || !content.IsArray() {
		return 0
	}
	count := 0
	for _, block := range content.Array() {
		if block.Get("type").String() == "image" {
			count++
		}
		if block.Get("type").String() == "tool_result" {
			count += countImageBlocksRecursive(block.Get("content"))
		}
	}
	return count
}

func TestRebuildToolResultWithContent(t *testing.T) {
	rawJSON := `{"type":"tool_result","tool_use_id":"x","content":[{"type":"text","text":"x"}]}`
	block := gjson.Parse(rawJSON)
	newContent := []any{map[string]any{"type": "text", "text": "y"}}

	rebuilt := rebuildToolResultWithContent(block, newContent)
	b, _ := json.Marshal(rebuilt)
	var m map[string]any
	json.Unmarshal(b, &m)
	if m["tool_use_id"] != "x" {
		t.Fatalf("tool_use_id should be x, got %v", m["tool_use_id"])
	}
	if m["type"] != "tool_result" {
		t.Fatalf("type should be tool_result, got %v", m["type"])
	}
	c, ok := m["content"].([]any)
	if !ok || len(c) != 1 {
		t.Fatalf("content should have 1 item, got %d", len(c))
	}
}
