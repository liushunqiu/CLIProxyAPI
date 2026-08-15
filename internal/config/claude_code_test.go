package config

import "testing"

func TestParseConfigBytesClaudeCodeModelListCloaking(t *testing.T) {
	tests := []struct {
		name string
		yaml string
		want bool
	}{
		{
			name: "defaults to enabled cloaking",
			yaml: "port: 8317\n",
			want: false,
		},
		{
			name: "disables model list cloaking",
			yaml: "claude-code:\n  disable-cloaking-model-list: true\n",
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, errParse := ParseConfigBytes([]byte(tt.yaml))
			if errParse != nil {
				t.Fatalf("ParseConfigBytes() error = %v", errParse)
			}
			if got := cfg.ClaudeCode.DisableCloakingModelList; got != tt.want {
				t.Fatalf("DisableCloakingModelList = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestParseConfigBytesClaudeCodeVisionSplit(t *testing.T) {
	tests := []struct {
		name       string
		yaml       string
		wantModel  string
		wantModels []string
	}{
		{
			name:      "defaults to empty vision model",
			yaml:      "port: 8317\n",
			wantModel: "",
		},
		{
			name: "parses vision model and rejecting list",
			yaml: `claude-code:
  vision-split:
    vision-model: "grok-4.6"
    image-rejecting-models:
      - "deepseek-v4-flash"
      - "deepseek-v4-flash:free"
`,
			wantModel:  "grok-4.6",
			wantModels: []string{"deepseek-v4-flash", "deepseek-v4-flash:free"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, errParse := ParseConfigBytes([]byte(tt.yaml))
			if errParse != nil {
				t.Fatalf("ParseConfigBytes() error = %v", errParse)
			}
			if got := cfg.ClaudeCode.VisionSplit.VisionModel; got != tt.wantModel {
				t.Fatalf("VisionModel = %q, want %q", got, tt.wantModel)
			}
			if len(cfg.ClaudeCode.VisionSplit.ImageRejectingModels) != len(tt.wantModels) {
				t.Fatalf("ImageRejectingModels = %v, want %v", cfg.ClaudeCode.VisionSplit.ImageRejectingModels, tt.wantModels)
			}
			for i := range tt.wantModels {
				if cfg.ClaudeCode.VisionSplit.ImageRejectingModels[i] != tt.wantModels[i] {
					t.Fatalf("ImageRejectingModels[%d] = %q, want %q", i, cfg.ClaudeCode.VisionSplit.ImageRejectingModels[i], tt.wantModels[i])
				}
			}
		})
	}
}
