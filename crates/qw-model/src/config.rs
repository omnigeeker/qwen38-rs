//! `config.json` parsing for Qwen3.8-27B (text tower only; the vision tower is
//! intentionally out of scope — this engine serves a text endpoint).

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::path::Path;

fn default_rms_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    // linear-attention (gated delta net) block
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    #[serde(default = "one")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "four")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParams>,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub num_experts: usize,
}

fn one() -> usize {
    1
}
fn four() -> usize {
    4
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(100_000.0)
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .unwrap_or(0.25)
    }

    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f64) * self.partial_rotary_factor()) as usize
    }

    /// `layer_types` from the checkpoint wins; otherwise derive from the interval.
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        if !self.layer_types.is_empty() && layer_idx < self.layer_types.len() {
            return self.layer_types[layer_idx] == "linear_attention";
        }
        (layer_idx + 1) % self.full_attention_interval != 0
    }

    pub fn num_linear_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|i| self.is_linear_layer(*i))
            .count()
    }

    pub fn num_full_layers(&self) -> usize {
        self.num_hidden_layers - self.num_linear_layers()
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp_num_hidden_layers > 0
    }

    /// Per-token KV cache elements for one full-attention layer (fp16 elements).
    pub fn kv_elems_per_token(&self, layer_idx: usize) -> usize {
        if self.is_linear_layer(layer_idx) {
            0
        } else {
            2 * self.num_key_value_heads * self.head_dim
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architectures: Option<Vec<String>>,
    pub model_type: Option<String>,
    pub text_config: TextConfig,
}

impl ModelConfig {
    pub fn from_path(p: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(p).map_err(|e| anyhow!("read {}: {e}", p.display()))?;
        let cfg: ModelConfig = serde_json::from_str(&txt)?;
        Ok(cfg)
    }

    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::from_path(&model_dir.join("config.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_shipped_config_shape() {
        // Synthetic config with the same shape as the checkpoint's.
        let json = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "hidden_size": 5120, "intermediate_size": 17408,
            "num_hidden_layers": 64, "num_attention_heads": 24,
            "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-06,
            "tie_word_embeddings": false, "attention_bias": false,
            "linear_num_value_heads": 48, "linear_num_key_heads": 16,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"]
          }
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 64);
        assert!(t.has_mtp());
        assert_eq!(t.rotary_dim(), 64);
        assert_eq!(t.rope_theta(), 10_000_000.0);
        // every 4th layer (index 3, 7, ...) is full attention
        assert!(!t.is_linear_layer(3));
        assert!(t.is_linear_layer(2));
        assert_eq!(t.kv_elems_per_token(3), 2 * 4 * 256);
        assert_eq!(t.kv_elems_per_token(2), 0);
        // with a 4-entry layer_types the remaining layers fall back to the interval
        assert!(!t.is_linear_layer(7));
    }

    #[test]
    fn loads_real_model_config_if_present() {
        let dir = std::path::Path::new("../../models/Qwen3.8-27B-4bit");
        if !dir.join("config.json").exists() {
            eprintln!("skipping: model not downloaded");
            return;
        }
        let cfg = ModelConfig::load(dir).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 64);
        assert_eq!(t.num_linear_layers(), 48);
        assert_eq!(t.num_full_layers(), 16);
        assert!(t.has_mtp());
        assert_eq!(t.vocab_size, 248320);
    }
}
