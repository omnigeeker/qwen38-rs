//! Tensor naming for the MLX-converted checkpoint layout.
//!
//! Observed key space of `mlx-community/Qwen3.8-27B-4bit`:
//! ```text
//! language_model.model.embed_tokens.{weight,scales,biases}
//! language_model.model.layers.{i}.input_layernorm.weight
//! language_model.model.layers.{i}.post_attention_layernorm.weight
//! language_model.model.layers.{i}.mlp.{gate,up,down}_proj.{weight,scales,biases}
//! language_model.model.layers.{i}.linear_attn.in_proj_qkv.{weight,scales,biases}
//! language_model.model.layers.{i}.linear_attn.in_proj_z|b|a.{weight,scales,biases}
//! language_model.model.layers.{i}.linear_attn.out_proj.{weight,scales,biases}
//! language_model.model.layers.{i}.linear_attn.conv1d.weight
//! language_model.model.layers.{i}.linear_attn.norm.weight
//! language_model.model.layers.{i}.linear_attn.{A_log,dt_bias}
//! language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
//! language_model.model.layers.{i}.self_attn.{q,k}_norm.weight
//! language_model.model.norm.weight
//! language_model.lm_head.{weight,scales,biases}
//! ```
//! MTP tensors (`mtp.*`) are absent from the MLX 4-bit export; they are pulled
//! from the bf16 release and quantised locally (see `qw-engine::mtp`).

/// Accessor for the checkpoint's tensor names.
#[derive(Debug, Clone)]
pub struct WeightLayout {
    pub prefix: String,
}

impl Default for WeightLayout {
    fn default() -> Self {
        Self {
            prefix: "language_model".to_string(),
        }
    }
}

impl WeightLayout {
    pub fn embed_tokens(&self) -> String {
        format!("{}.model.embed_tokens.weight", self.prefix)
    }
    pub fn final_norm(&self) -> String {
        format!("{}.model.norm.weight", self.prefix)
    }
    pub fn lm_head(&self) -> String {
        format!("{}.lm_head.weight", self.prefix)
    }
    pub fn layer(&self, i: usize) -> String {
        format!("{}.model.layers.{}", self.prefix, i)
    }
    pub fn input_norm(&self, i: usize) -> String {
        format!("{}.input_layernorm.weight", self.layer(i))
    }
    pub fn post_attn_norm(&self, i: usize) -> String {
        format!("{}.post_attention_layernorm.weight", self.layer(i))
    }
    // ---- SwiGLU MLP ----
    pub fn mlp_gate(&self, i: usize) -> String {
        format!("{}.mlp.gate_proj.weight", self.layer(i))
    }
    pub fn mlp_up(&self, i: usize) -> String {
        format!("{}.mlp.up_proj.weight", self.layer(i))
    }
    pub fn mlp_down(&self, i: usize) -> String {
        format!("{}.mlp.down_proj.weight", self.layer(i))
    }
    // ---- gated delta net ----
    pub fn gdn(&self, i: usize) -> String {
        format!("{}.linear_attn", self.layer(i))
    }
    pub fn gdn_in_qkv(&self, i: usize) -> String {
        format!("{}.in_proj_qkv.weight", self.gdn(i))
    }
    pub fn gdn_in_z(&self, i: usize) -> String {
        format!("{}.in_proj_z.weight", self.gdn(i))
    }
    pub fn gdn_in_b(&self, i: usize) -> String {
        format!("{}.in_proj_b.weight", self.gdn(i))
    }
    pub fn gdn_in_a(&self, i: usize) -> String {
        format!("{}.in_proj_a.weight", self.gdn(i))
    }
    pub fn gdn_out(&self, i: usize) -> String {
        format!("{}.out_proj.weight", self.gdn(i))
    }
    pub fn gdn_conv(&self, i: usize) -> String {
        format!("{}.conv1d.weight", self.gdn(i))
    }
    pub fn gdn_norm(&self, i: usize) -> String {
        format!("{}.norm.weight", self.gdn(i))
    }
    pub fn gdn_a_log(&self, i: usize) -> String {
        format!("{}.A_log", self.gdn(i))
    }
    pub fn gdn_dt_bias(&self, i: usize) -> String {
        format!("{}.dt_bias", self.gdn(i))
    }
    // ---- full attention ----
    pub fn attn(&self, i: usize) -> String {
        format!("{}.self_attn", self.layer(i))
    }
    pub fn attn_q(&self, i: usize) -> String {
        format!("{}.q_proj.weight", self.attn(i))
    }
    pub fn attn_k(&self, i: usize) -> String {
        format!("{}.k_proj.weight", self.attn(i))
    }
    pub fn attn_v(&self, i: usize) -> String {
        format!("{}.v_proj.weight", self.attn(i))
    }
    pub fn attn_o(&self, i: usize) -> String {
        format!("{}.o_proj.weight", self.attn(i))
    }
    pub fn attn_q_norm(&self, i: usize) -> String {
        format!("{}.q_norm.weight", self.attn(i))
    }
    pub fn attn_k_norm(&self, i: usize) -> String {
        format!("{}.k_norm.weight", self.attn(i))
    }
    // ---- MTP block (bf16 in the official release) ----
    pub fn mtp_pre_norm_embedding(&self) -> String {
        "mtp.pre_fc_norm_embedding.weight".to_string()
    }
    pub fn mtp_pre_norm_hidden(&self) -> String {
        "mtp.pre_fc_norm_hidden.weight".to_string()
    }
    pub fn mtp_fc(&self) -> String {
        "mtp.fc.weight".to_string()
    }
    pub fn mtp_norm(&self) -> String {
        "mtp.norm.weight".to_string()
    }
    pub fn mtp_layer(&self, i: usize) -> String {
        format!("mtp.layers.{i}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_checkpoint_layout() {
        let l = WeightLayout::default();
        assert_eq!(l.embed_tokens(), "language_model.model.embed_tokens.weight");
        assert_eq!(l.lm_head(), "language_model.lm_head.weight");
        assert_eq!(l.final_norm(), "language_model.model.norm.weight");
        assert_eq!(
            l.input_norm(7),
            "language_model.model.layers.7.input_layernorm.weight"
        );
        assert_eq!(
            l.gdn_in_qkv(0),
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(
            l.attn_q(3),
            "language_model.model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(l.mtp_fc(), "mtp.fc.weight");
    }
}
