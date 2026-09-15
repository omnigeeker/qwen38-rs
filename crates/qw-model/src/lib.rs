//! Qwen3.8-27B model definition.
//!
//! Text-only path of `Qwen3_5ForConditionalGeneration`:
//!   * 64 decoder layers, `full_attention_interval = 4`
//!     (48 gated-delta-net "linear attention" layers + 16 GQA full-attention layers)
//!   * MLP = SwiGLU, RMSNorm pre-norm on both sub-layers
//!   * 1 MTP layer (`mtp.*`) used for speculative decoding
//!
//! The numerical contract (verified against `mlx-lm`'s `qwen3_5.py`, which is
//! the reference oracle) is:
//!   * RMSNorm weights are stored **zero-centred** in this checkpoint family and
//!     must be shifted by +1.0 at load time (see `WeightLayout::norm_shift`).
//!   * `q = inv_scale^2 * rms_norm(q)`, `k = inv_scale * rms_norm(k)` inside the
//!     gated delta net, with `inv_scale = head_k_dim^-0.5`.
//!   * `beta = sigmoid(b)`, `g = exp(-exp(A_log) * softplus(a + dt_bias))`.

pub mod config;
pub mod linear;
pub mod naming;

pub use config::{ModelConfig, TextConfig};
pub use linear::QLinear;
pub use naming::WeightLayout;
