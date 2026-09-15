//! Inference engine: tokenisation, sampling, KV/state management, MTP
//! speculative decoding and the request scheduler.

pub mod sampler;
pub mod tokenizer;

pub use sampler::{Sampler, SamplingParams};
pub use tokenizer::Tokenizer;

/// Engine build status — replaced as milestones land.
pub const MILESTONES: &[&str] = &[
    "M0 scaffold + Metal runtime JIT (done)",
    "M1 single-token forward parity with mlx-lm",
    "M2 prefill + KV cache + greedy parity",
    "M3 4-bit GEMV perf: >=450 GB/s effective",
    "M4 MTP speculative decoding: >=40 tok/s single stream",
    "M5 OpenAI + Anthropic endpoints with streaming",
    "M6 continuous batching",
];
