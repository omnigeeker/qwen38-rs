//! Protocol layer: OpenAI Chat Completions / Completions and Anthropic
//! Messages, mapped onto one internal request type.
//!
//! The wire types live here so that both protocols can be exercised by tests
//! before the model backend is wired in (`backend` feature).

pub mod anthropic;
pub mod engine;
pub mod openai;
pub mod server;

pub use server::{build_router, serve, AppState};
