//! Generation backend: one thread owns the `Qwen38` engine and the tokenizer,
//! and serves requests off a channel.
//!
//! The Metal engine is not shareable across threads, and it must be loaded once
//! (~10 s for the 4-bit weights) rather than per request.  So the model lives on
//! a dedicated thread and every HTTP request becomes a job on a `mpsc` queue;
//! generated text comes back as a stream of pieces.

use anyhow::Result;
use qw_engine::tokenizer::{Message, Tokenizer};
use qw_model::runner::Qwen38;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// A prompt either arrives already rendered (OpenAI `/v1/completions`) or as
/// chat messages that the tokenizer's template renders (`/v1/chat/completions`,
/// `/v1/messages`).
pub enum Prompt {
    Text(String),
    Chat(Vec<Message>),
}

/// What the engine thread sends back: the prompt length first (so usage can
/// report it), then one event per generated piece.
pub enum EngineEvent {
    Prompt(usize),
    Piece(String),
}

pub struct Job {
    pub prompt: Prompt,
    pub max_tokens: usize,
    /// Rows for the key/value and delta-net state.  `max_t` must cover the
    /// prompt plus the completion.
    pub pieces: UnboundedSender<Result<EngineEvent, String>>,
}

/// Handle to the engine thread.  Cheap to clone.
#[derive(Clone)]
pub struct Engine {
    tx: Sender<Job>,
}

impl Engine {
    /// Start the engine thread and block until the weights are loaded, so that
    /// the caller can honestly report readiness.
    pub fn spawn(model_dir: PathBuf, max_t: usize) -> Result<Self> {
        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name("qw-engine".to_string())
            .spawn(move || {
                let loaded = (|| -> Result<(Qwen38, Tokenizer)> {
                    let model = Qwen38::load(&model_dir, max_t)?;
                    let tok = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
                    Ok((model, tok))
                })();
                let (mut model, tok) = match loaded {
                    Ok(v) => {
                        let _ = ready_tx.send(Ok(()));
                        v
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    let text = match &job.prompt {
                        Prompt::Text(s) => s.clone(),
                        Prompt::Chat(msgs) => tok.apply_chat_template(msgs),
                    };
                    if let Err(e) = run_job(&mut model, &tok, &text, job.max_tokens, &job.pieces) {
                        let _ = job.pieces.send(Err(e.to_string()));
                    }
                }
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx }),
            Ok(Err(e)) => anyhow::bail!("engine failed to load: {e}"),
            Err(_) => anyhow::bail!("engine thread died during load"),
        }
    }

    pub fn submit(
        &self,
        prompt: Prompt,
        max_tokens: usize,
    ) -> UnboundedReceiver<Result<EngineEvent, String>> {
        let (pieces, rx) = unbounded_channel();
        let _ = self.tx.send(Job {
            prompt,
            max_tokens,
            pieces,
        });
        rx
    }
}

/// Prompt tokens are pushed through the decode path one at a time (the causal
/// model gives the same answer as a batched prefill), then tokens are emitted
/// until EOS or the budget runs out.
fn run_job(
    model: &mut Qwen38,
    tok: &Tokenizer,
    text: &str,
    max_tokens: usize,
    pieces: &UnboundedSender<Result<EngineEvent, String>>,
) -> Result<()> {
    model.reset();
    let ids = tok.encode(text, false)?;
    let _ = pieces.send(Ok(EngineEvent::Prompt(ids.len())));
    for (p, id) in ids.iter().enumerate() {
        model.set_token(*id)?;
        model.forward(p)?;
    }
    // Decoding one id at a time mangles multi-byte characters that straddle two
    // tokens, so decode the running prefix and emit only what is new.
    let mut all: Vec<u32> = Vec::new();
    let mut sent_len = 0usize;
    for pos in ids.len()..(ids.len() + max_tokens) {
        let next = model.argmax();
        if tok.is_eos(next) {
            break;
        }
        all.push(next);
        let full = tok.decode(&all, true).unwrap_or_default();
        if full.len() > sent_len && full.is_char_boundary(sent_len) {
            // Hold back a trailing U+FFFD: the tokenizer emits a replacement
            // character when a multi-byte character is split across a token
            // boundary, and the remaining bytes only arrive with the next token.
            let piece = &full[sent_len..];
            let cut = piece.trim_end_matches('\u{FFFD}').len();
            if cut > 0 {
                sent_len += cut;
                if pieces
                    .send(Ok(EngineEvent::Piece(piece[..cut].to_string())))
                    .is_err()
                {
                    break;
                }
            }
        }
        model.set_token(next)?;
        model.forward(pos)?;
    }
    Ok(())
}
