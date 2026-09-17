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
                    if let Err(e) =
                        run_job(&mut model, &tok, &text, job.max_tokens, max_t, &job.pieces)
                    {
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
    max_t: usize,
    pieces: &UnboundedSender<Result<EngineEvent, String>>,
) -> Result<()> {
    model.reset();
    let ids = tok.encode(text, false)?;
    // Every row the decode touches has to exist: the KV cache and the delta-net
    // state were sized for `max_t` positions.  A prompt that already fills the
    // context, or a completion budget that would run past it, is rejected here
    // rather than silently walking off the end of the buffers.
    if ids.len() >= max_t {
        anyhow::bail!(
            "prompt is {} tokens but the context is {max_t}; restart with a larger --max-ctx",
            ids.len()
        );
    }
    let max_tokens = max_tokens.min(max_t - ids.len());
    let _ = pieces.send(Ok(EngineEvent::Prompt(ids.len())));
    // When the MTP head is present the drafts come from its own attention state,
    // so the prefill has to warm that state alongside the target's: feed the head
    // each token's predecessor exactly as the CLI does, or every draft is garbage
    // and speculation only costs time.
    let spec = model.has_mtp() && std::env::var("QW_NO_SPEC").is_err();
    if spec {
        model.enable_spec_snap();
    }
    for (p, id) in ids.iter().enumerate() {
        model.set_token(*id)?;
        model.forward(p)?;
        if spec && p + 1 < ids.len() {
            model.mtp_step(ids[p + 1], p + 1, false)?;
        }
    }
    // Decoding one id at a time mangles multi-byte characters that straddle two
    // tokens, so decode the running prefix and emit only what is new.
    let mut all: Vec<u32> = Vec::new();
    let mut sent_len = 0usize;
    // Speculative decoding when the MTP head is present: the same step the CLI
    // benchmark uses, so the endpoint serves at the tuned throughput.
    let mut pos = ids.len();
    let mut next = model.argmax();
    let mut emitted = 0usize;
    while emitted < max_tokens && !tok.is_eos(next) {
        let mut step: Vec<u32> = Vec::new();
        if spec {
            let (p, n, _, _) = model.spec_step(pos, next, &mut step)?;
            pos = p;
            next = n;
        } else {
            step.push(next);
            model.set_token(next)?;
            model.forward(pos)?;
            pos += 1;
            next = model.argmax();
        }
        let mut full_pass = false;
        for t in step {
            if tok.is_eos(t) || emitted >= max_tokens {
                full_pass = true;
                break;
            }
            all.push(t);
            emitted += 1;
        }
        {
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
        }
        if full_pass {
            break;
        }
    }
    Ok(())
}
