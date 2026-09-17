//! Generation backend: one thread owns the `Qwen38` engine and the tokenizer, and
//! serves up to `MAX_BATCH` requests concurrently.
//!
//! The Metal engine is not shareable across threads, and it must be loaded once
//! (~10 s for the 4-bit weights) rather than per request, so the model lives on a
//! dedicated thread and every HTTP request becomes a job.
//!
//! Requests are not queued behind each other.  Up to `MAX_BATCH` of them hold live
//! state at once, in one sequence slot each, and every round advances all active
//! slots by one token with a SINGLE `forward_rows` pass - one read of the 14.4 GB
//! of weights for the whole batch instead of one per request.  A single request
//! therefore costs what it always did, and sixteen concurrent requests cost
//! roughly one sweep per round between them rather than sixteen.
//!
//! Speculative decoding is off on this path: the MTP draft head still keeps a
//! single sequence's state, so the batch runs plain decode.

use anyhow::Result;
use qw_engine::tokenizer::{Message, Tokenizer};
use qw_model::runner::Qwen38;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// How many prompt tokens one pass may carry.  The projection kernel computes
/// four rows from a single read of the weights, so a pass that feeds it one token
/// throws three quarters of the work away - and since the client is waiting for its
/// first token, that waste is exactly what it experiences as a hang.  It matches
/// the convolution ring, which holds `conv_k` history rows plus this many.
const PREFILL_CHUNK: usize = qw_model::runner::TILE;

/// A prefix cache that needs no storage of its own.
///
/// Two properties of the engine make this possible.  The KV cache is never cleared -
/// `reset_seq` rewinds only the delta-net state and the convolution window - and
/// positions are written once and read only up to the current position.  So if a new
/// prompt begins with exactly the token sequence a slot has already consumed, that
/// slot is *already* in the state the prefix would have produced, and the prefix can
/// simply not be recomputed.  No snapshot, no copy, no eviction policy: either the
/// slot's state matches the prefix or the slot is rewound as before.
///
/// This is the shape agent frameworks produce, because every turn re-sends the whole
/// conversation, so the expensive part of the prompt has usually been seen already.
/// What a slot can skip for a given prompt, and where that comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reuse {
    /// Nothing reusable; the slot has to be rewound and the prompt run in full.
    None,
    /// The live state already represents this prefix, so nothing has to be copied.
    Live(usize),
    /// A copy taken at the end of an earlier prefill; it has to be restored first.
    Boundary(usize),
}

#[derive(Default)]
struct PrefixCache {
    /// Per slot, the tokens its state currently represents: the prompt tokens it has
    /// actually consumed followed by everything it generated.
    hist: Vec<Option<Vec<u32>>>,
    /// Per slot, the prompt whose end-of-prefill snapshot sits in the runner's copy.
    boundary: Vec<Option<Vec<u32>>>,
    lookups: usize,
    hits: usize,
    reused: usize,
    prompted: usize,
}

impl PrefixCache {
    fn new(batch: usize) -> Self {
        Self {
            hist: (0..batch).map(|_| None).collect(),
            boundary: (0..batch).map(|_| None).collect(),
            ..Default::default()
        }
    }

    /// The longest prefix this slot can skip, which is all of it or none of it: a
    /// partial match is useless because the recurrence cannot be rewound to a middle
    /// position, so only an exact match of a recorded history helps.
    ///
    /// The live state is preferred because it reaches further - it includes whatever
    /// the slot generated after the prompt - while the snapshot stops at the prompt.
    fn lookup(&self, slot: usize, ids: &[u32]) -> Reuse {
        if let Some(n) = is_prefix(&self.hist[slot], ids) {
            return Reuse::Live(n);
        }
        if let Some(n) = is_prefix(&self.boundary[slot], ids) {
            return Reuse::Boundary(n);
        }
        Reuse::None
    }

    fn record_boundary(&mut self, slot: usize, seq: Vec<u32>) {
        self.boundary[slot] = Some(seq);
    }

    fn record(&mut self, slot: usize, seq: Vec<u32>) {
        self.hist[slot] = Some(seq);
    }

    fn forget(&mut self, slot: usize) {
        self.hist[slot] = None;
    }

    /// The number worth quoting is the token-weighted one: a request that reuses nine
    /// tenths of its prompt is a hit, but a request-count hit rate would hide that
    /// almost all of the work was done twice.
    fn report(&self, slot: usize, skip: usize, ids_len: usize, src: &str) {
        let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
        tracing::info!(
            "slot {slot}: prefix cache {}{src} - skipped {skip} of {ids_len} prompt tokens ({:.0}%), cumulative KV hit rate {:.1}% of {} tokens over {} request(s), {} reused",
            if skip > 0 { "HIT" } else { "miss" },
            pct(skip, ids_len),
            pct(self.reused, self.prompted),
            self.prompted,
            self.lookups,
            self.hits,
        );
    }
}

/// `Some(n)` when `ids` starts with the whole of `hist`, which is the only match a
/// running recurrence can be resumed from.
fn is_prefix(hist: &Option<Vec<u32>>, ids: &[u32]) -> Option<usize> {
    let h = hist.as_ref()?;
    if !h.is_empty() && h.len() <= ids.len() && ids[..h.len()] == h[..] {
        Some(h.len())
    } else {
        None
    }
}

/// How many requests the engine will hold in flight at once.  It matches the
/// widest row tile the model can carry in one weight sweep.
pub const MAX_BATCH: usize = 16;

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
    pub pieces: UnboundedSender<Result<EngineEvent, String>>,
}

/// One request holding a sequence slot.
struct Active {
    job: Job,
    /// Prompt tokens, fed one per round until `pf` reaches the end.
    ids: Vec<u32>,
    pf: usize,
    /// Next position to feed during decoding.
    pos: usize,
    /// The token produced for this slot by the last pass.  It is emitted and then
    /// fed back in on the following round, which is where the causal model needs
    /// it.
    feed: u32,
    /// Set once the prompt is fully in, which is what makes `feed` meaningful.
    ready: bool,
    /// The client went away, so this slot is nobody's work any more.
    dead: bool,
    all: Vec<u32>,
    sent_len: usize,
    emitted: usize,
    max_tokens: usize,
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
        // QW_BATCH=<n> narrows the width, which is worth doing on a small machine:
        // every slot carries a full KV cache, so sixteen of them cost several GB
        // more than one.
        let batch = std::env::var("QW_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|b| *b > 0 && *b <= MAX_BATCH)
            .unwrap_or(MAX_BATCH);
        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name("qw-engine".to_string())
            .spawn(move || {
                let loaded = (|| -> Result<(Qwen38, Tokenizer)> {
                    let model = Qwen38::load_batch(&model_dir, max_t, batch)?;
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
                serve(&mut model, &tok, rx, max_t, batch);
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

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Decode the running prefix and send whatever is new.
///
/// Decoding one id at a time mangles multi-byte characters that straddle two
/// tokens, so the whole prefix is decoded and only the new suffix is emitted.
/// Until the slot retires, a trailing U+FFFD is held back: the tokenizer produces
/// one when a multi-byte character is split across a token boundary, and the rest
/// of it only arrives with the next token.
fn emit(tok: &Tokenizer, a: &mut Active, flush: bool) {
    let full = tok.decode(&a.all, true).unwrap_or_default();
    if full.len() <= a.sent_len || !full.is_char_boundary(a.sent_len) {
        return;
    }
    let piece = &full[a.sent_len..];
    let cut = if flush {
        piece.len()
    } else {
        piece.trim_end_matches('\u{FFFD}').len()
    };
    if cut > 0 {
        a.sent_len += cut;
        let _ = a
            .job
            .pieces
            .send(Ok(EngineEvent::Piece(piece[..cut].to_string())));
    }
}

/// Turn a queued job into a live slot, or report why it cannot be served.
fn prepare(
    model: &mut Qwen38,
    tok: &Tokenizer,
    job: Job,
    max_t: usize,
    slot: usize,
    cache: &mut PrefixCache,
    snapshot: bool,
) -> Option<Active> {
    let text = match &job.prompt {
        Prompt::Text(s) => s.clone(),
        Prompt::Chat(msgs) => tok.apply_chat_template(msgs),
    };
    let ids = match tok.encode(&text, false) {
        Ok(v) => v,
        Err(e) => {
            let _ = job.pieces.send(Err(e.to_string()));
            return None;
        }
    };
    // Every row the decode touches has to exist: the KV cache and the delta-net
    // state were sized for `max_t` positions.  A prompt that already fills the
    // context, or a completion budget that would run past it, is rejected here
    // rather than silently walking off the end of the buffers.
    if ids.is_empty() || ids.len() >= max_t {
        tracing::warn!(
            "slot {slot}: rejected a {} token prompt, the context is {max_t}; raise --max-ctx",
            ids.len()
        );
        let _ = job.pieces.send(Err(format!(
            "prompt is {} tokens but the context is {max_t}; restart with a larger --max-ctx",
            ids.len()
        )));
        return None;
    }
    // If this prompt starts with exactly what the slot has already consumed, the
    // slot is already in the right state and the whole prefix can be skipped.  Only
    // a complete match can be used: the recurrent state corresponds to the end of
    // what the slot consumed and cannot be rewound to a position in the middle.
    let reuse = if snapshot {
        cache.lookup(slot, &ids)
    } else {
        match is_prefix(&cache.hist[slot], &ids) {
            Some(n) => Reuse::Live(n),
            None => Reuse::None,
        }
    };
    let skip = match reuse {
        Reuse::None => 0,
        Reuse::Live(n) | Reuse::Boundary(n) => n,
    };
    cache.lookups += 1;
    cache.prompted += ids.len();
    cache.reused += skip;
    let src = match reuse {
        Reuse::None => {
            // The slot may have served an unrelated request.  Its recurrent state has
            // to go back to the initial condition first.
            if let Err(e) = model.reset_seq(slot) {
                let _ = job.pieces.send(Err(e.to_string()));
                return None;
            }
            cache.forget(slot);
            ""
        }
        Reuse::Live(_) => " (live state)",
        Reuse::Boundary(_) => {
            cache.hits += 1;
            // The slot has moved on since that prefill, so put the recurrent state and
            // the convolution window back the way they were.  The copy is not
            // consumed, so other requests can still resume from the same boundary.
            if let Err(e) = model.load_prefix(slot) {
                let _ = job.pieces.send(Err(e.to_string()));
                return None;
            }
            " (saved boundary)"
        }
    };
    if skip > 0 && src == " (live state)" {
        cache.hits += 1;
    }
    cache.report(slot, skip, ids.len(), src);
    let max_tokens = job.max_tokens.min(max_t - ids.len());
    if max_tokens < job.max_tokens {
        // Worth saying out loud: an agent framework that asks for 32000 tokens
        // against a small context gets silently truncated mid-task, which looks
        // like the model giving up rather than a setting.
        tracing::info!(
            "slot {slot}: prompt is {} tokens, so max_tokens is cut from {} to {}",
            ids.len(),
            job.max_tokens,
            max_tokens
        );
    } else {
        tracing::info!(
            "slot {slot}: prompt {} tokens, up to {max_tokens} to generate",
            ids.len()
        );
    }
    let _ = job.pieces.send(Ok(EngineEvent::Prompt(ids.len())));
    Some(Active {
        job,
        ids,
        pf: skip,
        pos: skip,
        feed: 0,
        ready: false,
        dead: false,
        all: Vec::new(),
        sent_len: 0,
        emitted: 0,
        max_tokens,
    })
}

fn serve(model: &mut Qwen38, tok: &Tokenizer, rx: Receiver<Job>, max_t: usize, batch: usize) {
    // Prefill a chunk per pass by default.  Paired on one machine in one thermal
    // state, for a 1314-token prompt: chunk=1 gave a TTFT of 130.08 s and chunk=4
    // gave 36.10 s, a 3.60x speedup, with byte-identical output.  The reason is
    // bandwidth: a one-row pass still has to stream all 14.4 GB of weights, so
    // feeding it a single token pays that whole read for one token.  QW_PREFILL_CHUNK
    // overrides it, and 1 restores the old behaviour for comparison.
    let chunk_cap = std::env::var("QW_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|c| *c > 0)
        .unwrap_or(PREFILL_CHUNK)
        .min(PREFILL_CHUNK);
    tracing::info!("prefill chunk: {chunk_cap} token(s) per pass");
    // Taking a copy of the recurrent state at the end of every prefill is what lets a
    // byte-identical retry hit: after decoding, the slot's state sits past the prompt
    // boundary and a recurrence cannot be rewound, so without the copy a retry has to
    // run the whole prompt again.  The mechanism is implemented and builds clean, but
    // it is NOT verified end to end - the cold-prefill time on this machine swings
    // between 230 s and over 600 s for the same 6866-token prompt, and the run that was
    // meant to check it did not finish inside the round.  An unverified optimisation
    // must not be the default, so it is opt-in until a paired measurement says it is
    // right; QW_PREFIX_SNAPSHOT=1 turns it on.
    let snapshot = std::env::var("QW_PREFIX_SNAPSHOT").is_ok();
    tracing::info!(
        "prefix cache: saved-boundary snapshots {}",
        if snapshot { "on" } else { "off" }
    );
    let mut slots: Vec<Option<Active>> = (0..batch).map(|_| None).collect();
    let mut cache = PrefixCache::new(batch);
    let mut waiting: VecDeque<Job> = VecDeque::new();
    loop {
        // ---- admit ----
        // Block only when there is nothing at all to do.  Otherwise take whatever
        // has already arrived, so a burst is batched instead of serialised.
        if waiting.is_empty() && slots.iter().all(|s| s.is_none()) {
            match rx.recv() {
                Ok(j) => waiting.push_back(j),
                // Every sender is gone, so the server is shutting down.
                Err(_) => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            waiting.push_back(j);
        }
        for (slot, entry) in slots.iter_mut().enumerate() {
            if entry.is_some() {
                continue;
            }
            let Some(job) = waiting.pop_front() else {
                break;
            };
            if let Some(a) = prepare(model, tok, job, max_t, slot, &mut cache, snapshot) {
                *entry = Some(a);
            }
        }

        // ---- one batched pass over every slot that can take a step ----
        // One pass carries at most MAX_BATCH rows.  A decoding slot needs exactly
        // one of them; a prefilling slot wants a full chunk.  Reserve the decoders
        // first, then split what is left between the prefillers.
        let want_one = |a: &Active| {
            a.pf >= a.ids.len()
                && a.ready
                && a.emitted < a.max_tokens
                && !tok.is_eos(a.feed)
                && !a.job.pieces.is_closed()
        };
        let decoding = slots
            .iter()
            .filter(|e| e.as_ref().is_some_and(want_one))
            .count();
        let prefilling = slots
            .iter()
            .filter(|e| {
                e.as_ref()
                    .is_some_and(|a| a.pf < a.ids.len() && !a.job.pieces.is_closed())
            })
            .count();
        let room = MAX_BATCH.saturating_sub(decoding);
        let share = room.checked_div(prefilling).unwrap_or(0);
        let chunk = if prefilling == 0 {
            0
        } else {
            chunk_cap.min(share).max(1)
        };
        let mut rows: Vec<(usize, usize)> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        let mut row_slot: Vec<usize> = Vec::new();
        for (slot, entry) in slots.iter_mut().enumerate() {
            let Some(a) = entry.as_mut() else { continue };
            // The client can vanish: a cancelled request in a TUI, a killed
            // curl, a framework that gave up and moved on.  Without this check
            // nothing would ever stop the slot - the only other stop conditions
            // are running out of max_tokens or emitting EOS, and OpenCode asks
            // for 32000 tokens, so an abandoned request would keep generating for
            // the better part of twenty minutes, sharing every round with real
            // work.  Sixteen abandoned requests would leave the engine entirely
            // occupied by nobody's work, which is indistinguishable from a hang.
            if a.job.pieces.is_closed() {
                if !a.dead {
                    tracing::info!(
                        "slot {slot}: client went away, releasing ({} of {} tokens emitted)",
                        a.emitted,
                        a.max_tokens
                    );
                }
                a.dead = true;
                continue;
            }
            if a.pf < a.ids.len() {
                // Prefill a chunk, not a token.  The rows are consecutive
                // positions of this one sequence and are appended in order, so
                // each one sees the previous ones in the KV cache and in the
                // convolution ring exactly as it would have if it had been fed on
                // its own pass.
                let take = (a.ids.len() - a.pf).min(chunk);
                for k in 0..take {
                    rows.push((slot, a.pf + k));
                    toks.push(a.ids[a.pf + k]);
                    row_slot.push(slot);
                }
                a.pf += take;
                continue;
            }
            if a.ready && a.emitted < a.max_tokens && !tok.is_eos(a.feed) {
                rows.push((slot, a.pos));
                toks.push(a.feed);
                row_slot.push(slot);
                a.pos += 1;
                a.all.push(a.feed);
                a.emitted += 1;
            }
        }
        if rows.is_empty() {
            continue;
        }
        let mut failed = match model
            .set_tokens(&toks)
            .and_then(|_| model.forward_rows(&rows))
        {
            Ok(()) => None,
            Err(e) => Some(e.to_string()),
        };
        for (i, &slot) in row_slot.iter().enumerate() {
            let Some(a) = slots[slot].as_mut() else {
                continue;
            };
            if let Some(msg) = &failed {
                let _ = a.job.pieces.send(Err(msg.clone()));
                a.emitted = a.max_tokens;
                continue;
            }
            let next = argmax(&model.logits_row(i));
            if a.pf >= a.ids.len() {
                if !a.ready && snapshot {
                    // The recurrent state sits exactly at the end of the prompt right
                    // now, which is the position a later request will want to resume
                    // from.  Copy it before decoding moves it on, or the only way back
                    // is to run the whole prompt again.
                    if let Err(e) = model.save_prefix(slot) {
                        failed = Some(e.to_string());
                    } else {
                        cache.record_boundary(slot, a.ids.clone());
                    }
                }
                a.ready = true;
            }
            if a.ready {
                a.feed = next;
            }
            emit(tok, a, false);
        }

        // ---- retire ----
        for (slot, entry) in slots.iter_mut().enumerate() {
            let done = match entry.as_ref() {
                Some(a) => a.dead || (a.ready && (a.emitted >= a.max_tokens || tok.is_eos(a.feed))),
                None => false,
            };
            if done {
                if let Some(a) = entry.as_mut() {
                    emit(tok, a, true);
                    // Record exactly what this slot consumed: the prompt tokens it
                    // actually prefilled plus what it generated.  Recording the whole
                    // prompt would be wrong for a request abandoned part way through
                    // prefill, because the state would not correspond to it and the
                    // next request would skip work it still needs.
                    let mut seq: Vec<u32> = a.ids[..a.pf.min(a.ids.len())].to_vec();
                    seq.extend_from_slice(&a.all);
                    cache.record(slot, seq);
                }
                *entry = None;
            }
        }
    }
}
