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
// A pass carries this many rows of one sequence.  It is the whole point of the
// wide row path: the per-pass cost is a fixed chain of dependent dispatches that
// does not scale with the row count, so 32 rows per pass amortise it eight-fold
// against the old value of TILE.
const PREFILL_CHUNK: usize = qw_model::runner::PASS_ROWS_MAX;
/// Widest prefill pass this engine may ask for.  It is not a tuning knob: the row
/// kernels admit at most `(conv_k + TILE).next_power_of_two() - conv_k` rows of one
/// sequence, which is 4 with the convolution ring at 8, and a wider pass is
/// rejected outright.  Asking for more is therefore clamped rather than honoured,
/// and the clamp is logged so the request is not silently ignored.
const PREFILL_CHUNK_MAX: usize = 32;

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
#[derive(Clone, PartialEq, Eq)]
enum Reuse {
    /// Nothing reusable; the slot has to be rewound and the prompt run in full.
    None,
    /// The live state already represents this prefix, so nothing has to be copied.
    Live(usize),
    /// A copy taken at the end of an earlier prefill; it has to be restored first.
    Boundary(usize),
    /// A prefix persisted by an earlier **process**, with its serialised state.
    Disk(usize, Vec<u8>),
}

/// Largest in-memory boundary snapshot to keep, in MB.  About 0.1 MB a token, so
/// this covers a 40k-token prefix; past it the boundary falls back to the
/// state-and-window copy instead of holding the blob.
const PREFIX_BLOB_MAX_MB: usize = 4096;

/// Distances from the end of a prompt at which a boundary is persisted.
///
/// One boundary at the very end is not enough.  Measured against real OpenCode,
/// two sessions' prompts shared 6566 of 6888 tokens and diverged in the last
/// ~320 - so a boundary at 6884 matches nothing, while one a little further
/// back lands inside the shared region and lets the request resume there.
/// The spacing is deliberately finer near the end, where the variation is.
const PREFIX_LADDER: [usize; 10] = [0, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096];

/// Prefixes on disk, keyed by content rather than by slot, so they outlive the
/// process and are shared by every slot and every session.
mod prefixes {
    use std::io::Read;
    use std::path::PathBuf;

   pub struct DiskPrefix {
        dir: PathBuf,
    }

    fn fnv(ids: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in ids {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    impl DiskPrefix {
        pub fn from_env() -> Option<Self> {
            let dir = PathBuf::from(std::env::var("QW_PREFIX_DISK").ok()?);
            std::fs::create_dir_all(&dir).ok()?;
            Some(Self { dir })
        }

        /// Longest stored prefix that `ids` starts with, plus its blob.
        ///
        /// Reads the header first and only pulls in the blob once the ids match,
        /// because a blob is a few hundred megabytes and the directory holds one
        /// file per distinct prompt.
        pub fn find(&self, ids: &[u32]) -> Option<(usize, Vec<u8>)> {
            let entries = std::fs::read_dir(&self.dir).ok()?;
            let mut best: Option<(usize, Vec<u8>)> = None;
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("pfx") {
                    continue;
                }
                let Some((n, stored, mut f)) = Self::header(&p) else {
                    continue;
                };
                if n == 0 || n > ids.len() || stored[..] != ids[..n] {
                    continue;
                }
                if best.as_ref().is_some_and(|(bn, _)| n <= *bn) {
                    continue;
                }
                let mut blob = Vec::new();
                if f.read_to_end(&mut blob).is_ok() {
                    best = Some((n, blob));
                }
            }
            best
        }

        fn header(p: &std::path::Path) -> Option<(usize, Vec<u32>, std::fs::File)> {
            let mut f = std::fs::File::open(p).ok()?;
            let mut h = [0u8; 12];
            f.read_exact(&mut h).ok()?;
            if &h[0..8] != b"Q38DSK1\0" {
                return None;
            }
            let n = u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize;
            if n == 0 || n > 1 << 22 {
                return None;
            }
            let mut raw = vec![0u8; n * 4];
            f.read_exact(&mut raw).ok()?;
            let ids = raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Some((n, ids, f))
        }

        pub fn store(&self, ids: &[u32], blob: &[u8]) -> std::io::Result<()> {
            let path = self.dir.join(format!("{:016x}-{}.pfx", fnv(ids), ids.len()));
            // The content hash names the file, so an existing path means we
            // already hold exactly this prefix.  Rewriting a 1.4 GB blob of
            // identical bytes on every turn is pure waste.
            if path.exists() {
                return Ok(());
            }
            let mut o = Vec::with_capacity(12 + ids.len() * 4 + blob.len());
            o.extend_from_slice(b"Q38DSK1\0");
            o.extend_from_slice(&(ids.len() as u32).to_le_bytes());
            for t in ids {
                o.extend_from_slice(&t.to_le_bytes());
            }
            o.extend_from_slice(blob);
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, &o)?;
            std::fs::rename(&tmp, &path)?;
            self.evict()
        }

        /// Keep the directory bounded.
        ///
        /// An agent's prompt grows every turn, so it produces a fresh, slightly
        /// longer prefix each time - without a cap this grows without limit.
        /// Oldest first, by mtime, until the total is back under the cap.
        fn evict(&self) -> std::io::Result<()> {
            let mb: u64 = std::env::var("QW_PREFIX_DISK_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8192);
            let cap = mb * 1024 * 1024;
            let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
            let mut total = 0u64;
            for e in std::fs::read_dir(&self.dir)?.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("pfx") {
                    continue;
                }
                let Ok(m) = e.metadata() else { continue };
                total += m.len();
                files.push((m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len(), p));
            }
            if total <= cap {
                return Ok(());
            }
            files.sort_by_key(|(t, _, _)| *t);
            for (_, len, p) in files {
                if total <= cap {
                    break;
                }
                if std::fs::remove_file(&p).is_ok() {
                    total = total.saturating_sub(len);
                }
            }
            Ok(())
        }
    }
}

#[derive(Default)]
struct PrefixCache {
    /// Per slot, the tokens its state currently represents: the prompt tokens it has
    /// actually consumed followed by everything it generated.
    hist: Vec<Option<Vec<u32>>>,
    /// Per slot, the prompt whose end-of-prefill snapshot sits in the runner's copy.
    boundary: Vec<Option<Vec<u32>>>,
    /// Per slot, the FULL snapshot of `boundary`: recurrent state, convolution window
    /// **and the KV for positions `0..n`**, in the same serialised form the disk cache
    /// uses.
    ///
    /// `copy_seq`-based restore is not enough on its own.  It copies only the
    /// recurrent state and the window, on the theory that the KV needs no restoring
    /// because it is keyed by position and written once.  That theory assumes the
    /// slot's KV still holds this prompt's content, and measured, a boundary hit
    /// resting on that assumption returns an answer that does not depend on the
    /// prompt at all.  Carrying the KV makes the resume independent of whatever the
    /// slot happened to hold.
    blob: Vec<Option<Vec<u8>>>,
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
            blob: (0..batch).map(|_| None).collect(),
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

    fn record_boundary(&mut self, slot: usize, seq: Vec<u32>, blob: Option<Vec<u8>>) {
        self.boundary[slot] = Some(seq);
        self.blob[slot] = blob;
    }

    fn record(&mut self, slot: usize, seq: Vec<u32>) {
        self.hist[slot] = Some(seq);
    }

    fn forget(&mut self, slot: usize) {
        self.blob[slot] = None;
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
    /// A chunk of decoded text, and how many TOKENS it stands for.
    ///
    /// The count cannot be derived from the text: speculative decoding settles up
    /// to four tokens per step and hands them over as one piece, so counting
    /// pieces under-reported `completion_tokens` by two thirds on a four-token
    /// request (measured: usage said 2 while the text was the same four tokens the
    /// plain path billed as 4).  Clients use this number for billing and for
    /// context budgeting, so it has to be tokens.
    Piece(String, usize),
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
    /// Whether the token in `feed` has already been handed to the client.
    /// `spec_step` re-emits the token it is given (it is "settled at `pos` but not
    /// yet emitted"), so the speculative path has to know that the pass which
    /// finished the prompt already emitted this one - otherwise every speculative
    /// request starts with that token twice, measured as " with with the founding".
    feed_emitted: bool,
    /// This request resumed from a stored prefix instead of prefilling from
    /// nothing.  Only used by the QW_PREFIX_DUMP instrument, which compares the
    /// state a resume reaches against the state a cold prefill reaches.
    restored: bool,
    /// The position a hit restored the slot to, if any.  At that exact position
    /// the blob in the cache already IS the live state, so re-exporting it is
    /// pure waste - measured, 17.8 ms and a 208 MB clone on every hit, against
    /// the 3.5 ms the restore itself costs.
    restored_at: Option<usize>,
    /// The client went away, so this slot is nobody's work any more.
    dead: bool,
    all: Vec<u32>,
    sent_len: usize,
    /// How many entries of `all` have already been accounted for in a `Piece`.
    /// `sent_len` tracks characters, which is what the decoder needs for the
    /// incremental UTF-8 boundary; this tracks tokens, which is what the client
    /// is billed for.
    sent_tokens: usize,
    emitted: usize,
    max_tokens: usize,
    /// When the request was admitted.  A cold agent prompt can take minutes to
    /// produce its first token, and until now the log said nothing between
    /// admission and the answer, so a working request was indistinguishable from a
    /// hung one - which is exactly how it was being reported.
    started: std::time::Instant,
    /// Last prefill milestone reported, as a percentage.
    logged_pct: usize,
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
        let new_tokens = a.all.len().saturating_sub(a.sent_tokens);
        a.sent_tokens = a.all.len();
        let _ = a
            .job
            .pieces
            .send(Ok(EngineEvent::Piece(piece[..cut].to_string(), new_tokens)));
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
    // Nothing in memory, but an earlier process may have left this prefix on
    // disk.  Keyed by content, so a fresh session finds it cold.
    let reuse = match reuse {
        Reuse::None => match prefixes::DiskPrefix::from_env().and_then(|d| d.find(&ids)) {
            Some((n, blob)) => Reuse::Disk(n, blob),
            None => Reuse::None,
        },
        other => other,
    };
    let mut restored_at: Option<usize> = None;
    let skip = match reuse {
        Reuse::None => 0,
        Reuse::Live(n) | Reuse::Boundary(n) | Reuse::Disk(n, _) => n,
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
        Reuse::Disk(n, ref blob) => {
            // Write the stored state straight into this slot's slice.  Only
            // positions 0..n are touched; the prefill that follows overwrites
            // anything past n, so no clearing is needed.
            restored_at = Some(n);
            let _t = std::time::Instant::now();
            if let Err(e) = model.import_prefix(slot, n, blob) {
                let _ = job.pieces.send(Err(e.to_string()));
                return None;
            }
            if std::env::var_os("QW_PREFIX_TIME").is_some() {
                eprintln!("prefix time: disk import {} tok {} MB {:?}", n, blob.len() / 1048576, _t.elapsed());
            }
            cache.hits += 1;
            " (disk prefix)"
        }
        Reuse::Live(_) => " (live state)",
        Reuse::Boundary(n) => {
            cache.hits += 1;
            // Put the slot back to that prefill: recurrent state, window and the KV
            // for 0..n.  Restoring the KV matters - without it the resume depends on
            // the slot still holding this prompt, which is not something this cache
            // may assume.  The copy is not consumed, so other requests can still
            // resume from the same boundary.
            // A boundary is recorded only together with its blob, so this is the
            // whole restore: recurrent state, window and the KV for 0..n.
            restored_at = Some(n);
            let restored = match cache.blob[slot].as_ref() {
                Some(b) => {
                    let _t = std::time::Instant::now();
                    let r = model.import_prefix(slot, n, b);
                    if std::env::var_os("QW_PREFIX_TIME").is_some() {
                        eprintln!("prefix time: boundary import {} tok {} MB {:?}", n, b.len() / 1048576, _t.elapsed());
                    }
                    r
                }
                None => model.reset_seq(slot),
            };
            if let Err(e) = restored {
                let _ = job.pieces.send(Err(e.to_string()));
                return None;
            }
            // QW_PREFIX_DUMP round-trip check: read the state straight back out.
            // If this does not match the blob that was just written in, the fault
            // is in export/import itself and nothing downstream can be trusted;
            // if it does match, the fault is in recomputing forward from here.
            if let Some(dir) = std::env::var_os("QW_PREFIX_DUMP") {
                match model.export_prefix(slot, n) {
                    Ok(back) => {
                        let path = std::path::Path::new(&dir)
                            .join(format!("{n}.roundtrip.bin"));
                        match std::fs::write(&path, &back) {
                            Ok(()) => tracing::info!(
                                "prefix dump: round-trip {} bytes -> {}",
                                back.len(),
                                path.display()
                            ),
                            Err(e) => tracing::warn!("prefix dump round-trip: {e}"),
                        }
                    }
                    Err(e) => tracing::warn!("prefix dump round-trip export: {e}"),
                }
            }
            " (saved boundary)"
        }
    };
    if skip > 0 && src == " (live state)" {
        cache.hits += 1;
    }
    cache.report(slot, skip, ids.len(), src);
    // Distinct from `skip > 0`: a Live hit copies nothing at all, because the
    // slot already sits on this prefix.  Only Boundary and Disk actually restore.
    let did_restore = matches!(reuse, Reuse::Boundary(_) | Reuse::Disk(_, _));
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
    // The absolute length of the prompt, captured before `ids` moves into the
    // slot.  This is the position the first generated token belongs at, and it is
    // a property of the prompt alone, so a cold run and a run that reused a prefix
    // produce the same value.
    let prompt_len = ids.len();
    Some(Active {
        job,
        ids,
        pf: skip,
        // NOT `skip`.  A hit's entire effect is setting `pf` to the reusable
        // length; `pos` is a different thing and must not move with it.
        // Instrumented, the positions handed to the model during generation are
        // `pos, pos+1, pos+2, ...`, and the kernels use them - the row loop in
        // `forward_rows` indexes the GDN convolution ring as `pos % conv_ring`.
        // Seeding `pos` from `skip` therefore drove the model with a position
        // sequence shifted by however much the request skipped, so the same prompt
        // reaching the same place gave a different state depending only on whether
        // a prefix was reused.  That is the whole cache bug.  Starting at zero
        // makes the sequence identical either way, and leaves a cold start
        // (`skip == 0`) bit-for-bit unchanged.
        // Zero is NOT the position the model expects.  A prefill pass feeds
        // absolute positions (`pf`, `pf + 1`, ...), so starting generation at zero
        // restarts the sequence mid-prompt: the first generated token is announced
        // as position 0 while it really sits at `prompt_len`, and both the rotary
        // embedding and the GDN convolution ring (`pos % conv_ring`) are driven
        // from it.  Measured on the same twelve prompt tokens with greedy decoding,
        // the CLI - which feeds absolute positions and is what the mlx-lm oracle
        // pins - answers " with the founding of the city of Rome in 753 BC and ends
        // with the fall of the Western Roman Empire in 476 AD", while the server
        // restarting at zero degenerates into " with the Roman Empire. The Roman
        // Empire was a vast and powerful state ... The Roman Empire was a vast and
        // powerful state".  Starting at `prompt_len` restores the absolute
        // sequence, and because it is a property of the prompt alone a cache hit
        // and a cold start of the same prompt stay identical.
        pos: prompt_len,
        feed: 0,
        ready: false,
        feed_emitted: false,
        restored: did_restore,
        restored_at,
        dead: false,
        all: Vec::new(),
        sent_len: 0,
        sent_tokens: 0,
        emitted: 0,
        max_tokens,
        started: std::time::Instant::now(),
        logged_pct: 0,
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
        .min(PREFILL_CHUNK_MAX);
    if let Some(want) = std::env::var("QW_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        if want != chunk_cap {
            tracing::warn!(
                "prefill chunk: QW_PREFILL_CHUNK={want} is not usable, clamped to {chunk_cap} \
                 (a pass may carry at most {PREFILL_CHUNK_MAX} rows of one sequence, set by the \
                 convolution ring; the old note that a wider pass buys nothing described the \
                 batch-`b` row path, which the tile kernel in a loop replaced)"
            );
        }
    }
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
    // The live-state cache: a slot whose own history already covers the new
    // prompt's prefix skips it outright, with nothing copied.  That is correct by
    // construction - the state really is the state after those tokens - so it is
    // safe to have on by default, and QW_PREFIX_SNAPSHOT=0 disables it.
    //
    // On by default, memory only.  It is now verified: with `pos` seeded from zero
    // rather than from `skip`, three identical requests at 63, 113 and 213 tokens
    // return the same text as each other and the same text as QW_PREFIX_SNAPSHOT=0,
    // on both the live and the boundary path.  Before that fix a hit returned an
    // answer that did not depend on the prompt at all.  QW_PREFIX_SNAPSHOT=0 (or
    // off/false/no) disables it.
    let snapshot = match std::env::var("QW_PREFIX_SNAPSHOT") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => true,
    };
    // The boundary cache saves a state snapshot and restores it for a later
    // request.  It carries the case the live path cannot: `hist` holds the prompt
    // plus everything generated, so for a straight repeat of a prompt `hist` is
    // LONGER than the prompt and `is_prefix` declines, leaving no live hit at all -
    // measured, three identical 913-token requests were three misses without this.
    // It is also correct now that `pos` is fixed, verified the same way as the live
    // path: repeats agree with each other and with a cold start.
    //
    // On by default, memory only, as asked.  It is the one part that is not free:
    // the blob is about 0.1 MB a token and one is kept per slot, and anything past
    // PREFIX_BLOB_MAX_MB is dropped rather than retained, so a slot that serves a
    // 4k-token prefix holds roughly 400 MB.  Disk persistence stays behind
    // QW_PREFIX_DISK, because that writes gigabytes into the user's home directory.
    // QW_PREFIX_BOUNDARY=0 disables it.
    let boundary = match std::env::var("QW_PREFIX_BOUNDARY") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => true,
    };
    tracing::info!(
        "prefix cache: live-state reuse {} (default), saved-boundary restore {} (QW_PREFIX_BOUNDARY)",
        if snapshot { "on" } else { "off" },
        if boundary { "on" } else { "off" }
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
        // The row budget for a pass is `PASS_ROWS_MAX`, not `MAX_BATCH`.  They were
        // the same constant, which silently capped a single prefilling slot at 16 rows
        // and left the wide row path carrying 16 instead of 32 - visible in the
        // dispatch histogram as four tile-kernel dispatches per linear rather than
        // eight, and as 768 per-row GDN dispatches a pass (48 layers times 16 rows).
        // Decoupling them keeps the slot count and the KV allocation untouched while
        // letting one prefiller use the whole row budget.
        let room = qw_model::runner::PASS_ROWS_MAX.saturating_sub(decoding);
        let share = room.checked_div(prefilling).unwrap_or(0);
        let chunk = if prefilling == 0 {
            0
        } else {
            chunk_cap.min(share).max(1)
        };
        // ---- speculative decoding is only safe for a lone sequence ----
        // `spec_step` drafts with the MTP head, which keeps ONE k/v cache, and it
        // verifies with `forward2`, which hardcodes sequence 0, and rewinds with
        // `commit_row`, which copies into sequence 0's recurrent state.  So it may
        // run only when exactly one slot is decoding and that slot is slot 0 - the
        // case a single-request client hits.  Anything else takes the batched pass.
        // Whether to keep the draft head's own cache in step during prefill.  The
        // head consumes the decoder's hidden state for the position BEFORE the token
        // it is fed, so it can only be advanced while that state is still in the
        // scratch residual stream - which is during this pass, in row order.  Left
        // unwarmed, the head attends over a cache that was never written, its drafts
        // are junk, and the server was measured to diverge from the CLI at character
        // 36 of a 64-token answer even though the plain path matched exactly.
        let spec_warm = std::env::var("QW_SPEC").is_ok() && model.has_mtp();
        let spec_ok = std::env::var("QW_SPEC").is_ok()
            && decoding == 1
            && prefilling == 0
            && slots
                .first()
                .and_then(|e| e.as_ref())
                .is_some_and(want_one);
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
                // Snapshot the recurrent state *before* the final chunk, not at the
                // end of the prompt.  At the prompt end the state has already consumed
                // the last token, but the logits for the first generated token exist
                // only in a scratch buffer that save_prefix does not copy - so a
                // resumed request would sample from stale logits and never converge.
                // Stopping one chunk early costs at most `chunk` rows of re-prefill
                // (about 80 ms) and lets an ordinary prefill rebuild the state, the KV
                // entries and the logits correctly.  The `pf > 0` guard keeps a short
                // request from clobbering a long prompt's snapshot with an empty one.
                let rem = a.ids.len() - a.pf;
                let rem_after = rem - take;
                // A boundary is due when this chunk crosses one of the ladder
                // offsets, so each fires exactly once per prefill.  Re-persisting
                // the same prefix is free: store() keys on content and returns
                // early if the file is already there.
                // A hit restored the slot to exactly `a.pf`, so the blob already
                // held is bit-for-bit the state standing here.  Exporting it again
                // costs 17.8 ms and a 208 MB clone and stores nothing new, so it is
                // skipped; the boundary and its blob recorded by the earlier request
                // are still in place, and the prefix bookkeeping is unchanged because
                // this request's `ids[..a.pf]` is the very prefix it restored from.
                if boundary
                    && snapshot
                    && a.pf > 0
                    && a.restored_at != Some(a.pf)
                    && PREFIX_LADDER.iter().any(|&k| rem > k && rem_after <= k)
                {
                    // `export_prefix` reads the live state, window and KV, so it has
                    // to run here, while the live state is at `a.pf`.  It is the ONLY
                    // thing captured: the blob already carries the recurrent state and
                    // the window, so the separate `save_prefix` GPU copy is redundant,
                    // and keeping it was what made an in-process boundary hit disagree
                    // with a cold run while the byte-identical disk path agreed
                    // exactly.  One snapshot, one restore, same code either way.
                    let _t = std::time::Instant::now();
                    match model.export_prefix(slot, a.pf) {
                        Ok(blob) => {
                            if std::env::var_os("QW_PREFIX_TIME").is_some() {
                                eprintln!("prefix time: export {} tok {} MB {:?}", a.pf, blob.len() / 1048576, _t.elapsed());
                            }
                            // Past a size cap no boundary is recorded at all.  Recording
                            // one without its KV would leave a resume that depends on
                            // the slot still holding this prompt - the very assumption
                            // that produced prompt-independent answers.
                            if blob.len() <= PREFIX_BLOB_MAX_MB * 1024 * 1024 {
                                cache.record_boundary(
                                    slot,
                                    a.ids[..a.pf].to_vec(),
                                    Some(blob.clone()),
                                );
                                if let Some(dir) = std::env::var_os("QW_PREFIX_DUMP") {
                                    let path = std::path::Path::new(&dir)
                                        .join(format!("{}.saved.bin", a.pf));
                                    if std::fs::write(&path, &blob).is_ok() {
                                        tracing::info!(
                                            "prefix dump: saved {} bytes at {}",
                                            blob.len(),
                                            path.display()
                                        );
                                    }
                                }
                                // And persist it, so the next process starts warm.
                                // Only for a prefix long enough to be a real agent
                                // system prompt: the blob costs about 0.1 MB a token.
                                if a.pf >= 128 {
                                    if let Some(d) = prefixes::DiskPrefix::from_env() {
                                        match d.store(&a.ids[..a.pf], &blob) {
                                            Ok(()) => tracing::info!(
                                                "slot {slot}: persisted {} tokens ({} MB) to disk",
                                                a.pf,
                                                blob.len() / (1024 * 1024)
                                            ),
                                            Err(e) => tracing::warn!("prefix store: {e}"),
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = a.job.pieces.send(Err(e.to_string()));
                        }
                    }
                }
                for k in 0..take {
                    rows.push((slot, a.pf + k));
                    toks.push(a.ids[a.pf + k]);
                    row_slot.push(slot);
                }
                a.pf += take;
                continue;
            }
            if a.ready && a.emitted < a.max_tokens && !tok.is_eos(a.feed) {
                if a.emitted == 0 && std::env::var_os("QW_POS_DEBUG").is_some() {
                    eprintln!(
                        "pos debug: slot {slot} START GENERATION pf={} pos={} skip_was={}",
                        a.pf, a.pos, a.restored
                    );
                }
                if std::env::var_os("QW_POS_DEBUG").is_some() && a.emitted < 3 {
                    eprintln!("pos debug: slot {slot} generation row emitted={} pos={} feed={}", a.emitted, a.pos, a.feed);
                }
                if spec_ok && slot == 0 {
                    // One speculative step drafts `TILE - 1` tokens and verifies them
                    // in a single `TILE`-row weight sweep, which is where the speed
                    // comes from: the decode pass already reads all 14.4 GB, so
                    // settling up to four tokens for it is nearly free.  Correctness
                    // does not depend on the drafts being good - a rejected draft is
                    // simply not emitted, and `next` is always the model's own
                    // prediction from the row that broke the run.
                    let mut out: Vec<u32> = Vec::new();
                    match model.spec_step(a.pos, a.feed, &mut out) {
                        Ok((np, ntok, _draft_s, _pass_s)) => {
                            let mut stop = false;
                            // The first element is the token at `pos` again.  Drop it
                            // when the pass that finished the prompt already emitted
                            // it; keep it when it is this path's own fresh prediction.
                            let skip = if a.feed_emitted { 1 } else { 0 };
                            let out_len = out.len();
                            let old_pos = a.pos;
                            for t in out.into_iter().skip(skip) {
                                // Mirror the plain path: an end-of-sequence token is
                                // fed back but never recorded, and its presence in
                                // `feed` is what stops the slot being scheduled.
                                if a.emitted >= a.max_tokens || tok.is_eos(t) {
                                    a.feed = t;
                                    stop = true;
                                    break;
                                }
                                a.all.push(t);
                                a.emitted += 1;
                            }
                            if !stop {
                                a.feed = ntok;
                                a.feed_emitted = false;
                            }
                            if std::env::var_os("QW_SPEC_DEBUG").is_some() {
                                eprintln!(
                                    "spec dbg: out={} skip={} emitted={} max={} pos {}->{} all={}",
                                    out_len, skip, a.emitted, a.max_tokens, old_pos, np, a.all.len()
                                );
                            }
                            a.pos = np;
                            emit(tok, a, false);
                        }
                        Err(e) => {
                            let _ = a.job.pieces.send(Err(e.to_string()));
                            a.emitted = a.max_tokens;
                        }
                    }
                    continue;
                }
                rows.push((slot, a.pos));
                toks.push(a.feed);
                row_slot.push(slot);
                a.pos += 1;
                // The token is NOT recorded here any more.  Recording it at the
                // point of use meant the first token - which the pass that finishes
                // the prompt already produced - only reached `a.all` on the NEXT
                // iteration, so `emit` below could not send it until a second full
                // weight sweep had gone by.  Measured, that put the client's
                // time-to-first-token at two sweeps for every request, including a
                // one-token prompt.  It is recorded where it is produced instead.
            }
        }
        // The retire step at the end of this iteration is what FINISHES a request,
        // and speculative decoding never fills `rows` - it settles tokens without a
        // batched pass - so this used to `continue` straight past retirement and
        // hang every speculative request with its tokens generated and never sent
        // (the log said "4 of 4 tokens emitted" and the client waited forever).  A
        // labelled block keeps the skip local to the pass.
        'pass: {
        if rows.is_empty() {
            break 'pass;
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
            // Keep the draft head in step, in the same order the rows were computed.
            // The MTP head writes its internal norm into row 0's slot, so rows above
            // zero are untouched and increasing order is safe.  `gen` warms positions
            // 0..len-2 with the token that follows, never the last one - there is no
            // token after the prompt yet - and this mirrors that exactly.
            if spec_warm {
                let p = rows[i].1;
                let nxt = if i + 1 < row_slot.len() && row_slot[i + 1] == slot {
                    Some(toks[i + 1])
                } else {
                    a.ids.get(a.pf).copied()
                };
                if let Some(t) = nxt {
                    // Warming is a throughput aid, not a correctness one, so a failure
                    // must not take the request down: the next verify simply rejects
                    // the drafts it produced.
                    if let Err(e) = model.mtp_step_at(i, t, p + 1, false) {
                        tracing::warn!("draft-head warm failed at row {i}: {e}");
                    }
                }
            }
            let next = argmax(&model.logits_row(i));
            if a.pf < a.ids.len() {
                let pct = a.pf * 100 / a.ids.len().max(1);
                if pct >= a.logged_pct + 25 {
                    a.logged_pct = pct;
                    tracing::info!(
                        "slot {slot}: prefill {}/{} ({}%) after {:.1}s",
                        a.pf,
                        a.ids.len(),
                        pct,
                        a.started.elapsed().as_secs_f64()
                    );
                }
            }
            if a.pf >= a.ids.len() {
                if !a.ready {
                    // Time to first token, which is the number an agent framework
                    // actually feels.  The prompt is fully in and this pass is the
                    // one that produces the answer's first token, so the elapsed
                    // time here is exactly it - and until now only the total was
                    // recorded, which made a warm hit indistinguishable from a cold
                    // start in the log.
                    tracing::info!(
                        "slot {slot}: prompt in after {:.1}s (time to first token, {} tokens)",
                        a.started.elapsed().as_secs_f64(),
                        a.ids.len()
                    );
                }
                // NOTE: `a.pos` is deliberately NOT set here.  Instrumented, a
                // 213-token prompt with no reuse reaches generation with pf=213 but
                // pos=0, so the positions handed to the model are 0,1,2,...  Setting
                // it to `a.ids.len()` changes the model's output for some prompts
                // while still passing the oracle 6/6, so the oracle cannot tell the
                // two conventions apart and neither may be adopted on guesswork.
                // What IS proven is the inconsistency that matters for the cache:
                // the sequence begins at `skip`, so a request that skipped n tokens
                // drives the model with positions shifted by n against a cold run of
                // the same prompt.  QW_POS_DEBUG=1 prints the sequence.
                a.ready = true;
            }
            if a.ready {
                a.feed = next;
            }
        }
        // Record and send once per slot, after every row of this pass has been
        // consumed.  Recording inside the row loop above put one token per ROW into
        // `a.all`, which is wrong for a prefill chunk of more than one row - and the
        // last chunk of a prompt is exactly such a pass, so the first token came out
        // as the last row's argmax repeated.  Measured against the previous build on
        // four prompts, that changed the emitted sequence, which is why this is done
        // here instead.
        //
        // Doing it here rather than at the start of the NEXT pass is what makes
        // time-to-first-token one weight sweep instead of two: the pass that finishes
        // the prompt already produced the answer's first token, so it can be handed
        // to the client now.
        for (idx, &slot) in row_slot.iter().enumerate() {
            if row_slot[idx + 1..].contains(&slot) {
                continue;
            }
            let Some(a) = slots[slot].as_mut() else {
                continue;
            };
            if a.ready && a.emitted < a.max_tokens && !tok.is_eos(a.feed) {
                a.all.push(a.feed);
                a.emitted += 1;
                a.feed_emitted = true;
            }
            emit(tok, a, false);
        }

        // ---- QW_PREFIX_DUMP: is a resume equivalent to a cold prefill? ----
        //
        // At QW_PREFIX_DUMP_POS=L, write this slot's recurrent state, window and
        // KV to $QW_PREFIX_DUMP/L.<cold|resumed>.bin.  Both runs of the same
        // prompt in one process therefore land in the same directory, and
        // comparing the two files answers directly whether resuming from a
        // boundary and recomputing forward reproduces what prefilling from
        // nothing produces.  Comparing whole files first says whether they differ
        // at all; a byte offset and stride arithmetic then names the tensor.
        if let Some(dir) = std::env::var_os("QW_PREFIX_DUMP") {
            let want: usize = std::env::var("QW_PREFIX_DUMP_POS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if want > 0 {
                for (slot, entry) in slots.iter().enumerate() {
                    let Some(a) = entry.as_ref() else { continue };
                    if a.pf != want {
                        continue;
                    }
                    let _t = std::time::Instant::now();
                    match model.export_prefix(slot, a.pf) {
                        Ok(blob) => {
                            let tag = if a.restored { "resumed" } else { "cold" };
                            let path =
                                std::path::Path::new(&dir).join(format!("{want}.{tag}.bin"));
                            match std::fs::write(&path, &blob) {
                                Ok(()) => tracing::info!(
                                    "prefix dump: {} bytes -> {}",
                                    blob.len(),
                                    path.display()
                                ),
                                Err(e) => tracing::warn!("prefix dump: {e}"),
                            }
                        }
                        Err(e) => tracing::warn!("prefix dump export: {e}"),
                    }
                }
            }
        }

        } // end 'pass

        // ---- retire ----
        for (slot, entry) in slots.iter_mut().enumerate() {
            let done = match entry.as_ref() {
                Some(a) => a.dead || (a.ready && (a.emitted >= a.max_tokens || tok.is_eos(a.feed))),
                None => false,
            };
            if done {
                if let Some(a) = entry.as_mut() {
                    emit(tok, a, true);
                    tracing::info!(
                        "slot {slot}: done - {}/{} prompt tokens prefilled, {} generated, {:.1}s total",
                        a.pf.min(a.ids.len()),
                        a.ids.len(),
                        a.emitted,
                        a.started.elapsed().as_secs_f64()
                    );
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
