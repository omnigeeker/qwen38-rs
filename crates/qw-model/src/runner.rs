//! Single-token (decode) forward pass for Qwen3.8-27B.
//!
//! Layer schedule and per-op math follow `mlx_lm/models/qwen3_5.py` exactly —
//! see `docs/M1_NOTES.md` for the transcription and the norm-weight convention.
//!
//! Every step of one token is encoded into **one command buffer**. Kernels that
//! consume another kernel's output are separated by `barrier()` (an encoder
//! boundary), which is what makes the ordering observable on Metal.

use anyhow::{bail, Context, Result};
use half::f16;
use qw_metal::{msl, msl_ops, CommandBatch, Dispatch, GpuBuffer, GpuDevice, Kernel};
use qw_weights::safetensors::WeightStore;
use std::path::Path;

use crate::config::ModelConfig;
use crate::linear::QLinear;
use crate::naming::WeightLayout;

/// Every kernel the forward pass needs, compiled once at load time.
struct Kernels {
    q4_gemv: Kernel,
    /// Compile-time k=2 specialisation (round 6): the runtime-k kernel spills
    /// its accumulators to thread-local memory.
    q4_gemv_tile: Kernel,
    /// 16-row grid-mapped variant, for batch passes wider than `TILE`.
    q4_gemv_b16: Kernel,
    rmsnorm: Kernel,
    rmsnorm_ws: Kernel,
    rmsnorm_nw: Kernel,
    rmsnorm_tile: Kernel,
    rmsnorm_gated: Kernel,
    rope: Kernel,
    silu_mul: Kernel,
    ewise_add: Kernel,
    gate_mul: Kernel,
    kv_append: Kernel,
    attn_scores: Kernel,
    attn_out: Kernel,
    attn_scores_rows: Kernel,
    attn_out_rows: Kernel,
    gdn_seq4: Kernel,
    conv1d_ring: Kernel,
    conv1d_ring_tile: Kernel,
    gdn: Kernel,
    gdn_seq: Kernel,
    rmsnorm_ws_rows: Kernel,
    gate_mul_rows: Kernel,
    kv_append_rows: Kernel,
    rope_rows: Kernel,
    copy: Kernel,
    round_bf16: Kernel,
}

/// Lanes per GDN value in `gdn_step_seq4`; also sets the compiled state-array
/// size, so it is fixed for the life of the process.
fn gdn_q() -> usize {
    static Q: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *Q.get_or_init(|| msl_ops::gdn_tiles()[0].max(1) as usize)
}

/// Whether the register-resident GDN scan is used.
fn gdn_seq4_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QW_GDN_SEQ4").ok().as_deref() != Some("0"))
}

/// Whether the row-batched attention kernels are used.  Read once: this is
/// consulted for every full-attention layer of every pass.
fn attn_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QW_ATTN_ROWS").ok().as_deref() != Some("0"))
}

struct FullAttn {
    q: QLinear<'static>,
    k: QLinear<'static>,
    v: QLinear<'static>,
    o: QLinear<'static>,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
    /// head-major `[n_kv][max_t][head_dim]`
    k_cache: GpuBuffer,
    v_cache: GpuBuffer,
}

struct Gdn {
    in_qkv: QLinear<'static>,
    in_z: QLinear<'static>,
    in_b: QLinear<'static>,
    in_a: QLinear<'static>,
    out_proj: QLinear<'static>,
    conv_w: GpuBuffer,
    norm_w: GpuBuffer,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    /// fp32 `[Hv][Dv][Dk]`
    state: GpuBuffer,
    /// `[conv_ring][conv_dim]` ring of raw pre-conv rows.  A pass writes `TILE`
    /// rows but only the first `k + 1` survive a rejection, and the next pass's
    /// first row still reads the three rows before it, so the ring has to be
    /// wider than the convolution window.  With exactly `conv_k` slots a rejected
    /// pass evicted the row the next pass was about to read; that stayed hidden
    /// because it only shows up as an argmax flipping on a near-tie.
    /// Slot of position `p` is `p % conv_ring`; the convolution reads the last four.
    window: GpuBuffer,
    /// `TILE` per-row snapshots of `state`, taken only while
    /// speculative verification is on.  A verify pass advances the recurrence
    /// through every drafted token, so on a partial acceptance the state must be
    /// committed for the longest accepted prefix; without these the only correct
    /// recovery is a full re-run of the accepted row, which costs more than the
    /// draft ever saves.
    snap: GpuBuffer,
    /// Snapshot of `state` taken at the end of a prefill, per sequence.
    cache_state: GpuBuffer,
    /// Snapshot of `window` taken at the same moment, per sequence.
    cache_window: GpuBuffer,
}

enum Kind {
    Full(Box<FullAttn>),
    Gdn(Box<Gdn>),
}

struct Layer {
    input_norm: GpuBuffer,
    post_norm: GpuBuffer,
    gate: QLinear<'static>,
    up: QLinear<'static>,
    down: QLinear<'static>,
    kind: Kind,
}

/// Scratch buffers, all fp16 unless stated.
/// Number of tokens a single forward pass can carry (see docs/PLAN_K2.md).
pub const TILE: usize = 4;

/// The one switch for the whole speculative path.
///
/// It used to be read as `QW_SPEC` being *present* in three independent places:
/// the engine's decode loop, this crate's per-row snapshot flag, and the CLI's
/// `gen`.  Turning it on by default in the engine alone left the verify pass
/// running without the per-row recurrent snapshots it needs, so a rejected draft
/// left the state wrong and the server diverged from the CLI at character 5 -
/// but only when `QW_SPEC` was unset, which is why passing `QW_SPEC=1` explicitly
/// never reproduced it.  On by default; `QW_SPEC=0` disables it everywhere.
pub fn spec_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QW_SPEC").ok().as_deref() != Some("0"))
}

/// The most rows of one sequence a single pass may carry.  The convolution ring
/// has to hold `conv_k` rows of history plus every row this pass writes, so the
/// ring - and therefore the prefill chunk - is sized from this.  Raising it from
/// the old value of `TILE` is what lets a prefill amortise the per-pass overhead,
/// which is a fixed cost of roughly 1216 dependent dispatches that does not care
/// how many rows ride along.
// 128, not 32.  The GEMM's grid is (out_f/BM)*128 by rows/BN, so a 32-row pass
// gives it a single y-slice and only out_f/32 threadgroups - about two waves on
// 40 SMs - which is exactly where the weight staging collapses to 39 GB/s.  The
// same kernel at 128 rows reaches 61 GB/s and costs 7.427 ms per token against
// 11.548, because each weight read is then amortised over four times the tokens.
// This is the lever the narrow-tile experiment got wrong: what matters is tokens
// per weight read, not threadgroups on their own.
pub const PASS_ROWS_MAX: usize = 1020;

/// Widest row tile a single pass can carry.  `TILE` is the speculative-verify
/// width; a batch-serving pass puts one row per sequence in the same structure.
// 1020, not 128.  This is the cap on rows in ONE pass, and the pass is what
// amortises a single sweep of all 14.4 GB of weights over every row it carries.
// At 128 rows a 602-token prompt needs five passes and therefore five weight
// sweeps; at 1020 it needs one.  The convolution ring is sized as
// `(conv_k + PASS_ROWS_MAX).next_power_of_two()`, so it stays at 1024 and the
// extra memory is a few hundred megabytes of scratch.
pub const BATCH_MAX: usize = 1020;

struct Scratch {
    x: GpuBuffer,
    h: GpuBuffer,
    qg: GpuBuffer,
    pk: GpuBuffer,
    pv: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    attn_out: GpuBuffer,
    attn_gated: GpuBuffer,
    proj_out: GpuBuffer,
    z: GpuBuffer,
    a: GpuBuffer,
    b: GpuBuffer,
    conv_out: GpuBuffer,
    gdn_y: GpuBuffer,
    gdn_gated: GpuBuffer,
    mlp_gate: GpuBuffer,
    mlp_up: GpuBuffer,
    mlp_act: GpuBuffer,
    logits: GpuBuffer,
    /// fp32 `[n_heads][max_t]`
    scores: GpuBuffer,
    /// `[TILE][conv_dim]` raw pre-conv rows of the pass in flight
    qkv_cur: GpuBuffer,
    /// Scratch for the keep-warm tick, deliberately NOT `logits`.
    ///
    /// The tick used to write eight halves into `logits`, the buffer the sampler
    /// reads.  Ordering on the engine thread makes that look safe, but any path
    /// that reads logits without a fresh forward - the spec verify reusing the
    /// previous round's logits, or a cache-hit restore - would be corrupted by a
    /// tick landing between two requests.
    tick: GpuBuffer,
}

/// The multi-token-prediction head: one `fc` that fuses the embedding of the
/// next token with the decoder's hidden state, one full-attention decoder layer
/// and its own KV cache.
///
/// The draft is only a proposal - the decoder verifies it - so a mistake here
/// costs acceptance rate, never correctness.
struct Mtp {
    fc: QLinear<'static>,
    q: QLinear<'static>,
    k: QLinear<'static>,
    v: QLinear<'static>,
    o: QLinear<'static>,
    gate: QLinear<'static>,
    up: QLinear<'static>,
    down: QLinear<'static>,
    pre_norm_e: GpuBuffer,
    pre_norm_h: GpuBuffer,
    input_norm: GpuBuffer,
    post_norm: GpuBuffer,
    norm: GpuBuffer,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
    k_cache: GpuBuffer,
    v_cache: GpuBuffer,
    /// MTP residual stream, `[hidden]`
    hid: GpuBuffer,
    /// `[2 * hidden]` input of `fc`
    cat: GpuBuffer,
}

impl Mtp {
    /// Load the locally quantised head written by `tools/mtp_quantize.py`.
    ///
    /// The norms in that file are already shifted by the reference's `+1`
    /// (the bf16 release stores them unshifted), so no shift is applied here.
    fn load(
        dev: &GpuDevice,
        store: &'static WeightStore,
        cfg: &crate::config::TextConfig,
        max_t: usize,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let l = WeightLayout::default();
        let layer = l.mtp_layer(0);
        let attn = |p: &str| format!("{layer}.self_attn.{p}.weight");
        Ok(Self {
            fc: QLinear::from_store(store, &l.mtp_fc())?,
            q: QLinear::from_store(store, &attn("q_proj"))?,
            k: QLinear::from_store(store, &attn("k_proj"))?,
            v: QLinear::from_store(store, &attn("v_proj"))?,
            o: QLinear::from_store(store, &attn("o_proj"))?,
            gate: QLinear::from_store(store, &format!("{layer}.mlp.gate_proj.weight"))?,
            up: QLinear::from_store(store, &format!("{layer}.mlp.up_proj.weight"))?,
            down: QLinear::from_store(store, &format!("{layer}.mlp.down_proj.weight"))?,
            pre_norm_e: norm_buf(dev, store, &l.mtp_pre_norm_embedding(), 0.0)?,
            pre_norm_h: norm_buf(dev, store, &l.mtp_pre_norm_hidden(), 0.0)?,
            input_norm: norm_buf(dev, store, &format!("{layer}.input_layernorm.weight"), 0.0)?,
            post_norm: norm_buf(
                dev,
                store,
                &format!("{layer}.post_attention_layernorm.weight"),
                0.0,
            )?,
            norm: norm_buf(dev, store, &l.mtp_norm(), 0.0)?,
            q_norm: norm_buf(dev, store, &attn("q_norm"), 0.0)?,
            k_norm: norm_buf(dev, store, &attn("k_norm"), 0.0)?,
            // The MTP draft head keeps a single sequence's cache for now; batch
            // serving runs the decoder only, and scaling this needs `batch`
            // threaded into Mtp::load.
            k_cache: dev.buffer(nkv * max_t * hd * 2),
            v_cache: dev.buffer(nkv * max_t * hd * 2),
            hid: dev.buffer(h * 2),
            cat: dev.buffer(2 * h * 2),
        })
    }
}

/// Where the quantised MTP head lives: `$QW_MTP_DIR`, else a sibling of the
/// model directory.  `None` means the engine runs without MTP.
/// Resident bytes of the safetensors shards under `dir`.
fn weights_bytes(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    rd.flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Physical memory, or 0 when it cannot be determined.
fn physical_bytes() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// What a given `max_t` and batch width cost, split into the part that does not
/// grow with the context (weights and delta-net state) and the part that does.
struct Budget {
    /// Resident weights, including the MTP head.
    weights: u64,
    /// Per slot: the delta-net recurrence, its convolution ring, and the
    /// speculative rollback snapshot, which is TILE copies of the recurrence.
    per_slot: u64,
    /// Per token across the whole batch: both caches of every full-attention
    /// layer, the attention score buffer, and the MTP head's own cache.
    per_token: u64,
}

/// Driver and load-time staging overhead that is in none of the buffers above.
/// Measured on this machine: resident size is 41 GB at ctx 8192 and 65 GB at ctx
/// 32768, i.e. 33 GB fixed plus 1.0 MB per token.  The token slope matches the
/// model almost exactly, but the modelled fixed cost is only ~26 GB, so about 8 GB
/// of the process is the Metal driver and staging rather than cache.  Ignoring that
/// would make the advice this check prints optimistic by about 12%.
const MEM_OVERHEAD: u64 = 8 << 30;

impl Budget {
    fn fixed(&self) -> u64 {
        self.weights + self.per_slot + MEM_OVERHEAD
    }
    fn total(&self, max_t: usize) -> u64 {
        self.fixed() + self.per_token * max_t as u64
    }
}

fn budget(cfg: &crate::config::TextConfig, dir: &Path, batch: usize) -> Budget {
    let hd = cfg.head_dim;
    let nkv = cfg.num_key_value_heads;
    let hv = cfg.linear_num_value_heads;
    let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
    let key_dim = cfg.linear_num_key_heads * dk;
    let conv_dim = key_dim * 2 + hv * dv;
    let conv_ring = (cfg.linear_conv_kernel_dim + PASS_ROWS_MAX).next_power_of_two();
    let (mut n_full, mut n_gdn) = (0usize, 0usize);
    for i in 0..cfg.num_hidden_layers {
        if cfg.is_linear_layer(i) {
            n_gdn += 1;
        } else {
            n_full += 1;
        }
    }
    let per_token = (2 * n_full * nkv * hd * 2 * batch
        + cfg.num_attention_heads * 4 * BATCH_MAX
        + 2 * nkv * hd * 2) as u64;
    let per_slot = (n_gdn * hv * dv * dk * 4
        + n_gdn * conv_ring * conv_dim * 2
        + n_gdn * hv * dv * dk * 4 * TILE) as u64
        * batch as u64;
    let weights = weights_bytes(dir) + mtp_dir(dir).map(|d| weights_bytes(&d)).unwrap_or(0);
    Budget {
        weights,
        per_slot,
        per_token,
    }
}

/// Refuse a configuration the machine cannot hold, instead of dying silently.
///
/// Every slot carries its own KV cache and all of them are allocated up front, so
/// the cost is `max-ctx x batch`.  Overshoot and the OS kills the process while it
/// is still allocating - no error, no message, the port never bound - and the only
/// symptom is `curl: (7) Failed to connect to 127.0.0.1 port 8080`.  That is a
/// miserable way to find out that `--max-ctx 1000000` wants a terabyte, so say so
/// plainly, before touching a single weight.
fn check_memory(
    cfg: &crate::config::TextConfig,
    dir: &Path,
    max_t: usize,
    batch: usize,
) -> Result<()> {
    let phys = physical_bytes();
    if phys == 0 {
        return Ok(()); // cannot tell; do not block a load we cannot judge
    }
    let b = budget(cfg, dir, batch);
    let needed = b.total(max_t);
    // Leave a margin: the OS and the Metal driver need room too, and the failure
    // mode when they do not get it is a kill, not an error.
    let ceiling = (phys as f64 * 0.85) as u64;
    if needed <= ceiling {
        return Ok(());
    }
    let gb = |bytes: u64| bytes as f64 / 1073741824.0;
    // Widest context that fits, at this width and at one slot, capped at what the
    // model itself supports.
    let fits = |width: usize| -> usize {
        let bb = budget(cfg, dir, width);
        let room = (ceiling as f64 - bb.fixed() as f64).max(0.0) as u64;
        ((room / bb.per_token.max(1)) as usize).min(cfg.max_position_embeddings)
    };
    let at_width = fits(batch);
    let at_one = fits(1);
    let mut msg = format!(
        "max-ctx {max_t} does not fit: it needs about {:.0} GB and this machine has {:.0} GB.\n\
         \x20 {:.0} GB of weights, {:.0} GB of KV cache, {:.0} GB of delta-net state.\n\
         \x20 The KV cache costs {} KB per token for the whole batch of {batch}, i.e. {:.0} KB\n\
         \x20 per token PER SLOT.  Every slot keeps its own and they are all allocated up\n\
         \x20 front, so the cost is multiplied by both max-ctx and the batch width.",
        gb(needed),
        gb(phys),
        gb(b.weights),
        gb(b.per_token * max_t as u64),
        gb(b.per_slot),
        b.per_token / 1024,
        b.per_token as f64 / 1024.0 / batch as f64,
    );
    if max_t > cfg.max_position_embeddings {
        msg.push_str(&format!(
            "\n\x20 Note the model itself only supports {} tokens.",
            cfg.max_position_embeddings
        ));
    }
    msg.push_str(&format!(
        "\n\x20 Largest that fits at batch {batch}: --max-ctx {at_width} (about {:.0} GB).\n\
         \x20 Or QW_BATCH=1 --max-ctx {at_one} (about {:.0} GB), trading concurrency for context.",
        gb(budget(cfg, dir, batch).total(at_width)),
        gb(budget(cfg, dir, 1).total(at_one)),
    ));
    bail!(msg)
}

fn mtp_dir(dir: &Path) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("QW_MTP_DIR") {
        return Some(std::path::PathBuf::from(p));
    }
    let cand = dir.join("..").join("Qwen3.8-27B-mtp-4bit");
    cand.is_dir().then_some(cand)
}

pub struct Qwen38 {
    pub cfg: crate::config::TextConfig,
    pub max_t: usize,
    pub vocab: usize,
    dev: GpuDevice,
    layers: Vec<Layer>,
    scratch: Scratch,
    kernels: Kernels,
    embed_w: qw_weights::TensorHandle<'static>,
    embed_s: qw_weights::TensorHandle<'static>,
    embed_b: qw_weights::TensorHandle<'static>,
    lm_head: QLinear<'static>,
    final_norm: GpuBuffer,
    layout: WeightLayout,
    /// Draft head for speculative decoding, when its weights are present.
    mtp: Option<Mtp>,
    last_dispatches: usize,
    /// `[layer][hidden]` copies of the residual stream (dumped on demand)
    debug: Option<GpuBuffer>,
    /// round the residual stream through bf16 after every layer, matching the
    /// bf16 numerics of the mlx-lm reference (see docs/M1_NOTES.md)
    pub bf16_residual: bool,
    /// take per-row recurrent snapshots during a verify pass (QW_SPEC=1)
    pub spec_snap: bool,
    /// Number of independent sequence slots whose recurrent and KV state is
    /// allocated up front.  `load` allocates one slot; `load_batch` allocates
    /// many, so several requests can hold live state at the same time instead of
    /// queueing behind a single `reset()`.
    pub batch: usize,
    /// Per-sequence byte stride of each state buffer: its length divided by
    /// `batch`.  Every dispatch that touches sequence state is offset by
    /// `seq * stride`, which is what leaves the kernels untouched - from inside a
    /// kernel the cache simply begins at this sequence's slice.
    kv_stride: usize,
    win_stride: usize,
    state_stride: usize,
    snap_stride: usize,
}

const NT: usize = 256;

/// Bisect mask for the token-batched attention operators: bit 1 the q/k norms,
/// bit 2 the output gate, bit 4 rope, bit 8 the kv append.  All four on by
/// default; a mask of 0 must reproduce the pre-batching code exactly, which is
/// what makes it a usable control.
fn attn_batch_mask() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("QW_ATTN_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15)
    })
}

fn f16_buf(dev: &GpuDevice, vals: &[f32]) -> GpuBuffer {
    let v: Vec<f16> = vals.iter().map(|x| f16::from_f32(*x)).collect();
    dev.buffer_from_bytes(&v)
}

fn f32_buf(dev: &GpuDevice, vals: &[f32]) -> GpuBuffer {
    dev.buffer_from_bytes(vals)
}

/// Read a norm tensor and (optionally) add the reference's `+1.0` shift.
fn norm_buf(dev: &GpuDevice, store: &WeightStore, name: &str, shift: f32) -> Result<GpuBuffer> {
    let h = store.handle(name).with_context(|| format!("norm {name}"))?;
    let mut vals = tensor_f32(&h)?;
    if shift != 0.0 {
        for v in vals.iter_mut() {
            *v += shift;
        }
    }
    Ok(f16_buf(dev, &vals))
}

fn tensor_f32(h: &qw_weights::TensorHandle<'_>) -> Result<Vec<f32>> {
    match h.info.dtype.as_str() {
        "F32" => Ok(h
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()),
        "F16" => Ok(h.as_f16().iter().map(|v| v.to_f32()).collect()),
        "BF16" => Ok(h.as_bf16_f32()),
        other => bail!("{}: unsupported dtype {other}", h.info.dtype),
    }
}

fn copy_dispatch(
    batch: &mut CommandBatch,
    k: &Kernel,
    src: &GpuBuffer,
    src_off: usize,
    dst: &GpuBuffer,
    dst_off: usize,
    n: usize,
) {
    batch.encode(
        Dispatch::new(k, (n, 1, 1), (NT, 1, 1))
            .buf(0, src)
            .buf(1, dst)
            .scalar(2, n as i32)
            .scalar(3, src_off as i32)
            .scalar(4, dst_off as i32),
    );
}

impl Qwen38 {
    pub fn load(dir: &Path, max_t: usize) -> Result<Self> {
        // QW_BATCH=16 allocates sixteen sequence slots instead of one.  It costs
        // 16x the state memory (most of it the full-attention KV cache), so it is
        // opt-in rather than the default.
        let batch = std::env::var("QW_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|b| *b > 0)
            .unwrap_or(1);
        Self::load_batch(dir, max_t, batch)
    }

    /// Load with `batch` independent sequence slots allocated.  Every slot holds
    /// a full set of recurrent state: GDN `state`, the convolution ring window,
    /// and the full-attention KV cache.
    pub fn load_batch(dir: &Path, max_t: usize, batch: usize) -> Result<Self> {
        let mut dev = GpuDevice::new()?;
        let cfg = ModelConfig::from_path(&dir.join("config.json"))?.text_config;
        check_memory(&cfg, dir, max_t, batch)?;
        let layout = WeightLayout::default();
        // The store must outlive every QLinear; the model is process-lifetime.
        let store: &'static WeightStore = Box::leak(Box::new(
            WeightStore::load_dir(&dev, dir).context("load weights")?,
        ));

        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_k = cfg.linear_conv_kernel_dim;
        // one ring for the window plus the rows a pass writes ahead
        let conv_ring = (conv_k + PASS_ROWS_MAX).next_power_of_two(); // power of two: the kernel masks
        if conv_k != 4 {
            bail!("conv kernel {conv_k} != 4 is not implemented");
        }
        let vocab = cfg.vocab_size;

        // Two-token tile (see docs/PLAN_K2.md).  Step 1 only *reserves* the
        // space: every dispatch still addresses row 0, so nothing observable can
        // change until the k=2 kernels are wired in.  `window` is deliberately
        // not doubled - it is the 4-row convolution history shared across steps,
        // not a per-token activation tile.
        let tile = BATCH_MAX;
        let scratch = Scratch {
            x: dev.buffer(h * 2 * tile),
            h: dev.buffer(h * 2 * tile),
            qg: dev.buffer(nh * hd * 2 * 2 * tile),
            pk: dev.buffer(nkv * hd * 2 * tile),
            pv: dev.buffer(nkv * hd * 2 * tile),
            q: dev.buffer(nh * hd * 2 * tile),
            k: dev.buffer(key_dim * 2 * tile),
            attn_out: dev.buffer(nh * hd * 2 * tile),
            attn_gated: dev.buffer(nh * hd * 2 * tile),
            proj_out: dev.buffer(h * 2 * tile),
            z: dev.buffer(value_dim * 2 * tile),
            a: dev.buffer(hv * 2 * tile),
            b: dev.buffer(hv * 2 * tile),
            conv_out: dev.buffer(conv_dim * 2 * tile),
            gdn_y: dev.buffer(value_dim * 2 * tile),
            gdn_gated: dev.buffer(value_dim * 2 * tile),
            mlp_gate: dev.buffer(cfg.intermediate_size * 2 * tile),
            mlp_up: dev.buffer(cfg.intermediate_size * 2 * tile),
            mlp_act: dev.buffer(cfg.intermediate_size * 2 * tile),
            logits: dev.buffer(vocab * 2 * tile),
            scores: dev.buffer(nh * max_t * 4 * tile),
            qkv_cur: dev.buffer(conv_dim * 2 * tile),
            tick: dev.buffer(16),
        };

        // MTP weights (bf16 repo) need the +1 norm shift; this export does not.
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let is_linear = cfg.is_linear_layer(i);
            let kind = if is_linear {
                Kind::Gdn(Box::new(Gdn {
                    in_qkv: QLinear::from_store(store, &layout.gdn_in_qkv(i))?,
                    in_z: QLinear::from_store(store, &layout.gdn_in_z(i))?,
                    in_b: QLinear::from_store(store, &layout.gdn_in_b(i))?,
                    in_a: QLinear::from_store(store, &layout.gdn_in_a(i))?,
                    out_proj: QLinear::from_store(store, &layout.gdn_out(i))?,
                    conv_w: norm_buf(&dev, store, &layout.gdn_conv(i), 0.0)?,
                    norm_w: norm_buf(&dev, store, &layout.gdn_norm(i), 0.0)?,
                    a_log: f32_buf(&dev, &tensor_f32(&store.handle(&layout.gdn_a_log(i))?)?),
                    dt_bias: f32_buf(&dev, &tensor_f32(&store.handle(&layout.gdn_dt_bias(i))?)?),
                    state: dev.buffer(hv * dv * dk * 4 * batch),
                    window: dev.buffer(conv_ring * conv_dim * 2 * batch),
                    snap: dev.buffer(hv * dv * dk * 4 * TILE * batch),
                    // Mirror of `state` and `window`, one slot's worth per sequence,
                    // held for the prefix cache.  165 MB in total against the 14.4 GB
                    // of weights, and it is what makes a prompt reusable after the
                    // slot has gone on to decode.
                    cache_state: dev.buffer(hv * dv * dk * 4 * batch),
                    cache_window: dev.buffer(conv_ring * conv_dim * 2 * batch),
                }))
            } else {
                Kind::Full(Box::new(FullAttn {
                    q: QLinear::from_store(store, &layout.attn_q(i))?,
                    k: QLinear::from_store(store, &layout.attn_k(i))?,
                    v: QLinear::from_store(store, &layout.attn_v(i))?,
                    o: QLinear::from_store(store, &layout.attn_o(i))?,
                    q_norm: norm_buf(&dev, store, &layout.attn_q_norm(i), 0.0)?,
                    k_norm: norm_buf(&dev, store, &layout.attn_k_norm(i), 0.0)?,
                    k_cache: dev.buffer(batch * nkv * max_t * hd * 2),
                    v_cache: dev.buffer(batch * nkv * max_t * hd * 2),
                }))
            };
            layers.push(Layer {
                input_norm: norm_buf(&dev, store, &layout.input_norm(i), 0.0)?,
                post_norm: norm_buf(&dev, store, &layout.post_attn_norm(i), 0.0)?,
                gate: QLinear::from_store(store, &layout.mlp_gate(i))?,
                up: QLinear::from_store(store, &layout.mlp_up(i))?,
                down: QLinear::from_store(store, &layout.mlp_down(i))?,
                kind,
            });
        }

        zero(&scratch.x);

        let mtp = match mtp_dir(dir) {
            Some(mdir) => {
                let st: &'static WeightStore = Box::leak(Box::new(
                    WeightStore::load_dir(&dev, &mdir)
                        .with_context(|| format!("load mtp weights from {}", mdir.display()))?,
                ));
                tracing::info!("mtp head loaded from {}", mdir.display());
                Some(Mtp::load(&dev, st, &cfg, max_t)?)
            }
            None => None,
        };

        let embed_stem = layout.embed_tokens();
        let stem = embed_stem.trim_end_matches(".weight");
        let kernels = {
            let mut b = dev.batch();
            Kernels {
                q4_gemv: b.kernel(msl::COMMON, msl::K_Q4_GEMV_HX)?,
                // Unblocked: see QLinear::encode_tile for why row blocking was tried
                // and rejected.
                // 16-byte weight loads instead of 8: the sweep is memory-latency bound and
                // this buys memory-level parallelism per instruction.  Round 043 paired
                // sweep at a reproducible clock plateau: median ratio 0.9304, 76/80 wins.
                q4_gemv_b16: b.kernel(msl::COMMON, msl::K_Q4_GEMV_B16)?,
                q4_gemv_tile: b.kernel(
                    msl::COMMON,
                    match TILE {
                        6 => msl::K_Q4_GEMV_K6_U4H,
                        4 => msl::K_Q4_GEMV_K4_U4HX,
                        _ => msl::K_Q4_GEMV_K3_U4HX,
                    },
                )?,
                rmsnorm: b.kernel(msl::COMMON, msl::K_RMSNORM)?,
                rmsnorm_ws: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_WS)?,
                rmsnorm_nw: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_NW)?,
                rmsnorm_tile: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_TILE)?,
                rmsnorm_gated: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_GATED)?,
                rope: b.kernel(msl::FUSED, msl::K_ROPE_PARTIAL)?,
                silu_mul: b.kernel(msl::FUSED, msl::K_SILU_MUL)?,
                ewise_add: b.kernel(msl::COMMON, msl::K_EWISE_ADD)?,
                gate_mul: b.kernel(msl_ops::GDN, msl_ops::K_GATE_MUL)?,
                kv_append: b.kernel(msl_ops::ATTN, msl_ops::K_KV_APPEND)?,
                attn_scores: b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_SCORES_SOFTMAX)?,
                attn_out: b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_OUT)?,
                attn_scores_rows: b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_SCORES_SOFTMAX_ROWS)?,
                attn_out_rows: b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_OUT_ROWS)?,
                gdn_seq4: b.kernel(
                    &msl_ops::gdn_src((cfg.linear_key_head_dim / gdn_q()) as i32),
                    msl_ops::K_GDN_STEP_SEQ4,
                )?,
                conv1d_ring: b.kernel(msl_ops::GDN, msl_ops::K_CONV1D_SILU_RING)?,
                conv1d_ring_tile: b.kernel(msl_ops::GDN, msl_ops::K_CONV1D_SILU_RING_TILE)?,
                gdn: b.kernel(msl_ops::GDN, msl_ops::K_GDN_STEP)?,
                gdn_seq: b.kernel(msl_ops::GDN, msl_ops::K_GDN_STEP_SEQ)?,
                rmsnorm_ws_rows: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_WS_ROWS)?,
                gate_mul_rows: b.kernel(msl_ops::GDN, msl_ops::K_GATE_MUL_ROWS)?,
                kv_append_rows: b.kernel(msl_ops::ATTN, msl_ops::K_KV_APPEND_ROWS)?,
                rope_rows: b.kernel(msl::FUSED, msl::K_ROPE_PARTIAL_ROWS)?,
                copy: b.kernel(msl_ops::GDN, msl_ops::K_COPY)?,
                round_bf16: b.kernel(msl_ops::GDN, msl_ops::K_ROUND_BF16)?,
            }
        };

        let final_norm = norm_buf(&dev, store, &layout.final_norm(), 0.0)?;
        let mut model = Self {
            cfg,
            max_t,
            vocab,
            dev,
            layers,
            scratch,
            kernels,
            embed_w: store.handle(&embed_stem)?,
            embed_s: store.handle(&format!("{stem}.scales"))?,
            embed_b: store.handle(&format!("{stem}.biases"))?,
            lm_head: QLinear::from_store(store, &layout.lm_head())?,
            final_norm,
            layout,
            mtp,
            last_dispatches: 0,
            debug: None,
            bf16_residual: std::env::var("QW_BF16_ROUND").is_ok(),
            spec_snap: spec_enabled(),
            batch,
            kv_stride: 0,
            win_stride: 0,
            state_stride: 0,
            snap_stride: 0,
        };
        // Derive the per-sequence strides from the buffers that were just sized.
        // Reading them back keeps this correct no matter how the sizes are
        // computed, and a wrong stride cannot survive the slot gate: the same
        // prompt run in two different slots has to produce the same output.
        let (st, wn, sn, kv) = {
            let (mut st, mut wn, mut sn, mut kv) = (0usize, 0usize, 0usize, 0usize);
            for layer in &model.layers {
                match &layer.kind {
                    Kind::Gdn(g) => {
                        if st == 0 {
                            st = g.state.len_bytes() / batch;
                            wn = g.window.len_bytes() / batch;
                            sn = g.snap.len_bytes() / batch;
                        }
                    }
                    Kind::Full(a) => {
                        if kv == 0 {
                            kv = a.k_cache.len_bytes() / batch;
                        }
                    }
                }
            }
            (st, wn, sn, kv)
        };
        model.state_stride = st;
        model.win_stride = wn;
        model.snap_stride = sn;
        model.kv_stride = kv;
        if std::env::var_os("QW_STRIDE_DEBUG").is_some() {
            for layer in &model.layers {
                if let Kind::Gdn(g) = &layer.kind {
                    eprintln!(
                        "stride dbg: batch={} TILE={} state.len={} snap.len={} state_stride={} snap_stride={} snap/TILE={} state/2={}",
                        batch,
                        TILE,
                        g.state.len_bytes(),
                        g.snap.len_bytes(),
                        st,
                        sn,
                        sn / TILE,
                        g.state.len_bytes() / 2
                    );
                    break;
                }
            }
        }
        model.reset();
        Ok(model)
    }

    /// Start recording the residual stream after every layer (diagnostics).
    pub fn enable_debug(&mut self) {
        let n = self.cfg.num_hidden_layers;
        self.debug = Some(self.dev.buffer(n * self.cfg.hidden_size * 2));
    }

    /// Write every recorded layer as f32 (layer-major) for offline comparison.
    pub fn write_vectors(&self, path: &std::path::Path) -> Result<()> {
        let dbg = self.debug.as_ref().context("debug capture not enabled")?;
        let h = self.cfg.hidden_size;
        let n = self.cfg.num_hidden_layers;
        let mut bytes = Vec::with_capacity(n * h * 4);
        for i in 0..n {
            for v in dbg.to_vec::<f16>(i * h, h) {
                bytes.extend_from_slice(&v.to_f32().to_le_bytes());
            }
        }
        std::fs::write(path, &bytes)?;
        Ok(())
    }

    /// Per-layer mean / std / absmax of the recorded residual stream.
    pub fn debug_stats(&self) -> Vec<(f32, f32, f32)> {
        let dbg = match &self.debug {
            Some(d) => d,
            None => return Vec::new(),
        };
        let h = self.cfg.hidden_size;
        let mut out = Vec::with_capacity(self.cfg.num_hidden_layers);
        for i in 0..self.cfg.num_hidden_layers {
            let v: Vec<f16> = dbg.to_vec(i * h, h);
            let n = v.len() as f64;
            let mean = v.iter().map(|x| x.to_f32() as f64).sum::<f64>() / n;
            let var = v
                .iter()
                .map(|x| {
                    let d = x.to_f32() as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            let absmax = v.iter().fold(0f32, |a, x| a.max(x.to_f32().abs()));
            out.push((mean as f32, var.sqrt() as f32, absmax));
        }
        out
    }

    /// Zero every recurrent / KV cache so a new prompt starts clean.
    pub fn reset(&mut self) {
        for layer in &self.layers {
            match &layer.kind {
                Kind::Gdn(g) => {
                    zero(&g.state);
                    zero(&g.window);
                }
                Kind::Full(a) => {
                    zero(&a.k_cache);
                    zero(&a.v_cache);
                }
            }
        }
        zero(&self.scratch.x);
        if let Some(m) = self.mtp.as_ref() {
            zero(&m.k_cache);
            zero(&m.v_cache);
        }
    }

    /// Zero one sequence slot's recurrent state, leaving every other slot alone.
    ///
    /// The KV cache needs no reset: attention at position `p` reads entries
    /// `0..=p` and `p` was written by `kv_append` in the same pass, so a stale
    /// entry beyond the sequence's own position is never reached.  The GDN state
    /// and the convolution ring are different - they are running summaries, not
    /// position-indexed tables - so a slot that is handed to a new request has to
    /// start them from zero.
    ///
    /// This is what makes the slot pool recyclable.  Without it a served slot
    /// could only be reused by zeroing every slot, which would destroy the state
    /// of the requests still running in the others.
    pub fn reset_seq(&mut self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Ok(());
        }
        // Zero in place.  This used to copy from a zeroed staging buffer, which
        // cost a 21 MB host allocation, a 21 MB host memset and upload, and then
        // read 1.1 GB of staging back on the GPU - 145 ms of a 1.60 s cold
        // time-to-first-token, for what is a pure write.  `zero_half` writes and
        // reads nothing.
        let _t0 = std::time::Instant::now();
        let mut b = CommandBatch::new(&mut self.dev);
        let k = b.kernel(qw_metal::msl_ops::GDN, qw_metal::msl_ops::K_ZERO_HALF)?;
        let _t1 = std::time::Instant::now();
        for layer in &self.layers {
            if let Kind::Gdn(g) = &layer.kind {
                for (buf, stride) in [(&g.state, self.state_stride), (&g.window, self.win_stride)] {
                    if stride == 0 {
                        continue;
                    }
                    // `zero_half` moves `half` elements, so lengths and offsets are
                    // in halves, not bytes.
                    let n = stride / 2;
                    b.encode(
                        Dispatch::new(&k, (n.div_ceil(8), 1, 1), (256, 1, 1))
                            .buf(1, buf)
                            .scalar(2, n as i32)
                            .scalar(4, (seq * stride / 2) as i32),
                    );
                }
            }
        }
        // The draft head's cache too.  It is a single cache addressed by absolute
        // position - its `kv_append` gets `pos` and no slot offset - and `reset`
        // has always zeroed it while `reset_seq`, the per-request reset, did not.
        // So the second request's decode attended over the first request's keys
        // and values at every position its own prefill had not just rewritten,
        // which is why the server's spec path was correct only on the first
        // request to a fresh server and diverged at character 5 after that.
        if let Some(m) = self.mtp.as_ref() {
            for buf in [&m.k_cache, &m.v_cache] {
                let n = (buf.len_bytes() / 2) as i32;
                if n <= 0 {
                    continue;
                }
                b.encode(
                    Dispatch::new(&k, ((n as usize).div_ceil(8), 1, 1), (256, 1, 1))
                        .buf(1, buf)
                        .scalar(2, n)
                        .scalar(4, 0),
                );
            }
        }
        let _t2 = std::time::Instant::now();
        b.finish(true);
        if std::env::var_os("QW_STEP_TIME").is_some() {
            eprintln!(
                "reset_seq: new+compile {:?}, encode {:?}, finish {:?}",
                _t1 - _t0,
                _t2 - _t1,
                _t2.elapsed()
            );
        }
        Ok(())
    }

    /// Keep the GPU out of its deepest idle state with a trivial dispatch.
    ///
    /// A server idle for a couple of seconds pays roughly 0.27 s of clock ramp on
    /// its next request - measured by removing the harness's two-second pause,
    /// which took a 602-token cold request from 1.783 s to 1.467 s and the
    /// `reset_seq` inside it from 205 ms to 5.8 ms.  Reissuing a few bytes every
    /// 100 ms stops the GPU from downclocking that far.  It is off by default
    /// because it is a continuous idle-power cost.
    pub fn keepwarm_tick(&mut self) -> Result<()> {
        let mut b = CommandBatch::new(&mut self.dev);
        let k = b.kernel(qw_metal::msl_ops::GDN, qw_metal::msl_ops::K_ZERO_HALF)?;
        b.encode(
            Dispatch::new(&k, (1, 1, 1), (256, 1, 1))
                .buf(1, &self.scratch.tick)
                .scalar(2, 8)
                .scalar(4, 0),
        );
        b.finish(true);
        Ok(())
    }

    /// Copy one slot's recurrent state and convolution window aside, so the position
    /// it is currently at can be returned to later.
    ///
    /// The delta-net recurrence is a running quantity and cannot be rewound, so the
    /// only way back to a position a slot has already passed is to have kept a copy
    /// from when it was there.  The prefix cache takes one of these the moment a
    /// prompt has been fully prefilled, which is exactly the position a later request
    /// will want to resume from.
    pub fn save_prefix(&mut self, seq: usize) -> Result<()> {
        self.copy_seq(seq, false)
    }

    /// Put a slot back at a position recorded by [`Self::save_prefix`].
    ///
    /// The copy is not consumed: it stays valid, so several requests can resume from
    /// the same boundary.  The KV cache is deliberately left alone - it is keyed by
    /// position, positions are written once, and a pass reads only up to the current
    /// position, so entries past the boundary are simply never read and are
    /// overwritten as the new prompt is prefilled.
    pub fn load_prefix(&mut self, seq: usize) -> Result<()> {
        self.copy_seq(seq, true)
    }

    /// Serialise everything a later *process* needs to resume slot `seq` at
    /// `pos`: the GDN recurrent state and convolution window of every linear
    /// layer, and the KV entries for positions `0..pos` of every full layer.
    ///
    /// The KV cache is head-major - `k[hk * max_t * hd + t * hd + d]` - so one
    /// head's live prefix is a single contiguous run.  That makes both
    /// directions a memcpy per head rather than a per-position gather.
    ///
    /// Layers are walked twice, all linear layers then all full ones, and
    /// `import_prefix` must walk them in exactly the same order.
    pub fn export_prefix(&mut self, seq: usize, pos: usize) -> Result<Vec<u8>> {
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let mut out = Vec::new();
        out.extend_from_slice(b"Q38PFX1\0");
        out.extend_from_slice(&(pos as u64).to_le_bytes());
        for layer in &self.layers {
            if let Kind::Gdn(g) = &layer.kind {
                for (buf, stride) in [(&g.state, self.state_stride), (&g.window, self.win_stride)] {
                    if stride == 0 {
                        continue;
                    }
                    out.extend_from_slice(&buf.read_at(seq * stride, stride));
                }
            }
        }
        for layer in &self.layers {
            if let Kind::Full(a) = &layer.kind {
                for cache in [&a.k_cache, &a.v_cache] {
                    for hk in 0..nkv {
                        let off = seq * self.kv_stride + hk * self.max_t * hd * 2;
                        out.extend_from_slice(&cache.read_at(off, pos * hd * 2));
                    }
                }
            }
        }
        Ok(out)
    }

    /// [`Self::export_prefix`], plus the logits the pass that consumed the final
    /// prompt row produced.
    ///
    /// A blob exported here describes the state *at the prompt end*, so a request
    /// that restores it has nothing left to prefill - but it still needs the first
    /// generated token, and those logits lived only in a scratch buffer that the
    /// plain export does not copy.  That is precisely why the cache used to stop
    /// one chunk short of the end, which cost every hit a re-prefill of the last
    /// chunk: one weight sweep, measured at 40-50 ms and half of warm
    /// time-to-first-token.  Carrying the logits removes that.
    ///
    /// The magic changes to `Q38PFX2` so [`Self::import_prefix`] can tell a blob
    /// that has the tail from one that does not.
    pub fn export_prefix_with_logits(&mut self, seq: usize, pos: usize, row: usize) -> Result<Vec<u8>> {
        anyhow::ensure!(row < TILE, "logits row {row} is past the {TILE}-row tile");
        let mut blob = self.export_prefix(seq, pos)?;
        blob[0..8].copy_from_slice(b"Q38PFX2\0");
        blob.extend_from_slice(&self.scratch.logits.read_at(row * self.vocab * 2, self.vocab * 2));
        Ok(blob)
    }

    /// Inverse of [`Self::export_prefix`].  Only positions `0..pos` are written;
    /// everything past `pos` is left as it was and is overwritten by the prefill
    /// that follows, which is why the KV cache needs no clearing.
    ///
    /// A `Q38PFX2` blob also carries the first generated token's logits, which are
    /// installed at row 0 for the caller to sample.
    pub fn import_prefix(&mut self, seq: usize, pos: usize, blob: &[u8]) -> Result<()> {
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        anyhow::ensure!(blob.len() >= 16, "prefix blob: too short");
        anyhow::ensure!(
            &blob[0..8] == b"Q38PFX1\0" || &blob[0..8] == b"Q38PFX2\0",
            "prefix blob: bad magic"
        );
        let stored = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
        anyhow::ensure!(
            stored == pos,
            "prefix blob holds position {stored}, asked to restore {pos}"
        );
        let mut o = 16usize;
        for layer in &self.layers {
            if let Kind::Gdn(g) = &layer.kind {
                for (buf, stride) in [(&g.state, self.state_stride), (&g.window, self.win_stride)] {
                    if stride == 0 {
                        continue;
                    }
                    anyhow::ensure!(o + stride <= blob.len(), "prefix blob: truncated state");
                    buf.write_at(seq * stride, &blob[o..o + stride]);
                    o += stride;
                }
            }
        }
        for layer in &self.layers {
            if let Kind::Full(a) = &layer.kind {
                for cache in [&a.k_cache, &a.v_cache] {
                    for hk in 0..nkv {
                        let off = seq * self.kv_stride + hk * self.max_t * hd * 2;
                        let n = pos * hd * 2;
                        anyhow::ensure!(o + n <= blob.len(), "prefix blob: truncated kv");
                        cache.write_at(off, &blob[o..o + n]);
                        o += n;
                    }
                }
            }
        }
        if self.blob_has_logits(blob) {
            let n = self.vocab * 2;
            anyhow::ensure!(o + n <= blob.len(), "prefix blob: truncated logits");
            self.scratch.logits.write_at(0, &blob[o..o + n]);
        }
        Ok(())
    }

    /// Does this prefix blob end with the first generated token's logits?
    ///
    /// True for a blob written by [`Self::export_prefix_with_logits`], which
    /// describes the state *at the prompt end* and can therefore be resumed with
    /// no prefill at all.  A `Q38PFX1` blob stops one chunk short of the end and
    /// always needs that chunk recomputed.
    pub fn blob_has_logits(&self, blob: &[u8]) -> bool {
        blob.len() >= 8 && &blob[0..8] == b"Q38PFX2\0"
    }

    fn copy_seq(&mut self, seq: usize, restore: bool) -> Result<()> {
        if seq >= self.batch {
            return Ok(());
        }
        let mut b = self.dev.batch();
        let k = b.kernel(qw_metal::msl_ops::GDN, qw_metal::msl_ops::K_COPY)?;
        // `copy_off` moves `half` elements, so lengths and offsets are in halves.
        for layer in &self.layers {
            if let Kind::Gdn(g) = &layer.kind {
                for (live, saved, stride) in [
                    (&g.state, &g.cache_state, self.state_stride),
                    (&g.window, &g.cache_window, self.win_stride),
                ] {
                    if stride == 0 {
                        continue;
                    }
                    let n = stride / 2;
                    let off = seq * stride / 2;
                    let (src, dst) = if restore {
                        (saved, live)
                    } else {
                        (live, saved)
                    };
                    copy_dispatch(&mut b, &k, src, off, dst, off, n);
                }
            }
        }
        b.finish(true);
        Ok(())
    }

    /// Make the recurrent state match the longest accepted prefix of a verify
    /// pass: `state` becomes what it was after row `row`
    /// (0-based).  Row `TILE - 1` is already current and needs no copy.
    pub fn commit_row(&mut self, row: usize) -> Result<()> {
        if row + 1 >= TILE {
            return Ok(());
        }
        let mut b = self.dev.batch();
        for layer in &self.layers {
            if let Kind::Gdn(g) = &layer.kind {
                // One step of one sequence's recurrent state.
                //
                // The single-sequence value is `len_bytes() / 2` and is exactly
                // what the shipped speculative path has always used, verified
                // byte-identical to the plain path over 300 tokens at a 57 per cent
                // draft-acceptance rate.  What it silently lacked was the slot
                // count: `state` holds `batch` sequences back to back, so on the
                // sixteen-slot server the same expression named EIGHT sequences'
                // worth, and the rewind copied that much out of the snapshot into
                // sequence 0.  Measured, that is the whole reason the server
                // diverged from the CLI whenever a draft was accepted (and matched
                // it exactly under QW_NO_ACCEPT, where no rewind happens): with
                // batch 1 the value is unchanged, byte for byte.
                let sh = g.state.len_bytes() / self.batch / 2;
                copy_dispatch(
                    &mut b,
                    &self.kernels.copy,
                    &g.snap,
                    row * sh,
                    &g.state,
                    0,
                    sh,
                );
            }
        }
        b.finish(true);
        Ok(())
    }

    /// Dequantise one embedding row on the host (10 KB) into the input buffer.
    fn embed_row(&self, token: u32) -> Result<Vec<f16>> {
        let h = self.cfg.hidden_size;
        let groups = h / 64;
        let words_per_row = h / 8;
        let wb = self.embed_w.bytes();
        let row = token as usize;
        if (row + 1) * words_per_row * 4 > wb.len() {
            bail!("token {token} out of range");
        }
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                wb[row * words_per_row * 4..].as_ptr() as *const u32,
                words_per_row,
            )
        };
        // Read this row's quantisation metadata straight out of the tensor bytes.
        // `as_bf16_f32()` converts the *whole* table - 19.9M scales plus 19.9M
        // biases, ~160 MB of allocation - and it was being paid once per token on
        // both the decode and the draft path, which is several milliseconds a call.
        let sb = self.embed_s.bytes();
        let bb = self.embed_b.bytes();
        let base = row * groups;
        if (base + groups) * 2 > sb.len() || (base + groups) * 2 > bb.len() {
            bail!("token {token} out of range");
        }
        let bf16_at = |bytes: &[u8], i: usize| -> f32 {
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) as u32;
            f32::from_bits(bits << 16)
        };
        let scales: Vec<f32> = (0..groups).map(|g| bf16_at(sb, base + g)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| bf16_at(bb, base + g)).collect();
        let mut vals = vec![0f32; h];
        for (wi, word) in words.iter().enumerate() {
            for v in 0..8 {
                let idx = wi * 8 + v;
                let q = ((word >> (4 * v)) & 0xF) as f32;
                vals[idx] = q * scales[idx / 64] + biases[idx / 64];
            }
        }
        Ok(vals.iter().map(|v| f16::from_f32(*v)).collect())
    }

    /// Embed one token per row of the input tile (`x`).
    pub fn set_tokens(&self, tokens: &[u32]) -> Result<()> {
        let mut out: Vec<f16> = Vec::with_capacity(tokens.len() * self.cfg.hidden_size);
        for t in tokens {
            out.extend_from_slice(&self.embed_row(*t)?);
        }
        self.scratch.x.copy_from(&out);
        Ok(())
    }

    /// Embed a single token into row 0 (the single-token entry point).
    pub fn set_token(&self, token: u32) -> Result<()> {
        let out = self.embed_row(token)?;
        self.scratch.x.copy_from(&out);
        Ok(())
    }

    /// Run one decode step at `pos`; the caches must be at that position.
    pub fn forward(&mut self, pos: usize) -> Result<()> {
        // QW_SLOT=<n> runs the pass against slot n's state.  With QW_BATCH=16 the
        // same prompt in slot 0 and slot 7 must produce identical output, which is
        // exactly what proves the per-sequence offsets are right.
        let seq = std::env::var("QW_SLOT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        self.forward_seq(seq, pos)
    }

    /// One token of sequence `seq`, sitting at position `pos` of that sequence's
    /// own state.  All the kernels stay single-sequence; the sequence is selected
    /// purely with byte offsets into the state buffers.
    pub fn forward_seq(&mut self, seq: usize, pos: usize) -> Result<()> {
        let Self {
            cfg,
            dev,
            layers,
            scratch,
            kernels,
            lm_head,
            final_norm,
            debug,
            bf16_residual,
            ..
        } = self;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let rot_dim = cfg.rotary_dim() as i32;
        let eps = cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let t = (pos + 1) as i32;
        let max_t = self.max_t as i32;
        let kv_off = seq * self.kv_stride;
        let win_off = seq * self.win_stride;
        let st_off = seq * self.state_stride;

        let mut b = CommandBatch::new(dev);
        for (i, layer) in layers.iter().enumerate() {
            // ---- pre-norm ----
            b.encode(
                Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &layer.input_norm)
                    .buf(2, &scratch.h)
                    .scalar(3, h as i32)
                    .scalar(4, eps),
            );
            b.barrier();

            match &layer.kind {
                Kind::Full(a) => {
                    // q (with output gate), k, v projections
                    a.q.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.qg);
                    b.barrier();
                    a.k.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.pk);
                    a.v.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.pv);
                    b.barrier();
                    // q_norm: heads live at stride 2*hd inside the q_proj output
                    // (each head emits [query | gate]).  Unlike the delta net
                    // there is NO extra query scale here — the reference only
                    // applies `scale = head_dim**-0.5` inside SDPA.
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_ws, (nh * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.qg)
                            .buf(1, &a.q_norm)
                            .buf(2, &scratch.q)
                            .scalar(3, hd as i32)
                            .scalar(4, (2 * hd) as i32)
                            .scalar(5, eps)
                            .scalar(6, 1.0f32)
                            .scalar(7, 1),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_ws, (nkv * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.pk)
                            .buf(1, &a.k_norm)
                            .buf(2, &scratch.k)
                            .scalar(3, hd as i32)
                            .scalar(4, hd as i32)
                            .scalar(5, eps)
                            .scalar(6, 1.0f32)
                            .scalar(7, 1),
                    );
                    b.barrier();
                    // partial RoPE (non-traditional pairing)
                    b.encode(
                        Dispatch::new(&kernels.rope, (nh * 64, 1, 1), (64, 1, 1))
                            .buf(0, &scratch.q)
                            .buf(1, &scratch.q)
                            .scalar(2, nh as i32)
                            .scalar(3, hd as i32)
                            .scalar(4, rot_dim)
                            .scalar(5, cfg.rope_theta() as f32)
                            .scalar(6, pos as i32),
                    );
                    b.encode(
                        Dispatch::new(&kernels.rope, (nkv * 64, 1, 1), (64, 1, 1))
                            .buf(0, &scratch.k)
                            .buf(1, &scratch.k)
                            .scalar(2, nkv as i32)
                            .scalar(3, hd as i32)
                            .scalar(4, rot_dim)
                            .scalar(5, cfg.rope_theta() as f32)
                            .scalar(6, pos as i32),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.kv_append, (nkv * hd, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.k)
                            .buf(1, &scratch.pv)
                            .buf_offset(2, &a.k_cache, kv_off)
                            .buf_offset(3, &a.v_cache, kv_off)
                            .scalar(4, pos as i32)
                            .scalar(5, max_t)
                            .scalar(6, nkv as i32)
                            .scalar(7, hd as i32),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.attn_scores, (nh * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.q)
                            .buf_offset(1, &a.k_cache, kv_off)
                            .buf(2, &scratch.scores)
                            .scalar(3, t)
                            .scalar(4, max_t)
                            .scalar(5, nh as i32)
                            .scalar(6, nkv as i32)
                            .scalar(7, hd as i32)
                            .scalar(8, scale),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.attn_out, (nh * hd, 1, 1), (hd, 1, 1))
                            .buf(0, &scratch.scores)
                            .buf_offset(1, &a.v_cache, kv_off)
                            .buf(2, &scratch.attn_out)
                            .scalar(3, t)
                            .scalar(4, max_t)
                            .scalar(5, nh as i32)
                            .scalar(6, nkv as i32)
                            .scalar(7, hd as i32),
                    );
                    b.barrier();
                    // out * sigmoid(gate) where gate sits after each head's query
                    b.encode(
                        Dispatch::new(&kernels.gate_mul, (nh * hd, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.attn_out)
                            .buf(1, &scratch.qg)
                            .buf(2, &scratch.attn_gated)
                            .scalar(3, hd as i32),
                    );
                    b.barrier();
                    a.o.encode(
                        &mut b,
                        &kernels.q4_gemv,
                        &scratch.attn_gated,
                        &scratch.proj_out,
                    );
                }
                Kind::Gdn(g) => {
                    // qkv projection writes straight into the conv window's ring slot
                    let conv_ring = (cfg.linear_conv_kernel_dim + PASS_ROWS_MAX).next_power_of_two(); // power of two: the kernel masks
                    let slot = (t - 1).rem_euclid(conv_ring as i32) as usize;
                    b.encode(
                        Dispatch::new(
                            &kernels.q4_gemv,
                            ((key_dim * 2 + value_dim) * 32, 1, 1),
                            (32, 1, 1),
                        )
                        .buf_offset(0, g.in_qkv.weight.buf, g.in_qkv.weight.offset)
                        .buf_offset(1, g.in_qkv.scales.buf, g.in_qkv.scales.offset)
                        .buf_offset(2, g.in_qkv.biases.buf, g.in_qkv.biases.offset)
                        .buf(3, &scratch.h)
                        .buf_offset(4, &g.window, win_off + slot * conv_dim * 2)
                        .scalar(5, g.in_qkv.in_f as i32),
                    );
                    g.in_z
                        .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.z);
                    g.in_b
                        .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.b);
                    g.in_a
                        .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.a);
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.conv1d_ring, (conv_dim, 1, 1), (NT, 1, 1))
                            .buf_offset(0, &g.window, win_off)
                            .buf(1, &g.conv_w)
                            .buf(2, &scratch.conv_out)
                            .scalar(3, conv_dim as i32)
                            .scalar(4, slot as i32)
                            .scalar(5, conv_ring as i32),
                    );
                    b.barrier();
                    let inv = 1.0f32 / (dk as f32).sqrt();
                    // q = inv^2 * rms_norm(q), k = inv * rms_norm(k)  (no weight)
                    // rmsnorm_s ABI: 0=x 1=w(optional) 2=y 3=D 4=in_stride 5=eps 6=scale 7=has_weight
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_nw, (hk * NT, 1, 1), (NT, 1, 1))
                            .buf_offset(0, &scratch.conv_out, 0)
                            .buf_offset(1, &scratch.conv_out, 0)
                            .buf(2, &scratch.q)
                            .scalar(3, dk as i32)
                            .scalar(4, dk as i32)
                            .scalar(5, 1e-6f32)
                            .scalar(6, inv * inv)
                            .scalar(7, 0),
                    );
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_nw, (hk * NT, 1, 1), (NT, 1, 1))
                            .buf_offset(0, &scratch.conv_out, key_dim * 2)
                            .buf_offset(1, &scratch.conv_out, key_dim * 2)
                            .buf(2, &scratch.k)
                            .scalar(3, dk as i32)
                            .scalar(4, dk as i32)
                            .scalar(5, 1e-6f32)
                            .scalar(6, inv)
                            .scalar(7, 0),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.gdn, (hv * dv, 1, 1), (dv, 1, 1))
                            .buf(0, &scratch.q)
                            .buf(1, &scratch.k)
                            .buf_offset(2, &scratch.conv_out, 2 * key_dim * 2)
                            .buf(3, &scratch.a)
                            .buf(4, &scratch.b)
                            .buf(5, &g.a_log)
                            .buf(6, &g.dt_bias)
                            .buf_offset(7, &g.state, st_off)
                            .buf(8, &scratch.gdn_y)
                            .scalar(9, hk as i32)
                            .scalar(10, hv as i32)
                            .scalar(11, dk as i32)
                            .scalar(12, dv as i32),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_gated, (hv * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.gdn_y)
                            .buf(1, &g.norm_w)
                            .buf(2, &scratch.z)
                            .buf(3, &scratch.gdn_gated)
                            .scalar(4, dv as i32)
                            .scalar(5, eps)
                            .scalar(6, 0),
                    );
                    b.barrier();
                    g.out_proj.encode(
                        &mut b,
                        &kernels.q4_gemv,
                        &scratch.gdn_gated,
                        &scratch.proj_out,
                    );
                }
            }
            b.barrier();
            // residual
            b.encode(
                Dispatch::new(&kernels.ewise_add, (h, 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &scratch.proj_out)
                    .buf(2, &scratch.x),
            );
            b.barrier();

            // ---- MLP ----
            b.encode(
                Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &layer.post_norm)
                    .buf(2, &scratch.h)
                    .scalar(3, h as i32)
                    .scalar(4, eps),
            );
            b.barrier();
            layer
                .gate
                .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.mlp_gate);
            layer
                .up
                .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.mlp_up);
            b.barrier();
            b.encode(
                Dispatch::new(&kernels.silu_mul, (cfg.intermediate_size, 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.mlp_gate)
                    .buf(1, &scratch.mlp_up)
                    .buf(2, &scratch.mlp_act),
            );
            b.barrier();
            layer.down.encode(
                &mut b,
                &kernels.q4_gemv,
                &scratch.mlp_act,
                &scratch.proj_out,
            );
            b.barrier();
            b.encode(
                Dispatch::new(&kernels.ewise_add, (h, 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &scratch.proj_out)
                    .buf(2, &scratch.x),
            );
            b.barrier();
            if *bf16_residual {
                b.encode(
                    Dispatch::new(&kernels.round_bf16, (h, 1, 1), (NT, 1, 1))
                        .buf(0, &scratch.x)
                        .buf(1, &scratch.x),
                );
                b.barrier();
            }
            if let Some(dbg) = debug.as_ref() {
                copy_dispatch(&mut b, &kernels.copy, &scratch.x, 0, dbg, i * h, h);
                b.barrier();
            }
        }

        // final norm + lm head
        b.encode(
            Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.x)
                .buf(1, final_norm)
                .buf(2, &scratch.h)
                .scalar(3, h as i32)
                .scalar(4, eps),
        );
        b.barrier();
        lm_head.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.logits);
        self.last_dispatches = b.dispatches();
        b.finish(true);
        Ok(())
    }

    /// Two-token forward: row 0 is position `pos`, row 1 is `pos + 1`.
    ///
    /// The projections run once over both rows (`TILE` wide, using the
    /// compile-time k=2 kernels); every per-row operation is issued once per
    /// row with a `row * ROW_BYTES` buffer offset.  The gated-delta-net branch
    /// stays per-row inside a single loop because its `in_proj_qkv` writes into
    /// the shared four-row convolution window.
    pub fn forward2(&mut self, pos: usize) -> Result<()> {
        let rows: Vec<(usize, usize)> = (0..TILE).map(|r| (0, pos + r)).collect();
        self.forward_rows(&rows)
    }

    /// Run `n` rows through a SINGLE weight sweep, where row `i` belongs to
    /// sequence `rows[i].0` at position `rows[i].1`.
    ///
    /// With `n == n` and every row on sequence 0 this is the speculative verify
    /// pass and is byte-for-byte what it always was.  With one row per sequence it
    /// is the batch-serving pass: the projections still read all 14.4 GB of
    /// weights exactly once for the whole tile - which is the entire point, since
    /// that is what makes 16 tokens cost one sweep instead of sixteen - while
    /// every per-row dispatch picks up its own sequence's state through a byte
    /// offset.
    ///
    /// The row count is capped at `BATCH_MAX`, and a pass may only carry more than
    /// `n` rows when they are DISTINCT sequences: the convolution ring has to
    /// hold `conv_k` rows plus everything the pass writes, which one position per
    /// sequence satisfies and sixteen consecutive positions of one sequence does
    /// not.
    pub fn forward_rows(&mut self, rows: &[(usize, usize)]) -> Result<()> {
        let n = rows.len();
        anyhow::ensure!(
            n > 0 && n <= BATCH_MAX,
            "rows must be 1..={BATCH_MAX}, got {n}"
        );
        // A pass may carry several consecutive positions of the same sequence -
        // that is exactly what chunked prefill needs - but only as many as the
        // convolution ring can hold next to the history it reads: `conv_k` rows of
        // history plus whatever this pass writes.
        let conv_k = self.cfg.linear_conv_kernel_dim;
        let max_per_seq = (conv_k + PASS_ROWS_MAX).next_power_of_two() - conv_k;
        for (i, r) in rows.iter().enumerate() {
            anyhow::ensure!(
                r.0 < self.batch,
                "row {i} uses sequence {} of {}",
                r.0,
                self.batch
            );
            let same = rows.iter().filter(|o| o.0 == r.0).count();
            anyhow::ensure!(
                same <= max_per_seq,
                "a pass may carry at most {max_per_seq} rows of one sequence, {same} were given"
            );
        }
        // Read the strides out before `self` is destructured, so the per-row
        // dispatches below never have to borrow `self` again.
        let kv_stride = self.kv_stride;
        let win_stride = self.win_stride;
        let st_stride = self.state_stride;
        let snap_stride = self.snap_stride;
        let Self {
            cfg,
            dev,
            layers,
            scratch,
            kernels,
            lm_head,
            final_norm,
            debug,
            bf16_residual,
            ..
        } = self;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let rot_dim = cfg.rotary_dim() as i32;
        let eps = cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let max_t = self.max_t as i32;

        let mut b = CommandBatch::new(dev);

        for (i, layer) in layers.iter().enumerate() {
            // ---- pre-norm ----
            b.encode(
                Dispatch::new(&kernels.rmsnorm, (n * (NT), 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &layer.input_norm)
                    .buf(2, &scratch.h)
                    .scalar(3, h as i32)
                    .scalar(4, eps),
            );
            b.barrier();
            match &layer.kind {
                Kind::Full(a) => {
                    // A prefill pass is one sequence's consecutive positions, so rope
                    // and the kv append can take the first row's position and count
                    // up; a batch pass cannot and keeps the per-row loops for those
                    // two.  The norms and the gate are per-token independent and are
                    // batched either way.
                    let one_seq = rows.iter().all(|r| r.0 == rows[0].0);
                    let am = attn_batch_mask();
                    // q (with output gate), k, v projections
                    a.q.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.qg,
                        n,
                    );
                    b.barrier();
                    a.k.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.pk,
                        n,
                    );
                    a.v.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.pv,
                        n,
                    );
                    b.barrier();
                    // q_norm: heads live at stride 2*hd inside the q_proj output
                    // (each head emits [query | gate]).  Unlike the delta net there
                    // is NO extra query scale here — the reference only applies
                    // `scale = head_dim**-0.5` inside SDPA.
                    if am & 1 != 0 {
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_ws_rows, (nh * n * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.qg)
                            .buf(1, &a.q_norm)
                            .buf(2, &scratch.q)
                            .scalar(3, hd as i32)
                            .scalar(4, (2 * hd * 2) as i32)
                            .scalar(5, eps)
                            .scalar(6, 1.0f32)
                            .scalar(7, 1)
                            .scalar(8, nh as i32)
                            .scalar(9, (nh * hd * 4) as i32)
                            .scalar(10, (nh * hd * 2) as i32),
                    );
                    } else {
                    for (row, _) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.rmsnorm_ws, (nh * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.qg, row * (nh * hd * 4))
                                .buf(1, &a.q_norm)
                                .buf_offset(2, &scratch.q, row * (nh * hd * 2))
                                .scalar(3, hd as i32)
                                .scalar(4, (2 * hd) as i32)
                                .scalar(5, eps)
                                .scalar(6, 1.0f32)
                                .scalar(7, 1),
                        );
                    }
                    }
                    b.barrier();
                    if am & 1 != 0 {
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_ws_rows, (nkv * n * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.pk)
                            .buf(1, &a.k_norm)
                            .buf(2, &scratch.k)
                            .scalar(3, hd as i32)
                            .scalar(4, (hd * 2) as i32)
                            .scalar(5, eps)
                            .scalar(6, 1.0f32)
                            .scalar(7, 1)
                            .scalar(8, nkv as i32)
                            .scalar(9, (nkv * hd * 2) as i32)
                            .scalar(10, (key_dim * 2) as i32),
                    );
                    } else {
                    for (row, _) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.rmsnorm_ws, (nkv * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.pk, row * (nkv * hd * 2))
                                .buf(1, &a.k_norm)
                                .buf_offset(2, &scratch.k, row * (key_dim * 2))
                                .scalar(3, hd as i32)
                                .scalar(4, hd as i32)
                                .scalar(5, eps)
                                .scalar(6, 1.0f32)
                                .scalar(7, 1),
                        );
                    }
                    }
                    b.barrier();
                    if one_seq && am & 4 != 0 {
                        let p0 = rows[0].1 as i32;
                        b.encode(
                            Dispatch::new(&kernels.rope_rows, (nh * n * 64, 1, 1), (64, 1, 1))
                                .buf(0, &scratch.q)
                                .buf(1, &scratch.q)
                                .scalar(2, nh as i32)
                                .scalar(3, hd as i32)
                                .scalar(4, rot_dim)
                                .scalar(5, cfg.rope_theta() as f32)
                                .scalar(6, p0)
                                .scalar(7, (nh * hd * 2) as i32),
                        );
                        b.encode(
                            Dispatch::new(&kernels.rope_rows, (nkv * n * 64, 1, 1), (64, 1, 1))
                                .buf(0, &scratch.k)
                                .buf(1, &scratch.k)
                                .scalar(2, nkv as i32)
                                .scalar(3, hd as i32)
                                .scalar(4, rot_dim)
                                .scalar(5, cfg.rope_theta() as f32)
                                .scalar(6, p0)
                                .scalar(7, (key_dim * 2) as i32),
                        );
                    } else {
                    for (row, &(_, pos)) in rows.iter().enumerate() {
                        // partial RoPE (non-traditional pairing)
                        b.encode(
                            Dispatch::new(&kernels.rope, (nh * 64, 1, 1), (64, 1, 1))
                                .buf_offset(0, &scratch.q, row * (nh * hd * 2))
                                .buf_offset(1, &scratch.q, row * (nh * hd * 2))
                                .scalar(2, nh as i32)
                                .scalar(3, hd as i32)
                                .scalar(4, rot_dim)
                                .scalar(5, cfg.rope_theta() as f32)
                                .scalar(6, pos as i32),
                        );
                        b.encode(
                            Dispatch::new(&kernels.rope, (nkv * 64, 1, 1), (64, 1, 1))
                                .buf_offset(0, &scratch.k, row * (key_dim * 2))
                                .buf_offset(1, &scratch.k, row * (key_dim * 2))
                                .scalar(2, nkv as i32)
                                .scalar(3, hd as i32)
                                .scalar(4, rot_dim)
                                .scalar(5, cfg.rope_theta() as f32)
                                .scalar(6, pos as i32),
                        );
                    }
                    }
                    b.barrier();
                    if one_seq && am & 8 != 0 {
                        b.encode(
                            Dispatch::new(&kernels.kv_append_rows, (nkv * hd * n, 1, 1), (NT, 1, 1))
                                .buf(0, &scratch.k)
                                .buf(1, &scratch.pv)
                                .buf_offset(2, &a.k_cache, rows[0].0 * kv_stride)
                                .buf_offset(3, &a.v_cache, rows[0].0 * kv_stride)
                                .scalar(4, rows[0].1 as i32)
                                .scalar(5, max_t)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32)
                                .scalar(8, key_dim as i32)
                                .scalar(9, (nkv * hd) as i32),
                        );
                    } else {
                    for (row, &(seq, pos)) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.kv_append, (nkv * hd, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.k, row * (key_dim * 2))
                                .buf_offset(1, &scratch.pv, row * (nkv * hd * 2))
                                .buf_offset(2, &a.k_cache, seq * kv_stride)
                                .buf_offset(3, &a.v_cache, seq * kv_stride)
                                .scalar(4, pos as i32)
                                .scalar(5, max_t)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32),
                        );
                    }
                    }
                    b.barrier();
                    // One dispatch for the whole pass instead of one per token:
                    // these two kernels were 94% of every dispatch the prefill
                    // made.  Row `r` attends to keys 0..pos0+r, so the causal
                    // mask is a shorter loop bound - but that is only the same
                    // computation when the pass is one sequence with consecutive
                    // positions, which is what `contiguous` checks.
                    let contiguous = attn_rows_enabled()
                        && one_seq
                        && rows
                            .iter()
                            .enumerate()
                            .all(|(r, &(_, p))| p == rows[0].1 + r);
                    if contiguous {
                        b.encode(
                            Dispatch::new(
                                &kernels.attn_scores_rows,
                                (n * nh * NT, 1, 1),
                                (NT, 1, 1),
                            )
                            .buf(0, &scratch.q)
                            .buf_offset(1, &a.k_cache, rows[0].0 * kv_stride)
                            .buf(2, &scratch.scores)
                            .scalar(3, max_t)
                            .scalar(4, nh as i32)
                            .scalar(5, nkv as i32)
                            .scalar(6, hd as i32)
                            .scalar(7, scale)
                            .scalar(8, rows[0].1 as i32),
                        );
                        b.barrier();
                        b.encode(
                            Dispatch::new(
                                &kernels.attn_out_rows,
                                (n * nh * hd, 1, 1),
                                (hd, 1, 1),
                            )
                            .buf(0, &scratch.scores)
                            .buf_offset(1, &a.v_cache, rows[0].0 * kv_stride)
                            .buf(2, &scratch.attn_out)
                            .scalar(3, max_t)
                            .scalar(4, nh as i32)
                            .scalar(5, nkv as i32)
                            .scalar(6, hd as i32)
                            .scalar(7, rows[0].1 as i32),
                        );
                    } else {
                    for (row, &(seq, pos)) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.attn_scores, (nh * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.q, row * (nh * hd * 2))
                                .buf_offset(1, &a.k_cache, seq * kv_stride)
                                .buf_offset(2, &scratch.scores, row * (nh * (max_t as usize) * 4))
                                .scalar(3, (pos + 1) as i32)
                                .scalar(4, max_t)
                                .scalar(5, nh as i32)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32)
                                .scalar(8, scale),
                        );
                    }
                    b.barrier();
                    for (row, &(seq, pos)) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.attn_out, (nh * hd, 1, 1), (hd, 1, 1))
                                .buf_offset(0, &scratch.scores, row * (nh * (max_t as usize) * 4))
                                .buf_offset(1, &a.v_cache, seq * kv_stride)
                                .buf_offset(2, &scratch.attn_out, row * (nh * hd * 2))
                                .scalar(3, (pos + 1) as i32)
                                .scalar(4, max_t)
                                .scalar(5, nh as i32)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32),
                        );
                    }
                    }
                    b.barrier();
                    // out * sigmoid(gate) where gate sits after each head's query
                    if am & 2 != 0 {
                    b.encode(
                        Dispatch::new(&kernels.gate_mul_rows, (n * (nh * hd), 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.attn_out)
                            .buf(1, &scratch.qg)
                            .buf(2, &scratch.attn_gated)
                            .scalar(3, hd as i32)
                            .scalar(4, (nh * hd) as i32),
                    );
                    } else {
                    for (row, _) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.gate_mul, (nh * hd, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.attn_out, row * (nh * hd * 2))
                                .buf_offset(1, &scratch.qg, row * (nh * hd * 4))
                                .buf_offset(2, &scratch.attn_gated, row * (nh * hd * 2))
                                .scalar(3, hd as i32),
                        );
                    }
                    }
                    b.barrier();
                    a.o.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.attn_gated,
                        &scratch.proj_out,
                        n,
                    );
                }
                Kind::Gdn(g) => {
                    g.in_z.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.z,
                        n,
                    );
                    g.in_b.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.b,
                        n,
                    );
                    g.in_a.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.a,
                        n,
                    );
                    let conv_ring = (cfg.linear_conv_kernel_dim + PASS_ROWS_MAX).next_power_of_two();
                    // Phase 1: project all n rows in one launch into the staging
                    // buffer.  The tiled kernel walks the same groups in the same
                    // order and reduces each row with the same simd_sum as the k=1
                    // kernel, so every row is bit-identical to its own launch - but
                    // the weights are read once instead of n times.
                    g.in_qkv.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.h,
                        &scratch.qkv_cur,
                        n,
                    );
                    b.barrier();
                    // Phase 2: one convolution PER ROW.  The tile-wide kernel shares
                    // a single convolution window across the whole tile, which is
                    // right for a verify pass (n consecutive positions of one
                    // sequence) and wrong for a batch pass (n different sequences,
                    // each with its own ring).  Each row's raw qkv is first moved out
                    // of the staging buffer into its own ring slot - the single-row
                    // path gets that for free by projecting straight into the window.
                    if rows.iter().all(|r| r.0 == rows[0].0) {
                        // conv1d_silu_ring_tile reads the current pass's rows straight
                        // out of the staging buffer and folds the ring update in, so
                        // it replaces BOTH the per-row copy and the per-row
                        // convolution: 12,288 launches per pass become 48.  Every
                        // window read is for a position before pos0, at least three
                        // slots behind the ones this dispatch writes, so no thread can
                        // read a slot another thread is writing.  It only applies when
                        // the rows are consecutive positions of one sequence; a batch
                        // pass of unrelated sequences keeps the per-row pair.
                        b.encode(
                            Dispatch::new(
                                &kernels.conv1d_ring_tile,
                                (n * conv_dim, 1, 1),
                                (NT, 1, 1),
                            )
                            .buf_offset(0, &g.window, rows[0].0 * win_stride)
                            .buf(1, &g.conv_w)
                            .buf(2, &scratch.conv_out)
                            .scalar(3, conv_dim as i32)
                            .scalar(4, (rows[0].1 % conv_ring) as i32)
                            .scalar(5, conv_ring as i32)
                            .buf(6, &scratch.qkv_cur)
                            .scalar(7, rows[0].1 as i32),
                        );
                    } else {
                    for (row, &(seq, pos)) in rows.iter().enumerate() {
                        let slot = pos % conv_ring;
                        let woff = seq * win_stride;
                        copy_dispatch(
                            &mut b,
                            &kernels.copy,
                            &scratch.qkv_cur,
                            row * conv_dim,
                            &g.window,
                            woff / 2 + slot * conv_dim,
                            conv_dim,
                        );
                        b.encode(
                            Dispatch::new(&kernels.conv1d_ring, (conv_dim, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &g.window, woff)
                                .buf(1, &g.conv_w)
                                .buf_offset(2, &scratch.conv_out, row * (conv_dim * 2))
                                .scalar(3, conv_dim as i32)
                                .scalar(4, slot as i32)
                                .scalar(5, conv_ring as i32),
                        );
                    }
                    }
                    b.barrier();
                    b.barrier();
                    let inv = 1.0f32 / (dk as f32).sqrt();
                    // q = inv^2 * rms_norm(q), k = inv * rms_norm(k)  (no weight)
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_tile, (hk * n * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.conv_out)
                            .buf(1, &scratch.q)
                            .scalar(2, dk as i32)
                            .scalar(3, dk as i32)
                            .scalar(4, conv_dim as i32)
                            .scalar(5, (nh * hd) as i32)
                            .scalar(6, 1e-6f32)
                            .scalar(7, inv * inv)
                            .scalar(8, hk as i32),
                    );
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_tile, (hk * n * NT, 1, 1), (NT, 1, 1))
                            .buf_offset(0, &scratch.conv_out, key_dim * 2)
                            .buf(1, &scratch.k)
                            .scalar(2, dk as i32)
                            .scalar(3, dk as i32)
                            .scalar(4, conv_dim as i32)
                            .scalar(5, key_dim as i32)
                            .scalar(6, 1e-6f32)
                            .scalar(7, inv)
                            .scalar(8, hk as i32),
                    );
                    b.barrier();
                    // Phase 3: the recurrence itself.  Every row of a prefill pass
                    // is a consecutive position of ONE sequence, so the rows share a
                    // single state slice and must be applied in order - which the
                    // fused kernel does internally, with no barrier between tokens
                    // because each thread owns its own slice.  A batch pass of
                    // unrelated sequences has no such ordering and keeps the
                    // per-row loop.
                    let one_seq = rows.iter().all(|r| r.0 == rows[0].0);
                    // The register-resident scan needs Dk and Dv to divide its
                    // tile; both are 128 here, but a different checkpoint need
                    // not be, and the old kernel is always correct.
                    let gq = gdn_q();
                    let gdvq = msl_ops::gdn_tiles()[1].max(1) as usize;
                    let seq4 = gdn_seq4_enabled()
                        && one_seq
                        && gq > 0
                        && dk % gq == 0
                        && dv % gdvq == 0;
                    if seq4 {
                        b.encode(
                            // The grid is in THREADS, not threadgroups:
                            // Hv * (Dv/DVQ) threadgroups of DVQ*Q threads.
                            Dispatch::new(
                                &kernels.gdn_seq4,
                                (hv * dv * gq, 1, 1),
                                ((gdvq * gq) as usize, 1, 1),
                            )
                                .buf(0, &scratch.q)
                                .buf(1, &scratch.k)
                                .buf_offset(2, &scratch.conv_out, 2 * key_dim * 2)
                                .buf(3, &scratch.a)
                                .buf(4, &scratch.b)
                                .buf(5, &g.a_log)
                                .buf(6, &g.dt_bias)
                                .buf_offset(7, &g.state, rows[0].0 * st_stride)
                                .buf(8, &scratch.gdn_y)
                                .scalar(9, hk as i32)
                                .scalar(10, hv as i32)
                                .scalar(11, dk as i32)
                                .scalar(12, dv as i32)
                                .buf_offset(13, &g.snap, rows[0].0 * snap_stride)
                                .scalar(14, if self.spec_snap { 1 } else { 0 })
                                .scalar(15, n as i32)
                                .scalar(16, (nh * hd) as i32)
                                .scalar(17, key_dim as i32)
                                .scalar(18, conv_dim as i32)
                                .scalar(19, hv as i32)
                                .scalar(20, value_dim as i32)
                                .scalar(21, (snap_stride / n) as i32),
                        );
                        b.barrier();
                    } else if one_seq {
                        b.encode(
                            Dispatch::new(&kernels.gdn_seq, (hv * dv, 1, 1), (dv, 1, 1))
                                .buf(0, &scratch.q)
                                .buf(1, &scratch.k)
                                .buf_offset(2, &scratch.conv_out, 2 * key_dim * 2)
                                .buf(3, &scratch.a)
                                .buf(4, &scratch.b)
                                .buf(5, &g.a_log)
                                .buf(6, &g.dt_bias)
                                .buf_offset(7, &g.state, rows[0].0 * st_stride)
                                .buf(8, &scratch.gdn_y)
                                .scalar(9, hk as i32)
                                .scalar(10, hv as i32)
                                .scalar(11, dk as i32)
                                .scalar(12, dv as i32)
                                .buf_offset(13, &g.snap, rows[0].0 * snap_stride)
                                .scalar(14, if self.spec_snap { 1 } else { 0 })
                                .scalar(15, n as i32)
                                .scalar(16, (nh * hd) as i32)
                                .scalar(17, key_dim as i32)
                                .scalar(18, conv_dim as i32)
                                .scalar(19, hv as i32)
                                .scalar(20, value_dim as i32)
                                .scalar(21, (snap_stride / n) as i32),
                        );
                        b.barrier();
                    } else {
                    for (row, &(seq, _)) in rows.iter().enumerate() {
                        b.encode(
                            Dispatch::new(&kernels.gdn, (hv * dv, 1, 1), (dv, 1, 1))
                                .buf_offset(0, &scratch.q, row * (nh * hd * 2))
                                .buf_offset(1, &scratch.k, row * (key_dim * 2))
                                .buf_offset(
                                    2,
                                    &scratch.conv_out,
                                    row * (conv_dim * 2) + (2 * key_dim * 2),
                                )
                                .buf_offset(3, &scratch.a, row * (hv * 2))
                                .buf_offset(4, &scratch.b, row * (hv * 2))
                                .buf(5, &g.a_log)
                                .buf(6, &g.dt_bias)
                                .buf_offset(7, &g.state, seq * st_stride)
                                .buf_offset(8, &scratch.gdn_y, row * (value_dim * 2))
                                .scalar(9, hk as i32)
                                .scalar(10, hv as i32)
                                .scalar(11, dk as i32)
                                .scalar(12, dv as i32)
                                .buf_offset(
                                    13,
                                    &g.snap,
                                    seq * snap_stride + row * (snap_stride / n),
                                )
                                .scalar(14, if self.spec_snap { 1 } else { 0 }),
                        );
                        b.barrier();
                    }
                    }
                    // The gated norm has no cross-row dependency, so it comes out
                    // of the recurrence loop entirely: one dispatch over all n rows
                    // instead of n, and n-1 fewer barriers per layer.  The
                    // recurrence above still needs a barrier per row - every row
                    // updates the same state slice - but this does not.
                    b.encode(
                        Dispatch::new(&kernels.rmsnorm_gated, (hv * n * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.gdn_y)
                            .buf(1, &g.norm_w)
                            .buf(2, &scratch.z)
                            .buf(3, &scratch.gdn_gated)
                            .scalar(4, dv as i32)
                            .scalar(5, eps)
                            .scalar(6, value_dim as i32),
                    );
                    b.barrier();
                    g.out_proj.encode_rows(
                        &mut b,
                        &kernels.q4_gemv,
                        &kernels.q4_gemv_tile,
                        &kernels.q4_gemv_b16,
                        &scratch.gdn_gated,
                        &scratch.proj_out,
                        n,
                    );
                }
            }
            b.barrier();
            // residual
            b.encode(
                Dispatch::new(&kernels.ewise_add, (n * (h), 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &scratch.proj_out)
                    .buf(2, &scratch.x),
            );
            b.barrier();

            // ---- MLP ----
            b.encode(
                Dispatch::new(&kernels.rmsnorm, (n * (NT), 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &layer.post_norm)
                    .buf(2, &scratch.h)
                    .scalar(3, h as i32)
                    .scalar(4, eps),
            );
            b.barrier();
            layer.gate.encode_rows(
                &mut b,
                &kernels.q4_gemv,
                &kernels.q4_gemv_tile,
                &kernels.q4_gemv_b16,
                &scratch.h,
                &scratch.mlp_gate,
                n,
            );
            layer.up.encode_rows(
                &mut b,
                &kernels.q4_gemv,
                &kernels.q4_gemv_tile,
                &kernels.q4_gemv_b16,
                &scratch.h,
                &scratch.mlp_up,
                n,
            );
            b.barrier();
            b.encode(
                Dispatch::new(
                    &kernels.silu_mul,
                    (n * (cfg.intermediate_size), 1, 1),
                    (NT, 1, 1),
                )
                .buf(0, &scratch.mlp_gate)
                .buf(1, &scratch.mlp_up)
                .buf(2, &scratch.mlp_act),
            );
            b.barrier();
            layer.down.encode_rows(
                &mut b,
                &kernels.q4_gemv,
                &kernels.q4_gemv_tile,
                &kernels.q4_gemv_b16,
                &scratch.mlp_act,
                &scratch.proj_out,
                n,
            );
            b.barrier();
            b.encode(
                Dispatch::new(&kernels.ewise_add, (n * (h), 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.x)
                    .buf(1, &scratch.proj_out)
                    .buf(2, &scratch.x),
            );
            b.barrier();
            for (row, _) in rows.iter().enumerate() {
                if *bf16_residual {
                    b.encode(
                        Dispatch::new(&kernels.round_bf16, (h, 1, 1), (NT, 1, 1))
                            .buf_offset(0, &scratch.x, row * (h * 2))
                            .buf_offset(1, &scratch.x, row * (h * 2)),
                    );
                    b.barrier();
                }
                if let Some(dbg) = debug.as_ref() {
                    copy_dispatch(&mut b, &kernels.copy, &scratch.x, 0, dbg, i * h, h);
                    b.barrier();
                }
            }
        }
        // final norm for both rows in one dispatch, then the head
        b.encode(
            Dispatch::new(&kernels.rmsnorm, (n * NT, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.x)
                .buf(1, final_norm)
                .buf(2, &scratch.h)
                .scalar(3, h as i32)
                .scalar(4, eps),
        );
        b.barrier();
        lm_head.encode_rows(
            &mut b,
            &kernels.q4_gemv,
            &kernels.q4_gemv_tile,
            &kernels.q4_gemv_b16,
            &scratch.h,
            &scratch.logits,
            n,
        );
        self.last_dispatches = b.dispatches();
        b.finish(true);
        Ok(())
    }

    /// After a two-row verify pass the hidden state for the last processed
    /// position sits in row 1 of the tile, but the MTP head reads row 0, so a
    /// speculative loop that consumed both rows must promote it before drafting
    /// the next token.  Rows are disjoint, so one dispatch is enough.
    /// Copy row `row` of the pass's hidden states down into row 0, where a
    /// following `mtp_step` looks for "the hidden of the previous position".
    pub fn promote_hidden(&mut self, row: usize) -> Result<()> {
        if row == 0 {
            return Ok(());
        }
        let h = self.cfg.hidden_size;
        let mut b = self.dev.batch();
        copy_dispatch(
            &mut b,
            &self.kernels.copy,
            &self.scratch.h,
            row * h,
            &self.scratch.h,
            0,
            h,
        );
        b.finish(true);
        Ok(())
    }

    /// Peek at row 1 of the input tile (debug aid for the two-row path).
    pub fn peek_row1(&self, n: usize) -> Vec<f32> {
        self.scratch
            .x
            .to_vec::<f16>(self.cfg.hidden_size, n)
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    /// FNV checksum of layer 0's input-norm weight (diagnostic only).
    pub fn norm_ck(&self) -> u64 {
        let b = &self.layers[0].input_norm;
        let n = b.len_bytes().min(4096);
        let bytes = b.to_vec::<u8>(0, n);
        let mut h: u64 = 1469598103934665603;
        for x in &bytes {
            h = (h ^ *x as u64).wrapping_mul(1099511628211);
        }
        h
    }

    /// True when the quantised draft head is loaded.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Advance the MTP head by one token and return its draft logits.
    ///
    /// `next_token` is the token the decoder chose for position `pos`, and the
    /// decoder's hidden state for the *previous* position must still be in
    /// `scratch.x` (call this immediately after the `forward` that produced
    /// `next_token`).  The head appends its own k/v at `pos`, so its cache stays
    /// in step with the decoder as long as it is called once per committed
    /// token; the returned logits draft `pos + 1`.
    ///
    /// With `want_logits == false` the head only advances its cache (the final
    /// head sweep is skipped), which is what prefill wants.
    /// Bytes of k (and of v) per sequence in the draft head's cache.
    fn mtp_cache_stride(&self) -> usize {
        match &self.mtp {
            Some(m) => m.k_cache.len_bytes() / self.batch.max(1),
            None => 0,
        }
    }

    pub fn mtp_step(&mut self, next_token: u32, pos: usize, want_logits: bool) -> Result<Vec<f32>> {
        self.mtp_step_at(0, next_token, pos, want_logits, 0)
    }

    /// `mtp_step`, but reading the decoder's hidden state from row `hrow` of the
    /// pass instead of row 0.
    ///
    /// The draft head consumes the decoder's hidden state for the position
    /// *before* the token being fed, and that state sits at `hrow * hidden` in
    /// the scratch residual stream once a multi-row pass has run.  A chunked
    /// prefill computes up to `PASS_ROWS_MAX` rows in one weight sweep, so it
    /// can only keep the draft head in step with the decoder if it can point
    /// the head at whichever row produced each token - which is what this
    /// offset is for.  Rows must be warmed in increasing order, because the
    /// head writes its own internal norm back into row 0's slot; every row
    /// above 0 is untouched by that write.
    /// `seq` selects which sequence's k/v region the head reads and writes.
    ///
    /// The cache is allocated `batch * nkv * max_t * hd` halves, but the kernels
    /// address it as `hk * maxT * D + pos * D + d`, i.e. they only ever see
    /// sequence 0's slice.  Binding the buffer at a byte offset of
    /// `seq * nkv * max_t * hd * 2` gives every sequence its own region without
    /// touching a single kernel - the layout already fits the allocation exactly.
    pub fn mtp_step_at(
        &mut self,
        hrow: usize,
        next_token: u32,
        pos: usize,
        want_logits: bool,
        seq: usize,
    ) -> Result<Vec<f32>> {
        let e = self.embed_row(next_token)?;
        let max_t = self.max_t as i32;
        let vocab = self.vocab;
        // Computed before `self` is destructured below.
        let koff = seq * self.mtp_cache_stride();
        let Self {
            cfg,
            dev,
            scratch,
            kernels,
            mtp,
            lm_head,
            ..
        } = self;
        let m = mtp.as_mut().context("MTP head not loaded")?;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let rot_dim = cfg.rotary_dim() as i32;
        let eps = cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let t = (pos + 1) as i32;

        // cat = [ norm(embed) | norm(hidden) ] (or the reverse, see QW_MTP_SWAP)
        let eb = dev.buffer_from_bytes(&e);
        let swap = std::env::var("QW_MTP_SWAP").is_ok();
        // The head consumes the decoder's *post*-final-norm hidden state (what
        // vLLM hands its MTP module).  QW_MTP_PRENORM=1 selects the pre-norm
        // residual instead, which costs ~22 points of acceptance.
        let postnorm = std::env::var("QW_MTP_PRENORM").is_err();
        // Ablation switch for bringing the head up: "attn" and/or "mlp" drop
        // that sub-block's contribution to the MTP residual stream.
        let skip = std::env::var("QW_MTP_SKIP").unwrap_or_default();
        let (e_off, h_off) = if swap { (h, 0usize) } else { (0usize, h) };
        let mut b = CommandBatch::new(dev);
        copy_dispatch(&mut b, &kernels.copy, &eb, 0, &m.cat, e_off, h);
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                .buf_offset(0, &m.cat, e_off * 2)
                .buf(1, &m.pre_norm_e)
                .buf_offset(2, &m.cat, e_off * 2)
                .scalar(3, h as i32)
                .scalar(4, eps),
        );
        b.encode(
            Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                .buf_offset(0, if postnorm { &scratch.h } else { &scratch.x }, hrow * h * 2)
                .buf(1, &m.pre_norm_h)
                .buf_offset(2, &m.cat, h_off * 2)
                .scalar(3, h as i32)
                .scalar(4, eps),
        );
        b.barrier();
        m.fc.encode(&mut b, &kernels.q4_gemv, &m.cat, &m.hid);
        b.barrier();
        // one full-attention decoder layer on the MTP residual stream
        b.encode(
            Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                .buf(0, &m.hid)
                .buf(1, &m.input_norm)
                .buf(2, &scratch.h)
                .scalar(3, h as i32)
                .scalar(4, eps),
        );
        b.barrier();
        m.q.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.qg);
        b.barrier();
        m.k.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.pk);
        m.v.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.pv);
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.rmsnorm_ws, (nh * NT, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.qg)
                .buf(1, &m.q_norm)
                .buf(2, &scratch.q)
                .scalar(3, hd as i32)
                .scalar(4, (2 * hd) as i32)
                .scalar(5, eps)
                .scalar(6, 1.0f32)
                .scalar(7, 1),
        );
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.rmsnorm_ws, (nkv * NT, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.pk)
                .buf(1, &m.k_norm)
                .buf(2, &scratch.k)
                .scalar(3, hd as i32)
                .scalar(4, hd as i32)
                .scalar(5, eps)
                .scalar(6, 1.0f32)
                .scalar(7, 1),
        );
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.rope, (nh * 64, 1, 1), (64, 1, 1))
                .buf(0, &scratch.q)
                .buf(1, &scratch.q)
                .scalar(2, nh as i32)
                .scalar(3, hd as i32)
                .scalar(4, rot_dim)
                .scalar(5, cfg.rope_theta() as f32)
                .scalar(6, pos as i32),
        );
        b.encode(
            Dispatch::new(&kernels.rope, (nkv * 64, 1, 1), (64, 1, 1))
                .buf(0, &scratch.k)
                .buf(1, &scratch.k)
                .scalar(2, nkv as i32)
                .scalar(3, hd as i32)
                .scalar(4, rot_dim)
                .scalar(5, cfg.rope_theta() as f32)
                .scalar(6, pos as i32),
        );
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.kv_append, (nkv * hd, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.k)
                .buf(1, &scratch.pv)
                .buf_offset(2, &m.k_cache, koff)
                .buf_offset(3, &m.v_cache, koff)
                .scalar(4, pos as i32)
                .scalar(5, max_t)
                .scalar(6, nkv as i32)
                .scalar(7, hd as i32),
        );
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.attn_scores, (nh * NT, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.q)
                .buf_offset(1, &m.k_cache, koff)
                .buf(2, &scratch.scores)
                .scalar(3, t)
                .scalar(4, max_t)
                .scalar(5, nh as i32)
                .scalar(6, nkv as i32)
                .scalar(7, hd as i32)
                .scalar(8, scale),
        );
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.attn_out, (nh * hd, 1, 1), (hd, 1, 1))
                .buf(0, &scratch.scores)
                .buf_offset(1, &m.v_cache, koff)
                .buf(2, &scratch.attn_out)
                .scalar(3, t)
                .scalar(4, max_t)
                .scalar(5, nh as i32)
                .scalar(6, nkv as i32)
                .scalar(7, hd as i32),
        );
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.gate_mul, (nh * hd, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.attn_out)
                .buf(1, &scratch.qg)
                .buf(2, &scratch.attn_gated)
                .scalar(3, hd as i32),
        );
        b.barrier();
        m.o.encode(
            &mut b,
            &kernels.q4_gemv,
            &scratch.attn_gated,
            &scratch.proj_out,
        );
        b.barrier();
        if !skip.contains("attn") {
            b.encode(
                Dispatch::new(&kernels.ewise_add, (h, 1, 1), (NT, 1, 1))
                    .buf(0, &m.hid)
                    .buf(1, &scratch.proj_out)
                    .buf(2, &m.hid),
            );
            b.barrier();
        }
        b.encode(
            Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                .buf(0, &m.hid)
                .buf(1, &m.post_norm)
                .buf(2, &scratch.h)
                .scalar(3, h as i32)
                .scalar(4, eps),
        );
        b.barrier();
        m.gate
            .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.mlp_gate);
        m.up.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.mlp_up);
        b.barrier();
        b.encode(
            Dispatch::new(&kernels.silu_mul, (cfg.intermediate_size, 1, 1), (NT, 1, 1))
                .buf(0, &scratch.mlp_gate)
                .buf(1, &scratch.mlp_up)
                .buf(2, &scratch.mlp_act),
        );
        b.barrier();
        m.down.encode(
            &mut b,
            &kernels.q4_gemv,
            &scratch.mlp_act,
            &scratch.proj_out,
        );
        b.barrier();
        if !skip.contains("mlp") {
            b.encode(
                Dispatch::new(&kernels.ewise_add, (h, 1, 1), (NT, 1, 1))
                    .buf(0, &m.hid)
                    .buf(1, &scratch.proj_out)
                    .buf(2, &m.hid),
            );
        }
        if want_logits {
            b.barrier();
            b.encode(
                Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                    .buf(0, &m.hid)
                    .buf(1, &m.norm)
                    .buf(2, &scratch.h)
                    .scalar(3, h as i32)
                    .scalar(4, eps),
            );
            b.barrier();
            lm_head.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.logits);
        }
        b.finish(true);
        if !want_logits {
            return Ok(Vec::new());
        }
        let v: Vec<f16> = scratch.logits.to_vec(0, vocab);
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// Temporary diagnostic: the first values of every MTP norm.
    pub fn mtp_norm_stats(&self) -> Vec<(&'static str, Vec<f32>)> {
        let m = match self.mtp.as_ref() {
            Some(m) => m,
            None => return Vec::new(),
        };
        let f = |b: &GpuBuffer| -> Vec<f32> {
            let v: Vec<f16> = b.to_vec(0, 4);
            v.iter().map(|x| x.to_f32()).collect()
        };
        vec![
            ("pre_norm_e", f(&m.pre_norm_e)),
            ("pre_norm_h", f(&m.pre_norm_h)),
            ("input_norm", f(&m.input_norm)),
            ("post_norm", f(&m.post_norm)),
            ("norm", f(&m.norm)),
            ("q_norm", f(&m.q_norm)),
            ("k_norm", f(&m.k_norm)),
        ]
    }

    /// Diagnostic: `(name, min, max, non-finite count)` for the MTP state and
    /// the decoder logits after a step.  Returned as one structured value so a
    /// print cannot mis-align it against its format string.
    pub fn mtp_dump(&self) -> Vec<(&'static str, f32, f32, usize)> {
        let m = match self.mtp.as_ref() {
            Some(m) => m,
            None => return Vec::new(),
        };
        let h = self.cfg.hidden_size;
        let stat = |name: &'static str, v: Vec<f32>| {
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            let mn = v
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::MAX, f32::min);
            let mx = v
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::MIN, f32::max);
            (name, mn, mx, bad)
        };
        let take = |b: &GpuBuffer, n: usize| -> Vec<f32> {
            b.to_vec::<f16>(0, n).iter().map(|x| x.to_f32()).collect()
        };
        vec![
            stat("cat", take(&m.cat, 2 * h)),
            stat("hid", take(&m.hid, h)),
            stat("normed", take(&self.scratch.h, h)),
            stat("logits", take(&self.scratch.logits, self.vocab)),
        ]
    }

    /// Index of the largest logit (host side).
    /// Speculative decoding rewinds the recurrent state after a rejected draft,
    /// which needs the per-row snapshots.  Every front end that calls
    /// `spec_step` has to switch this on: it used to be tied to the `QW_SPEC`
    /// environment variable, which the server never sets, so `commit_row` was a
    /// silent no-op there and each rejected draft stayed in the GDN state.
    pub fn enable_spec_snap(&mut self) {
        self.spec_snap = true;
    }

    /// One speculative step, shared by every front end.
    ///
    /// `next` is a token already settled at `pos` but not yet emitted.  The step
    /// drafts `TILE - 1` tokens ahead with the MTP head (each chained on the
    /// previous draft), verifies all of them in one `TILE`-wide forward pass,
    /// appends every accepted token to `out`, and returns the position and
    /// settled token for the next call, plus the draft and verify times in
    /// seconds so a caller can report them.
    pub fn spec_step(
        &mut self,
        pos: usize,
        next: u32,
        out: &mut Vec<u32>,
        seq: usize,
    ) -> Result<(usize, u32, f64, f64)> {
        use std::time::Instant;
        out.push(next);
        let t_draft = Instant::now();
        let mut d = [0u32; TILE - 1];
        for i in 0..TILE - 1 {
            let tok_in = if i == 0 { next } else { d[i - 1] };
            d[i] = Self::argmax_of(&self.mtp_step_at(0, tok_in, pos + i, true, seq)?);
        }
        let draft = t_draft.elapsed().as_secs_f64();
        let t_verify = Instant::now();
        let mut toks = Vec::with_capacity(TILE);
        toks.push(next);
        toks.extend_from_slice(&d);
        let t_set = Instant::now();
        self.set_tokens(&toks)?;
        let set_ms = t_set.elapsed().as_secs_f64() * 1e3;
        let t_fwd = Instant::now();
        self.forward2(pos)?;
        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
        let t_log = Instant::now();
        let mut r = [0u32; TILE];
        for (i, slot) in r.iter_mut().enumerate() {
            *slot = Self::argmax_of(&self.logits_row(i));
        }
        let log_ms = t_log.elapsed().as_secs_f64() * 1e3;
        let verify = t_verify.elapsed().as_secs_f64();
        if std::env::var("QW_TAIL").is_ok() {
            eprintln!(
                "  verify: set_tokens {:.2} ms | forward2 {:.2} ms | {}x(logits pull + argmax) {:.2} ms",
                set_ms,
                fwd_ms,
                TILE,
                log_ms
            );
        }
        // The longest prefix of drafts the target agrees with.  Row `k` settles
        // the token after the last accepted draft, so it becomes the next `next`
        // for free.
        let t_tail = Instant::now();
        let mut k = 0usize;
        while k < TILE - 1 && d[k] == r[k] {
            k += 1;
        }
        // Diagnostic: force the pass to accept nothing.  The emitted sequence then
        // has to match the plain path exactly, which bisects the acceptance
        // machinery against the shared forward.
        if std::env::var("QW_NO_ACCEPT").is_ok() {
            k = 0;
        }
        out.extend(d.iter().take(k));
        // Row `TILE - 1` is already current and needs no rewind.
        let t_commit = Instant::now();
        if k + 1 < TILE {
            self.commit_row(k)?;
        }
        let commit_ms = t_commit.elapsed().as_secs_f64() * 1e3;
        let t_promo = Instant::now();
        // The chained drafts after the first were computed from row 0, which still
        // held the hidden of `pos`; re-append the last accepted one with the hidden
        // it actually needs.  Dropping this costs 6 points of acceptance.
        if k >= 1 {
            self.promote_hidden(k - 1)?;
            self.mtp_step_at(0, d[k - 1], pos + k, false, seq)?;
        }
        self.promote_hidden(k)?;
        if std::env::var("QW_TAIL").is_ok() {
            eprintln!(
                "  tail: commit {:.2} ms | promote+redraft {:.2} ms | acceptance+bookkeeping {:.2} ms | tail total {:.2} ms",
                commit_ms,
                t_promo.elapsed().as_secs_f64() * 1e3,
                t_promo.elapsed().as_secs_f64() * 1e3 - commit_ms - 0.0,
                t_tail.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok((pos + k + 1, r[k], draft, verify))
    }

    pub fn argmax_of(v: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, x) in v.iter().enumerate() {
            if *x > v[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Logits of one row of the last two-token forward, copied to the host.
    pub fn logits_row(&self, row: usize) -> Vec<f32> {
        let v: Vec<f16> = self.scratch.logits.to_vec(row * self.vocab, self.vocab);
        v.iter().map(|x| x.to_f32()).collect()
    }

    /// Logits of the last `forward` call, copied to the host.
    pub fn logits(&self) -> Vec<f32> {
        let v: Vec<f16> = self.scratch.logits.to_vec(0, self.vocab);
        v.iter().map(|x| x.to_f32()).collect()
    }

    /// Re-run the LM head on the current hidden state in its own command
    /// buffer (debug aid: separates "bad input" from "bad ordering").
    pub fn probe_lm_head(&mut self) -> Result<Vec<f32>> {
        let Self {
            dev,
            scratch,
            kernels,
            lm_head,
            ..
        } = self;
        let mut b = CommandBatch::new(dev);
        lm_head.encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.logits);
        b.finish(true);
        Ok(scratch
            .logits
            .to_vec::<f16>(0, 4)
            .iter()
            .map(|v| v.to_f32())
            .collect())
    }

    /// Debug view of an internal buffer ("x", "h", "logits").
    pub fn peek(&self, which: &str, n: usize) -> Vec<f32> {
        let buf = match which {
            "x" => &self.scratch.x,
            "h" => &self.scratch.h,
            "gdn_gated" => &self.scratch.gdn_gated,
            "attn_gated" => &self.scratch.attn_gated,
            _ => &self.scratch.logits,
        };
        buf.to_vec::<f16>(0, n).iter().map(|v| v.to_f32()).collect()
    }

    /// Number of dispatches encoded by the last forward pass.
    /// Time the two attention kernels, and a projection as a control, in isolation.
    ///
    /// Section 15 of `docs/PLAN_BATCH16.md` leaves one thing unexplained: a pass at
    /// position 6000 costs about 20 ms per row more than one at position 1000, and no
    /// accounting for attention over the history reaches that number.  Rather than infer
    /// it a third time, dispatch the kernels by themselves and see which one grows.
    ///
    /// Measurement only: it writes into scratch buffers that the next pass overwrites
    /// anyway, and touches no recurrent state, no KV cache and no position.
    pub fn bench_attn(&mut self, t: usize, iters: usize) -> Result<(f64, f64, f64)> {
        anyhow::ensure!(t > 0 && t <= self.max_t, "t must be within max_t");
        anyhow::ensure!(iters > 0, "iters must be positive");
        let max_t = self.max_t;
        let Self {
            dev,
            layers,
            scratch,
            kernels,
            cfg,
            ..
        } = self;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let kv_off = 0usize;
        let full = layers
            .iter()
            .find_map(|l| match &l.kind {
                Kind::Full(a) => Some(a.as_ref()),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("model has no full-attention layer"))?;

        let mut b = CommandBatch::new(dev);
        for _ in 0..iters {
            b.encode(
                Dispatch::new(&kernels.attn_scores, (nh * NT, 1, 1), (NT, 1, 1))
                    .buf(0, &scratch.q)
                    .buf_offset(1, &full.k_cache, kv_off)
                    .buf(2, &scratch.scores)
                    .scalar(3, t as i32)
                    .scalar(4, max_t as i32)
                    .scalar(5, nh as i32)
                    .scalar(6, nkv as i32)
                    .scalar(7, hd as i32)
                    .scalar(8, scale),
            );
            b.barrier();
        }
        let t0 = std::time::Instant::now();
        b.finish(true);
        let scores = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let mut b = CommandBatch::new(dev);
        for _ in 0..iters {
            b.encode(
                Dispatch::new(&kernels.attn_out, (nh * hd, 1, 1), (hd, 1, 1))
                    .buf(0, &scratch.scores)
                    .buf_offset(1, &full.v_cache, kv_off)
                    .buf(2, &scratch.attn_out)
                    .scalar(3, t as i32)
                    .scalar(4, max_t as i32)
                    .scalar(5, nh as i32)
                    .scalar(6, nkv as i32)
                    .scalar(7, hd as i32),
            );
            b.barrier();
        }
        let t0 = std::time::Instant::now();
        b.finish(true);
        let out = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Control: a projection at four rows.  It reads the same weights whatever `t`
        // is, so if this moves with `t` the drift is the machine, not the attention.
        let mut b = CommandBatch::new(dev);
        for _ in 0..iters {
            full.q.encode_rows(
                &mut b,
                &kernels.q4_gemv,
                &kernels.q4_gemv_tile,
                &kernels.q4_gemv_b16,
                &scratch.h,
                &scratch.q,
                TILE,
            );
            b.barrier();
        }
        let t0 = std::time::Instant::now();
        b.finish(true);
        let proj = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        Ok((scores, out, proj))
    }

    pub fn last_dispatches(&self) -> usize {
        self.last_dispatches
    }

    pub fn argmax(&self) -> u32 {
        let v: Vec<f16> = self.scratch.logits.to_vec(0, self.vocab);
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, x) in v.iter().enumerate() {
            let f = x.to_f32();
            if f > best_v {
                best_v = f;
                best = i;
            }
        }
        best as u32
    }

    pub fn top_k(&self, k: usize) -> Vec<(u32, f32)> {
        let v: Vec<f16> = self.scratch.logits.to_vec(0, self.vocab);
        let mut items: Vec<(u32, f32)> = v
            .iter()
            .enumerate()
            .map(|(i, x)| (i as u32, x.to_f32()))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items.truncate(k);
        items
    }

    /// Number of dispatches the last forward encoded (perf bookkeeping).
    pub fn layout_name(&self, i: usize) -> String {
        format!("{}.{}", self.layout.prefix, i)
    }
}

fn zero(buf: &GpuBuffer) {
    let bytes = vec![0u8; buf.len_bytes()];
    buf.copy_from(&bytes);
}
