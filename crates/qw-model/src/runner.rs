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
    q4_gemv_k2: Kernel,
    rmsnorm: Kernel,
    rmsnorm_ws: Kernel,
    rmsnorm_nw: Kernel,
    rmsnorm_gated: Kernel,
    rope: Kernel,
    silu_mul: Kernel,
    ewise_add: Kernel,
    gate_mul: Kernel,
    kv_append: Kernel,
    attn_scores: Kernel,
    attn_out: Kernel,
    conv1d: Kernel,
    gdn: Kernel,
    copy: Kernel,
    round_bf16: Kernel,
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
    /// `[conv_k - 1][conv_dim]` raw pre-conv rows
    conv_hist: GpuBuffer,
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
pub const TILE: usize = 2;

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
    window: GpuBuffer,
    conv_out: GpuBuffer,
    gdn_y: GpuBuffer,
    gdn_gated: GpuBuffer,
    mlp_gate: GpuBuffer,
    mlp_up: GpuBuffer,
    mlp_act: GpuBuffer,
    logits: GpuBuffer,
    /// fp32 `[n_heads][max_t]`
    scores: GpuBuffer,
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
    last_dispatches: usize,
    /// `[layer][hidden]` copies of the residual stream (dumped on demand)
    debug: Option<GpuBuffer>,
    /// round the residual stream through bf16 after every layer, matching the
    /// bf16 numerics of the mlx-lm reference (see docs/M1_NOTES.md)
    pub bf16_residual: bool,
}

const NT: usize = 256;

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
        let mut dev = GpuDevice::new()?;
        let cfg = ModelConfig::from_path(&dir.join("config.json"))?.text_config;
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
        if conv_k != 4 {
            bail!("conv kernel {conv_k} != 4 is not implemented");
        }
        let vocab = cfg.vocab_size;

        // Two-token tile (see docs/PLAN_K2.md).  Step 1 only *reserves* the
        // space: every dispatch still addresses row 0, so nothing observable can
        // change until the k=2 kernels are wired in.  `window` is deliberately
        // not doubled - it is the 4-row convolution history shared across steps,
        // not a per-token activation tile.
        let tile = TILE;
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
            window: dev.buffer(conv_k * conv_dim * 2),
            conv_out: dev.buffer(conv_dim * 2 * tile),
            gdn_y: dev.buffer(value_dim * 2 * tile),
            gdn_gated: dev.buffer(value_dim * 2 * tile),
            mlp_gate: dev.buffer(cfg.intermediate_size * 2 * tile),
            mlp_up: dev.buffer(cfg.intermediate_size * 2 * tile),
            mlp_act: dev.buffer(cfg.intermediate_size * 2 * tile),
            logits: dev.buffer(vocab * 2 * tile),
            scores: dev.buffer(nh * max_t * 4 * tile),
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
                    state: dev.buffer(hv * dv * dk * 4),
                    conv_hist: dev.buffer((conv_k - 1) * conv_dim * 2),
                }))
            } else {
                Kind::Full(Box::new(FullAttn {
                    q: QLinear::from_store(store, &layout.attn_q(i))?,
                    k: QLinear::from_store(store, &layout.attn_k(i))?,
                    v: QLinear::from_store(store, &layout.attn_v(i))?,
                    o: QLinear::from_store(store, &layout.attn_o(i))?,
                    q_norm: norm_buf(&dev, store, &layout.attn_q_norm(i), 0.0)?,
                    k_norm: norm_buf(&dev, store, &layout.attn_k_norm(i), 0.0)?,
                    k_cache: dev.buffer(nkv * max_t * hd * 2),
                    v_cache: dev.buffer(nkv * max_t * hd * 2),
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

        let embed_stem = layout.embed_tokens();
        let stem = embed_stem.trim_end_matches(".weight");
        let kernels = {
            let mut b = dev.batch();
            Kernels {
                q4_gemv: b.kernel(msl::COMMON, msl::K_Q4_GEMV)?,
                q4_gemv_k2: b.kernel(msl::COMMON, msl::K_Q4_GEMV_K2)?,
                rmsnorm: b.kernel(msl::COMMON, msl::K_RMSNORM)?,
                rmsnorm_ws: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_WS)?,
                rmsnorm_nw: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_NW)?,
                rmsnorm_gated: b.kernel(msl_ops::GDN, msl_ops::K_RMSNORM_GATED)?,
                rope: b.kernel(msl::FUSED, msl::K_ROPE_PARTIAL)?,
                silu_mul: b.kernel(msl::FUSED, msl::K_SILU_MUL)?,
                ewise_add: b.kernel(msl::COMMON, msl::K_EWISE_ADD)?,
                gate_mul: b.kernel(msl_ops::GDN, msl_ops::K_GATE_MUL)?,
                kv_append: b.kernel(msl_ops::ATTN, msl_ops::K_KV_APPEND)?,
                attn_scores: b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_SCORES_SOFTMAX)?,
                attn_out: b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_OUT)?,
                conv1d: b.kernel(msl_ops::GDN, msl_ops::K_CONV1D_SILU)?,
                gdn: b.kernel(msl_ops::GDN, msl_ops::K_GDN_STEP)?,
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
            last_dispatches: 0,
            debug: None,
            bf16_residual: std::env::var("QW_BF16_ROUND").is_ok(),
        };
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
                    zero(&g.conv_hist);
                }
                Kind::Full(a) => {
                    zero(&a.k_cache);
                    zero(&a.v_cache);
                }
            }
        }
        zero(&self.scratch.x);
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
        let scales = self.embed_s.as_bf16_f32();
        let biases = self.embed_b.as_bf16_f32();
        let scales = &scales[row * groups..(row + 1) * groups];
        let biases = &biases[row * groups..(row + 1) * groups];
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
                            .buf(2, &a.k_cache)
                            .buf(3, &a.v_cache)
                            .scalar(4, pos as i32)
                            .scalar(5, max_t)
                            .scalar(6, nkv as i32)
                            .scalar(7, hd as i32),
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.attn_scores, (nh * NT, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.q)
                            .buf(1, &a.k_cache)
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
                            .buf(1, &a.v_cache)
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
                    // qkv projection writes straight into the conv window's last row
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
                        .buf_offset(4, &scratch.window, 3 * conv_dim * 2)
                        .scalar(5, g.in_qkv.in_f as i32),
                    );
                    g.in_z
                        .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.z);
                    g.in_b
                        .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.b);
                    g.in_a
                        .encode(&mut b, &kernels.q4_gemv, &scratch.h, &scratch.a);
                    b.barrier();
                    copy_dispatch(
                        &mut b,
                        &kernels.copy,
                        &g.conv_hist,
                        0,
                        &scratch.window,
                        0,
                        3 * conv_dim,
                    );
                    b.barrier();
                    b.encode(
                        Dispatch::new(&kernels.conv1d, (conv_dim, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.window)
                            .buf(1, &g.conv_w)
                            .buf(2, &scratch.conv_out)
                            .scalar(3, conv_dim as i32),
                    );
                    b.encode(
                        Dispatch::new(&kernels.copy, (3 * conv_dim, 1, 1), (NT, 1, 1))
                            .buf(0, &scratch.window)
                            .buf(1, &g.conv_hist)
                            .scalar(2, (3 * conv_dim) as i32)
                            .scalar(3, conv_dim as i32)
                            .scalar(4, 0),
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
                            .buf(7, &g.state)
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
                            .scalar(5, eps),
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

        let mut b = CommandBatch::new(dev);

        for (i, layer) in layers.iter().enumerate() {
            for row in 0..TILE {
                // ---- pre-norm ----
                b.encode(
                    Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                        .buf_offset(0, &scratch.x, row * (h * 2))
                        .buf(1, &layer.input_norm)
                        .buf_offset(2, &scratch.h, row * (h * 2))
                        .scalar(3, h as i32)
                        .scalar(4, eps),
                );
            }
            b.barrier();
            match &layer.kind {
                Kind::Full(a) => {
                    // q (with output gate), k, v projections
                    a.q.encode_k(&mut b, &kernels.q4_gemv_k2, &scratch.h, &scratch.qg, TILE);
                    b.barrier();
                    a.k.encode_k(&mut b, &kernels.q4_gemv_k2, &scratch.h, &scratch.pk, TILE);
                    a.v.encode_k(&mut b, &kernels.q4_gemv_k2, &scratch.h, &scratch.pv, TILE);
                    b.barrier();
                    for row in 0..TILE {
                        // q_norm: heads live at stride 2*hd inside the q_proj output
                        // (each head emits [query | gate]).  Unlike the delta net
                        // there is NO extra query scale here — the reference only
                        // applies `scale = head_dim**-0.5` inside SDPA.
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
                    b.barrier();
                    for row in 0..TILE {
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
                    b.barrier();
                    for row in 0..TILE {
                        // partial RoPE (non-traditional pairing)
                        b.encode(
                            Dispatch::new(&kernels.rope, (nh * 64, 1, 1), (64, 1, 1))
                                .buf_offset(0, &scratch.q, row * (nh * hd * 2))
                                .buf_offset(1, &scratch.q, row * (nh * hd * 2))
                                .scalar(2, nh as i32)
                                .scalar(3, hd as i32)
                                .scalar(4, rot_dim)
                                .scalar(5, cfg.rope_theta() as f32)
                                .scalar(6, (pos + row) as i32),
                        );
                        b.encode(
                            Dispatch::new(&kernels.rope, (nkv * 64, 1, 1), (64, 1, 1))
                                .buf_offset(0, &scratch.k, row * (key_dim * 2))
                                .buf_offset(1, &scratch.k, row * (key_dim * 2))
                                .scalar(2, nkv as i32)
                                .scalar(3, hd as i32)
                                .scalar(4, rot_dim)
                                .scalar(5, cfg.rope_theta() as f32)
                                .scalar(6, (pos + row) as i32),
                        );
                    }
                    b.barrier();
                    for row in 0..TILE {
                        b.encode(
                            Dispatch::new(&kernels.kv_append, (nkv * hd, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.k, row * (key_dim * 2))
                                .buf_offset(1, &scratch.pv, row * (nkv * hd * 2))
                                .buf(2, &a.k_cache)
                                .buf(3, &a.v_cache)
                                .scalar(4, (pos + row) as i32)
                                .scalar(5, max_t)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32),
                        );
                    }
                    b.barrier();
                    for row in 0..TILE {
                        b.encode(
                            Dispatch::new(&kernels.attn_scores, (nh * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.q, row * (nh * hd * 2))
                                .buf(1, &a.k_cache)
                                .buf_offset(2, &scratch.scores, row * (nh * (max_t as usize) * 4))
                                .scalar(3, t + row as i32)
                                .scalar(4, max_t)
                                .scalar(5, nh as i32)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32)
                                .scalar(8, scale),
                        );
                    }
                    b.barrier();
                    for row in 0..TILE {
                        b.encode(
                            Dispatch::new(&kernels.attn_out, (nh * hd, 1, 1), (hd, 1, 1))
                                .buf_offset(0, &scratch.scores, row * (nh * (max_t as usize) * 4))
                                .buf(1, &a.v_cache)
                                .buf_offset(2, &scratch.attn_out, row * (nh * hd * 2))
                                .scalar(3, t + row as i32)
                                .scalar(4, max_t)
                                .scalar(5, nh as i32)
                                .scalar(6, nkv as i32)
                                .scalar(7, hd as i32),
                        );
                    }
                    b.barrier();
                    for row in 0..TILE {
                        // out * sigmoid(gate) where gate sits after each head's query
                        b.encode(
                            Dispatch::new(&kernels.gate_mul, (nh * hd, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.attn_out, row * (nh * hd * 2))
                                .buf_offset(1, &scratch.qg, row * (nh * hd * 4))
                                .buf_offset(2, &scratch.attn_gated, row * (nh * hd * 2))
                                .scalar(3, hd as i32),
                        );
                    }
                    b.barrier();
                    a.o.encode_k(
                        &mut b,
                        &kernels.q4_gemv_k2,
                        &scratch.attn_gated,
                        &scratch.proj_out,
                        TILE,
                    );
                }
                Kind::Gdn(g) => {
                    g.in_z
                        .encode_k(&mut b, &kernels.q4_gemv_k2, &scratch.h, &scratch.z, TILE);
                    g.in_b
                        .encode_k(&mut b, &kernels.q4_gemv_k2, &scratch.h, &scratch.b, TILE);
                    g.in_a
                        .encode_k(&mut b, &kernels.q4_gemv_k2, &scratch.h, &scratch.a, TILE);
                    for row in 0..TILE {
                        // qkv projection writes straight into the conv window's last row
                        b.encode(
                            Dispatch::new(
                                &kernels.q4_gemv,
                                ((key_dim * 2 + value_dim) * 32, 1, 1),
                                (32, 1, 1),
                            )
                            .buf_offset(0, g.in_qkv.weight.buf, g.in_qkv.weight.offset)
                            .buf_offset(1, g.in_qkv.scales.buf, g.in_qkv.scales.offset)
                            .buf_offset(2, g.in_qkv.biases.buf, g.in_qkv.biases.offset)
                            .buf_offset(3, &scratch.h, row * (h * 2))
                            .buf_offset(4, &scratch.window, 3 * conv_dim * 2)
                            .scalar(5, g.in_qkv.in_f as i32),
                        );
                        b.barrier();
                        copy_dispatch(
                            &mut b,
                            &kernels.copy,
                            &g.conv_hist,
                            0,
                            &scratch.window,
                            0,
                            3 * conv_dim,
                        );
                        b.barrier();
                        b.encode(
                            Dispatch::new(&kernels.conv1d, (conv_dim, 1, 1), (NT, 1, 1))
                                .buf(0, &scratch.window)
                                .buf(1, &g.conv_w)
                                .buf_offset(2, &scratch.conv_out, row * (conv_dim * 2))
                                .scalar(3, conv_dim as i32),
                        );
                        b.encode(
                            Dispatch::new(&kernels.copy, (3 * conv_dim, 1, 1), (NT, 1, 1))
                                .buf(0, &scratch.window)
                                .buf(1, &g.conv_hist)
                                .scalar(2, (3 * conv_dim) as i32)
                                .scalar(3, conv_dim as i32)
                                .scalar(4, 0),
                        );
                        b.barrier();
                        let inv = 1.0f32 / (dk as f32).sqrt();
                        // q = inv^2 * rms_norm(q), k = inv * rms_norm(k)  (no weight)
                        // rmsnorm_s ABI: 0=x 1=w(optional) 2=y 3=D 4=in_stride 5=eps 6=scale 7=has_weight
                        b.encode(
                            Dispatch::new(&kernels.rmsnorm_nw, (hk * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.conv_out, row * (conv_dim * 2))
                                .buf_offset(1, &scratch.conv_out, row * (conv_dim * 2))
                                .buf_offset(2, &scratch.q, row * (nh * hd * 2))
                                .scalar(3, dk as i32)
                                .scalar(4, dk as i32)
                                .scalar(5, 1e-6f32)
                                .scalar(6, inv * inv)
                                .scalar(7, 0),
                        );
                        b.encode(
                            Dispatch::new(&kernels.rmsnorm_nw, (hk * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(
                                    0,
                                    &scratch.conv_out,
                                    row * (conv_dim * 2) + (key_dim * 2),
                                )
                                .buf_offset(
                                    1,
                                    &scratch.conv_out,
                                    row * (conv_dim * 2) + (key_dim * 2),
                                )
                                .buf_offset(2, &scratch.k, row * (key_dim * 2))
                                .scalar(3, dk as i32)
                                .scalar(4, dk as i32)
                                .scalar(5, 1e-6f32)
                                .scalar(6, inv)
                                .scalar(7, 0),
                        );
                        b.barrier();
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
                                .buf(7, &g.state)
                                .buf_offset(8, &scratch.gdn_y, row * (value_dim * 2))
                                .scalar(9, hk as i32)
                                .scalar(10, hv as i32)
                                .scalar(11, dk as i32)
                                .scalar(12, dv as i32),
                        );
                        b.barrier();
                        b.encode(
                            Dispatch::new(&kernels.rmsnorm_gated, (hv * NT, 1, 1), (NT, 1, 1))
                                .buf_offset(0, &scratch.gdn_y, row * (value_dim * 2))
                                .buf(1, &g.norm_w)
                                .buf_offset(2, &scratch.z, row * (value_dim * 2))
                                .buf_offset(3, &scratch.gdn_gated, row * (value_dim * 2))
                                .scalar(4, dv as i32)
                                .scalar(5, eps),
                        );
                        b.barrier();
                    }
                    g.out_proj.encode_k(
                        &mut b,
                        &kernels.q4_gemv_k2,
                        &scratch.gdn_gated,
                        &scratch.proj_out,
                        TILE,
                    );
                }
            }
            b.barrier();
            for row in 0..TILE {
                // residual
                b.encode(
                    Dispatch::new(&kernels.ewise_add, (h, 1, 1), (NT, 1, 1))
                        .buf_offset(0, &scratch.x, row * (h * 2))
                        .buf_offset(1, &scratch.proj_out, row * (h * 2))
                        .buf_offset(2, &scratch.x, row * (h * 2)),
                );
            }
            b.barrier();
            for row in 0..TILE {
                // ---- MLP ----
                b.encode(
                    Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                        .buf_offset(0, &scratch.x, row * (h * 2))
                        .buf(1, &layer.post_norm)
                        .buf_offset(2, &scratch.h, row * (h * 2))
                        .scalar(3, h as i32)
                        .scalar(4, eps),
                );
            }
            b.barrier();
            layer.gate.encode_k(
                &mut b,
                &kernels.q4_gemv_k2,
                &scratch.h,
                &scratch.mlp_gate,
                TILE,
            );
            layer.up.encode_k(
                &mut b,
                &kernels.q4_gemv_k2,
                &scratch.h,
                &scratch.mlp_up,
                TILE,
            );
            b.barrier();
            for row in 0..TILE {
                b.encode(
                    Dispatch::new(&kernels.silu_mul, (cfg.intermediate_size, 1, 1), (NT, 1, 1))
                        .buf_offset(0, &scratch.mlp_gate, row * (cfg.intermediate_size * 2))
                        .buf_offset(1, &scratch.mlp_up, row * (cfg.intermediate_size * 2))
                        .buf_offset(2, &scratch.mlp_act, row * (cfg.intermediate_size * 2)),
                );
            }
            b.barrier();
            layer.down.encode_k(
                &mut b,
                &kernels.q4_gemv_k2,
                &scratch.mlp_act,
                &scratch.proj_out,
                TILE,
            );
            b.barrier();
            for row in 0..TILE {
                b.encode(
                    Dispatch::new(&kernels.ewise_add, (h, 1, 1), (NT, 1, 1))
                        .buf_offset(0, &scratch.x, row * (h * 2))
                        .buf_offset(1, &scratch.proj_out, row * (h * 2))
                        .buf_offset(2, &scratch.x, row * (h * 2)),
                );
            }
            b.barrier();
            for row in 0..TILE {
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
        // final norm + lm head for both rows
        for row in 0..TILE {
            b.encode(
                Dispatch::new(&kernels.rmsnorm, (NT, 1, 1), (NT, 1, 1))
                    .buf_offset(0, &scratch.x, row * h * 2)
                    .buf(1, final_norm)
                    .buf_offset(2, &scratch.h, row * h * 2)
                    .scalar(3, h as i32)
                    .scalar(4, eps),
            );
        }
        b.barrier();
        lm_head.encode_k(
            &mut b,
            &kernels.q4_gemv_k2,
            &scratch.h,
            &scratch.logits,
            TILE,
        );
        self.last_dispatches = b.dispatches();
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

    /// Read one embedding row straight from the weight buffer, on the host.
    /// Used to detect GPU writes that land outside their buffer.
    pub fn embed_probe(&self, token: u32) -> Vec<f32> {
        self.embed_row(token)
            .map(|v| v.iter().map(|x| x.to_f32()).collect())
            .unwrap_or_default()
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
