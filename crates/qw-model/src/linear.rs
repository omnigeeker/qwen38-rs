//! Quantised linear layers bound directly to the zero-copy weight store.

use anyhow::{anyhow, bail, Result};
use qw_metal::{msl, CommandBatch, Dispatch, GpuBuffer, Kernel};
use qw_weights::{QuantSpec, TensorHandle, WeightStore};

/// An affine 4-bit `nn.Linear` (no bias), resolved to GPU buffer offsets.
pub struct QLinear<'a> {
    pub name: String,
    pub weight: TensorHandle<'a>,
    pub scales: TensorHandle<'a>,
    pub biases: TensorHandle<'a>,
    pub out_f: usize,
    pub in_f: usize,
    pub spec: QuantSpec,
}

impl<'a> QLinear<'a> {
    /// `weight_name` is the `.weight` key; `scales`/`biases` are derived.
    pub fn from_store(store: &'a WeightStore, weight_name: &str) -> Result<Self> {
        if !weight_name.ends_with(".weight") {
            bail!("expected a .weight name, got {weight_name}");
        }
        let stem = &weight_name[..weight_name.len() - ".weight".len()];
        let weight = store.handle(weight_name)?;
        let scales = store.handle(&format!("{stem}.scales"))?;
        let biases = store.handle(&format!("{stem}.biases"))?;

        let wshape = weight.shape();
        if wshape.len() != 2 {
            bail!("{weight_name}: expected 2-D packed weight, got {wshape:?}");
        }
        let out_f = wshape[0];
        let in_f = wshape[1] * QuantSpec::MLX_4BIT.values_per_word();
        let sshape = scales.shape();
        if sshape != [out_f, in_f / QuantSpec::MLX_4BIT.group_size] {
            bail!(
                "{weight_name}: scales shape {sshape:?} != [{out_f}, {}]",
                in_f / QuantSpec::MLX_4BIT.group_size
            );
        }
        Ok(Self {
            name: stem.to_string(),
            weight,
            scales,
            biases,
            out_f,
            in_f,
            spec: QuantSpec::MLX_4BIT,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.spec.bytes_for(self.out_f * self.in_f)
            + self.out_f * (self.in_f / self.spec.group_size) * 4
    }

    /// Check that every pointer the kernel will load from is suitably aligned
    /// for 16-byte `half4` / 4-byte `uint` vector loads.
    pub fn alignment_ok(&self) -> bool {
        self.weight.offset.is_multiple_of(4)
            && self.scales.offset.is_multiple_of(8)
            && self.biases.offset.is_multiple_of(8)
    }

    /// Encode `y = W x` for a single token (`x` is `[in_f]` fp16, `y` is `[out_f]`).
    pub fn encode(&self, batch: &mut CommandBatch, kernel: &Kernel, x: &GpuBuffer, y: &GpuBuffer) {
        let d = Dispatch::new(kernel, (self.out_f * 32, 1, 1), (32, 1, 1))
            .buf_offset(0, self.weight.buf, self.weight.offset)
            .buf_offset(1, self.scales.buf, self.scales.offset)
            .buf_offset(2, self.biases.buf, self.biases.offset)
            .buf(3, x)
            .buf(4, y)
            .scalar(5, self.in_f as i32);
        batch.encode(d);
    }

    /// Encode `y = W x` for `k` tokens at once: `x` is `[k][in_f]`, `y` is
    /// `[k][out_f]`.  The weights are read once and reused for every token,
    /// which is what makes speculative verification bandwidth-free.
    pub fn encode_k(
        &self,
        batch: &mut CommandBatch,
        kernel: &Kernel,
        x: &GpuBuffer,
        y: &GpuBuffer,
        k: usize,
    ) {
        let d = Dispatch::new(kernel, (self.out_f * 32, 1, 1), (32, 1, 1))
            .buf_offset(0, self.weight.buf, self.weight.offset)
            .buf_offset(1, self.scales.buf, self.scales.offset)
            .buf_offset(2, self.biases.buf, self.biases.offset)
            .buf(3, x)
            .buf(4, y)
            .scalar(5, self.in_f as i32)
            .scalar(6, k as i32)
            .scalar(7, self.out_f as i32);
        batch.encode(d);
    }

    /// Single switch point for the `TILE`-token verify GEMV.
    ///
    /// Row blocking was tried here - `TILE_ROWS` output rows per threadgroup, which
    /// divides the x load traffic by that factor - and measured about 5% SLOWER
    /// end-to-end in the model over 8 tightly interleaved pairs (6 of 8 slower, and
    /// slower in every pair where the baseline itself was not an outlier), even
    /// though the isolated sweep behind `bench --rows 9` preferred it.  The sweep
    /// reuses one input buffer for all 497 linears, so x stays cache-hot there and
    /// the very traffic row blocking removes is understated; in the model the kernel
    /// also has to cover linears with tiny `out_f` (the GDN a/b projections are 48
    /// rows), where blocking leaves the grid nearly empty.  The variants stay in
    /// msl.rs, and `--rows 9` can revisit them if the reading is ever reversed.
    pub fn encode_tile(
        &self,
        batch: &mut CommandBatch,
        kernel: &Kernel,
        x: &GpuBuffer,
        y: &GpuBuffer,
        k: usize,
    ) {
        self.encode_k(batch, kernel, x, y, k);
    }

    /// Multi-token GEMV with `rows` output rows per threadgroup and the x slice
    /// held in registers (benchmark A/B switch, see `qwen38 bench --rows`).
    pub fn encode_kr(
        &self,
        batch: &mut CommandBatch,
        kernel: &Kernel,
        x: &GpuBuffer,
        y: &GpuBuffer,
        k: usize,
        rows: usize,
    ) {
        let d = Dispatch::new(kernel, (self.out_f.div_ceil(rows) * 32, 1, 1), (32, 1, 1))
            .buf_offset(0, self.weight.buf, self.weight.offset)
            .buf_offset(1, self.scales.buf, self.scales.offset)
            .buf_offset(2, self.biases.buf, self.biases.offset)
            .buf(3, x)
            .buf(4, y)
            .scalar(5, self.in_f as i32)
            .scalar(6, k as i32)
            .scalar(7, self.out_f as i32)
            .scalar(8, rows as i32);
        batch.encode(d);
    }

    /// Batch-`b` GEMV where one weight row is shared by `b` consecutive
    /// threadgroups (see `msl::q4_gemv_b16`).  `x` is `[b][in_f]`, `y` is
    /// `[b][out_f]`.  Unlike `encode_k` the grid grows with the batch, because the
    /// parallelism is what buys the L2 reuse.
    pub fn encode_b(
        &self,
        batch: &mut CommandBatch,
        kernel: &Kernel,
        x: &GpuBuffer,
        y: &GpuBuffer,
        b: usize,
    ) {
        let d = Dispatch::new(kernel, (b * self.out_f * 32, 1, 1), (32, 1, 1))
            .buf_offset(0, self.weight.buf, self.weight.offset)
            .buf_offset(1, self.scales.buf, self.scales.offset)
            .buf_offset(2, self.biases.buf, self.biases.offset)
            .buf(3, x)
            .buf(4, y)
            .scalar(5, self.in_f as i32)
            .scalar(6, b as i32)
            .scalar(7, self.out_f as i32);
        batch.encode(d);
    }

    /// Project `rows` activation rows, dispatching to the kernel that can
    /// actually do that many.  The tile kernels are compile-time specialisations
    /// (`K4_U4HX` is exactly four rows - feeding it sixteen silently projects only
    /// the first four and leaves the rest of the tile stale), so anything wider
    /// has to go through the grid-mapped kernel, which puts the extra parallelism
    /// in the grid and shares one weight row between consecutive threadgroups.
    /// Both write `y[row * out_f + r]`, so callers cannot tell which one ran.
    pub fn encode_rows(
        &self,
        batch: &mut CommandBatch,
        single_k: &Kernel,
        tile_k: &Kernel,
        batch_k: &Kernel,
        x: &GpuBuffer,
        y: &GpuBuffer,
        rows: usize,
    ) {
        if rows == 1 {
            // A one-row pass has nothing for the TILE-wide kernel to share, and
            // `K4_U4HX` is a four-row specialisation: fed a single row it still
            // computes four.  The plain single-token kernel does one.  Measured
            // over a real decode step, this is the whole difference between the
            // server's 497 tile dispatches and `gen`'s 497 `q4_gemv_hx` ones.
            self.encode(batch, single_k, x, y);
        } else if rows <= crate::runner::TILE {
            self.encode_tile(batch, tile_k, x, y, rows);
        } else {
            // Matrix units, when asked for.  Off by default because the two paths
            // round differently - the GEMM's worst relative error over all 497
            // linears is 9.662e-4 and the tile kernel's max_abs runs 5.0e-4 to
            // 1.6e-3 on the same reference - so switching changes the emitted
            // text.  The acceptance test is therefore the mlx-lm oracle, not
            // byte-identity with the scalar path.  One dispatch per linear
            // instead of rows/4, and the MACs go through the matrix units: the
            // MMA runs at 36 TFLOPS against the scalar path's 5.4.
            if rows >= gemm_min_rows() && self.out_f >= gemm_min_out_f() {
                if let Err(e) = batch.kernel(msl::COMMON, msl::K_Q4_GEMM_TILE) {
                    eprintln!("q4_gemm_tile FAILED TO BUILD ({e:?}); its dispatch is skipped and y stays zero");
                }
                // Compile check for the MetalPerformancePrimitives tensor-op path.
                // Silent when it works; loud if a future edit breaks it, because
                // that path is the way out of the sixteen refuted staging ideas.
                if let Err(e) = batch.kernel(msl::MPP, "q4_mpp_probe") {
                    eprintln!("q4_mpp_probe FAILED TO BUILD: {e}");
                }
                if let Ok(gk) = batch.kernel(msl::COMMON, msl::K_Q4_GEMM_TILE) {
                    // Split-K.  With a small out_f the tile grid alone cannot fill
                    // the GPU, so a wide pass leans on grid.y for parallelism and
                    // ends up reading the whole 14.4 GB of weights once per 32
                    // tokens.  Cutting K four ways reads them once per pass and
                    // grid.z restores the threadgroup count to the 640 that the BN
                    // sweep measured as the bandwidth sweet spot.  QW_SPLITK=0
                    // turns it off, which is how it was A/B'd.
                    if splitk_enabled(self.out_f, self.in_f, rows) {
                        if let Ok(rk) = batch.kernel(msl::COMMON, msl::K_Q4_GEMM_REDUCE) {
                            const S: usize = 4;
                            let scratch = batch.buffer(S * rows * self.out_f * 4);
                            let d = Dispatch::new(
                                &gk,
                                (self.out_f.div_ceil(32) * 128, rows.div_ceil(32), S),
                                (128, 1, 1),
                            )
                            .buf_offset(0, self.weight.buf, self.weight.offset)
                            .buf_offset(1, self.scales.buf, self.scales.offset)
                            .buf_offset(2, self.biases.buf, self.biases.offset)
                            .buf(3, x)
                            .buf(4, &scratch)
                            .scalar(5, self.in_f as i32)
                            .scalar(6, rows as i32)
                            .scalar(7, self.out_f as i32)
                            .scalar(8, 4);
                            batch.encode(d);
                            let n = rows * self.out_f;
                            let dr = Dispatch::new(&rk, (n, 1, 1), (256, 1, 1))
                                .buf(0, &scratch)
                                .buf(1, y)
                                .scalar(2, n as i32);
                            batch.encode(dr);
                            return;
                        }
                    }
                    const BM: usize = 32;
                    const BN: usize = 32;
                    let d = Dispatch::new(
                        &gk,
                        (self.out_f.div_ceil(BM) * 128, rows.div_ceil(BN), 1),
                        (128, 1, 1),
                    )
                    .buf_offset(0, self.weight.buf, self.weight.offset)
                    .buf_offset(1, self.scales.buf, self.scales.offset)
                    .buf_offset(2, self.biases.buf, self.biases.offset)
                    .buf(3, x)
                    .buf(4, y)
                    .scalar(5, self.in_f as i32)
                    .scalar(6, rows as i32)
                    .scalar(7, self.out_f as i32)
                    .scalar(8, std::env::var("QW_GEMM_MODE").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0));
                    batch.encode(d);
                    return;
                }
            }
            // More rows than the tile kernel's fixed four.  `K4_U4HX` is unrolled
            // over four tokens and IGNORES the `k` scalar, so a partial chunk still
            // computes four rows and still writes four rows of `y`; the x and y
            // buffers are sized for `BATCH_MAX`, which is at least
            // `PASS_ROWS_MAX`, so the three extra rows of the last chunk land inside
            // the allocation and their outputs are simply never read.  Running the
            // tile kernel in a loop is what makes a wide pass cheap: the per-pass
            // overhead is a fixed chain of dependent dispatches that does not care
            // how many rows ride along, so carrying 32 rows instead of 4 amortises
            // it eight-fold.
            let mut off = 0usize;
            while off < rows {
                let d = Dispatch::new(tile_k, (self.out_f * 32, 1, 1), (32, 1, 1))
                    .buf_offset(0, self.weight.buf, self.weight.offset)
                    .buf_offset(1, self.scales.buf, self.scales.offset)
                    .buf_offset(2, self.biases.buf, self.biases.offset)
                    .buf_offset(3, x, off * self.in_f * 2)
                    .buf_offset(4, y, off * self.out_f * 2)
                    .scalar(5, self.in_f as i32)
                    .scalar(6, crate::runner::TILE as i32)
                    .scalar(7, self.out_f as i32);
                batch.encode(d);
                off += crate::runner::TILE;
            }
            let _ = batch_k;
        }
    }

    /// Compile (or fetch) the multi-token GEMV entry point.
    pub fn kernel_k(batch: &mut CommandBatch) -> Result<Kernel> {
        batch.kernel(msl::COMMON, msl::K_Q4_GEMV_K)
    }

    /// Compile (or fetch) the GEMV entry point.
    pub fn kernel(batch: &mut CommandBatch) -> Result<Kernel> {
        batch.kernel(msl::COMMON, msl::K_Q4_GEMV)
    }

    /// CPU reference: dequantise this layer's real weights and multiply.
    pub fn cpu_reference(&self, x: &[half::f16]) -> Result<Vec<f32>> {
        if x.len() != self.in_f {
            bail!("x has {} elements, expected {}", x.len(), self.in_f);
        }
        let packed: &[u8] = self.weight.bytes();
        // scales/biases are BF16 in this checkpoint (see qw-metal::msl docs)
        let scales = self.scales.as_bf16_f32();
        let biases = self.biases.as_bf16_f32();
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u32,
                packed.len() / std::mem::size_of::<u32>(),
            )
        };
        if words.len() < self.out_f * (self.in_f / 8) {
            return Err(anyhow!(
                "{}: packed weight too small ({} words)",
                self.name,
                words.len()
            ));
        }
        let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        let mut out = vec![0f32; self.out_f];
        let groups = self.in_f / self.spec.group_size;
        for (row, o) in out.iter_mut().enumerate() {
            let wrow = &words[row * (self.in_f / 8)..(row + 1) * (self.in_f / 8)];
            let srow = &scales[row * groups..(row + 1) * groups];
            let brow = &biases[row * groups..(row + 1) * groups];
            let mut acc = 0f32;
            for (wi, word) in wrow.iter().enumerate() {
                for v in 0..8 {
                    let idx = wi * 8 + v;
                    let q = ((word >> (4 * v)) & 0xF) as f32;
                    let w = q * srow[idx / self.spec.group_size] + brow[idx / self.spec.group_size];
                    acc += w * xf[idx];
                }
            }
            *o = acc;
        }
        Ok(out)
    }
}

/// Row count at or above which `encode_rows` uses the GEMM instead of looping the
/// four-row tile kernel.  `QW_GEMM_MIN_ROWS` selects it; unset means never, which
/// is the shipped default because the two paths round differently.
fn gemm_min_out_f() -> usize {
    // The GEMM's grid is (out_f / BM) * 128 threads, so a small out_f yields very
    // few threadgroups: 5120 rows is 160 of them, about two waves across 40 SMs,
    // and that is exactly the regime where the weight staging only reaches 39 GB/s
    // against 61 GB/s once there are four times as many.  The four-row tile kernel
    // launches out_f * 32 threads - eight times as many - so it may well win on the
    // small tensors.  0 keeps the GEMM everywhere, which is the measured default
    // until an interleaved A/B says otherwise.
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("QW_GEMM_MIN_OUT_F")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Split-K is only worth it where the tile grid alone cannot fill the GPU.  This
/// model has exactly four out_f values - 48, 5120, 17408 and 248320 - and at
/// 17408 and above grid.x already supplies 544 or more threadgroups, so only 5120
/// needs the help.  It also needs a wide pass: with few rows the extra reduction
/// dispatch costs more than the weight traffic it saves.  QW_SPLITK=0 disables it.
/// OFF by default: `QW_SPLITK=1` opts in.  With BN left at 32 the traffic algebra
/// says splitting K changes nothing - W*(rows/BN) does not depend on how K is cut -
/// and that is exactly what it measured: 5/8 with a median of minus 1 per cent.
/// It was built to pair with a 128-wide tile, but that combination measured 72 per
/// cent slower because sixteen accumulators per simdgroup cost more than the saved
/// traffic.  The code stays, proved correct by the gates, but it is not the default
/// because it has never been shown to be faster.
fn splitk_enabled(out_f: usize, in_f: usize, rows: usize) -> bool {
    if std::env::var("QW_SPLITK").ok().as_deref() != Some("1") {
        return false;
    }
    rows >= 32 && out_f >= 1024 && out_f <= 5120 && in_f >= 512
}

fn gemm_min_rows() -> usize {
    // Read once: this is consulted for every linear of every pass, so a plain
    // `std::env::var` here would be 497 allocations on a prefill's hot path.
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("QW_GEMM_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            // 8 = on by default.  Measured on the same tensors and the same CPU
            // reference, the GEMM is more accurate than the four-row kernel we used
            // to ship (max_abs better by 1.3-1.9x, rel by 1.5-1.8x, on all four
            // checked tensors), and 2.2x faster on a cold prefill.  Set
            // QW_GEMM_MIN_ROWS very high to fall back to the scalar loop.
            .unwrap_or(8)
    })
}
