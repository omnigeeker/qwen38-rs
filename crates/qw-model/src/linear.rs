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
        wide_k: &Kernel,
        mpp_k: Option<&Kernel>,
        mpp_small_k: Option<&Kernel>,
        mpp_mid_k: Option<&Kernel>,
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
                // The `q4_gemm_tile` compile check that used to sit here is gone.
                // It resolved the kernel, threw the handle away and printed on
                // failure - once per quantised linear, on every pass.  That is 497
                // full `fxhash`es of the COMMON translation unit per prefill, for
                // a result nothing read, and `QW_ENCODE_TIME` priced it at 62.9 ms
                // of CPU encode on a 598-row pass.  The fall-through below still
                // resolves and dispatches the kernel, and still degrades to a
                // zeroed `y` if it does not build, which is the same outcome the
                // check could produce.
                // QW_MPP=3: the affine tensor-op GEMM.  Tile M(tokens)=128,
                // N(rows)=64, K=32, weight tile only in shared memory, and the
                // activations read straight from device memory by the tensor op.
                // Proved numerically by `qwen38 mpp-test` (rel 1.8e-7) before
                // anything was built on it.
                // The tensor-op GEMM is the default path: it is numerically
                // identical to the hand-written kernel on every shape checked
                // (same max_abs to the last digit, out_f=48 included) and 2.15x
                // faster on a cold prefill, because the tensor op reads the
                // activations straight from device memory and runs the MACs at
                // 44 TFLOPS against simdgroup_matrix's 22.  QW_MPP=0 forces the
                // old kernel back on; if the MPP source ever fails to compile the
                // old kernel is used anyway, because the block below falls through.
                //
                // The kernel handle arrives resolved.  It used to be looked up
                // here - twice, with a `String` clone of the source each time -
                // and `pipeline()` hashes the whole source to key its cache, so
                // 497 linears meant ~994 full hashes of the MPP translation unit
                // per pass.  `QW_ENCODE_TIME` priced that at 100 ms of CPU encode
                // on a 598-row prefill, against 0.85 ms for a whole decode pass.
                let use_mpp = matches!(
                    std::env::var("QW_MPP").ok().as_deref(),
                    None | Some("3")
                );
                if use_mpp {
                    // Pick the tile whose M extent actually covers this pass.  The
                    // descriptor's M is a compile-time constant, so the wide kernel
                    // spends a 256-token tile on a sixteen-row pass; the narrow one
                    // spends 32.  Measured over the real 497 linears, cost is
                    // `0.10 s + 0.0013 * NRB` while one M-block covers the pass, so
                    // the narrow kernel is worth it up to the point where the extra
                    // M-blocks cost more than the width saves - which the sweep
                    // puts at 128 rows.
                    let (mpp_sel, tiles) = if mpp_small_k.is_some() && rows <= MPP_SMALL_MAX_ROWS {
                        (mpp_small_k, msl::mpp_tiles_small())
                    } else if mpp_mid_k.is_some() && rows <= MPP_NARROW_MAX_ROWS {
                        (mpp_mid_k, msl::mpp_tiles_mid())
                    } else {
                        (mpp_k, msl::mpp_tiles())
                    };
                    if let Some(mk) = mpp_sel {
                        let [_, nra, nrb, nt] = tiles;
                        let (nra, nrb, nt) = (nra as usize, nrb as usize, nt as usize);
                        batch.encode(
                            Dispatch::new(
                                &mk,
                                (rows.div_ceil(nrb) * nt, self.out_f.div_ceil(nra), 1),
                                (nt, 1, 1),
                            )
                            .buf_offset(0, self.weight.buf, self.weight.offset)
                            .buf_offset(1, self.scales.buf, self.scales.offset)
                            .buf_offset(2, self.biases.buf, self.biases.offset)
                            .buf(3, x)
                            .buf(4, y)
                            .scalar(5, self.in_f as i32)
                            .scalar(6, rows as i32)
                            .scalar(7, self.out_f as i32)
                            .scalar(
                                8,
                                std::env::var("QW_GEMM_MODE")
                                    .ok()
                                    .and_then(|v| v.parse::<i32>().ok())
                                    .unwrap_or(0),
                            ),
                        );
                        return;
                    }
                }
                // mode 7: run the MetalPerformancePrimitives probe instead of the
                // hand-written tile kernel.  Its output is WRONG - it computes
                // `A x q` and ignores the per-group scale and bias - so this exists
                // purely to measure what the tensor op costs on our shapes before
                // investing in the affine epilogue.  The lm_head is skipped because
                // its float output tile would be 127 MB.
                // mode 8: the affine-aware tensor path.  Exact, unlike mode 7:
                // one matmul per 64-weight group with the per-(column, group)
                // scale and bias applied to the in-register accumulator.
                // Selected by its own variable, not QW_GEMM_MODE: that one is also
                // handed to q4_gemm_tile as a scalar, so reusing it made the old
                // kernel take a debug branch and poisoned every comparison.
                if std::env::var("QW_MPP").ok().as_deref() == Some("2")
                    && self.out_f <= 17408
                {
                    let ng = self.in_f / 64;
                    let gsum = batch.buffer(rows * ng * 4);
                    let cacc = batch.buffer(rows * self.out_f * 4);
                    if let Ok(gs) = batch.kernel(msl::MPP, "q4_group_sums") {
                        batch.encode(
                            Dispatch::new(&gs, (rows * ng, 1, 1), (256, 1, 1))
                                .buf(0, x)
                                .buf(1, &gsum)
                                .scalar(2, self.in_f as i32)
                                .scalar(3, ng as i32),
                        );
                    }
                    batch.barrier();
                    if let Err(e) = batch.kernel(msl::MPP, "q4_mpp_affine_v2") {
                        eprintln!("q4_mpp_affine FAILED TO BUILD: {e}");
                    }
                    if let Ok(ak) = batch.kernel(msl::MPP, "q4_mpp_affine_v2") {
                        batch.encode(
                            Dispatch::new(
                                &ak,
                                (self.out_f.div_ceil(32) * 128, rows.div_ceil(64), 1),
                                (128, 1, 1),
                            )
                            .buf(0, x)
                            .buf_offset(1, self.weight.buf, self.weight.offset)
                            .buf_offset(2, self.scales.buf, self.scales.offset)
                            .buf_offset(3, self.biases.buf, self.biases.offset)
                            .buf(4, &gsum)
                            .buf(5, &cacc)
                            .scalar(6, ng as i32)
                            .scalar(7, self.out_f as i32)
                            .buf(8, &cacc),
                        );
                    }
                    batch.barrier();
                    if let Ok(sk) = batch.kernel(msl::MPP, "q4_store_half") {
                        let n = rows * self.out_f;
                        batch.encode(
                            Dispatch::new(&sk, (n, 1, 1), (256, 1, 1))
                                .buf(0, &cacc)
                                .buf(1, y)
                                .scalar(2, n as i32),
                        );
                        return;
                    }
                }
                if std::env::var("QW_MPP").ok().as_deref() == Some("1")
                    && self.out_f <= 17408
                {
                    if let Ok(pk) = batch.kernel(msl::MPP, "q4_mpp_probe") {
                        batch.encode(
                            Dispatch::new(
                                &pk,
                                (self.out_f.div_ceil(32) * 128, rows.div_ceil(64), 1),
                                (128, 1, 1),
                            )
                            .buf(0, x)
                            .buf_offset(1, self.weight.buf, self.weight.offset)
                            .buf(2, y),
                        );
                        return;
                    }
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
            //
            // Round 190: the wide tile (`wide_k`, NR=2 x NK=8) halves the number of
            // weight sweeps for rows 5..31 by carrying eight rows per sweep instead
            // of four.  Measured on the real 497-linears sweep, paired in one
            // process with a duplicate arm as the instrument's own bias:
            //
            //   tokens  engine today        wide tile          ratio   calibration
            //     8     71.02 ms (2 x k4)   55.34 ms (1 x t28) 0.7792  0.9878
            //    16    140.27 ms (4 x k4)  107.80 ms (2 x t28) 0.7685  0.9987
            //
            // At four tokens the tiled kernel is neutral (35.30 against 36.24,
            // 0.9741 raw) and NR=4 is worse, so `rows <= TILE` keeps the k4 path.
            //
            // A sweep of eight is not free: it costs 55.34 ms against 35.62 ms for
            // four, because the kernel becomes ALU-bound (8 tokens x 5.76 ms of
            // MACs) rather than bandwidth-bound.  So a wide sweep is only worth it
            // when it replaces TWO four-row sweeps, and the schedule below spends a
            // wide sweep on a remainder only when that remainder would otherwise
            // cost two narrow ones - which is the case for 5, 6, 7 and 13..15 rows.
            // The whole decision is made per dispatch from the measured constants,
            // not from a table, so a future re-measurement is a one-line change.
            const T28_MS: f64 = 55.34; // one NR=2 x NK=8 sweep
            const K4_MS: f64 = 35.62; // one four-row sweep
            let wide_ok = self.out_f % 2 == 0 && self.out_f >= wide_min_out_f();
            let mut off = 0usize;
            while off < rows {
                let rem = rows - off;
                let use_wide = wide_ok
                    && (rem as f64 / 8.0).ceil() * T28_MS < (rem as f64 / 4.0).ceil() * K4_MS;
                let (kern, step, grid_div) = if use_wide {
                    // NR=2 output rows per threadgroup, so the grid halves.
                    (wide_k, 8usize, 2usize)
                } else {
                    (tile_k, crate::runner::TILE, 1usize)
                };
                let d = Dispatch::new(
                    kern,
                    (self.out_f.div_ceil(grid_div) * 32, 1, 1),
                    (32, 1, 1),
                )
                    .buf_offset(0, self.weight.buf, self.weight.offset)
                    .buf_offset(1, self.scales.buf, self.scales.offset)
                    .buf_offset(2, self.biases.buf, self.biases.offset)
                    .buf_offset(3, x, off * self.in_f * 2)
                    .buf_offset(4, y, off * self.out_f * 2)
                    .scalar(5, self.in_f as i32)
                    .scalar(6, step as i32)
                    .scalar(7, self.out_f as i32);
                batch.encode(d);
                off += step;
            }
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

/// Smallest `out_f` for which `encode_rows` will spend a wide (NR=2 x NK=8) sweep
/// on a pass wider than four rows.
///
/// The wide kernel gives each threadgroup TWO output rows, so it launches half the
/// threadgroups the four-row kernel does: 256 for `out_f` 512, 24 for `out_f` 48.
/// 24 threadgroups on 40 SMs is under a wave and the sweep would be latency-bound
/// instead of bandwidth-bound - the same failure the rejected row-blocking variant
/// hit.  This model has one small `out_f` (48, the GDN a/b projections, 96 of the
/// 497 linears and 0.08 per cent of the weight bytes) and the rest are 1024 and up,
/// so a 256-row floor excludes exactly that case and nothing else.
/// `QW_T28_MIN_OUT_F` overrides it; 0 forces the wide path everywhere.
fn wide_min_out_f() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("QW_T28_MIN_OUT_F")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// Largest pass that uses the NRB=32 MPP tile.
pub const MPP_SMALL_MAX_ROWS: usize = 32;

/// Largest pass that uses the NRB=64 MPP tile.  Above this the NRB=256 tile wins.
///
/// The tile has to track the row count, because the descriptor's M is a
/// compile-time constant: measured over a whole forward pass, 48 and 64 rows are
/// fastest at NRB=64 (0.1499 and 0.1575 against 0.1754 and 0.1843 at NRB=32) and
/// 96 and 128 at NRB=128 (0.2172 and 0.2333 against 0.2576 and 0.3327).  64 is
/// within 7 per cent of the best across the whole range, so one middle tile
/// covers it.
pub const MPP_NARROW_MAX_ROWS: usize = 128;

fn gemm_min_rows() -> usize {
    // Read once: this is consulted for every linear of every pass, so a plain
    // `std::env::var` here would be 497 allocations on a prefill's hot path.
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("QW_GEMM_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            // 14 = on by default.  It was 32 because the MPP tile used to be
            // NRB=256 wide, so a sixteen-row pass spent a 256-token tile and cost
            // 0.436 s - far worse than the 0.120 s the wide GEMV takes.  With the
            // narrow (NRB=32) tile resolved in `Kernels::q4_mpp_small` the same
            // pass costs 0.109 s, and three alternating rounds on a near-cool box
            // gave 0.1093 / 0.1096 / 0.1089 against 0.1196 / 0.1233 / 0.1203 for
            // the wide GEMV: 1.10x, and at 32 rows the two are identical (0.1184
            // against 0.1182).  The MPP floor is the weight sweep, about 0.105 s
            // regardless of row count, and the wide GEMV costs about 7.5 ms per
            // token, so the crossover is ~14 rows.  Below it the scalar path still
            // wins: at 8 rows the MPP floor is 1.66x the GEMV's 0.064 s.
            //
            // The old note here is kept because its failure mode still applies to
            // the wide tile.  Measured on the same tensors and the same CPU
            // reference, the GEMM is more accurate than the four-row kernel we used
            // to ship (max_abs better by 1.3-1.9x, rel by 1.5-1.8x, on all four
            // checked tensors), and 2.2x faster on a cold prefill.  Set
            // QW_GEMM_MIN_ROWS very high to fall back to the scalar loop.
            //
            // It was 8, which sent an eight-row pass - the shape a four-slot decode
            // actually uses - into the MPP tensor path.  That path's tile is
            // M(tokens)=128, so at eight rows it runs 16x oversized and wastes
            // almost all of the tensor work: an eight-row pass cost 385.4 ms at 8
            // and 99.5 ms at 32, a 3.9x difference, and the four-slot aggregate went
            // from 19.40 to 59.80 tok/s.  At 32 the GEMM still wins for the shapes
            // it was chosen for, because 1020 (never use it) gives 100.3 ms for the
            // same eight-row pass and only 47.16 aggregate.
            .unwrap_or(14)
    })
}
