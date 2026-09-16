//! `qwen38 bench` — full-model quantised-GEMV sweep over the **real** weights.
//!
//! This measures the bandwidth-bound part of decode: every token must read all
//! 15.13 GB of language-model weights exactly once.  It is the number M3/M4 are
//! graded on (attention/GDN/norm overheads are counted separately later).

use anyhow::Result;
use half::f16;
use qw_metal::GpuDevice;
use qw_model::QLinear;
use qw_weights::WeightStore;
use std::path::Path;
use std::time::Instant;

pub fn run(model_dir: &Path, iters: usize, k: usize, rows: usize) -> Result<()> {
    let mut dev = GpuDevice::new()?;
    let store = WeightStore::load_dir(&dev, model_dir)?;

    // Collect every quantised linear except the embedding table (a gather, not a GEMV).
    let mut names: Vec<String> = store
        .names()
        .filter(|n| n.ends_with(".weight"))
        .filter(|n| !n.contains("embed_tokens"))
        .filter(|n| store.has(&n.replace(".weight", ".scales")))
        .cloned()
        .collect();
    names.sort();

    let mut linears = Vec::new();
    let mut total_bytes = 0usize;
    let mut unaligned = 0usize;
    for n in &names {
        let l = QLinear::from_store(&store, n)?;
        if !l.alignment_ok() {
            unaligned += 1;
        }
        total_bytes += l.weight_bytes();
        linears.push(l);
    }
    println!(
        "quantised linears: {} ({} GB of weights, {} unaligned)",
        linears.len(),
        total_bytes as f64 / 1e9,
        unaligned
    );

    // Reusable inputs: in_f is 5120 (most) or 17408 (down_proj).
    let mut xs: Vec<(usize, qw_metal::GpuBuffer)> = Vec::new();
    for in_f in [5120usize, 17408, 10240, 6144] {
        if linears.iter().any(|l| l.in_f == in_f) {
            let v = vec![f16::from_f32(0.01); in_f * k];
            xs.push((in_f, dev.buffer_from_bytes(&v)));
        }
    }
    // QW_X_PER_LINEAR: give every linear its own x buffer, the way the model does,
    // instead of one shared buffer per in_f.  Isolates whether the shared (always
    // cache-hot) input is what makes this sweep look so much faster than the same
    // dispatches inside a real pass.
    let per_linear = std::env::var_os("QW_X_PER_LINEAR").is_some();
    let mut x_own: Vec<qw_metal::GpuBuffer> = Vec::new();
    if per_linear {
        for l in &linears {
            let v = vec![f16::from_f32(0.01); l.in_f * k];
            x_own.push(dev.buffer_from_bytes(&v));
        }
    }
    let ys: Vec<(usize, qw_metal::GpuBuffer)> = {
        let mut v = Vec::new();
        for out_f in linears
            .iter()
            .map(|l| l.out_f)
            .collect::<std::collections::BTreeSet<_>>()
        {
            v.push((out_f, dev.buffer(out_f * 2 * k)));
        }
        v
    };

    // rows == 9: round-robin the k=3 kernel variants inside one process, so a
    // comparison never pays the 14 GB weight load twice and thermal drift hits every
    // variant equally.
    if rows == 9 {
        // Only near-equal candidates go in one round: a variant that is 30% slower
        // swings the clock inside the round and corrupts the baseline it is paired
        // against (R=6/R=8 spill and did exactly that).
        // Row blocking is closed (round 043: 1.0266-1.0690, slower at every factor even
        // at a reproducible clock plateau).  What is left untested is the other way to
        // buy memory-level parallelism: more bytes in flight per instruction, i.e. a
        // 16-byte uint4 weight load instead of 8 bytes.
        // The 4th entry is a duplicate of the baseline.  Its ratio against the primary
        // baseline is the instrument's calibration: a value away from 1.0000 means the
        // method has an order or drift bias, and any candidate's number has to be read
        // against that, not against 1.0000.
        let variants: [(&str, &str, usize); 5] = [
            ("k3 (baseline)", qw_metal::msl::K_Q4_GEMV_K3, 1),
            ("k3 + u4 (16B) loads", qw_metal::msl::K_Q4_GEMV_K3_U4, 1),
            ("k3 + u4, 2 rows/tg", qw_metal::msl::K_Q4_GEMV_K3_R2U, 2),
            ("k3 + u4 + half dots", qw_metal::msl::K_Q4_GEMV_K3_U4H, 1),
            ("k3 (baseline dup)", qw_metal::msl::K_Q4_GEMV_K3, 1),
        ];
        // On this machine the GPU clock swings by 4x on the timescale of a single
        // sweep (battery + Low Power Mode), so absolute times mean nothing.  Measure
        // every variant once per round together with the baseline and compare the
        // paired ratio, which is immune to a multiplicative clock change, and count
        // how many rounds each variant actually won.
        let n = variants.len();
        let rounds = iters.max(1);
        let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut wins = vec![0usize; n];
        for round in 0..rounds {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..n).collect()
            } else {
                (0..n).rev().collect()
            };
            let mut t = vec![0.0f64; n];
            for i in order {
                let (_label, name, grid_rows) = variants[i];
                let t0 = Instant::now();
                {
                    let mut batch = dev.batch();
                    let kernel = batch.kernel(qw_metal::msl::COMMON, name)?;
                    for l in &linears {
                        let x = &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1;
                        let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                        l.encode_kr(&mut batch, &kernel, x, y, 3, grid_rows);
                    }
                    batch.finish(true);
                }
                t[i] = t0.elapsed().as_secs_f64() * 1000.0;
            }
            for i in 1..n {
                ratios[i].push(t[i] / t[0]);
                if t[i] < t[0] {
                    wins[i] += 1;
                }
            }
        }
        for v in ratios.iter_mut() {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let mut base_ms: Vec<f64> = Vec::new();
        // The last entry is a duplicate of the baseline, so its ratio is the
        // instrument's own bias - the order and drift inside a round.  Round 045
        // measured that bias at 0.9649, meaning the raw ratios flattered every
        // candidate by ~3.5% and sent round 044 chasing a 7% "win" that is really
        // ~1.2%.  Every ratio is therefore reported both raw and calibrated, and the
        // calibrated column is the one to believe.
        let bias = {
            let last = &ratios[variants.len() - 1];
            if last.is_empty() {
                1.0
            } else {
                last[last.len() / 2]
            }
        };
        for (i, (label, _n, gr)) in variants.iter().enumerate() {
            if i == 0 {
                println!("  {label:<20} grid_rows={gr}  (paired baseline)");
            } else if i + 1 == variants.len() {
                let m = ratios[i][ratios[i].len() / 2];
                println!(
                    "  {:<20} grid_rows={}  median ratio {:.4}  wins {}/{}  (CALIBRATION: this is the instrument bias)",
                    label, gr, m, wins[i], rounds
                );
            } else {
                let m = ratios[i][ratios[i].len() / 2];
                println!(
                    "  {:<20} grid_rows={}  median ratio {:.4}  calibrated {:.4}  wins {}/{}  ({})",
                    label,
                    gr,
                    m,
                    m / bias,
                    wins[i],
                    rounds,
                    if m / bias < 1.0 { "faster" } else { "slower" }
                );
            }
        }
        base_ms.clear();
        return Ok(());
    }

    // QW_BENCH_BURN=<seconds>: hammer the GPU with the same sweep until the deadline
    // before timing anything.  If the in-situ/in-isolation gap is the clock dropping
    // under sustained load rather than anything about the kernels, this reproduces it.
    if let Ok(secs) = std::env::var("QW_BENCH_BURN") {
        let secs: f64 = secs.parse().unwrap_or(0.0);
        let deadline = Instant::now() + std::time::Duration::from_secs_f64(secs);
        let mut burns = 0usize;
        while Instant::now() < deadline {
            let mut batch = dev.batch();
            let kernel = if k == 1 {
                QLinear::kernel(&mut batch)?
            } else {
                let name = match k {
                    2 => qw_metal::msl::K_Q4_GEMV_K2,
                    3 => qw_metal::msl::K_Q4_GEMV_K3,
                    4 => qw_metal::msl::K_Q4_GEMV_K4,
                    _ => qw_metal::msl::K_Q4_GEMV_K,
                };
                batch.kernel(qw_metal::msl::COMMON, name)?
            };
            for l in &linears {
                let x = &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1;
                let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                if k == 1 {
                    l.encode(&mut batch, &kernel, x, y);
                } else {
                    l.encode_kr(&mut batch, &kernel, x, y, k, 1);
                }
            }
            batch.finish(true);
            burns += 1;
        }
        eprintln!(
            "  burned {burns} sweeps over {secs:.0} s ({:.2} ms each)",
            secs * 1e3 / burns.max(1) as f64
        );
    }

    let mut best_ms = f64::INFINITY;
    for _ in 0..iters.max(1) {
        let t0 = Instant::now();
        {
            let mut batch = dev.batch();
            let kernel = if k == 1 {
                QLinear::kernel(&mut batch)?
            } else if rows == 1 {
                let name = match k {
                    2 => qw_metal::msl::K_Q4_GEMV_K2,
                    3 => qw_metal::msl::K_Q4_GEMV_K3,
                    4 => qw_metal::msl::K_Q4_GEMV_K4,
                    _ => qw_metal::msl::K_Q4_GEMV_K,
                };
                batch.kernel(qw_metal::msl::COMMON, name)?
            } else if rows == 2 {
                // k=3 with 16-byte weight loads
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K3_U4)?
            } else if rows == 3 {
                // k=3, two output rows per threadgroup (x loaded once, reused twice)
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K3_R2)?
            } else if rows == 4 {
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K3_R4)?
            } else {
                QLinear::kernel_k(&mut batch)?
            };
            for (li, l) in linears.iter().enumerate() {
                let x = if per_linear {
                    &x_own[li]
                } else {
                    &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1
                };
                let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                if k == 1 {
                    l.encode(&mut batch, &kernel, x, y);
                } else if rows >= 1 {
                    // rows selects the *kernel*; the grid blocking is 1 except for the
                    // row-blocked variants, which need out_f/R threadgroups.
                    let grid_rows = match rows {
                        3 => 2,
                        4 => 4,
                        _ => 1,
                    };
                    l.encode_kr(&mut batch, &kernel, x, y, k, grid_rows);
                } else {
                    l.encode_k(&mut batch, &kernel, x, y, k);
                }
            }
            batch.finish(true);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        best_ms = best_ms.min(ms);
    }

    let gbps = total_bytes as f64 / (best_ms / 1000.0) / 1e9;
    let tok_s = k as f64 * 1000.0 / best_ms;
    println!(
        "one full weight sweep over k={k} tokens, rows={rows}: {:.2} ms -> {:.1} tok/s linear path, {:.0} GB/s effective",
        best_ms, tok_s, gbps
    );
    // end_to_end = false: this sweep encodes every linear into one command
    // buffer with constant inputs, so it is the upper bound of the linear part,
    // not measured generation throughput.
    println!(
        "{{\"tok_per_s\": {:.2}, \"ms_per_token\": {:.3}, \"effective_gbps\": {:.1}, \"weight_bytes\": {}, \"end_to_end\": false, \"mode\": \"linear_sweep\", \"k\": {}}}",
        tok_s, best_ms / k as f64, gbps, total_bytes, k
    );
    Ok(())
}
