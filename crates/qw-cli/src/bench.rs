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

/// Median of a slice, without pulling in a stats crate.  Every comparison in this
/// file is paired inside one process, so the median of the per-round ratios is the
/// number to read, not a mean of absolute times.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

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
            ("k3 + u4 + half dots", qw_metal::msl::K_Q4_GEMV_K3_U4H, 1),
            ("k3 + u4 + x16B load", qw_metal::msl::K_Q4_GEMV_K3_U4HX, 1),
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

    // rows == 18: batch-16 against the serial alternative, both producing the same
    // 16 tokens from the same 14.4 GB of weights.  Four arms, alternating inside one
    // process because on this box the clock moves by up to 4x within a single run and
    // a cross-process A/B is meaningless.  Run it with --tokens 16.
    //
    //   b16   one sweep, grid-mapped   16 tokens from 1 weight read   (round 056)
    //   k16   one sweep, per-thread    16 tokens from 1 weight read   (round 055)
    //   k1x16 sixteen ordinary sweeps  16 tokens from 16 weight reads
    //   b16*  duplicate of the first, which is what calibrates the method
    if rows == 18 {
        const SERIAL: usize = 16;
        if k != SERIAL {
            anyhow::bail!("--rows 18 needs --tokens {SERIAL} (buffers are {k} rows wide)");
        }
        let mut t = [f64::INFINITY; 4];
        for r in 0..3usize {
            let order: [usize; 4] = if r % 2 == 0 {
                [0, 1, 2, 3]
            } else {
                [3, 2, 1, 0]
            };
            for which in order {
                let t0 = Instant::now();
                {
                    let mut batch = dev.batch();
                    match which {
                        2 => {
                            let kern =
                                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_HX)?;
                            for _ in 0..SERIAL {
                                for l in &linears {
                                    let x = &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1;
                                    let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                                    l.encode(&mut batch, &kern, x, y);
                                }
                            }
                        }
                        1 => {
                            let kern = batch
                                .kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K16_U4HX)?;
                            for l in &linears {
                                let x = &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1;
                                let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                                l.encode_k(&mut batch, &kern, x, y, SERIAL);
                            }
                        }
                        _ => {
                            let kern = batch
                                .kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_B16)?;
                            for l in &linears {
                                let x = &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1;
                                let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                                l.encode_b(&mut batch, &kern, x, y, SERIAL);
                            }
                        }
                    }
                    batch.finish(true);
                }
                t[which] = t[which].min(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let tok = |ms: f64| SERIAL as f64 * 1000.0 / ms;
        println!(
            "batch-16 vs serial over the full {:.1} GB of quantised weights, {SERIAL} tokens either way",
            total_bytes as f64 / 1e9
        );
        println!(
            "{:<42} {:>10} {:>10} {:>9}",
            "arm", "ms/sweep", "tok/s", "vs b16"
        );
        let labels: [String; 4] = [
            "b16  grid-shared, 1 weight read".to_string(),
            "k16  per-thread, 1 weight read".to_string(),
            format!("k1 x{SERIAL} serial, 16 weight reads"),
            "b16  duplicate (calibration)".to_string(),
        ];
        for i in 0..4 {
            println!(
                "{:<42} {:>10.2} {:>10.1} {:>8.3}x",
                labels[i],
                t[i],
                tok(t[i]),
                t[0] / t[i]
            );
        }
        let bias = t[3] / t[0];
        println!(
            "\nb16 over serial {:>.3}x raw, {:>.3}x calibrated (duplicate reads {bias:.3}x the original)",
            t[2] / t[0],
            (t[2] / t[0]) / bias
        );
        println!(
            "{{\"b16_ms\": {:.2}, \"k16_ms\": {:.2}, \"serial_ms\": {:.2}, \"gain_raw\": {:.3}, \"calibration\": {:.3}, \"gain_calibrated\": {:.3}, \"tokens\": {}}}",
            t[0],
            t[1],
            t[2],
            t[2] / t[0],
            bias,
            (t[2] / t[0]) / bias,
            SERIAL
        );
        return Ok(());
    }

    // rows == 21: the 2-D (features x tokens) tile, one weight read per pass.
    //
    // One sweep of all 14.412 GB is the floor and every arm here pays it exactly
    // once per chunk; what varies is how many tokens ride on that sweep and how
    // much redundant activation traffic the kernel issues.  The measurement that
    // matters is tokens per weight read, and the fair baseline for a given token
    // count is the arm the engine runs today: `rows <= TILE` calls the flat k=4
    // kernel once, anything wider loops it in fours.
    //
    // Round-robin inside one process, as for rows == 9, and the first arm is
    // repeated last so the ratio against it is the instrument's own bias rather
    // than a hard 1.0000.
    if rows == 21 {
        // (label, [(kernel, NK tokens, NR rows), ...])
        type Arm = (&'static str, &'static [(&'static str, usize, usize)]);
        const K4: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1)];
        const K8: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_K8_FLAT, 8, 1)];
        const T24: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T24, 4, 2)];
        const T44: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T44, 4, 4)];
        const T28: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T28, 8, 2)];
        const T48: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T48, 8, 4)];
        const T216: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T216, 16, 2)];
        const T28D: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T28_DIAG, 8, 2)];
        const T28W: &[(&str, usize, usize)] = &[(qw_metal::msl::K_Q4_GEMV_T28W, 8, 2)];
        // 12 + 4 = 16 tokens, so this is a like-for-like alternative to t28 x2.
        const T212T24: &[(&str, usize, usize)] = &[
            (qw_metal::msl::K_Q4_GEMV_T212, 12, 2),
            (qw_metal::msl::K_Q4_GEMV_T24, 4, 2),
        ];
        const K4X2: &[(&str, usize, usize)] = &[
            (qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1),
            (qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1),
        ];
        const K4X4: &[(&str, usize, usize)] = &[
            (qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1),
            (qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1),
            (qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1),
            (qw_metal::msl::K_Q4_GEMV_K4_FLAT, 4, 1),
        ];
        const T44X4: &[(&str, usize, usize)] = &[
            (qw_metal::msl::K_Q4_GEMV_T44, 4, 4),
            (qw_metal::msl::K_Q4_GEMV_T44, 4, 4),
            (qw_metal::msl::K_Q4_GEMV_T44, 4, 4),
            (qw_metal::msl::K_Q4_GEMV_T44, 4, 4),
        ];
        const T48X2: &[(&str, usize, usize)] = &[
            (qw_metal::msl::K_Q4_GEMV_T48, 8, 4),
            (qw_metal::msl::K_Q4_GEMV_T48, 8, 4),
        ];
        let arms: Vec<Arm> = match k {
            4 => vec![
                ("k4-flat  x1  (engine today)", K4),
                ("t24 (NR2) x1", T24),
                ("t44 (NR4) x1", T44),
                ("k4-flat  x1  (dup)", K4),
            ],
            8 => vec![
                ("k4-flat  x2  (engine today)", K4X2),
                ("k8-flat  x1", K8),
                ("t28 (NR2) x1", T28),
                ("t48 (NR4) x1", T48),
                ("t28-DIAG no-hi-unpack (WRONG)", T28D),
                ("t28w (NK8 NR2, 16B wload)", T28W),
                ("k4-flat  x2  (dup)", K4X2),
            ],
            16 => vec![
                ("k4-flat  x4  (engine today)", K4X4),
                ("t44 (NR4) x4", T44X4),
                ("t48 (NR4) x2", T48X2),
                ("t28 (NR2) x2", &[(qw_metal::msl::K_Q4_GEMV_T28, 8, 2); 2]),
                ("t216 (NK16 NR2) x1", T216),
                ("t28w (NK8 NR2) x2", &[(qw_metal::msl::K_Q4_GEMV_T28W, 8, 2); 2]),
                ("t212+t24 (NK12+4) =16", T212T24),
                ("k4-flat  x4  (dup)", K4X4),
            ],
            _ => anyhow::bail!("--rows 21 supports --tokens 4, 8 or 16 (got {k})"),
        };

        // out_f must be a multiple of NR for the tiled kernels; every quantised
        // linear in this model is a multiple of 4, but check rather than assume.
        for (label, chunks) in &arms {
            for (_kn, _nk, nr) in chunks.iter() {
                if *nr > 1 {
                    for l in &linears {
                        anyhow::ensure!(
                            l.out_f % nr == 0,
                            "arm {label}: out_f {} is not a multiple of NR {nr}",
                            l.out_f
                        );
                    }
                }
            }
        }

        let n = arms.len();
        let rounds = iters.max(1);
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); n];
        for round in 0..rounds {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..n).collect()
            } else {
                (0..n).rev().collect()
            };
            for i in order {
                let t0 = Instant::now();
                {
                    let mut batch = dev.batch();
                    let mut kernels = Vec::new();
                    for (kn, nk, nr) in arms[i].1.iter() {
                        let kern = batch.kernel(qw_metal::msl::COMMON, kn)?;
                        kernels.push((kern, *nk, *nr));
                    }
                    for l in &linears {
                        let x = &xs.iter().find(|(nn, _)| *nn == l.in_f).unwrap().1;
                        let y = &ys.iter().find(|(nn, _)| *nn == l.out_f).unwrap().1;
                        let mut off = 0usize;
                        for (kern, nk, nr) in &kernels {
                            let d = qw_metal::Dispatch::new(
                                kern,
                                ((l.out_f / nr) * 32, 1, 1),
                                (32, 1, 1),
                            )
                            .buf_offset(0, l.weight.buf, l.weight.offset)
                            .buf_offset(1, l.scales.buf, l.scales.offset)
                            .buf_offset(2, l.biases.buf, l.biases.offset)
                            .buf_offset(3, x, off * l.in_f * 2)
                            .buf_offset(4, y, off * l.out_f * 2)
                            .scalar(5, l.in_f as i32)
                            .scalar(6, *nk as i32)
                            .scalar(7, l.out_f as i32);
                            batch.encode(d);
                            off += *nk;
                        }
                        anyhow::ensure!(off == k, "arm {} covered {off} of {k} tokens", arms[i].0);
                    }
                    batch.finish(true);
                }
                times[i].push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let bias = {
            let last = &times[n - 1];
            if last.is_empty() { 1.0 } else { median(last) / median(&times[0]) }
        };
        println!(
            "2-D tile sweep over the full {:.3} GB of quantised weights, {k} tokens either way, {rounds} rounds",
            total_bytes as f64 / 1e9
        );
        println!(
            "{:<30} {:>10} {:>10} {:>10} {:>9}",
            "arm", "ms/min", "tok/s", "vs base", "calib"
        );
        for (i, (label, chunks)) in arms.iter().enumerate() {
            let m = median(&times[i]);
            let ratio = m / median(&times[0]);
            let tag = if i == 0 {
                "(paired baseline)".to_string()
            } else if i + 1 == n {
                format!("(CALIBRATION: instrument bias {bias:.4})")
            } else if ratio / bias < 1.0 {
                format!("faster, {} chunk(s)", chunks.len())
            } else {
                format!("slower, {} chunk(s)", chunks.len())
            };
            println!(
                "{:<30} {:>10.2} {:>10.1} {:>10.4} {:>9.4}  {tag}",
                label,
                m,
                k as f64 * 1000.0 / m,
                ratio,
                ratio / bias
            );
        }
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
            } else if rows == 7 {
                // The weight-stationary kernel with NK=6.  Same trap as rows 5: the
                // accumulator loop is unrolled over NK, so --tokens must be 6.
                anyhow::ensure!(
                    k == 6,
                    "--rows 7 is K_Q4_GEMV_K6_U4H, unrolled over a compile-time NK of 6, so \
                     --tokens must be 6 (got {k})."
                );
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K6_U4H)?
            } else if rows == 6 {
                // The weight-stationary kernel with NK=8.  `--tokens` must be 8 for
                // the timing to match the work done; anything else is rejected below.
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K8_U4HX)?
            } else if rows == 5 {
                // Exactly what the engine's prefill uses: `encode_rows` ->
                // `encode_tile` -> `encode_k(K_Q4_GEMV_K4_U4HX, k)`.  Until now no
                // `--rows` value could reach it, so the sweep numbers on record for
                // "the engine's kernel" were actually `K_Q4_GEMV_K` from the
                // `rows == 0` branch, which is a different kernel entirely.
                anyhow::ensure!(
                    k == 4,
                    "--rows 5 is K_Q4_GEMV_K4_U4HX, whose accumulator loop is unrolled over a \
                     compile-time NK of 4, so it processes four tokens no matter what --tokens \
                     says.  Asking for k={k} would report a speedup that is only the kernel \
                     doing less work than the timing assumes.  Use --rows 6 for k=8."
                );
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K4_U4HX)?
            } else if rows == 19 {
                // The flat-partition k=4 GEMV: same arithmetic as --rows 5 but lane L
                // takes the 8-weight chunk `iter * 32 + L`, so the x load is 512
                // contiguous bytes per warp instead of 32 lines of 16 bytes.
                anyhow::ensure!(k == 4, "--rows 19 is the flat k=4 kernel; use --tokens 4");
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K4_FLAT)?
            } else if rows == 20 {
                // DO NOT TRUST THIS NUMBER FOR THE ENGINE.  The flat k=8 kernel is
                // faster than the k=4 one here - 7.731/7.738 ms per token against
                // 8.824/8.834 - and it is still 1.28x SLOWER on a real eight-row
                // decode pass: 122.00 ms min against 95.24 ms for the 2 x k=4 the
                // engine actually runs.  This arm exists only to reproduce that
                // divergence; see docs/PLAN_BATCH16.md §72fb.  Wiring it into
                // `encode_rows` was tried and reverted.
                anyhow::ensure!(k == 8, "--rows 20 is the flat k=8 kernel; use --tokens 8");
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K8_FLAT)?
            } else if rows == 8 {
                // The row-amortising GEMM.  Unlike the GEMV variants its token count
                // is a runtime argument, so any `--tokens` is meaningful - this is the
                // measurement that says whether amortising the weight read pays.
                batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMM_TILE)?
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
                } else if rows == 8 {
                    let bm = 32usize;
                    let bn = 32usize;
                    let d = qw_metal::Dispatch::new(
                        &kernel,
                        (((l.out_f + bm - 1) / bm) * 128, (k + bn - 1) / bn, 1),
                        (128, 1, 1),
                    )
                    .buf_offset(0, l.weight.buf, l.weight.offset)
                    .buf_offset(1, l.scales.buf, l.scales.offset)
                    .buf_offset(2, l.biases.buf, l.biases.offset)
                    .buf(3, x)
                    .buf(4, y)
                    .scalar(5, l.in_f as i32)
                    .scalar(6, k as i32)
                    .scalar(7, l.out_f as i32)
                    .scalar(8, std::env::var("QW_GEMM_MODE").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0));
                    batch.encode(d);
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

/// Correctness check for the row-amortising GEMM: run `q4_gemm_tile` over real
/// weights and compare against the CPU dequantise-and-multiply reference.
///
/// The kernel must be verified before anything routes prefill through it, so this
/// deliberately exercises the awkward shapes: `out_f` values that are not a
/// multiple of the 32-row tile (the GDN a/b projections are 48 rows) and a token
/// count that is not a multiple of the 32-token tile.
pub fn gemm_check(model_dir: &Path) -> Result<()> {
    let mut dev = GpuDevice::new()?;
    let store = WeightStore::load_dir(&dev, model_dir)?;

    let mut names: Vec<String> = store
        .names()
        .filter(|n| n.ends_with(".weight"))
        .filter(|n| !n.contains("embed_tokens"))
        .filter(|n| store.has(&n.replace(".weight", ".scales")))
        .cloned()
        .collect();
    names.sort();

    // A spread of shapes: the big ones, and small/odd out_f that hit the tile tails.
    // `QW_GEMM_CHECK_ALL` sweeps every quantised linear instead, which is how the
    // end-to-end failure was tracked down - the four shapes here all passed while the
    // engine still produced the wrong answer.
    let mut picked: Vec<String> = Vec::new();
    let gpu_only = std::env::var("QW_GEMM_CHECK_ALL").is_ok();
    if gpu_only {
        picked = names.clone();
    }
    for want in if picked.is_empty() { vec![5120usize, 17408, 18432, 48, 96] } else { Vec::new() } {
        if let Some(n) = names
            .iter()
            .find(|n| QLinear::from_store(&store, n).map(|l| l.out_f == want).unwrap_or(false))
        {
            picked.push(n.clone());
        }
    }
    for n in names.iter().take(2) {
        if !picked.contains(n) {
            picked.push(n.clone());
        }
    }

    // 40 is deliberately not a multiple of the 32-token tile.  It is overridable
    // because the token count also decides how many M blocks the tile kernel runs:
    // at NRB=512 forty tokens is a single block, so a mis-tiled second block would go
    // unnoticed, and a tile change cannot be trusted without a run at a real prefill
    // length.
    let tokens = std::env::var("QW_GEMM_CHECK_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(40usize);
    println!("gemm-check: {} linears, {tokens} tokens per case", picked.len());

    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    let mut all_ok = true;

    for name in &picked {
        let l = QLinear::from_store(&store, name)?;
        if l.in_f % 64 != 0 {
            println!("  SKIP {name}: in_f {} is not a multiple of GROUP_SIZE", l.in_f);
            continue;
        }

        // Deterministic pseudo-random activations.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut xv = vec![f16::from_f32(0.0); tokens * l.in_f];
        // Real activations are not uniform: a small fraction of channels carry values
        // one to two orders of magnitude above the rest.  Those outliers are what the
        // per-64-value scale has to cover, and they are what a uniform distribution
        // cannot reproduce.  `QW_GEMM_CHECK_OUTLIER=<mult>` marks one channel in
        // sixty-four as an outlier, the same channel for every token, and scales it.
        let outlier: f32 = std::env::var("QW_GEMM_CHECK_OUTLIER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        for i in 0..xv.len() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            let c = i % l.in_f;
            let scale = if outlier != 1.0 && c % 64 == 7 { outlier } else { 1.0 };
            xv[i] = f16::from_f32((u * 2.0 - 1.0) * scale);
        }
        let xbuf = dev.buffer_from_bytes(&xv);
        let ybuf = dev.buffer_from_bytes(&vec![f16::from_f32(0.0); tokens * l.out_f]);

        let mut batch = qw_metal::CommandBatch::new(&mut dev);
        let kernel = batch.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMM_TILE)?;
        let bm = 32usize;
        let bn = 32usize;
        // `dispatch_threads` takes the grid in THREADS, not threadgroups, so the
        // x extent has to be multiplied by the threadgroup size.  Passing the tile
        // count directly launches a couple of threadgroups and silently leaves the
        // rest of the output untouched.
        let d = qw_metal::Dispatch::new(
            &kernel,
            (((l.out_f + bm - 1) / bm) * 128, (tokens + bn - 1) / bn, 1),
            (128, 1, 1),
        )
        .buf_offset(0, l.weight.buf, l.weight.offset)
        .buf_offset(1, l.scales.buf, l.scales.offset)
        .buf_offset(2, l.biases.buf, l.biases.offset)
        .buf(3, &xbuf)
        .buf(4, &ybuf)
        .scalar(5, l.in_f as i32)
        .scalar(6, tokens as i32)
        .scalar(7, l.out_f as i32);
        batch.encode(d);
        batch.finish(true);

        let got: Vec<f16> = ybuf.to_vec(0, tokens * l.out_f);

        // The CPU reference is far too slow to sweep 497 linears with, so in ALL mode
        // the reference is the GEMV tile kernel the engine actually uses - the same
        // four-row kernel `encode_rows` runs in a loop.  That is the equivalence that
        // matters for the engine: if the GEMM disagrees with this, the engine's answer
        // changes, which is exactly what the end-to-end test showed.
        // Fifth path: the affine tensor-op GEMM (`q4_mpp_mm`).  Tile
        // M(tokens)=128, N(rows)=64, K=32, weight tile only in shared memory,
        // activations read straight from device memory by the tensor op.
        let nbuf = dev.buffer_from_bytes(&vec![f16::from_f32(0.0); tokens * l.out_f]);
        let mut nb = qw_metal::CommandBatch::new(&mut dev);
        {
            let nk = nb.kernel(&qw_metal::msl::mpp_src(), "q4_mpp_mm")?;
            let [_, nra, nrb, nt] = qw_metal::msl::mpp_tiles();
            let (nra, nrb, nt) = (nra as usize, nrb as usize, nt as usize);
            nb.encode(
                qw_metal::Dispatch::new(
                    &nk,
                    (tokens.div_ceil(nrb) * nt, l.out_f.div_ceil(nra), 1),
                    (nt, 1, 1),
                )
                .buf_offset(0, l.weight.buf, l.weight.offset)
                .buf_offset(1, l.scales.buf, l.scales.offset)
                .buf_offset(2, l.biases.buf, l.biases.offset)
                .buf(3, &xbuf)
                .buf(4, &nbuf)
                .scalar(5, l.in_f as i32)
                .scalar(6, tokens as i32)
                .scalar(7, l.out_f as i32)
                .scalar(8, 0),
            );
        }
        nb.finish(true);
        let ngot: Vec<f16> = nbuf.to_vec(0, tokens * l.out_f);
        let mut n_abs = 0.0f64;
        let mut n_ref = 0.0f64;
        let mut n_bad = 0usize;
        let mut n_zer = 0usize;
        for t in 0..tokens {
            let xr: Vec<f16> = xv[t * l.in_f..(t + 1) * l.in_f].to_vec();
            let want = l.cpu_reference(&xr)?;
            for r in 0..l.out_f {
                let b = want[r] as f64;
                let nv = ngot[t * l.out_f + r].to_f32() as f64;
                if nv == 0.0 {
                    n_zer += 1;
                }
                let d = (nv - b).abs();
                if d > n_abs {
                    n_abs = d;
                }
                if b.abs() > n_ref {
                    n_ref = b.abs();
                }
                if d > 2e-3 * (if n_ref > 0.0 { n_ref } else { 1.0 }) {
                    n_bad += 1;
                }
            }
        }
        println!(
            "      mpp-mm max_abs {:.3e} rel {:.3e}  zero {n_zer}/{}  over-tol {n_bad}",
            n_abs,
            if n_ref > 0.0 { n_abs / n_ref } else { n_abs },
            tokens * l.out_f
        );

        // The MPP tile-op GEMM is the kernel the prefill path actually runs, so its
        // result has to gate the verdict.  It did not: `all_ok` below is set from the
        // GEMM and GEMV paths only, so raising QW_MPP_NRA from 32 to 128 - which makes
        // the tile write only a quarter of the output rows and report 120 TFLOPS
        // against the default's 36 - left this check printing PASSED.  Measured at 40
        // tokens: NRA=32 gives `zero 0/204800 over-tol 0`, NRA=64 gives 102400 zeros,
        // NRA=128 gives 153600.
        let mpp_ok = n_bad == 0;
        all_ok &= mpp_ok;
        if !mpp_ok {
            println!(
                "      ^ FAIL: the MPP tile GEMM put {n_bad} of {} elements outside tolerance",
                tokens * l.out_f
            );
        }

        let mut max_abs = 0.0f64;
        let mut max_ref = 0.0f64;
        if gpu_only {
            let refbuf = dev.buffer_from_bytes(&vec![f16::from_f32(0.0); tokens * l.out_f]);
            let mut rb = qw_metal::CommandBatch::new(&mut dev);
            let tk = rb.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K4_U4HX)?;
            let mut off = 0usize;
            while off < tokens {
                rb.encode(
                    qw_metal::Dispatch::new(&tk, (l.out_f * 32, 1, 1), (32, 1, 1))
                        .buf_offset(0, l.weight.buf, l.weight.offset)
                        .buf_offset(1, l.scales.buf, l.scales.offset)
                        .buf_offset(2, l.biases.buf, l.biases.offset)
                        .buf_offset(3, &xbuf, off * l.in_f * 2)
                        .buf_offset(4, &refbuf, off * l.out_f * 2)
                        .scalar(5, l.in_f as i32)
                        .scalar(6, 4i32)
                        .scalar(7, l.out_f as i32),
                );
                off += 4;
            }
            rb.finish(true);
            let want: Vec<f16> = refbuf.to_vec(0, tokens * l.out_f);
            for i in 0..tokens * l.out_f {
                let a = got[i].to_f32() as f64;
                let b = want[i].to_f32() as f64;
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(b.abs());
            }
        } else {
            // Same harness, same tensor, same reference, same metric: run the four-row
            // kernel the engine actually ships against the same CPU reference the GEMM
            // is being measured against.  Comparing the GEMM's `rel` with this kernel's
            // `max_abs` is what produced three wrong conclusions in a row; only these
            // two numbers side by side mean anything.
            let gbuf = dev.buffer_from_bytes(&vec![f16::from_f32(0.0); tokens * l.out_f]);
            let mut gb = qw_metal::CommandBatch::new(&mut dev);
            let tk = gb.kernel(qw_metal::msl::COMMON, qw_metal::msl::K_Q4_GEMV_K4_U4HX)?;
            let mut off = 0usize;
            while off < tokens {
                gb.encode(
                    qw_metal::Dispatch::new(&tk, (l.out_f * 32, 1, 1), (32, 1, 1))
                        .buf_offset(0, l.weight.buf, l.weight.offset)
                        .buf_offset(1, l.scales.buf, l.scales.offset)
                        .buf_offset(2, l.biases.buf, l.biases.offset)
                        .buf_offset(3, &xbuf, off * l.in_f * 2)
                        .buf_offset(4, &gbuf, off * l.out_f * 2)
                        .scalar(5, l.in_f as i32)
                        .scalar(6, 4i32)
                        .scalar(7, l.out_f as i32),
                );
                off += 4;
            }
            gb.finish(true);
            let ggot: Vec<f16> = gbuf.to_vec(0, tokens * l.out_f);
            let mut g_abs = 0.0f64;
            let mut g_ref = 0.0f64;
            // Third path: the MPP affine kernel, against the SAME CPU reference.
            // The engine can only be told apart from the reference by its tokens, so
            // a number here is the only way to see whether the per-group scale is
            // being applied at all.
            let ng = l.in_f / 64;
            let gsum = dev.buffer(tokens * ng * 4);
            let cacc = dev.buffer(tokens * l.out_f * 4);
            let mbuf = dev.buffer_from_bytes(&vec![f16::from_f32(0.0); tokens * l.out_f]);
            let mut mb = qw_metal::CommandBatch::new(&mut dev);
            {
                let gs = mb.kernel(qw_metal::msl::MPP, "q4_group_sums")?;
                mb.encode(
                    qw_metal::Dispatch::new(&gs, (tokens * ng, 1, 1), (256, 1, 1))
                        .buf(0, &xbuf)
                        .buf(1, &gsum)
                        .scalar(2, l.in_f as i32)
                        .scalar(3, ng as i32),
                );
                mb.barrier();
                let ak = mb.kernel(qw_metal::msl::MPP, "q4_mpp_affine_v2")?;
                mb.encode(
                    qw_metal::Dispatch::new(
                        &ak,
                        (l.out_f.div_ceil(32) * 128, tokens.div_ceil(64), 1),
                        (128, 1, 1),
                    )
                    .buf(0, &xbuf)
                    .buf_offset(1, l.weight.buf, l.weight.offset)
                    .buf_offset(2, l.scales.buf, l.scales.offset)
                    .buf_offset(3, l.biases.buf, l.biases.offset)
                    .buf(4, &gsum)
                    .buf(5, &cacc)
                    .scalar(6, ng as i32)
                    .scalar(7, l.out_f as i32)
                    .buf(8, &cacc),
                );
                mb.barrier();
                let shk = mb.kernel(qw_metal::msl::MPP, "q4_store_half")?;
                mb.encode(
                    qw_metal::Dispatch::new(&shk, (tokens * l.out_f, 1, 1), (256, 1, 1))
                        .buf(0, &cacc)
                        .buf(1, &mbuf)
                        .scalar(2, (tokens * l.out_f) as i32),
                );
            }
            mb.finish(true);
            // FOURTH path: the mode 7 probe - op.run into a DEVICE tensor, no
            // cooperative tensor anywhere.  It was reported as "3.2x faster" and as
            // proof that the device-tensor destination works; neither claim has ever
            // been checked against the CPU reference.  Do that here.
            let pbuf = dev.buffer_from_bytes(&vec![f16::from_f32(0.0); tokens * l.out_f]);
            // A `tensor` parameter with dynamic extents takes those extents from the
            // length of the buffer bound to it.  Binding the whole 14.4 GB weight
            // arena therefore hands the op extents that are not this matrix at all,
            // and the op silently does nothing - which is exactly what the zero
            // output above would look like.  Bind an exactly-sized copy instead.
            let wbytes: Vec<u8> = l.weight.buf.to_vec(l.weight.offset, l.out_f * l.in_f / 2);
            let wcopy = dev.buffer_from_bytes(&wbytes);
            let mut pb = qw_metal::CommandBatch::new(&mut dev);
            {
                let pk = pb.kernel(qw_metal::msl::MPP, "q4_mpp_probe")?;
                pb.encode(
                    qw_metal::Dispatch::new(
                        &pk,
                        (l.out_f.div_ceil(32) * 128, tokens.div_ceil(64), 1),
                        (128, 1, 1),
                    )
                    .buf(0, &xbuf)
                    .buf(1, &wcopy)
                    .buf(2, &pbuf),
                );
            }
            pb.finish(true);
            let pgot: Vec<f16> = pbuf.to_vec(0, tokens * l.out_f);
            let mut p_abs = 0.0f64;
            let mut p_ref = 0.0f64;
            for t2 in 0..tokens {
                let xr: Vec<f16> = xv[t2 * l.in_f..(t2 + 1) * l.in_f].to_vec();
                let want = l.cpu_reference(&xr)?;
                for r in 0..l.out_f {
                    let d = (pgot[t2 * l.out_f + r].to_f32() as f64 - want[r] as f64).abs();
                    if d > p_abs { p_abs = d; }
                    if (want[r] as f64).abs() > p_ref { p_ref = (want[r] as f64).abs(); }
                }
            }
            let mgot: Vec<f16> = mbuf.to_vec(0, tokens * l.out_f);
            let mut m_abs = 0.0f64;
            let mut m_ref = 0.0f64;
            for t in 0..tokens {
                let xr: Vec<f16> = xv[t * l.in_f..(t + 1) * l.in_f].to_vec();
                let want = l.cpu_reference(&xr)?;
                for r in 0..l.out_f {
                    let a = got[t * l.out_f + r].to_f32() as f64;
                    let b = want[r] as f64;
                    max_abs = max_abs.max((a - b).abs());
                    max_ref = max_ref.max(b.abs());
                    let g = ggot[t * l.out_f + r].to_f32() as f64;
                    g_abs = g_abs.max((g - b).abs());
                    g_ref = g_ref.max(b.abs());
                    let mm = mgot[t * l.out_f + r].to_f32() as f64;
                    m_abs = m_abs.max((mm - b).abs());
                    m_ref = m_ref.max(b.abs());
                }
            }
            println!(
                "      gemm max_abs {:.3e} rel {:.3e}   |   gemv max_abs {:.3e} rel {:.3e}   |   MPP max_abs {:.3e} rel {:.3e}   (same CPU ref, {} tokens)",
                max_abs,
                if max_ref > 0.0 { max_abs / max_ref } else { max_abs },
                g_abs,
                if g_ref > 0.0 { g_abs / g_ref } else { g_abs },
                m_abs,
                if m_ref > 0.0 { m_abs / m_ref } else { m_abs },
                tokens
            );
        }
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
        let ok = rel < 2e-3;
        all_ok &= ok;
        if rel > worst {
            worst = rel;
            worst_at = format!("{name} (out_f {}, in_f {})", l.out_f, l.in_f);
        }
        println!(
            "  {} {:<58} out_f {:>6} in_f {:>6}  max_abs {:.3e}  rel {:.3e}",
            if ok { "PASS" } else { "FAIL" },
            name,
            l.out_f,
            l.in_f,
            max_abs,
            rel
        );
    }

    println!(
        "gemm-check: worst relative error {:.3e} at {worst_at} -> {}",
        worst,
        if all_ok { "PASSED" } else { "FAILED" }
    );
    anyhow::ensure!(all_ok, "the GEMM kernel does not match the CPU reference");
    Ok(())
}

/// Prove the MetalPerformancePrimitives matmul call pattern against a CPU
/// reference, with no model and no quantisation in the way.
///
/// `run(left, right, dest)` wants `left` = (K, M), `right` = (K, N) under
/// `transpose_right = true`, and `dest` = (N, M).  Getting any of those three
/// axes wrong yields a kernel that runs, produces zeros, and looks exactly like
/// "this machine cannot execute the tensor op" - which is the conclusion an
/// earlier round drew.  This is the cheap test that separates the two.
pub fn mpp_test() -> Result<()> {
    let mut dev = GpuDevice::new()?;

    // 2x2 threadgroups of (tokens 128 x rows 64), 3 K tiles of 32.
    let m = 256usize; // tokens
    let n = 128usize; // weight rows
    let k = 96usize;

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    };
    let av: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
    let bv: Vec<f16> = (0..n * k).map(|_| f16::from_f32(next())).collect();

    let abuf = dev.buffer_from_bytes(&av);
    let bbuf = dev.buffer_from_bytes(&bv);
    let cbuf = dev.buffer_from_bytes(&vec![0.0f32; m * n]);

    let mut batch = qw_metal::CommandBatch::new(&mut dev);
    let kern = batch.kernel(qw_metal::msl::MPP, "mpp_test_mm")?;
    const NT: usize = 128;
    const NRA: usize = 64;
    const NRB: usize = 128;
    batch.encode(
        qw_metal::Dispatch::new(
            &kern,
            (m.div_ceil(NRB) * NT, n.div_ceil(NRA), 1),
            (NT, 1, 1),
        )
        .buf(0, &abuf)
        .buf(1, &bbuf)
        .buf(2, &cbuf)
        .scalar(3, m as i32)
        .scalar(4, n as i32)
        .scalar(5, k as i32),
    );
    batch.finish(true);

    let got: Vec<f32> = cbuf.to_vec(0, m * n);
    // CPU reference: c[tok][row] = sum_k a[tok][k] * b[row][k], in f64.
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    let mut nzero = 0usize;
    for tok in 0..m {
        for row in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += av[tok * k + kk].to_f64() * bv[row * k + kk].to_f64();
            }
            let d = (got[tok * n + row] as f64 - acc).abs();
            if d > max_abs {
                max_abs = d;
            }
            if acc.abs() > max_ref {
                max_ref = acc.abs();
            }
            if got[tok * n + row] != 0.0 {
                nzero += 1;
            }
        }
    }
    println!(
        "mpp-test: M {m} N {n} K {k}  nonzero {nzero}/{}  max_abs {:.3e}  rel {:.3e}",
        m * n,
        max_abs,
        if max_ref > 0.0 { max_abs / max_ref } else { max_abs }
    );
    anyhow::ensure!(nzero > 0, "the tensor op wrote nothing - the call pattern is still wrong");
    anyhow::ensure!(max_abs / max_ref < 1e-2, "the tensor op produced the wrong values");
    println!("mpp-test: PASSED");
    Ok(())
}

/// Replay every quantised linear's prefill GEMM on the real weights and time the
/// whole batch.
///
/// This exists because every other way of splitting a prefill into "the GEMMs"
/// and "everything else" that we tried turned out to be invalid.  Skipping a
/// kernel with `QW_SKIP_KERNEL`, or skipping the matmul inside `q4_mpp_mm` with
/// `QW_GEMM_MODE=2`, both change what the *later* kernels read, and reading
/// garbage is cheaper than reading real values - `QW_GEMM_MODE=6` claimed the
/// writeback was worth 0.36 s and the honest implementation of it was 1.8 %
/// *slower*.  Replaying the same 497 dispatches in isolation costs nothing that
/// the real pass also pays, so its number can be subtracted from a prefill.
pub fn gemm_bench(model_dir: &Path, tokens: usize, iters: usize) -> Result<()> {
    let mut dev = GpuDevice::new()?;
    let store = WeightStore::load_dir(&dev, model_dir)?;

    let mut names: Vec<String> = store
        .names()
        .filter(|n| n.ends_with(".weight"))
        .filter(|n| !n.contains("embed_tokens"))
        .filter(|n| store.has(&n.replace(".weight", ".scales")))
        .cloned()
        .collect();
    names.sort();

    let mut ls: Vec<QLinear> = Vec::with_capacity(names.len());
    for n in &names {
        let l = QLinear::from_store(&store, n)?;
        if l.in_f % 64 != 0 {
            continue;
        }
        ls.push(l);
    }
    let max_in = ls.iter().map(|l| l.in_f).max().unwrap();
    let max_out = ls.iter().map(|l| l.out_f).max().unwrap();
    let flops: f64 = ls
        .iter()
        .map(|l| 2.0 * tokens as f64 * l.in_f as f64 * l.out_f as f64)
        .sum();

    let x = dev.buffer(tokens * max_in * 2);
    let y = dev.buffer(tokens * max_out * 2);
    let src = qw_metal::msl::mpp_src();
    let [_, nra, nrb, nt] = qw_metal::msl::mpp_tiles();
    let (nra, nrb, nt) = (nra as usize, nrb as usize, nt as usize);

    println!(
        "gemm-bench: {} linears, {tokens} tokens, {:.1} GFLOP/pass, tile NRA={nra} NRB={nrb} NT={nt}",
        ls.len(),
        flops / 1e9
    );

    let mut best = f64::MAX;
    for it in 0..iters {
        let mut b = qw_metal::CommandBatch::new(&mut dev);
        let mk = b.kernel(&src, "q4_mpp_mm")?;
        for l in &ls {
            b.encode(
                qw_metal::Dispatch::new(
                    &mk,
                    (tokens.div_ceil(nrb) * nt, l.out_f.div_ceil(nra), 1),
                    (nt, 1, 1),
                )
                .buf_offset(0, l.weight.buf, l.weight.offset)
                .buf_offset(1, l.scales.buf, l.scales.offset)
                .buf_offset(2, l.biases.buf, l.biases.offset)
                .buf(3, &x)
                .buf(4, &y)
                .scalar(5, l.in_f as i32)
                .scalar(6, tokens as i32)
                .scalar(7, l.out_f as i32)
                .scalar(8, 0),
            );
            b.barrier();
        }
        let t = Instant::now();
        b.finish(true);
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
        println!("  iter {it}: {s:.4} s  {:.1} TFLOPS", flops / s / 1e12);
    }
    println!("  best: {best:.4} s  {:.1} TFLOPS", flops / best / 1e12);
    Ok(())
}

/// Time a real cold prefill in this same process, so it can be compared against
/// `gemm_bench` without the HTTP request, the tokenizer or the sampler in the way.
/// The difference between the two is the honest non-GEMM share.
pub fn prefill_bench(model_dir: &Path, tokens: usize, iters: usize, batch: &str) -> Result<()> {
    let batches: Vec<usize> = batch
        .split(',')
        .map(|x| x.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("--batch: {e}"))?;
    anyhow::ensure!(!batches.is_empty(), "--batch: empty");
    // One model per width, run round-robin, so a thermal drift between the two
    // hits both arms instead of whichever ran second.
    let mut models = Vec::new();
    for &b in &batches {
        models.push(qw_model::runner::Qwen38::load_batch(model_dir, 1020, b)?);
    }
    let rows: Vec<(usize, usize)> = (0..tokens).map(|i| (0usize, i)).collect();
    let mut best = vec![f64::MAX; batches.len()];
    let mut order: Vec<usize> = (0..batches.len()).collect();
    for it in 0..iters {
        if it % 2 == 1 {
            order.reverse();
        }
        for &i in &order {
            models[i].reset();
            let t = Instant::now();
            models[i].forward_rows(&rows)?;
            let s = t.elapsed().as_secs_f64();
            if s < best[i] {
                best[i] = s;
            }
            println!("  iter {it} batch {:3}: {s:.4} s", batches[i]);
        }
    }
    println!();
    let base = best[0];
    for (i, &b) in batches.iter().enumerate() {
        println!(
            "  batch {:3}: best {:.4} s for {tokens} tokens  ({:.3}x batch {})",
            b,
            best[i],
            best[i] / base,
            batches[0]
        );
    }
    Ok(())
}
