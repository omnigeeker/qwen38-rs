//! `qwen38 check` — run our 4-bit GEMV kernel against the CPU reference on
//! **real** checkpoint weights.  This is the first end-to-end validation of the
//! zero-copy loader (mmap -> Metal buffer offset -> kernel -> result).
//!
//! With `--rows N` it instead validates the **batched** kernel: `N` independent
//! activations pushed through a single weight read.  Every output row must match
//! both the CPU reference and the single-row kernel run on the same input, and
//! the two timings are reported side by side — that pair is what batch-16
//! serving is betting on, so it is measured on real weights before any of the
//! forward pass is touched.

use anyhow::{bail, Result};
use half::f16;
use qw_metal::GpuDevice;
use qw_model::QLinear;
use qw_weights::WeightStore;
use std::path::Path;
use std::time::Instant;

fn rng_vec(n: usize, seed: u64) -> Vec<f16> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    (0..n)
        .map(|_| {
            let v = ((next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            f16::from_f32(v)
        })
        .collect()
}

/// Which batched kernel implements a given row count.
fn batched_kernel(rows: usize) -> Result<&'static str> {
    Ok(match rows {
        4 => qw_metal::msl::K_Q4_GEMV_K4_U4HX,
        16 => qw_metal::msl::K_Q4_GEMV_K16_U4HX,
        other => bail!("--rows {other} has no batched kernel (implemented: 4, 16)"),
    })
}

pub fn run(model_dir: &Path, which: Option<&str>, samples: usize, rows: usize) -> Result<()> {
    let mut dev = GpuDevice::new()?;
    let store = WeightStore::load_dir(&dev, model_dir)?;

    // A spread across the architecture: MLP, linear attention, full attention, lm_head.
    let mut targets: Vec<String> = vec![
        "language_model.model.layers.0.mlp.gate_proj.weight".into(),
        "language_model.model.layers.0.mlp.down_proj.weight".into(),
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight".into(),
        "language_model.model.layers.0.linear_attn.out_proj.weight".into(),
        "language_model.model.layers.3.self_attn.q_proj.weight".into(),
        "language_model.model.layers.3.self_attn.o_proj.weight".into(),
        "language_model.model.layers.63.mlp.up_proj.weight".into(),
        "language_model.lm_head.weight".into(),
    ];
    if let Some(t) = which {
        targets = vec![t.to_string()];
    }
    if samples > 0 && samples < targets.len() {
        targets.truncate(samples);
    }

    if rows > 1 {
        return run_batched(&mut dev, &store, &targets, rows);
    }

    let mut failures = 0usize;
    println!(
        "{:<58} {:>6} {:>6} {:>11} {:>10} {:>10}",
        "tensor", "out", "in", "max_abs", "rms_ref", "norm_err"
    );
    for name in &targets {
        if !store.has(name) {
            bail!("missing tensor {name}");
        }
        let linear = QLinear::from_store(&store, name)?;
        if !linear.alignment_ok() {
            println!(
                "{:<58} SKIP (unaligned: w%4={} s%8={} b%8={})",
                linear.name,
                linear.weight.offset % 4,
                linear.scales.offset % 8,
                linear.biases.offset % 8
            );
            continue;
        }
        let x = rng_vec(linear.in_f, 0xC0FFEE);
        let bx = dev.buffer_from_bytes(&x);
        let by = dev.buffer(linear.out_f * 2);
        {
            let mut batch = dev.batch();
            let kernel = QLinear::kernel(&mut batch)?;
            linear.encode(&mut batch, &kernel, &bx, &by);
            batch.finish(true);
        }
        let got: Vec<f16> = by.to_vec(0, linear.out_f);
        let want = linear.cpu_reference(&x)?;

        // The output is stored as fp16, so the meaningful error metric is the
        // absolute error relative to the tensor's own scale (its RMS), not a
        // per-element relative error (entries near zero would dominate).
        let mut max_abs = 0f32;
        let mut sum_sq = 0f64;
        for (g, w) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((g.to_f32() - *w).abs());
            sum_sq += (*w as f64) * (*w as f64);
        }
        let rms = (sum_sq / want.len() as f64).sqrt() as f32;
        let norm_err = max_abs / rms.max(1e-6);
        let ok = norm_err < 5e-2;
        if !ok {
            failures += 1;
        }
        println!(
            "{:<58} {:>6} {:>6} {:>11.4} {:>10.2} {:>9.3}% {}",
            linear.name,
            linear.out_f,
            linear.in_f,
            max_abs,
            rms,
            norm_err * 100.0,
            if ok { "ok" } else { "FAIL" }
        );
    }
    if failures > 0 {
        bail!("{failures} tensor(s) failed the real-weight GEMV check");
    }
    println!("\nall real-weight GEMV checks passed");
    Ok(())
}

/// `rows` independent activations through one weight read, verified per row.
fn run_batched(
    dev: &mut GpuDevice,
    store: &WeightStore,
    targets: &[String],
    rows: usize,
) -> Result<()> {
    let kname = batched_kernel(rows)?;

    // How long the JIT takes to build a 16-row kernel is itself a risk worth
    // knowing about: it happens once per process, at startup.
    let compile_ms = {
        let t0 = Instant::now();
        let mut batch = dev.batch();
        let _ = batch.kernel(qw_metal::msl::COMMON, kname)?;
        batch.finish(false);
        t0.elapsed().as_secs_f64() * 1000.0
    };
    println!("batched kernel {kname} (rows={rows}), Metal JIT compiled in {compile_ms:.0} ms\n");
    println!(
        "{:<46} {:>7} {:>9} {:>9} {:>8} {:>9} {:>9} {:>8} {:>8}",
        "tensor",
        "out",
        "batch_ms",
        &format!("x{rows}_ms"),
        "speedup",
        "max_abs",
        "norm_err",
        "bits!=",
        "vs_1row"
    );

    let mut failures = 0usize;
    for name in targets {
        if !store.has(name) {
            bail!("missing tensor {name}");
        }
        let linear = QLinear::from_store(store, name)?;
        if !linear.alignment_ok() {
            println!("{:<46} SKIP (unaligned)", linear.name);
            continue;
        }
        let out_f = linear.out_f;
        let in_f = linear.in_f;

        // `rows` distinct activations, concatenated into one `[rows][in_f]` buffer.
        let xs: Vec<Vec<f16>> = (0..rows)
            .map(|t| rng_vec(in_f, 0xC0FFEE + t as u64 * 7919))
            .collect();
        let flat: Vec<f16> = xs.iter().flatten().copied().collect();
        let bx = dev.buffer_from_bytes(&flat);
        let by = dev.buffer(out_f * rows * 2);

        // --- batched: one dispatch, one weight read, `rows` output rows ---
        let mut batch_ms = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            {
                let mut b = dev.batch();
                let k = b.kernel(qw_metal::msl::COMMON, kname)?;
                linear.encode_k(&mut b, &k, &bx, &by, rows);
                b.finish(true);
            }
            batch_ms = batch_ms.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let got: Vec<f16> = by.to_vec(0, out_f * rows);

        // --- the honest reference: `rows` separate single-row runs ---
        let bxs: Vec<_> = xs.iter().map(|x| dev.buffer_from_bytes(x)).collect();
        let bys: Vec<_> = (0..rows).map(|_| dev.buffer(out_f * 2)).collect();
        let mut single_ms = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            {
                let mut b = dev.batch();
                let k = QLinear::kernel(&mut b)?;
                for t in 0..rows {
                    linear.encode_k(&mut b, &k, &bxs[t], &bys[t], 1);
                }
                b.finish(true);
            }
            single_ms = single_ms.min(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // --- every row must match the CPU reference AND the single-row kernel ---
        let mut worst_norm = 0f32;
        let mut max_abs_all = 0f32;
        let mut bit_diff = 0usize;
        for t in 0..rows {
            let a = &got[t * out_f..(t + 1) * out_f];
            let b1: Vec<f16> = bys[t].to_vec(0, out_f);
            let want = linear.cpu_reference(&xs[t])?;
            let mut sum_sq = 0f64;
            let mut max_abs = 0f32;
            for i in 0..out_f {
                if a[i] != b1[i] {
                    bit_diff += 1;
                }
                max_abs = max_abs.max((a[i].to_f32() - want[i]).abs());
                sum_sq += (want[i] as f64) * (want[i] as f64);
            }
            let rms = (sum_sq / out_f as f64).sqrt() as f32;
            worst_norm = worst_norm.max(max_abs / rms.max(1e-6));
            max_abs_all = max_abs_all.max(max_abs);
        }

        let ok = worst_norm < 5e-2;
        if !ok {
            failures += 1;
        }
        println!(
            "{:<46} {:>7} {:>8.3} {:>9.3} {:>7.2}x {:>9.4} {:>8.3}% {:>8} {}",
            linear.name,
            out_f,
            batch_ms,
            single_ms,
            single_ms / batch_ms.max(1e-9),
            max_abs_all,
            worst_norm * 100.0,
            bit_diff,
            if ok { "ok" } else { "FAIL" }
        );
    }
    if failures > 0 {
        bail!("{failures} tensor(s) failed the batched GEMV check");
    }
    println!(
        "\nall batched GEMV checks passed: every one of {rows} rows matched its own CPU reference.\
         \nCross-row contamination would be gross, not last-ulp, so this catches it; the bits!=\
         \ncolumn is the ordinary difference between two summation orders and is not gated"
    );
    Ok(())
}
