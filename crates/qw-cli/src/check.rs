//! `qwen38 check` — run our 4-bit GEMV kernel against the CPU reference on
//! **real** checkpoint weights.  This is the first end-to-end validation of the
//! zero-copy loader (mmap -> Metal buffer offset -> kernel -> result).

use anyhow::{bail, Result};
use half::f16;
use qw_metal::GpuDevice;
use qw_model::QLinear;
use qw_weights::WeightStore;
use std::path::Path;

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

pub fn run(model_dir: &Path, which: Option<&str>, samples: usize) -> Result<()> {
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
