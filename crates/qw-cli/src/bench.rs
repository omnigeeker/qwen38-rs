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

pub fn run(model_dir: &Path, iters: usize) -> Result<()> {
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
            let v = vec![f16::from_f32(0.01); in_f];
            xs.push((in_f, dev.buffer_from_bytes(&v)));
        }
    }
    let ys: Vec<(usize, qw_metal::GpuBuffer)> = {
        let mut v = Vec::new();
        for out_f in linears
            .iter()
            .map(|l| l.out_f)
            .collect::<std::collections::BTreeSet<_>>()
        {
            v.push((out_f, dev.buffer(out_f * 2)));
        }
        v
    };

    let mut best_ms = f64::INFINITY;
    for _ in 0..iters.max(1) {
        let t0 = Instant::now();
        {
            let mut batch = dev.batch();
            let kernel = QLinear::kernel(&mut batch)?;
            for l in &linears {
                let x = &xs.iter().find(|(n, _)| *n == l.in_f).unwrap().1;
                let y = &ys.iter().find(|(n, _)| *n == l.out_f).unwrap().1;
                l.encode(&mut batch, &kernel, x, y);
            }
            batch.finish(true);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        best_ms = best_ms.min(ms);
    }

    let gbps = total_bytes as f64 / (best_ms / 1000.0) / 1e9;
    let tok_s = 1000.0 / best_ms;
    println!(
        "one full weight sweep: {:.2} ms -> {:.1} tok/s linear path, {:.0} GB/s effective",
        best_ms, tok_s, gbps
    );
    // end_to_end = false: this sweep encodes every linear into one command
    // buffer with constant inputs, so it is the upper bound of the linear part,
    // not measured generation throughput.
    println!(
        "{{\"tok_per_s\": {:.2}, \"ms_per_token\": {:.3}, \"effective_gbps\": {:.1}, \"weight_bytes\": {}, \"end_to_end\": false, \"mode\": \"linear_sweep\"}}",
        tok_s, best_ms, gbps, total_bytes
    );
    Ok(())
}
