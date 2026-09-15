//! Greedy decode driver — the M1/M2 correctness vehicle.
//!
//! The prompt is pushed through the *decode* path one token at a time (a causal
//! model gives the same result as a batched prefill), which means this measures
//! true per-token latency rather than a prefill-optimised path.

use anyhow::Result;
use qw_engine::tokenizer::Tokenizer;
use qw_model::runner::Qwen38;
use std::path::Path;
use std::time::Instant;

pub struct GenOpts<'a> {
    pub model_dir: &'a Path,
    pub prompt: &'a str,
    pub max_tokens: usize,
    pub max_t: usize,
    pub dump_top: usize,
    pub dump_hidden: bool,
    pub dump_vectors: Option<String>,
    pub stop_at_eos: bool,
}

pub fn run(opts: GenOpts<'_>) -> Result<()> {
    let t_load = Instant::now();
    let mut model = Qwen38::load(opts.model_dir, opts.max_t)?;
    let tok = Tokenizer::from_file(&opts.model_dir.join("tokenizer.json"))?;
    eprintln!(
        "loaded in {:.1}s ({} layers, max_t {})",
        t_load.elapsed().as_secs_f32(),
        model.cfg.num_hidden_layers,
        opts.max_t
    );

    if opts.dump_hidden || opts.dump_vectors.is_some() {
        model.enable_debug();
    }
    let ids = tok.encode(opts.prompt, false)?;
    eprintln!("prompt ids: {ids:?}");

    let debug = std::env::var("QW_DEBUG").is_ok();
    let t0 = Instant::now();
    for (p, id) in ids.iter().enumerate() {
        model.set_token(*id)?;
        if debug {
            eprintln!(
                "  [dbg] after set_token({id}): x[0..6]={:?}",
                model.peek("x", 6)
            );
        }
        model.forward(p)?;
        if debug {
            eprintln!("  [dbg] probe lm_head alone: {:?}", model.probe_lm_head()?);
            eprintln!(
                "  [dbg] pos {p}: dispatches={} x[0..6]={:?} h[0..6]={:?} logits[0..4]={:?}",
                model.last_dispatches(),
                model.peek("x", 6),
                model.peek("h", 6),
                model.peek("logits", 4)
            );
        }
    }
    let prefill = t0.elapsed();

    if let Some(path) = &opts.dump_vectors {
        model.write_vectors(std::path::Path::new(path))?;
        eprintln!("wrote {path}");
    }

    if opts.dump_hidden {
        println!("hidden_stats:");
        for (i, (mean, std, absmax)) in model.debug_stats().iter().enumerate() {
            println!("  layer_{i:02}  mean={mean:+.6} std={std:.6} absmax={absmax:.6}");
        }
    }

    if opts.dump_top > 0 {
        println!("first_logits_topk:");
        for (id, v) in model.top_k(opts.dump_top) {
            let piece = tok.decode(&[id], true).unwrap_or_default();
            println!("  {id:>7}  {v:>10.4}  {:?}", piece);
        }
    }

    let mut out: Vec<u32> = Vec::new();
    let t1 = Instant::now();
    for pos in ids.len()..ids.len() + opts.max_tokens {
        let next = model.argmax();
        out.push(next);
        if opts.stop_at_eos && tok.is_eos(next) {
            break;
        }
        model.set_token(next)?;
        model.forward(pos)?;
    }
    let dt = t1.elapsed();

    println!("greedy_ids: {out:?}");
    println!("greedy_text: {}", tok.decode(&out, true)?);
    let n = out.len().max(1);
    println!(
        "timing: prefill {} tokens in {:.3}s ({:.1} tok/s) | decode {:.2} tok/s ({:.2} ms/token)",
        ids.len(),
        prefill.as_secs_f32(),
        ids.len() as f32 / prefill.as_secs_f32(),
        n as f32 / dt.as_secs_f32(),
        dt.as_secs_f32() * 1000.0 / n as f32
    );
    Ok(())
}

/// Compare a generated continuation against the mlx-lm oracle's greedy ids.
pub fn check_against_oracle(
    model_dir: &Path,
    oracle: &Path,
    max_t: usize,
    limit: Option<&str>,
) -> Result<()> {
    let raw = std::fs::read_to_string(oracle)?;
    let doc: serde_json::Value = serde_json::from_str(&raw)?;
    let cases = doc["cases"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("oracle has no cases"))?;

    let mut model = Qwen38::load(model_dir, max_t)?;
    let tok = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
    let mut pass = 0usize;
    let mut total = 0usize;
    for (name, c) in cases {
        if let Some(only) = limit {
            if only != name {
                continue;
            }
        }
        total += 1;
        let prompt = c["prompt"].as_str().unwrap_or("");
        let want_ids: Vec<u32> = c["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_greedy: Vec<u32> = c["greedy_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let ids = tok.encode(prompt, false)?;
        let tok_ok = ids == want_ids;

        // reset caches
        model.reset();
        for (p, id) in ids.iter().enumerate() {
            model.set_token(*id)?;
            model.forward(p)?;
        }
        let mut got = Vec::new();
        for pos in ids.len()..ids.len() + want_greedy.len() {
            let next = model.argmax();
            got.push(next);
            model.set_token(next)?;
            model.forward(pos)?;
        }
        let match_len = got
            .iter()
            .zip(want_greedy.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let ok = tok_ok && match_len == want_greedy.len();
        if ok {
            pass += 1;
        }
        println!(
            "{:<20} tokens={} greedy {}/{} {}   got={:?}",
            name,
            if tok_ok { "ok" } else { "MISMATCH" },
            match_len,
            want_greedy.len(),
            if ok { "PASS" } else { "FAIL" },
            &got[..match_len.min(got.len()).min(6)]
        );
    }
    println!("parity: {pass}/{total} cases");
    if pass != total {
        anyhow::bail!("oracle parity failed");
    }
    Ok(())
}
