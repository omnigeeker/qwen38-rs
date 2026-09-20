//! Greedy decode driver — the M1/M2 correctness vehicle.
//!
//! The prompt is pushed through the *decode* path one token at a time (a causal
//! model gives the same result as a batched prefill), which means this measures
//! true per-token latency rather than a prefill-optimised path.

use anyhow::Result;
use qw_engine::tokenizer::Tokenizer;
use qw_model::runner::Qwen38;
use qw_model::runner::TILE;
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

/// How often does the MTP head's top-1 draft equal the token the decoder really
/// picks?  That acceptance rate is what speculative decoding runs on, so it is
/// the only number that says whether the head is wired up correctly.
fn mtp_pos_off() -> i64 {
    std::env::var("QW_MTP_OFF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn mtp_check(model: &mut Qwen38, ids: &[u32], steps: usize) -> Result<()> {
    let off = mtp_pos_off();
    if !model.has_mtp() {
        println!("mtp check: head not loaded (set QW_MTP_DIR)");
        return Ok(());
    }
    model.reset();
    let n = ids.len();
    if n == 0 {
        return Ok(());
    }
    // Ride along with the prompt so the head's own cache covers the prompt too.
    let tp = Instant::now();
    for (p, id) in ids.iter().enumerate() {
        model.set_token(*id)?;
        model.forward(p)?;
        if p + 1 < n {
            let mpos = (p as i64 + 1 + off).max(0) as usize;
            model.mtp_step(ids[p + 1], mpos, false)?;
        }
    }
    println!(
        "mtp check: prompt {} tokens, head cache warm in {:.1} ms",
        n,
        tp.elapsed().as_secs_f64() * 1e3
    );

    println!("mtp check: norms {:?}", model.mtp_norm_stats());

    let mut pos = n - 1;
    let tok = model.argmax();
    let t = Instant::now();
    let mut draft = model.mtp_step(tok, (pos as i64 + 1 + off).max(0) as usize, true)?;
    let first_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("mtp check: dump {:?}", model.mtp_dump());

    let mut hits = 0usize;
    let mut total = 0usize;
    let mut log: Vec<(u32, u32)> = Vec::new();
    let t = Instant::now();
    for _ in 0..steps {
        model.set_token(tok)?;
        pos += 1;
        model.forward(pos)?;
        let actual = model.argmax();
        if !draft.is_empty() {
            total += 1;
            let guess = Qwen38::argmax_of(&draft);
            if guess == actual {
                hits += 1;
            }
            if log.len() < 8 {
                log.push((guess, actual));
            }
        }
        draft = model.mtp_step(actual, (pos as i64 + 1 + off).max(0) as usize, true)?;
    }
    let per_step = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!(
        "mtp check: acceptance {hits}/{total} = {:.1}% | first draft {:.1} ms | head+head sweep {:.2} ms/token",
        100.0 * hits as f64 / total.max(1) as f64,
        first_ms,
        per_step
    );
    println!("mtp check: first (draft, actual) pairs {log:?}");
    Ok(())
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
    // Speculative decoding needs the draft head's own k/v cache to cover the
    // prompt, and it can only be advanced while each position's hidden state is
    // still in the tile, so the warm-up rides along with the prefill.
    let spec = qw_model::runner::spec_enabled() && model.has_mtp();
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
        if spec && p + 1 < ids.len() {
            model.mtp_step(ids[p + 1], p + 1, false)?;
        }
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

    // Diagnostic: the prefill is twelve sequential full-weight passes, which is
    // enough GPU load to reach the thermally throttled plateau before the first
    // decode token is timed.  Idling here lets the part cool so the first verify
    // of the run measures the unthrottled clock, which is the closest this
    // machine (battery + Low Power Mode) offers to a mains figure.
    if let Ok(v) = std::env::var("QW_COOL_SLEEP") {
        let secs: f64 = v.parse().unwrap_or(0.0);
        if secs > 0.0 {
            eprintln!("  [cool] idling {secs:.0}s after the prefill before decoding");
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        }
    }

    if std::env::var("QW_MTP_CHECK").is_ok() {
        mtp_check(&mut model, &ids, 32)?;
    }

    if std::env::var("QW_TILE_CHECK").is_ok() {
        // The short-prefix check says nothing about a cache with hundreds of
        // entries in it, which is where the spec and plain paths were seen to
        // part.  Walk TILE rows greedily with the sequential path, then replay
        // the same tokens through the tiled path from a fresh prefill and compare
        // the logits row by row.
        let amax = |v: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, x) in v.iter().enumerate() {
                if *x > bv {
                    bv = *x;
                    bi = i;
                }
            }
            bi as u32
        };
        let md = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        // Depth matters: walk `n` tokens greedily first so the caches are as full
        // as they were when the two paths were seen to part.
        let n: usize = std::env::var("QW_TILE_CHECK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let mut p = ids.len();
        let mut gen: Vec<u32> = Vec::new();
        for _ in 0..n {
            let t = model.argmax();
            gen.push(t);
            model.set_token(t)?;
            model.forward(p)?;
            p += 1;
        }
        let mut toks = Vec::new();
        let mut refs: Vec<Vec<f32>> = Vec::new();
        for r in 0..TILE {
            let next = model.argmax();
            toks.push(next);
            model.set_token(next)?;
            model.forward(p + r)?;
            refs.push(model.logits());
        }
        println!("tilecheck: prefix {p} rows {TILE} tokens {toks:?}");
        model.reset();
        for (i, id) in ids.iter().enumerate() {
            model.set_token(*id)?;
            model.forward(i)?;
        }
        for (i, id) in gen.iter().enumerate() {
            model.set_token(*id)?;
            model.forward(ids.len() + i)?;
        }
        model.set_tokens(&toks)?;
        model.forward2(p)?;
        for (r, want) in refs.iter().enumerate() {
            let got = model.logits_row(r);
            println!(
                "tilecheck: row {r} max|dlogit|={:.6e} argmax {} vs {}",
                md(want, &got),
                amax(&got),
                amax(want)
            );
        }
        return Ok(());
    }

    if std::env::var("QW_K2_CHECK").is_ok() {
        let amax = |v: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, x) in v.iter().enumerate() {
                if *x > bv {
                    bv = *x;
                    bi = i;
                }
            }
            bi as u32
        };
        let md = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let t0 = ids[0];
        let mut base = Vec::new();
        for step in 0..2 {
            model.reset();
            model.set_token(t0)?;
            model.forward(0)?;
            let d = if step == 0 {
                0.0
            } else {
                md(&base, &model.logits())
            };
            base = model.logits();
            println!(
                "k2check: k=1 #{} argmax={} d={:.4}",
                step + 1,
                amax(&base),
                d
            );
        }
        let norm_ref = model.norm_ck();
        let t1 = amax(&base);
        model.reset();
        model.set_token(t0)?;
        model.forward(0)?;
        model.set_token(t1)?;
        model.forward(1)?;
        let ref1 = model.logits();
        for (name, toks) in [("self-pair", vec![t0, t0]), ("pair", vec![t0, t1])] {
            let mut run = |toks: &[u32]| -> Result<(u32, f32, u32, f32)> {
                model.reset();
                model.set_tokens(toks)?;
                model.forward2(0)?;
                let r0 = model.logits_row(0);
                let r1 = model.logits_row(1);
                Ok((amax(&r0), md(&base, &r0), amax(&r1), md(&ref1, &r1)))
            };
            for pass in 1..=2 {
                let a = run(&toks)?;
                println!(
                    "k2check: {name} run{pass} row0 ref={} got={} d={:.4} | row1 ref={} got={} d={:.4}",
                    amax(&base),
                    a.0,
                    a.1,
                    amax(&ref1),
                    a.2,
                    a.3
                );
            }
        }
        model.set_token(t0)?;
        model.forward(0)?;
        println!("k2check: dispatches k=1 = {}", model.last_dispatches());
        model.set_tokens(&[t0, t1])?;
        model.forward2(0)?;
        println!("k2check: dispatches k=2 = {}", model.last_dispatches());
        // per-pass cost at a fixed position, interleaved so thermal drift hits
        // both variants equally; the minimum is the least polluted sample
        let (mut k1, mut k2) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..10 {
                model.set_token(t0)?;
                model.forward(0)?;
            }
            k1 = k1.min(t.elapsed().as_secs_f64() / 10.0);
            let t = std::time::Instant::now();
            for _ in 0..10 {
                model.set_tokens(&[t0, t1])?;
                model.forward2(0)?;
            }
            k2 = k2.min(t.elapsed().as_secs_f64() / 10.0);
        }
        println!(
            "k2check: one weight sweep  k=1 {:.2} ms ({:.1} tok/s) | k=2 {:.2} ms ({:.2} ms/token, {:.1} tok/s)",
            k1 * 1e3,
            1.0 / k1,
            k2 * 1e3,
            k2 * 1e3 / 2.0,
            2.0 / k2
        );
        model.reset();
        model.set_token(t0)?;
        model.forward(0)?;
        let norm_after = model.norm_ck();
        println!(
            "k2check: k=1 after all k2 passes d={:.4}, norm weight intact: {}",
            md(&base, &model.logits()),
            norm_ref == norm_after
        );
        return Ok(());
    }

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
    let mut drafts = 0usize;
    let mut passes = 0usize;
    let t1 = Instant::now();
    if spec {
        model.enable_spec_snap();
        // `next` is a token the decoder has already settled (position `pos`) but
        // not yet emitted; the pass verifies a draft for `pos + 1` and, when the
        // draft is right, hands back the token for `pos + 2` for free from row 1.
        let mut pos = ids.len();
        let mut next = model.argmax();
        let (mut t_draft, mut t_pass) = (0.0f64, 0.0f64);
        // This machine throttles the GPU by ~3x within 20-30 s of sustained load and
        // recovers within ~2 minutes idle, so the average rate over a whole run mixes
        // two clock states and is not reproducible.  The steady-state window below is
        // measured after the throttled plateau is reached, which is.
        let mut steady: Option<(f64, usize, usize)> = None;
        while out.len() < opts.max_tokens {
            // `spec_step` emits the settled token itself, so the end-of-sequence
            // check has to happen here and emit it explicitly.
            if opts.stop_at_eos && tok.is_eos(next) {
                out.push(next);
                break;
            }
            let (p, n, dd, vv) = model.spec_step(pos, next, &mut out)?;
            t_draft += dd;
            t_pass += vv;
            pos = p;
            next = n;
            passes += 1;
            drafts += TILE - 1;
            if steady.is_none() && out.len() * 3 >= opts.max_tokens * 2 {
                steady = Some((t1.elapsed().as_secs_f64(), out.len(), passes));
            }
        }
        // a `TILE`-wide step can overshoot the requested length
        out.truncate(opts.max_tokens);
        // every pass emits its base token plus however many drafts it accepted
        let hits = out.len().saturating_sub(passes);
        let dp = drafts.max(1) as f64;
        eprintln!(
            "spec: draft {:.2} ms/pass, verify {:.2} ms/pass ({:.0}% of a pass)",
            1e3 * t_draft / dp,
            1e3 * t_pass / dp,
            100.0 * t_draft / (t_draft + t_pass).max(1e-9)
        );
        if let Some((t_s, n0, p0)) = steady {
            let dt = t1.elapsed().as_secs_f64() - t_s;
            let dn = out.len().saturating_sub(n0);
            let dp = passes.saturating_sub(p0);
            if dt > 0.0 && dn > 0 {
                eprintln!(
                    "spec: steady-state {:.2} tok/s ({dn} tokens in {dt:.2} s, {:.2} tokens/pass, {:.1} ms/token)",
                    dn as f64 / dt,
                    dn as f64 / dp.max(1) as f64,
                    1e3 * dt / dn as f64
                );
            }
        }
        eprintln!(
            "spec: {hits}/{drafts} drafts accepted ({:.1}%), {:.2} tokens/pass",
            if drafts == 0 {
                0.0
            } else {
                100.0 * hits as f64 / drafts as f64
            },
            out.len() as f64 / passes.max(1) as f64
        );
    } else {
        for pos in ids.len()..ids.len() + opts.max_tokens {
            let next = model.argmax();
            out.push(next);
            if opts.stop_at_eos && tok.is_eos(next) {
                break;
            }
            model.set_token(next)?;
            model.forward(pos)?;
        }
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
