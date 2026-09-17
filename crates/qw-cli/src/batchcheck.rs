//! `qwen38 batchcheck` - prove that one batched pass equals independent passes.
//!
//! The entire point of the batch work is that sixteen sequences advance one token
//! each from a SINGLE read of the 14.4 GB of weights.  That is worth nothing
//! unless the result is what sixteen separate runs would have produced, so this is
//! the gate.
//!
//! There are two phases, because one alone would be weak.  In the first every slot
//! is fed the same prompt, position by position.  In the second, one more pass
//! feeds each slot a DIFFERENT token, and that phase is the one that matters: if
//! the per-sequence offsets were wrong then every slot could read and write the
//! same state and still agree with the other slots.  Once the tokens differ the
//! states have to diverge, and the only way the batched result can match a slot's
//! independent run is if the slots are genuinely separate.

use anyhow::Result;
use qw_engine::tokenizer::Tokenizer;
use qw_model::runner::Qwen38;
use std::path::Path;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

pub fn run(model_dir: &Path, prompt: &str, slots: usize, max_t: usize) -> Result<()> {
    anyhow::ensure!((1..=16).contains(&slots), "--slots must be 1..=16");
    anyhow::ensure!(max_t >= 64, "--max-t must be at least 64");
    let mut model = Qwen38::load_batch(model_dir, max_t, slots)?;
    let tok = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
    let ids = tok.encode(prompt, false)?;
    anyhow::ensure!(
        !ids.is_empty() && ids.len() + 8 < max_t,
        "prompt must fit the context"
    );
    // One distinct token per slot for the divergence pass.
    let diverge: Vec<u32> = (0..slots).map(|s| 1000 + 7 * s as u32).collect();
    println!(
        "batch gate: {slots} slots, prompt {} tokens, divergence tokens {:?}",
        ids.len(),
        &diverge[..slots.min(4)]
    );

    // ---- reference: each slot completely alone ----
    let mut alone: Vec<Vec<f32>> = Vec::new();
    for (s, &dtok) in diverge.iter().enumerate() {
        model.reset_seq(s)?;
        for (p, id) in ids.iter().enumerate() {
            model.set_token(*id)?;
            model.forward_seq(s, p)?;
        }
        model.set_token(dtok)?;
        model.forward_seq(s, ids.len())?;
        alone.push(model.logits_row(0));
    }

    // ---- batched: one pass per position covering every slot ----
    for s in 0..slots {
        model.reset_seq(s)?;
    }
    for (p, id) in ids.iter().enumerate() {
        model.set_tokens(&vec![*id; slots])?;
        let rows: Vec<(usize, usize)> = (0..slots).map(|s| (s, p)).collect();
        model.forward_rows(&rows)?;
    }
    model.set_tokens(&diverge)?;
    let rows: Vec<(usize, usize)> = (0..slots).map(|s| (s, ids.len())).collect();
    model.forward_rows(&rows)?;
    let batched: Vec<Vec<f32>> = (0..slots).map(|s| model.logits_row(s)).collect();

    // ---- compare ----
    let mut bad = 0usize;
    let mut worst = 0.0f32;
    for s in 0..slots {
        let a = argmax(&alone[s]);
        let b = argmax(&batched[s]);
        let d = alone[s]
            .iter()
            .zip(&batched[s])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(d);
        if a != b {
            bad += 1;
            println!("  slot {s}: MISMATCH independent={a} batched={b}");
        }
    }
    println!(
        "greedy token identical in {} of {slots} slots; worst |logit difference| {worst:.4}",
        slots - bad
    );
    anyhow::ensure!(
        bad == 0,
        "{bad} of {slots} slots disagree between the batched pass and their independent runs"
    );
    println!("batch gate PASSED: one {slots}-row pass equals {slots} independent passes");
    Ok(())
}
