//! `qwen38 pos-bench` - what one pass costs, as a function of where it sits.
//!
//! Prefill costs more per token the further into the prompt it gets.  The server's
//! progress log measures `ms/token ~= 9.2 + 5.5e-3 * position` for a 6865-token
//! prompt, which makes the position term about 80% of a 7K cold prefill.  Neither
//! attention FLOPs (~7 ms for the whole prompt) nor attention memory traffic
//! (~1.5 s) explains that, so the point of this command is to cost a pass properly
//! before going anywhere near a kernel.
//!
//! The pass cost decomposes as `ms/token = C / rows + c_row + f(T)`, where `C` is
//! the fixed per-pass cost and `f(T)` is whatever scales with the cached history.
//! Sweeping with ONE row per pass and comparing the slope against the four-row
//! slope separates them: the same slope means `f(T)` is charged per ROW (a kernel
//! re-reading the history), four times steeper means it is charged per PASS.

use anyhow::Result;
use qw_model::runner::Qwen38;
use std::path::Path;

pub fn run(
    model_dir: &Path,
    tokens: usize,
    rows: usize,
    slots: usize,
    max_t: usize,
    report: usize,
) -> Result<()> {
    anyhow::ensure!((1..=16).contains(&rows), "--rows must be 1..=16");
    anyhow::ensure!((1..=16).contains(&slots), "--slots must be 1..=16");
    // One sequence can only contribute four consecutive positions per pass: the
    // convolution ring holds `conv_k` rows of history plus the rows a pass writes.
    // Several sequences can each contribute a row, up to the batch maximum, which is
    // what makes the marginal cost of rows 5..16 measurable BEFORE committing to
    // widening the ring.
    anyhow::ensure!(
        slots > 1 || rows <= 4,
        "--rows above 4 needs --slots above 1; one sequence is capped at 4 rows per pass"
    );
    anyhow::ensure!(
        slots == 1 || rows <= slots,
        "--rows must not exceed --slots when several sequences are used"
    );
    anyhow::ensure!(tokens >= 64, "--tokens must be at least 64");
    anyhow::ensure!(tokens + 8 < max_t, "--tokens must fit inside --max-t");
    anyhow::ensure!(report >= 1, "--report must be at least 1");

    let mut model = Qwen38::load_batch(model_dir, max_t, slots)?;
    // Ids inside the vocabulary; the values do not matter, only the positions do.
    let ids: Vec<u32> = (0..tokens)
        .map(|i| 1000 + ((i * 7919) % 90_000) as u32)
        .collect();
    for s in 0..slots {
        model.reset_seq(s)?;
    }

    println!(
        "pos-bench: {tokens} tokens, {rows} row(s) per pass over {slots} sequence(s), max_t {max_t}"
    );
    println!(
        "{:>7} {:>11} {:>11} {:>9} {:>10}",
        "pos", "ms/pass", "ms/token", "tok/s", "elapsed"
    );

    let started = std::time::Instant::now();
    let mut p = 0usize;
    let mut mark = 0usize;
    let mut seg_ms = 0.0f64;
    let mut seg_tok = 0usize;
    let mut seg_passes = 0usize;
    while p < tokens {
        let k = rows.min(tokens - p);
        // One sequence: `k` consecutive positions.  Several: one row each, all at `p`.
        let staged: Vec<u32> = if slots == 1 {
            ids[p..p + k].to_vec()
        } else {
            vec![ids[p]; k]
        };
        model.set_tokens(&staged)?;
        let rws: Vec<(usize, usize)> = if slots == 1 {
            (0..k).map(|i| (0usize, p + i)).collect()
        } else {
            (0..k).map(|s| (s, p)).collect()
        };
        let t0 = std::time::Instant::now();
        model.forward_rows(&rws)?;
        seg_ms += t0.elapsed().as_secs_f64() * 1000.0;
        seg_tok += k;
        seg_passes += 1;
        p += k;
        if p >= mark + report || p == tokens {
            println!(
                "{:>7} {:>11.2} {:>11.2} {:>9.1} {:>9.1}s",
                p,
                seg_ms / seg_passes as f64,
                seg_ms / seg_tok as f64,
                seg_tok as f64 / (seg_ms / 1000.0),
                started.elapsed().as_secs_f64()
            );
            mark = p;
            seg_ms = 0.0;
            seg_tok = 0;
            seg_passes = 0;
        }
    }
    println!(
        "done: {tokens} tokens in {:.1}s at {rows} row(s)/pass",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
