# Round 037 - one tiled launch for the GDN qkv projection

- date: 2026-09-15
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte G4=pass

## What changed

The GDN branch projected `in_qkv` with one k=1 launch *per tile row* - three launches reading
the same ~21 MB of weights per layer per pass, about 3 GB per pass and 19% of all traffic,
two thirds of it the same bytes again.  The only reason it was per row is that the three
output rows had to land in three consecutive ring slots, and `slot0 + 2` wraps for two of the
eight slot positions.

Now the tile is projected with the k=3 kernel into a contiguous `qkv_cur[TILE][conv_dim]`
staging buffer, and the tile convolution reads its own rows from that buffer (`pos >= pos0`)
while older rows still come from the ring.  Each layer then copies its own `TILE` rows from
the staging buffer into their ring slots, which is a pure memcpy.

This is safe to do because `Q4_GEMV_KS` is bit-identical per row to the k=1 kernel: the same
`g = lane; g < n_groups; g += 32` walk, the same `Q4_WORDS_PER_GROUP` unpack order, and the
same `simd_sum(acc[t])` reduction for each row.  A tile launch therefore performs exactly the
operations in exactly the order that three k=1 launches would.

## The bug I wrote and the gate caught

The first version copied the accepted rows into the ring at the *end of the pass*, from
`spec_step`, which is where the state rewind lives.  That is wrong because `qkv_cur` is a
single scratch buffer shared by every layer: by the end of the pass it holds the last layer's
rows, so every other layer's ring got that layer's data.  Output diverged at token 4, exactly
like the ring-size bug of round 035 - the byte-identical gate caught it in one run.  The copy
now happens inside each layer's own block, right after that layer's convolution.

(Keeping the acceptance-dependent version, which would copy only the `k + 1` surviving rows,
is not possible: `k` is only known after the logits are read, which is after the whole forward
has been encoded.  Copying all `TILE` rows is what the old per-row projection did, and the
ring is wide enough that three rows written ahead cannot reach what the next pass reads.)

## Result

Order-alternating A/B against the round-036 binary, four pairs in one cool window:

```
iter1 old 24.76 ms/token   new 24.29   new/old 0.981
iter2 old 24.75            new 24.15   0.976
iter3 old 25.21            new 25.72   1.020
iter4 old 28.38            new 27.11   0.955
median paired ratio = 0.978
```

About 2.2%, three pairs of four in favour.  Best readings 24.15 / 24.29 ms/token, i.e. ~41.4
tok/s, against 24.75 / 24.76 for the previous build.

Notably this is much less than the ~2 GB of traffic saved would suggest if the pass were
bandwidth-bound, which is consistent with the tiled GEMV being issue-bound rather than
memory-bound.  Two rounds of batching have now bought ~2% each; the remaining per-row
launches in a pass are the attention layer's `kv_append`, `attn_scores`, `attn_out` and
`gate_mul` (4 per row, 12 per attention layer, ~190 per pass), and those are the next
candidates.

## One unresolved observation

While checking the clippy fix, a single `verify` run reported `parity: 5/6` - with `spec` and
`plain` still byte-identical over 300 tokens on that same build, and with the change in
question (`(t - 1) as i32` to `t - 1`, where `t` is already `i32`, and the removal of two
struct fields that nothing read) incapable of altering arithmetic.  It has not reproduced:
six runs on that binary and the previous one, then ten more runs, all report 6/6, and the
failing case was not captured because the reporting run only grepped the summary line.

What is notable is the context of that one run: it was the last command of a shell invocation
that had just run `cargo build`, `cargo fmt`, `cargo clippy` and `cargo test --workspace` -
and the workspace tests load and run the model on the same GPU.  Every other gate run in this
project has been issued on an otherwise idle machine.  So the leading hypothesis is resource
contention with a failure mode that is not validated (device buffers are written without
error checking), not a numerical difference, and the operational rule that falls out of it is
that GPU gates should not share a process window with a build-and-test sweep.  This is flagged
for the next round rather than closed: an intermittent parity failure in a gate that the whole
correctness argument rests on deserves a real reproduction attempt, and the cheapest next step
is to loop `verify` many times while recording the failing case's identity and the load
average.
