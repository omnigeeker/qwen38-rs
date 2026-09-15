# Round 038 - contention experiment, and a batching attempt that measured slower

- date: 2026-09-15
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte G4=pass
- outcome: no code change.  The round's value is two negative results and one latent trap.

## The intermittent parity reading, from the other side

Round 037 flagged a single `parity: 5/6` that 13 runs could not reproduce.  The hypothesis
recorded there was resource contention with a failure mode that is not validated, because the
one anomalous run was the only gate issued in the same shell window as `cargo build`, `fmt`,
`clippy` and `cargo test --workspace` - and the workspace tests load and run the model on the
same GPU.

Deliberate reproduction attempt: 30 `verify` runs in a loop, each writing its full output to
its own file, with `sysctl vm.loadavg` recorded per run.  Five runs completed on the
unmodified binary under load 3.7 - 6.8 and all five reported 6/6.  The loop was then
contaminated by a rebuild (runs 6+ picked up a binary I was mid-edit on, which is why they
read 0/6), and was killed.  So the anomaly did not reproduce, and at 1 in ~19 observed runs it
stays open at the same low priority: the gate is not trustworthy enough to be ignored, not
unstable enough to block on.  The operational rule stands - do not issue GPU gates in the same
process window as a build-and-test sweep.

A useful side effect: comparing idle against loaded output showed the plain path is
load-independent (0 differences over 300 tokens with a concurrent generation running), so the
plain decode path is not the fragile one.

## A latent trap worth writing down

`scratch.k` is shared by two branches whose rows are NOT the same width.  The GDN branch
defines `key_dim = hk * dk`; the attention branch's k projection is `nkv * hd`.  Both are
plausible values for the same expression, and the attention branch's dispatch used the local
`key_dim` as the row offset while `scratch.pv` used `nkv * hd`.  Any refactor that assumes the
two names mean the same number will silently read the wrong row - and the failure looks like a
spec-vs-plain divergence at a token whose index moves around, not like an obvious corruption.
This cost most of the round.

## The batching attempt

With the strides made explicit, `kv_append` and `gate_mul` batch over the tile correctly and
byte-identically (parity 6/6, spec == plain, plain == round 037), removing 4 launches per
attention layer and 64 per pass.  It measured SLOWER:

```
iter 1: round-037 23.86 ms   attn-batched 24.85 ms   ratio 1.041
iter 2: round-037 23.08 ms   attn-batched 24.76 ms   ratio 1.073
mean paired ratio 1.057
```

Reverted.  The reason is the shape of the kernel: batching requires the row to be decoded from
the flattened index, i.e. one integer division per thread, while `gate_mul` does one load, one
`exp` and one store, and `kv_append` does two loads and two stores.  The division costs more
than the launch it removes.  This is the opposite of what happened with the convolution and
the norms in rounds 036-037, where each thread does enough work to hide the decode.

So the cheap batching lever is now exhausted: the remaining per-row launches in a pass are the
two `rmsnorm_ws` pairs, the two `rope` pairs, `attn_scores` and `attn_out`, and the first two
are the only ones whose per-thread work might still amortise a decode.  Since two rounds of
batching bought ~2% each and dispatch time is ~6.5 ms of a 61 ms pass, dispatch reduction
cannot be the main lever; the tiled GEMV's issue cost is.
