# Round 036 - batching the convolution and the norms over the tile rows

- date: 2026-09-15
- gates: G1=pass G2=parity 6/6 + spec==plain byte-for-byte G4=pass

## What changed

The GDN branch of a speculative pass used to run, for each of the `TILE` rows, its own
convolution and its own pair of weightless q/k norms: 6 dispatches per layer per pass that do
not depend on the recurrence.  They are now two kernels over the whole tile -

* `conv1d_silu_ring_tile`, grid `(conv_dim * TILE)`, row `r` reading ring slot `slot0 + r`;
* `rmsnorm_nw_tile`, grid `(hk * TILE * NT)`, threadgroup `(row, head)` flattened;

so `forward2`'s GDN branch is now three phases: qkv for every row into its ring slot (still
per row, because the slot is row dependent), then one convolution and two norms for the tile,
then the recurrence itself row by row (it is sequential in the state and must stay that way).
That removes ~288 of the ~1200 dispatches in a pass.

## A methodology fix worth recording

An intermediate run of the byte-identical gate reported IDENTICAL for both prompts - because
`gen` had died on an MSL compile error and the two empty outputs compared equal.  The gate now
asserts the token count, which is the same class of mistake as trusting a stale binary.  (The
compile error itself: MSL will not accept a kernel that mixes vector thread-position bindings
with scalar ones, so the flattened 1D grid above is not a style choice but a requirement.)

## Result

Order-alternating A/B against the pre-batching binary, four pairs in one cool window:

```
iter1 old 24.62 ms/token   new 24.30   new/old 0.987
iter2 new 24.21            old 25.03   0.967
iter3 old 25.64            new 25.66   1.001
iter4 new 25.95            old 27.22   0.953
median ratio = 0.977
```

So the batching is worth about 2.3% - real, but far less than the dispatch count suggested.
Best readings in that window: 41.31 tok/s (new) against 40.62 (old), which puts the engine at
a median of ~41 tok/s and its best runs just over 41.

## Where the time actually goes

A pass emits 2.51 tokens in ~61 ms; a plain step takes ~37 ms for one token.  The pass moves
roughly 13.5 GB of weights plus ~1.9 GB of lm_head (all three rows need logits to verify
against) plus ~1.4 GB of MTP drafts - about 16.8 GB in 61 ms, or ~275 GB/s, where the plain
step already reaches ~364 GB/s on the same weights.  **So the speculative pass is not at the
bandwidth ceiling; it has real headroom.**  Since the per-row sequential work is small, the
leading suspect is the three-row accumulator in the tiled GEMV costing occupancy, and that is
the next thing to attack - a pass that matched the plain step's bandwidth efficiency would
land near 60 tok/s.

## The next lever, with the reasoning already done

Reading `Q4_GEMV_KS` settles two things.

**The tiled GEMV is bit-exact per row versus the k=1 kernel.**  Both walk `g = lane; g <
n_groups; g += 32`, unpack the same `Q4_WORDS_PER_GROUP` words in order, and reduce with
`simd_sum(acc[t])` per row.  A tile launch therefore performs the same operations in the same
order for each row that three k=1 launches would - which is why `QW_TILE_CHECK` sees exactly
0.0 rather than "close", and it means a tile launch can replace k=1 launches without touching
a single bit of output.

**The GDN branch still reads its qkv weights three times per pass.**  `g.in_qkv` is
`(key_dim * 2 + value_dim) x hidden`, about 21 MB per layer, and it is projected with a
separate k=1 launch per row: ~3 GB per pass, ~19% of all traffic, of which two thirds is the
same bytes read again.  The only reason it is per row is that the three output rows have to
land in three consecutive ring slots, and `slot0 + 2` wraps for two of the eight slot
positions.

The way through is to stop writing the projection straight into the ring:

1. project the tile with the k=3 kernel into a contiguous `qkv_cur[TILE][conv_dim]` - one
   launch, one read of the weights, and by the argument above bit-identical to the three
   separate launches;
2. have the tile convolution read its older rows from the ring and its own rows from
   `qkv_cur`, which needs one extra pointer and one comparison in the kernel;
3. at the end of the pass, copy the `k + 1` accepted rows from `qkv_cur` into their ring
   slots - a pure memcpy, so still bit-identical, and only for rows that survive.

That is one launch where there are now three, and it removes roughly 2 GB of redundant weight
traffic per pass, which is ~12% of the pass - several times what the whole of round 036
bought.  It is a real change to the ring's bookkeeping, so it wants a full round rather than
the tail of this one.
