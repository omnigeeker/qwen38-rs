# Round 022 — the top-level ops become one dispatch each; and a cost model that
# says the next lever is more rows, not fewer dispatches

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6, and `forward2` bit-exact on two different prompts)** G3=small win G4=pass

## What changed

`rmsnorm` (both norms), `ewise_add` (both residuals) and `silu_mul` are issued once
for the whole tile instead of once per row.  No MSL changed: every one of those
kernels already derives its row from the threadgroup id and strides by the row
length, and a tile row *is* one row length, so a grid of `TILE * N` threads covers
both rows.  The generator emits those dispatches with the grid multiplied and no row
offsets.

Dispatch count for one pass: **2115 -> 1794** (the 321 predicted: five per layer plus
the final norm).  Bit-exactness is unchanged: both rows still `d = 0.0000`, on two
prompts, repeated runs, `reset()` afterwards clean.

## The measurement, and a warning about it

Timings at a fixed position, interleaved (k=1, k=2, k=1, k=2, ...) and reported as
the minimum of five:

```
run A:  k=1 39.79 ms (25.1 tok/s) | k=2 49.18 ms (24.59 ms/token, 40.7 tok/s)
run B:  k=1 41.78 ms (23.9 tok/s) | k=2 61.53 ms (30.77 ms/token, 32.5 tok/s)
```

Same binary.  The machine is being hammered by these very benchmarks and the spread
is 25%; the k=2/k=1 ratio moves from 1.24 to 1.47, which is what throttling should do
(it hurts the compute-bound per-row kernels more than the bandwidth-bound sweep).
**Only interleaved minima from a single run are comparable**, and the ratio is more
trustworthy than either absolute number.  An earlier reading of 55.7 ms after this
change was noise, not a regression - the change is a small win (2115 -> 1794
dispatches).

## The cost model this gives us

```
one row   39.8 ms   (of which ~27 ms is the weight sweep)
two rows  49.2 ms   -> the second row costs ~9.4 ms
```

So an extra row costs ~9.4 ms against a ~27 ms sweep that is paid once.  That is the
whole argument for drafting more tokens per pass:

```
k=1   39.8 ms / 1 token   = 25.1 tok/s
k=2   49.2 ms / 2 tokens  = 40.7 tok/s
k=3  ~58.6 ms / 3 tokens  = 51 tok/s
k=4  ~68.0 ms / 4 tokens  = 59 tok/s
```

...before the draft cost and before acceptance.  With per-token acceptance `a` and
`k` drafted tokens the pass yields `a + a^2 + ... + a^k + 1` tokens, and the MTP head
costs about 2.5 ms per drafted token (its own layer plus a 636 MB head sweep):

```
a = 0.7:  k=2 33 tok/s   k=3 38 tok/s   k=4 39 tok/s
a = 0.8:  k=2 35 tok/s   k=3 45 tok/s   k=4 43 tok/s
```

Two consequences.  First, `TILE` must become a parameter (the `q4_gemv_k3/k4` kernels
already exist), because with `a` in the usual 0.7-0.8 band most of the headroom above
40 comes from amortising the sweep over three or four rows.  Second, the acceptance
rate is worth more than any micro-optimisation left in the per-row kernels, which is
why the MTP head is next rather than more dispatch fusion.

## The problem speculative decoding has with a recurrent model

The delta-net layers carry a state that advances once per token, so accepting only a
prefix of the drafted rows leaves the state too far ahead.  Restoring by recomputing
is the usual answer but costs a whole extra pass here.  The cheap answer is a
snapshot: the state is 48 layers x 3 MB = 147 MB, so a per-row snapshot costs about
0.6 ms of traffic and one copy dispatch per layer per row, and a rejected prefix is
then repaired by a pointer switch instead of a recompute.  Worth its own round after
the MTP head exists and the real acceptance rate is measured.
