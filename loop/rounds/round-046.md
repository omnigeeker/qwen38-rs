# Round 046 - one idea refuted by measurement, and the contamination class that faked the first run

- date: 2026-09-16
- goal: raise the single-decoder target from 40 to 50 tok/s
- gates: G1=pass G2=parity 6/6 G3=no speedup claimed G4=pass
- outcome: the logits-argmax idea was implemented, measured, and **reverted as a 0.88% regression**.
  A stray `qwen38 serve` process corrupted the first attempt; with it killed the machine became
  reproducible to 0.0%.

## The idea, and why it was wrong

A speculative pass asks for the argmax of the full 248320-wide vocabulary **five times**: twice
for the MTP drafts (`mtp_step` returned a `Vec<f32>`, built by widening an f16 readback) and three
times for the verify rows (`logits_row` per row).  That is five readbacks, five 1 MB allocations
and 1.24 M f16->f32 widenings per pass, and only an index is wanted.

The change: return the f16 readback from `mtp_step_f16`, take the argmax straight off the f16, and
read all `TILE` verify rows in one `to_vec`.  The reasoning for correctness was sound - f16 -> f32
is monotonic and injective, so `a > b` in f16 exactly when `f32(a) > f32(b)`, ties break the same
way, and parity confirmed it at 6/6.

It is slower.  Three clean pairs, orders alternating:

```
pair1  old 18.28  new 18.07  ratio 1.0116
pair2  old 18.28  new 18.22  ratio 1.0033
pair3  old 18.28  new 18.12  ratio 1.0088
median paired ratio 1.0088  ->  f16 argmax is 0.88% SLOWER
```

The reason is that `f16`'s comparison operators are not free: `PartialOrd for f16` widens both
operands on every compare, so the f16 path pays **two conversions per element compared** inside a
loop that cannot be vectorised, where the f32 path paid **one bulk conversion per element** and
then compared raw f32s.  There are more comparisons than elements, so the trade is strictly
negative.  Reverted; parity re-confirmed at 6/6.

## The contamination class that faked the first A/B

The first attempt at that A/B gave a median paired ratio of 0.8927 - an apparent 12% win - driven
by two `u4` arm readings of 14.72 and 9.29 tok/s against a normal 18.5.  Cause:

```
60244 0.0 14.2 ./target/release/qwen38 serve --model ...
load averages: 2.78 2.88 3.00
```

A stray `qwen38 serve` from an earlier round was still resident holding ~18 GB.  This is the
**second** time a leftover server has corrupted a measurement in this session (round 040 killed
one), so from here a stray-process check belongs in front of every timed comparison, not just in
the acceptance script.  With it killed, the same binary read `18.28` three times in a row - the
paired protocol resolves effects well under 1% when nothing else is running.

## What this buys

A refuted optimisation and a known contamination class.  The saving was only ever projected at
1-2%, so nothing is lost, and the number that mattered - that the v4-kernel change is worth +1.7%
and not 7% - is unchanged from round 045.
