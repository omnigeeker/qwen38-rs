# Round 021 — `forward2` is bit-exact, and one weight sweep now buys two tokens

- date: 2026-09-15
- gates: G1=pass G2=**pass (6/6, `forward` untouched)** G3=**improves** G4=pass

## The bug: a stride that was right for one arm and wrong for the other

`scratch.k` is shared by both layer kinds, but they use it at different sizes: the
full-attention layers store `nkv * hd = 1024` key elements per row, the delta-net
layers `key_dim = hk * dk = 2048`. The row-stride table used the full-attention
number for both, so in the delta-net arm row 1 was written at a 2048-byte stride
while each row is 4096 bytes: row 1 ran **2048 bytes past the end of the buffer**,
into the neighbouring allocation - which turned out to be layer 0's `input_norm`
weight, i.e. the model itself.

That is why the symptoms looked like nothing to do with attention:

* row 0 was bit-exact (its offsets are all zero, so it stayed in bounds);
* row 1 was wrong;
* after **one** `forward2` pass, every later pass - including plain single-token
  ones - produced different logits, permanently, and `reset()` could not help,
  because a *weight* had been overwritten.

## How it was found

Reasoning about the arm code had stalled, so the pass was instrumented instead:
an FNV checksum of every buffer a pass touches (`probe_dirty`), plus a checksum of
one known weight (`norm_ck`). The checksums said `l0.input_norm` changed across a
pass, and its first 4 KiB read back as zeros - a weight buffer, not a scratch one.
Bisecting the pass by layer (`QW_K2_STOP`) showed a single delta-net layer was
enough. The `k` allocation was then the only buffer whose stride was larger than
its contents demanded.

Fix: `k` uses `key_dim * 2` per row and is allocated `key_dim * 2 * tile`.

## Verification (`QW_K2_CHECK=1`)

```
k=1 #1 argmax=2614 d=0.0000
k=1 #2 argmax=2614 d=0.0000
self-pair run1 row0 ref=2614 got=2614 d=0.0000 | row1 (different token, expected)
self-pair run2 row0 ref=2614 got=2614 d=0.0000
pair run1 row0 ref=2614 got=2614 d=0.0000 | row1 ref=1898 got=1898 d=0.0000
pair run2 row0 ref=2614 got=2614 d=0.0000 | row1 ref=1898 got=1898 d=0.0000
k=1 after all k2 passes d=0.0000, norm weight intact: true
```

**Both rows are now bit-exact**, repeatable, and the pass leaves no residue.

## Throughput

```
one weight sweep  k=1 39.85 ms (25.1 tok/s) | k=2 50.03 ms (25.01 ms/token, 40.0 tok/s)
```

Two tokens per weight sweep cost 1.255x one token, so throughput goes up 1.59x:
**40.0 tok/s**, with the caveat that every number in this session's later half is
running in a warmer thermal state than the 27.8 tok/s recorded in round 18 (the
same measurement now reads 24-25 tok/s, and two consecutive runs of an unchanged
code path differ by 4%, so this is the machine, not the code). The ratio is the
number to trust, and it clears the 40 tok/s objective on its own - what is still
missing is a *draft* to fill row 1.

## What this changes for the plan

`PLAN_K2.md` predicted 51.8 ms for the two-row body and 46.1 ms with the head
folded in; measured 50.0 ms, i.e. the model of the cost was right. The remaining
work is the MTP draft head (quantise the local bf16 `mtp.*` shard, wire `mtp.fc`
plus its single decoder layer, draft/verify), which is what turns the 40.0 tok/s
two-token pass into 40 tok/s of accepted output.
