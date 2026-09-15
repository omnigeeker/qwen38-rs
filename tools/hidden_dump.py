#!/usr/bin/env python3
"""Per-layer hidden-state dump from mlx-lm, fed **one token at a time**.

The batched prefill in `oracle.py --dump-hidden` uses different mlx kernels than
a decode step, so it is not the right reference for a decode-only engine.  This
script replays the prompt token by token and records, after every layer, the
statistics of the *last* token's hidden state — the exact same quantity our
engine prints with `qwen38 gen --dump-hidden`.
"""

import argparse
import json
import sys

DEFAULT_MODEL = "models/Qwen3.8-27B-4bit"


def hidden_stats(h):
    import mlx.core as mx

    flat = h.reshape(-1).astype(mx.float32)
    n = flat.size
    mean = float(mx.mean(flat))
    var = float(mx.mean((flat - mean) ** 2))
    return {
        "mean": mean,
        "std": var**0.5,
        "absmax": float(mx.max(mx.abs(flat))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="loop/artifacts/hidden_stepwise.json")
    ap.add_argument("--vectors", default="", help="write every layer's f32 vector here")
    args = ap.parse_args()

    from mlx_lm import load
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    model, tokenizer = load(args.model)
    lm = model.language_model
    inner = lm.model
    ids = tokenizer.encode(args.prompt)
    print(f"[hidden] prompt ids: {ids}", file=sys.stderr)

    cache = model.make_cache()
    stats = {"prompt_ids": ids, "layers": {}}
    vectors = []
    for step, tok in enumerate(ids):
        h = inner.embed_tokens(mx.array([[tok]]))
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        if step == 0:
            stats["embedding"] = hidden_stats(h)
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
            mx.eval(h)
            if step == len(ids) - 1:
                stats["layers"][f"layer_{i}_{'lin' if layer.is_linear else 'full'}"] = (
                    hidden_stats(h)
                )
                vectors.append(h.reshape(-1).astype(mx.float32))
    hn = inner.norm(h)
    mx.eval(hn)
    stats["final_norm"] = hidden_stats(hn)
    logits = lm.lm_head(hn)
    mx.eval(logits)
    last = logits[0, -1].astype(mx.float32)
    order = mx.argsort(-last)[:8]
    stats["top8"] = [[int(i), float(last[int(i)])] for i in order]

    if args.vectors:
        import numpy as np

        np.stack([np.array(v) for v in vectors]).astype("float32").tofile(args.vectors)
        print(f"[hidden] wrote {args.vectors}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"[hidden] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
