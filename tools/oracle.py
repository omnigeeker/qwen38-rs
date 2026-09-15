#!/usr/bin/env python3
"""Reference oracle for qwen38-rs.

Runs the *same* 4-bit checkpoint through mlx-lm (the reference implementation of
Qwen3.8, `mlx_lm/models/qwen3_5.py`) and dumps:

  * greedy token ids for every prompt in the suite (bit-exact parity target)
  * top-k logits of the first generated position (numerical tolerance target)
  * optional per-layer hidden-state statistics (to localise a mismatch)

Usage:
    python3 tools/oracle.py --out loop/artifacts/oracle.json
    python3 tools/oracle.py --dump-hidden --out loop/artifacts/oracle.json
"""

import argparse
import json
import os
import sys
import time

DEFAULT_MODEL = os.environ.get("MODEL_DIR", "models/Qwen3.8-27B-4bit")
DEFAULT_SUITE = "tools/prompts.json"


def load_suite(path):
    with open(path) as f:
        return json.load(f)


def hidden_stats(h):
    """Compact, comparable summary of a hidden-state tensor."""
    import mlx.core as mx

    flat = h.reshape(-1).astype(mx.float32)
    n = flat.size
    mean = float(mx.mean(flat))
    var = float(mx.mean((flat - mean) ** 2))
    absmax = float(mx.max(mx.abs(flat)))
    head = [float(v) for v in flat[:32]]
    # checksum insensitive to ordering but sensitive to values
    checksum = float(mx.sum(flat * mx.arange(1, n + 1, dtype=mx.float32)))
    return {
        "shape": list(h.shape),
        "mean": mean,
        "std": var ** 0.5,
        "absmax": absmax,
        "head32": head,
        "checksum": checksum,
    }


def run(args):
    import mlx.core as mx
    from mlx_lm import load

    suite = load_suite(args.suite)
    print(f"[oracle] loading {args.model}", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"[oracle] loaded in {time.time() - t0:.1f}s", flush=True)

    lm = model.language_model
    text_args = lm.args

    out = {"model": args.model, "suite": args.suite, "cases": {}, "hidden": {}}

    for case in suite["cases"]:
        name = case["name"]
        prompt = case["prompt"]
        max_tokens = case.get("max_tokens", 16)
        ids = tokenizer.encode(prompt)
        cache = model.make_cache()
        generated = []
        first_logits_topk = None
        t0 = time.time()
        logits = None
        if args.stepwise:
            # Feed the prompt one token at a time, exactly like a decode-only
            # engine does.  mlx uses different kernels for a batched prefill
            # (matmul over T tokens) than for a single-token step, so this is
            # the only apples-to-apples comparison for our engine.
            for tok in ids:
                logits = model(mx.array([[tok]]), cache=cache)
        else:
            cur = mx.array([ids])
        for step in range(max_tokens):
            # step 0 of the stepwise path already has the prompt's logits.
            if not (args.stepwise and step == 0):
                logits = model(cur, cache=cache)
            last = logits[:, -1, :].astype(mx.float32)
            if step == 0:
                top = mx.argsort(-last[0])[:8]
                first_logits_topk = [
                    [int(i), float(last[0, int(i)])] for i in top
                ]
            nxt = int(mx.argmax(last, axis=-1)[0])
            generated.append(nxt)
            cur = mx.array([[nxt]])
        dt = time.time() - t0
        out["cases"][name] = {
            "prompt": prompt,
            "prompt_ids": ids,
            "greedy_ids": generated,
            "greedy_text": tokenizer.decode(generated),
            "first_logits_topk": first_logits_topk,
            "seconds": dt,
            "tok_per_s": max_tokens / dt if dt > 0 else None,
        }
        print(
            f"[oracle] {name}: {max_tokens} tok in {dt:.2f}s "
            f"({max_tokens / dt:.1f} tok/s) -> {generated[:8]}",
            flush=True,
        )

        if args.dump_hidden:
            out["hidden"][name] = dump_hidden(model, text_args, ids)
            print(f"[oracle] hidden states dumped for {name}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[oracle] wrote {args.out}", flush=True)


def dump_hidden(model, text_args, ids):
    """Replay the decoder layer by layer and summarise every hidden state."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    lm = model.language_model
    inner = lm.model
    inputs = mx.array([ids])
    h = inner.embed_tokens(inputs)
    cache = model.make_cache()
    fa_idx = inner.fa_idx
    ssm_idx = inner.ssm_idx
    fa_mask = create_attention_mask(h, cache[fa_idx])
    ssm_mask = create_ssm_mask(h, cache[ssm_idx])

    stats = {"embedding": hidden_stats(h)}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        if i < 6 or i % 8 == 7 or i == len(inner.layers) - 1 or not layer.is_linear:
            stats[f"layer_{i}_{'lin' if layer.is_linear else 'full'}"] = hidden_stats(h)
    h = inner.norm(h)
    stats["final_norm"] = hidden_stats(h)
    logits = lm.lm_head(h)
    stats["lm_head_top8"] = {
        "shape": list(logits.shape),
        "top8": [
            [int(i), float(v)]
            for i, v in sorted(
                [(int(i), float(logits[0, -1, int(i)])) for i in mx.argsort(-logits[0, -1])[:8]],
                key=lambda x: -x[1],
            )
        ],
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--suite", default=DEFAULT_SUITE)
    ap.add_argument("--out", default="loop/artifacts/oracle.json")
    ap.add_argument("--dump-hidden", action="store_true")
    ap.add_argument(
        "--stepwise",
        action="store_true",
        help="run the prompt token-by-token (decode numerics) instead of one batched prefill",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    sys.exit(main())
