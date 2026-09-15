"""Quantise the MTP head from the official bf16 release into the MLX affine
4-bit layout the engine already reads for the decoder.

The MLX 4-bit export of Qwen3.8-27B has no `mtp.*` tensors, so they are pulled
from `Qwen3.8-27B-bf16-mtp` and quantised here with the same parameters as the
rest of the model (group_size 64, 4 bits).  Norms are copied verbatim, apart from
`mtp.layers.0.{input_layernorm,post_attention_layernorm}` and the attention
`{q,k}_norm`, which get the same +1 shift the 4-bit export applies to the decoder
layers (see the round-2 notes).

Usage: python3 tools/mtp_quantize.py
"""

import os

import mlx.core as mx

SRC = "models/Qwen3.8-27B-bf16-mtp/model-00018-of-00018.safetensors"
DST = "models/Qwen3.8-27B-mtp-4bit/mtp.safetensors"

MATS = [
    "mtp.fc.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]
NORMS = [
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
]


def shift(name: str) -> float:
    """+1 for the norms the 4-bit export stores pre-shifted.

    The bf16 release stores every RMSNorm weight as `w - 1`; mlx-lm applies the
    shift when it converts a checkpoint that carries `mtp.*` tensors, which is
    exactly this release.  It covers the MTP module's own norms too - their raw
    values are visibly negative, which no RMSNorm weight is.  `mtp.norm` is the
    exception: its raw values already sit around 1.
    """
    if name == "mtp.norm.weight":
        return 0.0
    if "norm" in name:
        return 1.0
    return 0.0


def main() -> None:
    src = mx.load(SRC)
    out = {}
    for name in MATS:
        # Quantise the bf16 tensor directly: MLX gives the scales/biases the
        # dtype of the input, and the engine reads them as bf16 (that is what
        # the 4-bit decoder export stores).
        w = src[name]
        q, s, b = mx.quantize(w, group_size=64, bits=4)
        base = name[: -len(".weight")]
        out[base + ".weight"] = q
        out[base + ".scales"] = s
        out[base + ".biases"] = b
        dq = mx.dequantize(q, s, b, group_size=64, bits=4).astype(mx.float32)
        wf = w.astype(mx.float32)
        err = mx.max(mx.abs(dq - wf)).item()
        rel = err / mx.max(mx.abs(wf)).item()
        print(
            f"{base:52} {str(w.shape):12} q={q.dtype} s={s.dtype} "
            f"max|dq-w|={err:.5f} rel={rel:.5f}"
        )
    for name in NORMS:
        w = src[name].astype(mx.float32)
        sh = shift(name)
        out[name] = (w + sh).astype(mx.bfloat16)
        print(f"{name:52} {str(w.shape):12} bf16 shift=+{sh}")
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    mx.save_safetensors(DST, out)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
