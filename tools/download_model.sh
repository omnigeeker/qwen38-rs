#!/usr/bin/env bash
# Download Qwen3.8-27B FP4 (MLX affine 4-bit) via the ModelScope CLI.
#
#   ./tools/download_model.sh            # weights only (~16 GB)
#   ./tools/download_model.sh --mtp      # also the bf16 MTP tensors
#
# ModelScope needs a writable cache; the repo keeps it inside the workspace so
# nothing is written to $HOME.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_REPO="${MODEL_REPO:-mlx-community/Qwen3.8-27B-4bit}"
MTP_REPO="${MTP_REPO:-Qwen/Qwen3.8-27B}"
DEST="${DEST:-$ROOT/models/Qwen3.8-27B-4bit}"
MTP_DEST="${MTP_DEST:-$ROOT/models/Qwen3.8-27B-bf16-mtp}"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/.cache/modelscope}"
mkdir -p "$DEST" "$MODELSCOPE_CACHE"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "modelscope CLI not found; pip install modelscope" >&2
  exit 1
fi

echo "==> downloading $MODEL_REPO -> $DEST"
modelscope download --model "$MODEL_REPO" --local_dir "$DEST"

if [[ "${1:-}" == "--mtp" ]]; then
  echo "==> downloading MTP shards from $MTP_REPO -> $MTP_DEST"
  # The MLX 4-bit export drops `mtp.*`; the official bf16 release keeps them.
  # Only the shard(s) that actually contain mtp.* are needed.
  mkdir -p "$MTP_DEST"
  modelscope download --model "$MTP_REPO" \
    --include 'model.safetensors.index.json' \
    --include 'config.json' \
    --local_dir "$MTP_DEST"
  python3 - "$MTP_DEST" "$MTP_REPO" <<'PY'
import json, subprocess, sys, os
dest, repo = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(dest, "model.safetensors.index.json")))["weight_map"]
files = sorted({v for k, v in idx.items() if k.startswith("mtp.")})
print("MTP tensors live in:", files)
for f in files:
    subprocess.check_call(["modelscope", "download", "--model", repo, "--include", f, "--local_dir", dest])
PY
fi

echo "==> done"
du -sh "$DEST" 2>/dev/null || true
