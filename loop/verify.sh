#!/usr/bin/env bash
# Gate G1 + G2: static checks, unit tests, and (when an oracle exists) the
# end-to-end greedy parity check against mlx-lm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. loop/env.sh

echo "=== G1a: cargo fmt ==="
cargo fmt --all --check

echo "=== G1b: clippy ==="
cargo clippy --workspace --all-targets -- -D warnings

echo "=== G1c: unit tests ==="
cargo test --workspace -- --nocapture

echo "=== G2: end-to-end parity (skipped until oracle exists) ==="
if [[ -f loop/artifacts/oracle.json ]]; then
  cargo run --release -p qw-cli --bin qwen38 -- \
    verify --oracle loop/artifacts/oracle.json || {
      echo "parity check FAILED"; exit 1; }
else
  echo "no loop/artifacts/oracle.json yet -> generate it with:"
  echo "  python3 tools/oracle.py --out loop/artifacts/oracle.json"
fi

echo "=== gates G1/G2 OK ==="
