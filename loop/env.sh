# Source this file (`. loop/env.sh`) before running cargo.
# The Rust toolchain is installed *inside* the workspace so that no writes
# outside the project directory are required.
export RUSTUP_HOME="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/.toolchain/rustup"
export CARGO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/.toolchain/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
export MODELSCOPE_CACHE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/.cache/modelscope"
export MODEL_DIR="${MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/models/Qwen3.8-27B-4bit}"
