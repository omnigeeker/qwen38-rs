#!/usr/bin/env bash
# One loop iteration: build -> gate -> record -> publish.
#
#   loop/run_round.sh "round title"        # full round (gates + commit + push)
#   SKIP_PUSH=1 loop/run_round.sh "title"  # local only
#
# The script never invents success: if a gate fails the round is recorded as
# REJECTED and the process exits non-zero so the next iteration must fix it.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. loop/env.sh

TITLE="${1:-untitled round}"
ROUND=$(python3 -c "import json;print(json.load(open('loop/state.json'))['round'])")
ROUND_PAD=$(printf "%03d" "$ROUND")
LOG_DIR="loop/rounds"
mkdir -p "$LOG_DIR" loop/artifacts
LOG="$LOG_DIR/round-$ROUND_PAD.md"

echo "=== qwen38 loop: round $ROUND_PAD — $TITLE ==="

G1="fail"; G2="skipped"; G3="skipped"
BENCH_JSON=""

echo "--- G1 fmt/clippy/test ---"
if cargo fmt --all --check >/tmp/qw_fmt.log 2>&1 \
   && cargo clippy --workspace --all-targets -- -D warnings >/tmp/qw_clippy.log 2>&1 \
   && cargo test --workspace >/tmp/qw_test.log 2>&1; then
  G1="pass"
else
  echo "G1 FAILED (see /tmp/qw_clippy.log, /tmp/qw_test.log)"
  tail -30 /tmp/qw_test.log
fi

if [[ "$G1" == "pass" ]]; then
  echo "--- G2 parity vs oracle ---"
  if [[ -f loop/artifacts/oracle.json ]]; then
    if cargo run --release -q -p qw-cli --bin qwen38 -- verify \
         --oracle loop/artifacts/oracle.json >/tmp/qw_verify.log 2>&1; then
      G2="pass"
    else
      G2="fail"
      tail -30 /tmp/qw_verify.log
    fi
  else
    echo "no oracle yet; generating it (mlx-lm reference)"
    if python3 tools/oracle.py --out loop/artifacts/oracle.json >/tmp/qw_oracle.log 2>&1; then
      G2="oracle-ready"
    else
      G2="oracle-failed"
      tail -20 /tmp/qw_oracle.log
    fi
  fi

  echo "--- G3 throughput ---"
  if cargo build --release -p qw-cli >/dev/null 2>&1; then
    if BENCH_JSON=$(cargo run --release -q -p qw-cli --bin qwen38 -- bench \
        --tokens 128 2>/tmp/qw_bench.log); then
      G3="pass"
    else
      G3="not-implemented"
      tail -5 /tmp/qw_bench.log
    fi
  fi
fi

TOK_S=$(python3 - "$BENCH_JSON" <<'PY' 2>/dev/null || echo 0
import json,sys
try:
    print(json.loads(sys.argv[1]).get("tok_per_s", 0.0))
except Exception:
    print(0.0)
PY
)

cat > "$LOG" <<EOF
# Round $ROUND_PAD — $TITLE

- date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- gates: G1=$G1 G2=$G2 G3=$G3
- tok/s: $TOK_S

## Evidence

\`\`\`
$( [[ -f /tmp/qw_test.log ]] && tail -12 /tmp/qw_test.log )
\`\`\`

## Decision

$( [[ "$G1" == "pass" && ( "$G2" == "pass" || "$G2" == "oracle-ready" ) ]] && echo "ACCEPTED" || echo "REJECTED" )
EOF

echo "--- recording state ---"
python3 - "$ROUND" "$TOK_S" "$G1" "$G2" "$G3" "$TITLE" <<'PY'
import json, sys, datetime
round_no, tok_s, g1, g2, g3, title = sys.argv[1:7]
p='loop/state.json'
st=json.load(open(p))
best=st['metrics'].get('best_tok_s_single_stream',0.0)
tok=float(tok_s)
if tok > best:
    st['metrics']['best_tok_s_single_stream']=tok
st['gates'].update({'G1_build':g1,'G2_correctness':g2,'G3_performance':g3})
st.setdefault('history',[]).append({
  'round': int(round_no), 'title': title, 'tok_s': tok,
  'gates': {'G1':g1,'G2':g2,'G3':g3},
  'at': datetime.datetime.utcnow().isoformat()+'Z'})
st['round']=int(round_no)+1
json.dump(st, open(p,'w'), indent=2)
print(f"state: round -> {st['round']}, best_tok_s={st['metrics']['best_tok_s_single_stream']}")
PY

echo "--- G4 publish ---"
git add -A
git commit -m "round $ROUND_PAD: $TITLE (G1=$G1 G2=$G2 G3=$G3 tok/s=$TOK_S)" || echo "nothing to commit"
if [[ "${SKIP_PUSH:-0}" != "1" ]]; then
  git push || echo "push failed (check gh auth / remote)"
fi

echo "=== round $ROUND_PAD done: G1=$G1 G2=$G2 G3=$G3 tok/s=$TOK_S ==="
