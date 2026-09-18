#!/usr/bin/env bash
# Manual acceptance for the qwen38 engine (Qwen3.8-27B FP4, MTP, Metal).
#
#   ./tools/accept.sh                 correctness gates + endpoint   (~4 min)
#   ./tools/accept.sh --perf          ... plus a throughput run      (~10 min, cool machine)
#
# Environment knobs:
#   ACCEPT_COOL=<seconds>   idle time before the throughput run (default 240, 0 to skip)
#   ACCEPT_TOKENS=<n>       tokens per throughput run (default 300)
#
# Every check prints PASS or FAIL and the script exits non-zero if any FAILed.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN=./target/release/qwen38
ORACLE=loop/artifacts/oracle.json
MODEL=models/Qwen3.8-27B-4bit
PORT=8199
PROMPT="The history of the Roman Empire is a long one that begins"
PERF=0
[ "${1:-}" = "--perf" ] && PERF=1
COOL=${ACCEPT_COOL:-240}
TOKENS=${ACCEPT_TOKENS:-300}
TMP=$(mktemp -d)
trap 'pkill -f "qwen38 serve --port $PORT" 2>/dev/null; rm -rf "$TMP"' EXIT

pass=0; fail=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
no()   { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
head1(){ printf '\n\033[1m== %s\033[0m\n' "$1"; }

ids() { # ids <outfile> <extra env...>  -- generate and keep only the token ids
  local out=$1; shift
  env "$@" $BIN gen --prompt "$PROMPT" --max-tokens "$TOKENS" --no-stop 2>/dev/null \
    | grep -o "greedy_ids: .*" > "$out"
}

# ---------------------------------------------------------------- build + checkpoint
head1 "build and checkpoint"
if [ -x "$BIN" ]; then ok "binary present ($(git log --oneline -1 2>/dev/null))"; else no "no $BIN - run: . loop/env.sh && cargo build --release -p qw-cli"; fi
if [ -d "$MODEL" ] && [ -f "$MODEL/.msc" ]; then
  ok "ModelScope checkpoint at $MODEL ($(du -sh "$MODEL" | cut -f1), .msc metadata present)"
else
  no "checkpoint or its .msc metadata missing at $MODEL"
fi
# Never pipe a producing command straight into `grep -q`: grep exits at its first match
# and closes the pipe, and Rust ignores SIGPIPE, so the producer dies with EPIPE
# ("Abort trap: 6") despite doing nothing wrong.  Capture first, then grep.
$BIN info --model-dir "$MODEL" > "$TMP/info.txt" 2>&1
if grep -q "expected-tensor check: 0 missing" "$TMP/info.txt"; then
  ok "info: layer plan readable, no missing tensors"
  grep -E "^(layers|MTP layers|linear attn)" "$TMP/info.txt" | sed 's/^/       /'
else
  no "info: could not read the layer plan (see $TMP/info.txt)"
fi

# ---------------------------------------------------------------- correctness
head1 "correctness"
$BIN verify --oracle "$ORACLE" --model-dir "$MODEL" > "$TMP/verify.txt" 2>&1
if grep -q "parity: 6/6" "$TMP/verify.txt"; then
  ok "oracle parity 6/6 (greedy ids match mlx-lm on all six cases)"
else
  no "oracle parity - expected 'parity: 6/6 cases' (see $TMP/verify.txt)"
fi

$BIN check --model-dir "$MODEL" > "$TMP/check.txt" 2>&1
if grep -q "all real-weight GEMV checks passed" "$TMP/check.txt"; then
  ok "4-bit GEMV matches the CPU reference on real weights ($(grep -c 'ok$' "$TMP/check.txt") tensors)"
else
  no "check: 4-bit GEMV did not match the CPU reference (see $TMP/check.txt)"
fi

ids "$TMP/plain.ids"
ids "$TMP/spec.ids" QW_SPEC=1
python3 - "$TMP/plain.ids" "$TMP/spec.ids" "$TOKENS" <<'PY'
import re, sys
def load(p): return [int(x) for x in re.findall(r'-?\d+', open(p).read())]
a, b, want = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
if len(a) != want or len(b) != want:
    print(f"  \033[31mFAIL\033[0m spec==plain: bad length plain={len(a)} spec={len(b)}, expected {want} (an empty run compares equal, so the count is asserted)")
    sys.exit(1)
diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
if diff:
    print(f"  \033[31mFAIL\033[0m spec==plain: {len(diff)} of {want} tokens differ, first at {diff[0]}")
    sys.exit(1)
print(f"  \033[32mPASS\033[0m spec==plain: byte-identical over {want} tokens (MTP does not change what is emitted)")
PY
if [ $? -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi

# ---------------------------------------------------------------- throughput
head1 "throughput"
LINE=$($BIN bench --iters 3 --tokens 3 2>/dev/null | grep -o '{"tok_per_s.*}')
if [ -n "$LINE" ]; then
  echo "  bench (linear weight sweep, k=3 rows, assumes all 3 accepted): $LINE"
  ok "kernel sweep measured (end_to_end=false - this is NOT the otps to quote)"
else
  no "bench produced no result line"
fi

if [ "$PERF" = "1" ]; then
  if [ "$COOL" != "0" ]; then
    echo "  idling ${COOL}s so the measurement is not taken on a hot machine..."
    sleep "$COOL"
  fi
  best=0
  for i in 1 2 3; do
    OUT=$(QW_SPEC=1 $BIN gen --prompt "$PROMPT" --max-tokens "$TOKENS" --no-stop 2>&1)
    TPS=$(echo "$OUT" | sed -nE 's/.*decode ([0-9.]+) tok\/s.*/\1/p' | tail -1)
    ACC=$(echo "$OUT" | sed -nE 's/.*accepted \(([0-9.]+)%\), ([0-9.]+) tokens\/pass.*/\1% accepted, \2 tok\/pass/p' | tail -1)
    echo "  run $i: ${TPS:-?} tok/s${ACC:+  ($ACC)}"
    awk -v a="$TPS" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$TPS
  done
  if awk -v b="$best" 'BEGIN{exit !(b>=40)}'; then
    ok "end-to-end single decoder: best ${best} tok/s (>= 40 required)"
  else
    no "end-to-end single decoder: best ${best} tok/s, below 40"
  fi
  echo "     note: the first run after idling is the one to quote; later runs drop as the M5 heats up."
else
  echo "  (skipped - rerun with --perf for the cooled end-to-end otps measurement)"
fi

# ---------------------------------------------------------------- endpoint
head1 "endpoint (OpenAI + Anthropic)"
pkill -f "qwen38 serve --port $PORT" 2>/dev/null
$BIN serve --port $PORT --model-dir "$MODEL" > "$TMP/serve.log" 2>&1 &
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done

BODY='{"model":"qwen3.8-27b-fp4","max_tokens":24,"messages":[{"role":"user","content":"Say hello in five words."}]}'
OA_BODY='{"model":"qwen3.8-27b-fp4","max_tokens":24,"messages":[{"role":"user","content":"Say hello in five words."}]}'

if curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c '
import json,sys
d=json.load(sys.stdin)
ids=[m["id"] for m in d.get("data",[])]
print(("  \033[32mPASS\033[0m " if ids else "  \033[31mFAIL\033[0m ")+f"/v1/models -> {ids}")
sys.exit(0 if ids else 1)'; then pass=$((pass+1)); else fail=$((fail+1)); fi

if curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "$OA_BODY" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print(f"  \033[31mFAIL\033[0m openai non-stream: unparseable ({e})"); sys.exit(1)
c=d.get("choices",[{}])[0]
obj=d.get("object"); fin=c.get("finish_reason"); usg=d.get("usage"); txt=c.get("message",{}).get("content","")[:70]
good = obj=="chat.completion" and fin and (usg or {}).get("completion_tokens") is not None
tag = "PASS" if good else "FAIL"; col = "\033[32m" if good else "\033[31m"
print(f"  {col}{tag}\033[0m openai non-stream -> object={obj} finish={fin} usage={usg}")
print(f"       text: {txt!r}")
sys.exit(0 if good else 1)'; then pass=$((pass+1)); else fail=$((fail+1)); fi

N=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
      -d '{"model":"qwen3.8-27b-fp4","max_tokens":24,"messages":[{"role":"user","content":"Say hello in five words."}],"stream":true}' | grep -c "^data:")
if [ "$N" -gt 1 ]; then ok "openai stream -> $N data: chunks"; else no "openai stream -> only $N data: chunks"; fi

if curl -s "http://127.0.0.1:$PORT/v1/messages" -H 'content-type: application/json' -d "$BODY" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print(f"  \033[31mFAIL\033[0m anthropic non-stream: unparseable ({e})"); sys.exit(1)
typ=d.get("type"); stop=d.get("stop_reason"); usg=d.get("usage"); txt=d.get("content",[{}])[0].get("text","")[:70]
good = typ=="message" and stop and (usg or {}).get("output_tokens") is not None
tag = "PASS" if good else "FAIL"; col = "\033[32m" if good else "\033[31m"
print(f"  {col}{tag}\033[0m anthropic non-stream -> type={typ} stop_reason={stop} usage={usg}")
print(f"       text: {txt!r}")
sys.exit(0 if good else 1)'; then pass=$((pass+1)); else fail=$((fail+1)); fi

EV=$(curl -s "http://127.0.0.1:$PORT/v1/messages" -H 'content-type: application/json' \
      -d '{"model":"qwen3.8-27b-fp4","max_tokens":24,"stream":true,"messages":[{"role":"user","content":"Say hello in five words."}]}' \
      | grep "^event:" | sed 's/event: //' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
for want in message_start content_block_start content_block_stop message_stop; do
  case "$EV" in *"$want"*) ;; *) no "anthropic stream missing $want"; EV=""; break;; esac
done
[ -n "$EV" ] && ok "anthropic stream -> $EV"

pkill -f "qwen38 serve --port $PORT" 2>/dev/null

# ------------------------------------------------- a hit must not change the answer
head1 "prefix cache determinism"
# The gate this whole class of bug needed and did not have.  Measured once, seeding
# the generation position from the amount a request skipped made a cache hit return
# an answer that did not even depend on the prompt, and the six-case oracle was
# blind to it.  A hit is allowed to be faster; it is not allowed to be different.
CACHE_PROMPT="The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. Reply with one word."
ask_once() {
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"qwen3.8-27b-fp4\",\"temperature\":0,\"max_tokens\":12,\"messages\":[{\"role\":\"user\",\"content\":\"$CACHE_PROMPT\"}]}" \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception: print("")'
}
serve_up() {
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
pkill -f "qwen38 serve --port $PORT" 2>/dev/null; sleep 2
$BIN serve --port $PORT --model-dir "$MODEL" > "$TMP/serve_cache.log" 2>&1 &
serve_up
A=$(ask_once); B=$(ask_once); C=$(ask_once)
pkill -f "qwen38 serve --port $PORT" 2>/dev/null; sleep 2
QW_PREFIX_SNAPSHOT=0 $BIN serve --port $PORT --model-dir "$MODEL" > "$TMP/serve_nocache.log" 2>&1 &
serve_up
D=$(ask_once)
pkill -f "qwen38 serve --port $PORT" 2>/dev/null
if [ -n "$A" ] && [ "$A" = "$B" ] && [ "$B" = "$C" ]; then
  ok "cache on: three identical requests agree"
else
  no "cache on: repeats disagree ($A | $B | $C)"
fi
if [ -n "$A" ] && [ "$A" = "$D" ]; then
  ok "cache hit matches the cold start"
else
  no "cache hit differs from the cold start ($A vs $D)"
fi

# ------------------------------------------------- prefill width must not change the answer
head1 "prefill width determinism"
# The gate the GEMM class of bug needed and did not have.  The six-case oracle is
# blind to it: every oracle prompt is five to twenty tokens, so its prefill pass is
# far below the width at which a pass switches kernels, and `batch-check`'s
# independent runs go through the single-row kernel.  Wiring the simdgroup GEMM in
# therefore passed verify 6/6, batch-check and the whole suite while producing a
# different answer for a long prompt.  A four-token chunk and a thirty-two-token
# chunk run different kernels on the same tokens; they must still agree.
WIDE_PROMPT="A team of engineers is designing a water tank for a small village. The tank must hold at least twenty thousand litres, sit on a concrete pad, and survive freezing winters. Describe the main design decisions they should make and why each one matters."
ask_wide() {
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json'     -d "{\"model\":\"qwen3.8-27b-fp4\",\"temperature\":0,\"max_tokens\":40,\"messages\":[{\"role\":\"user\",\"content\":\"$WIDE_PROMPT\"}]}"     | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception: print("")'
}
W4=""
W32=""
for W in 4 32; do
  pkill -f "qwen38 serve --port $PORT" 2>/dev/null; sleep 2
  QW_PREFILL_CHUNK=$W QW_PREFIX_SNAPSHOT=0 $BIN serve --port $PORT --model-dir "$MODEL" > "$TMP/serve_w$W.log" 2>&1 &
  serve_up
  R=$(ask_wide)
  pkill -f "qwen38 serve --port $PORT" 2>/dev/null; sleep 2
  if [ "$W" = 4 ]; then W4="$R"; else W32="$R"; fi
done
if [ -n "$W4" ] && [ "$W4" = "$W32" ]; then
  ok "chunk 4 and chunk 32 give the same answer"
else
  no "chunk 4 and chunk 32 disagree (${W4:0:40} vs ${W32:0:40})"
fi

# ---------------------------------------------------------------- verdict
head1 "verdict"
printf '  %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && printf '  \033[32mACCEPTED\033[0m\n' || printf '  \033[31mNOT ACCEPTED\033[0m\n'
exit $((fail > 0))
