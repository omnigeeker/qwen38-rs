#!/usr/bin/env bash
# Concurrency gate: the engine has to hold 16 requests in flight at once, and each
# response has to be the one that request gets on its own.
#
# A shared prompt would be a weak test - every slot could be reading the same state
# and still agree - so each request asks for a different thing.
set -uo pipefail
PORT=${PORT:-8099}
MODEL=${MODEL:-models/Qwen3.8-27B-4bit}
BIN=${BIN:-./target/release/qwen38}
N=${N:-16}
TOK=${TOK:-24}
TMP=$(mktemp -d)
trap 'pkill -f "qwen38 serve --port $PORT" 2>/dev/null; rm -rf "$TMP"' EXIT

pkill -f "qwen38 serve --port $PORT" 2>/dev/null
sleep 1
$BIN serve --port $PORT --model-dir "$MODEL" > "$TMP/serve.log" 2>&1 &
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "server never came up"; tail -5 "$TMP/serve.log"; exit 1; }

ask() {
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"qwen3.8-27b-fp4\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: the name of colour number $1.\"}],\"max_tokens\":$TOK}" > "$2"
}

echo "== $N requests at once"
T0=$(python3 -c 'import time;print(time.time())')
# Wait on the curls by pid.  A bare `wait` would also wait for the server started
# above, which never exits, and the script would hang forever with the work done.
pids=()
for i in $(seq 1 "$N"); do ask "$i" "$TMP/conc_$i.json" & pids+=($!); done
wait "${pids[@]}"
T1=$(python3 -c 'import time;print(time.time())')

echo "== the same $N requests one at a time"
T2=$(python3 -c 'import time;print(time.time())')
for i in $(seq 1 "$N"); do ask "$i" "$TMP/solo_$i.json"; done
T3=$(python3 -c 'import time;print(time.time())')

python3 - "$TMP" "$N" "$T0" "$T1" "$T2" "$T3" <<'PY'
import json, sys
tmp, n = sys.argv[1], int(sys.argv[2])
t0, t1, t2, t3 = (float(x) for x in sys.argv[3:7])
def text(p):
    d = json.load(open(p))
    return d["choices"][0]["message"]["content"]
bad = 0
for i in range(1, n + 1):
    c, s = text(f"{tmp}/conc_{i}.json"), text(f"{tmp}/solo_{i}.json")
    if c != s:
        bad += 1
        print(f"  request {i}: concurrent != solo")
        print(f"    concurrent: {c[:70]!r}")
        print(f"    solo      : {s[:70]!r}")
print(f"concurrent wall {t1-t0:6.2f}s   solo wall {t3-t2:6.2f}s   ratio {(t3-t2)/(t1-t0):.2f}x")
print(f"identical responses: {n-bad}/{n}")
sys.exit(1 if bad else 0)
PY
