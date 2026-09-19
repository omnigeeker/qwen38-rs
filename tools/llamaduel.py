"""Cold TTFT and output tok/s for llama.cpp's llama-server, measured the same way
as tools/duel.py measures ours and Ollama's, so the three are comparable.

The objective names both llama.cpp and Ollama, but only Ollama had ever been
measured; llama.cpp is installed at /opt/homebrew/bin/llama-server and the GGUF
is already on disk as Ollama's own blob, so no download is needed.

TTFT stops at the first NON-EMPTY content chunk (the server emits a role-only
chunk first, and stopping there gives a bogus ~0.06 s).  Token counts come from
the non-streaming response's usage.completion_tokens, because counting SSE chunks
measures chunks per second, not tokens per second - the mistake that made our own
otps figure wrong for many rounds.  A fresh random prompt every round, since both
llama.cpp and Ollama cache prefixes and a reused prompt gives a fake TTFT.
"""
import json, os, random, statistics, subprocess, time, urllib.request

PORT = 8220
BLOB = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-832df740370353951c0a569064a8034461fc7fe54fa1a000311ba609e83542a2")
BIN = "/opt/homebrew/bin/llama-server"
ROUNDS = int(os.environ.get("LL_ROUNDS", "5"))
COOL = int(os.environ.get("LL_COOL", "60"))
GEN = int(os.environ.get("LL_GEN", "128"))
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima"]


def prompt(seed):
    random.seed(seed)
    return " ".join(random.choice(WORDS) for _ in range(430)) + \
        " The history of the Roman Empire is a long one that begins"


def wait(path="/health"):
    for _ in range(1800):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def measure(p):
    sb = {"prompt": p, "max_tokens": GEN, "temperature": 0, "stream": True}
    t0 = time.time()
    first = last = None
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(sb).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        pl = line[5:].strip()
        if pl == "[DONE]":
            break
        try:
            j = json.loads(pl)
        except Exception:
            continue
        ch = (j.get("choices") or [{}])[0]
        txt = (ch.get("delta") or {}).get("content") or ch.get("text") or ""
        if txt:
            now = time.time() - t0
            if first is None:
                first = now
            last = now
    nb = {"prompt": p, "max_tokens": GEN, "temperature": 0}
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(nb).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    u = json.load(r).get("usage") or {}
    c = u.get("completion_tokens")
    otps = (c - 1) / (last - first) if c and first and last and last > first and c > 1 else None
    return {"ttft": first, "otps": otps, "pt": u.get("prompt_tokens"), "ct": c}


if __name__ == "__main__":
    if not os.path.exists(BLOB):
        raise SystemExit(f"GGUF blob missing: {BLOB}")
    print(f"starting {BIN} on {BLOB}", flush=True)
    srv = subprocess.Popen([BIN, "-m", BLOB, "--port", str(PORT), "-c", "8192",
                            "-ngl", "99", "--temp", "0", "--no-warmup"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t_load = time.time()
        if not wait():
            raise SystemExit("llama-server never became healthy")
        print(f"loaded in {time.time()-t_load:.1f} s; cooling {COOL}s...", flush=True)
        time.sleep(COOL)
        tt, ot = [], []
        for r in range(1, ROUNDS + 1):
            p = prompt(random.randrange(10**9))
            res = measure(p)
            print(f"  r{r} llama.cpp: TTFT {res['ttft']:.3f} s  otps "
                  f"{'None' if res['otps'] is None else round(res['otps'],2)}  "
                  f"pt={res['pt']} ct={res['ct']}", flush=True)
            if res["ttft"]:
                tt.append(res["ttft"])
            if res["otps"]:
                ot.append(res["otps"])
        if tt:
            print(f"\n== llama.cpp medians ==")
            print(f"  cold TTFT {statistics.median(tt):.3f} s  {[round(x,2) for x in tt]}")
            print(f"  otps      {statistics.median(ot):.2f} tok/s  {[round(x,2) for x in ot]}")
    finally:
        srv.terminate()
        srv.wait()
