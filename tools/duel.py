"""Head-to-head cold-prefill TTFT and output tokens/s: ours vs Ollama.

The old version counted SSE chunks and called them tokens, which made our otps
meaningless.  The streaming endpoint here is used ONLY for TTFT (the first
non-empty content chunk - the server emits a role-only chunk ~60 ms in, and
stopping the clock there gives a bogus 0.06 s).  The token count comes from a
second, non-streaming request with the same temperature-0 parameters, whose
`usage.completion_tokens` is authoritative; `stream_options.include_usage` does
not work on this server, but the non-streaming path does return usage.

Ollama reports its own eval_count/eval_duration, which is genuinely tokens/s, and
we cross-check it with the same first-to-last-content method.
"""
import json, os, random, statistics, subprocess, sys, time, urllib.request

PORT = 8214
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima"]
ROUNDS = int(os.environ.get("DUEL_ROUNDS", "4"))
COOL = int(os.environ.get("DUEL_COOL", "300"))
GEN = int(os.environ.get("DUEL_GEN", "128"))
SPEC = os.environ.get("DUEL_SPEC", "0") == "1"
OURS = os.environ.get("DUEL_OURS", "./target/release/qwen38")
OLLAMA = os.environ.get("DUEL_OLLAMA", "qwen38-27b:bench")


def prompt(seed):
    random.seed(seed)
    return " ".join(random.choice(WORDS) for _ in range(430)) + \
        " The history of the Roman Empire is a long one that begins"


def wait(p, path="/health"):
    for _ in range(900):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{p}{path}", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def stream_ttft(url, body, kind):
    """Return (ttft, t_last_content, n_content_chunks).  kind: 'oai' | 'ollama'."""
    t0 = time.time()
    first = last = None
    n = 0
    r = urllib.request.urlopen(urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if kind == "oai":
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
        else:
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            txt = j.get("response") or ""
        if txt:
            now = time.time() - t0
            if first is None:
                first = now
            last = now
            n += 1
    return first, last, n


def ours(p):
    e = dict(os.environ)
    for k in ("QW_PREFIX_DISK", "QW_PREFIX_SNAPSHOT"):
        e.pop(k, None)
    s = subprocess.Popen([OURS, "serve", "--model-dir", "models/Qwen3.8-27B-4bit",
                          "--port", str(PORT)], env=e,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait(PORT):
            return None
        sb = {"model": "m", "prompt": p, "max_tokens": GEN,
              "temperature": 0, "stream": True}
        ttft, last, _ = stream_ttft(f"http://127.0.0.1:{PORT}/v1/completions", sb, "oai")
        nb = {"model": "m", "prompt": p, "max_tokens": GEN, "temperature": 0}
        r = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(nb).encode(),
            headers={"Content-Type": "application/json"}), timeout=1800)
        u = json.load(r).get("usage") or {}
        c = u.get("completion_tokens")
        pt = u.get("prompt_tokens")
        otps = (c - 1) / (last - ttft) if c and last and ttft and last > ttft and c > 1 else None
        return {"ttft": ttft, "otps": otps, "chunks": None,
                "completion_tokens": c, "prompt_tokens": pt}
    finally:
        s.terminate()
        s.wait()
        for _ in range(120):
            if subprocess.run(["bash", "-c", f"lsof -nP -iTCP:{PORT} -sTCP:LISTEN"],
                              capture_output=True).returncode != 0:
                break
            time.sleep(0.5)


def ollama(p):
    body = {"model": OLLAMA, "prompt": p, "raw": True, "stream": True,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": GEN}}
    t0 = time.time()
    first = last = None
    n = 0
    ev = None
    r = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:11434/api/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    for raw in r:
        try:
            j = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        txt = j.get("response") or ""
        if txt:
            now = time.time() - t0
            if first is None:
                first = now
            last = now
            n += 1
        if j.get("done"):
            ev = j
    rate = None
    if ev and ev.get("eval_count") and ev.get("eval_duration"):
        rate = ev["eval_count"] / (ev["eval_duration"] / 1e9)
    return {"ttft": first, "otps": rate, "chunks": n,
            "completion_tokens": ev.get("eval_count") if ev else None,
            "prompt_tokens": ev.get("prompt_eval_count") if ev else None,
            "cross_check": (n - 1) / (last - first) if first and last and last > first and n > 1 else None}


def fmt(tag, res):
    if not res:
        return f"  {tag:7s}: FAILED"
    g = lambda k, p=3: ("None" if res.get(k) is None else round(res[k], p))
    extra = ""
    if res.get("cross_check") is not None:
        extra = f"  (chunk-rate {res['cross_check']:.2f}, ignore)"
    return (f"  {tag:7s}: TTFT {g('ttft')} s  otps {g('otps',2)}  "
            f"pt={res.get('prompt_tokens')} ct={res.get('completion_tokens')}{extra}")


if __name__ == "__main__":
    print(f"ours={OURS} spec={SPEC} gen={GEN} rounds={ROUNDS} cooling {COOL}s...",
          flush=True)
    time.sleep(COOL)
    acc = {"ours": [], "ollama": []}
    for r in range(1, ROUNDS + 1):
        p = prompt(random.randrange(10**9))
        seq = [("ours", ours), ("ollama", ollama)]
        if r % 2 == 0:
            seq = seq[::-1]
        for tag, fn in seq:
            res = fn(p)
            print(f"  r{r} " + fmt(tag, res).strip(), flush=True)
            if res and res.get("otps"):
                acc[tag].append(res["otps"])
    o, l = acc["ours"], acc["ollama"]
    if o and l:
        print(f"\n== otps medians over {min(len(o), len(l))} rounds ==")
        print(f"  ours   {statistics.median(o):.2f} tok/s  {[round(x,2) for x in o]}")
        print(f"  ollama {statistics.median(l):.2f} tok/s  {[round(x,2) for x in l]}")
        print(f"  ratio  {statistics.median(o)/statistics.median(l):.3f}x")
