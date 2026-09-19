"""Three-way interleaved cold-TTFT and output tok/s: ours vs Ollama vs llama.cpp.

All three servers stay loaded for the whole run and each round sends the SAME
fresh random prompt to each of them in a rotating order, so drift and thermal
load are shared rather than attributed to whichever engine happened to run
first.  A fresh prompt every round matters because all three cache prefixes and a
reused prompt produces a fake TTFT.

Measurement rules, learned the hard way:
  * TTFT stops at the first NON-EMPTY content chunk.  Our server emits a
    role-only chunk about 60 ms in; stopping there gave a bogus 0.06 s.
  * Token counts come from the non-streaming response's usage.completion_tokens.
    Counting SSE chunks measures chunks per second, not tokens per second - the
    error that made our own otps figure wrong for many rounds.
"""
import json, os, random, statistics, subprocess, time, urllib.request

OURS_PORT, LLAMA_PORT, OLLAMA_PORT = 8199, 8220, 11434
BLOB = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-832df740370353951c0a569064a8034461fc7fe54fa1a000311ba609e83542a2")
ROUNDS = int(os.environ.get("TRI_ROUNDS", "6"))
COOL = int(os.environ.get("TRI_COOL", "90"))
GEN = int(os.environ.get("TRI_GEN", "128"))
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima"]


def prompt(seed):
    random.seed(seed)
    return " ".join(random.choice(WORDS) for _ in range(430)) + \
        " The history of the Roman Empire is a long one that begins"


def wait(port, path="/health"):
    for _ in range(1800):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def oai_measure(port, p):
    sb = {"model": "m", "prompt": p, "max_tokens": GEN,
          "temperature": 0, "stream": True}
    t0 = time.time()
    first = last = None
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(sb).encode(),
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
    nb = {"model": "m", "prompt": p, "max_tokens": GEN, "temperature": 0}
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(nb).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    u = json.load(r).get("usage") or {}
    c = u.get("completion_tokens")
    return {"ttft": first, "pt": u.get("prompt_tokens"), "ct": c,
            "otps": (c - 1) / (last - first) if c and first and last and last > first and c > 1 else None}


def ollama_measure(p):
    body = {"model": "qwen38-27b:bench", "prompt": p, "raw": True, "stream": True,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": GEN}}
    t0 = time.time()
    first = last = None
    n = 0
    ev = None
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{OLLAMA_PORT}/api/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    for raw in r:
        try:
            j = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if j.get("response"):
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
    return {"ttft": first, "otps": rate, "ct": ev.get("eval_count") if ev else None,
            "pt": ev.get("prompt_eval_count") if ev else None}


if __name__ == "__main__":
    procs = []
    env = dict(os.environ)
    for k in ("QW_PREFIX_DISK", "QW_PREFIX_SNAPSHOT"):
        env.pop(k, None)
    print("starting all three servers...", flush=True)
    procs.append(subprocess.Popen(
        ["./target/release/qwen38", "serve", "--model-dir", "models/Qwen3.8-27B-4bit",
         "--port", str(OURS_PORT)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    procs.append(subprocess.Popen(
        ["/opt/homebrew/bin/llama-server", "-m", BLOB, "--port", str(LLAMA_PORT),
         "-c", "8192", "-ngl", "99", "--temp", "0", "--no-warmup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    try:
        for port, name in ((OURS_PORT, "ours"), (LLAMA_PORT, "llama.cpp")):
            print(f"  waiting for {name}...", flush=True)
            if not wait(port):
                raise SystemExit(f"{name} never became healthy")
        print(f"all up; cooling {COOL}s...", flush=True)
        time.sleep(COOL)
        acc = {"ours": {"ttft": [], "otps": []},
               "ollama": {"ttft": [], "otps": []},
               "llama.cpp": {"ttft": [], "otps": []}}
        for r in range(1, ROUNDS + 1):
            p = prompt(random.randrange(10**9))
            order = [("ours", lambda: oai_measure(OURS_PORT, p)),
                     ("ollama", lambda: ollama_measure(p)),
                     ("llama.cpp", lambda: oai_measure(LLAMA_PORT, p))]
            order = order[r % 3:] + order[:r % 3]
            for tag, fn in order:
                try:
                    res = fn()
                except Exception as ex:
                    print(f"  r{r} {tag:9s}: ERROR {ex}", flush=True)
                    continue
                g = lambda k, d=3: ("None" if res.get(k) is None else round(res[k], d))
                print(f"  r{r} {tag:9s}: TTFT {g('ttft')} s  otps {g('otps',2)}  "
                      f"pt={res.get('pt')} ct={res.get('ct')}", flush=True)
                if res.get("ttft"):
                    acc[tag]["ttft"].append(res["ttft"])
                if res.get("otps"):
                    acc[tag]["otps"].append(res["otps"])
                # Warm TTFT: the identical prompt again.  All three engines cache
                # prefixes, so this is the fair "second request" comparison, and
                # for a served model with a system prompt it is the common case.
                try:
                    w = fn()
                except Exception as ex:
                    print(f"  r{r} {tag:9s}: warm ERROR {ex}", flush=True)
                    continue
                ww = w.get("ttft")
                print(f"  r{r} {tag:9s}: WARM TTFT "
                      f"{'None' if ww is None else round(ww, 4)} s", flush=True)
                if ww:
                    acc[tag].setdefault("warm", []).append(ww)
        print("\n== medians ==")
        for tag in ("ours", "llama.cpp", "ollama"):
            a = acc[tag]
            wm = statistics.median(a["warm"]) if a.get("warm") else None
            if a["ttft"]:
                print(f"  {tag:9s} cold TTFT {statistics.median(a['ttft']):6.3f} s"
                      f"   warm TTFT {('None' if wm is None else format(wm, '.4f')):>7s} s"
                      f"   otps {statistics.median(a['otps']):6.2f} tok/s"
                      f"   (n={len(a['otps'])})")
        print(f"\n  raw ttft ours={[round(x,2) for x in acc['ours']['ttft']]}")
        print(f"  raw otps ours={[round(x,2) for x in acc['ours']['otps']]}")
        print(f"  raw otps llama.cpp={[round(x,2) for x in acc['llama.cpp']['otps']]}")
        print(f"  raw otps ollama={[round(x,2) for x in acc['ollama']['otps']]}")
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            pr.wait()
