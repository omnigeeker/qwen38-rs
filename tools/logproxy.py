#!/usr/bin/env python3
"""Transparent logging proxy: see what an agent framework actually sends.

Sits on the port the framework is already configured for and forwards to the real
server, so nothing on the client side has to change.  Logs the request shape
(message count, character count, tool count, max_tokens, stream) and, for a
streamed reply, the time to the first byte - which is where a slow prefill shows up.
"""
import http.client
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ("127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 8081)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _shape(self, body):
        try:
            d = json.loads(body)
        except Exception as e:
            return f"(body is not json: {e})"
        msgs = d.get("messages") or []
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)
        return (
            f"msgs={len(msgs)} chars={chars} tools={len(d.get('tools') or [])} "
            f"max_tokens={d.get('max_tokens')} stream={d.get('stream')} "
            f"model={d.get('model')}"
        )

    def _relay(self, method):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else None
        t0 = time.time()
        tail = self._shape(body) if body else ""
        print(f"[{time.strftime('%H:%M:%S')}] {method} {self.path} bytes={n} {tail}", flush=True)
        conn = http.client.HTTPConnection(*UPSTREAM, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        conn.request(method, self.path, body=body, headers=headers)
        r = conn.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("connection", "close")
        self.end_headers()
        first, total = None, 0
        while True:
            # read1 returns as soon as anything is available, which is what makes
            # the first-byte time meaningful for a streamed reply.
            chunk = r.read1(65536)
            if not chunk:
                break
            if first is None:
                first = time.time() - t0
                print(f"           first byte after {first:.1f}s", flush=True)
            total += len(chunk)
            self.wfile.write(chunk)
            self.wfile.flush()
        print(f"           done: {total} bytes in {time.time() - t0:.1f}s", flush=True)
        conn.close()

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"proxy :{port} -> {UPSTREAM[0]}:{UPSTREAM[1]}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
