"""Logging reverse proxy for an OpenAI-compatible server; records agent traces for replay.

Stdlib only: one thread per request is plenty for an agent's handful of concurrent calls.

Every /v1/chat/completions request is logged with the session it belongs to (the recorder
sets it with `POST /session/<id>`) and `deps_done`, the number of responses in that session
that had already completed when the request arrived. That single number captures an agent's
dependency structure without knowing its framework: parallel calls share a value, a
sequential pipeline counts 0, 1, 2, ...

    python -m record.proxy --upstream http://127.0.0.1:8000 --out traces/accord.jsonl
"""

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHAT_PATH = "/v1/chat/completions"


class Recorder:
    def __init__(self, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.out = open(out_path, "w")
        self.lock = threading.Lock()
        self.session = "default"
        self.completed = Counter()

    def arrive(self):
        with self.lock:
            return self.session, self.completed[self.session]

    def finish(self, session, deps_done, body, status):
        with self.lock:
            self.completed[session] += 1
            row = {"session": session, "deps_done": deps_done, "status": status, "body": body}
            self.out.write(json.dumps(row) + "\n")
            self.out.flush()


def make_handler(upstream, rec):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self, body):
            headers = {h: self.headers[h] for h in ("Content-Type", "Authorization") if h in self.headers}
            req = urllib.request.Request(upstream + self.path, data=body, method=self.command, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.status, r.read(), r.headers.get("Content-Type")
            except urllib.error.HTTPError as e:
                return e.code, e.read(), e.headers.get("Content-Type")
            except urllib.error.URLError as e:
                return 502, str(e).encode(), "text/plain"

        def reply(self, status, payload, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.reply(*self.forward(None))

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/session/"):
                with rec.lock:
                    rec.session = self.path[len("/session/"):]
                return self.reply(200, b"ok", "text/plain")
            if self.path != CHAT_PATH:
                return self.reply(*self.forward(body))
            session, deps_done = rec.arrive()
            status, payload, content_type = self.forward(body)
            rec.finish(session, deps_done, json.loads(body), status)
            if status != 200:
                print(f"warning: upstream returned {status} in session {session}: {payload[:200]!r}", file=sys.stderr)
            self.reply(status, payload, content_type)

    return Handler


def set_session(proxy_url, session_id):
    """Called by the recorders before each agent run."""
    urllib.request.urlopen(urllib.request.Request(f"{proxy_url}/session/{session_id}", method="POST")).read()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", required=True, help="model server base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--out", required=True, help="trace file to write, e.g. traces/accord.jsonl")
    args = p.parse_args()
    rec = Recorder(args.out)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.upstream.rstrip("/"), rec))
    server.daemon_threads = True
    print(f"proxy on :{args.port} -> {args.upstream}, writing {args.out}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        rec.out.close()


if __name__ == "__main__":
    main()
