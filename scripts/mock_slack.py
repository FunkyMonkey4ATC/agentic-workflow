#!/usr/bin/env python3
"""Mock Slack incoming-webhook sink for offline tests. Logs every post to data/mock_slack.jsonl.
Start/stop it to simulate the network going away and coming back.

  scripts/mock_slack.py [--port 7999] [--fail]   (--fail: accept the request but return 500)
Point the agent at it with SLACK_WEBHOOK_URL=http://127.0.0.1:7999/hook
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7999)
ap.add_argument("--fail", action="store_true")
args = ap.parse_args()
LOG = Path(__file__).resolve().parents[1] / "data" / "mock_slack.jsonl"
LOG.parent.mkdir(exist_ok=True)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n).decode()
        with LOG.open("a") as f:
            f.write(json.dumps({"t": time.time(), "path": self.path, "body": body}) + "\n")
        print(f"[mock-slack] {self.path} :: {body[:100]}", flush=True)
        self.send_response(500 if args.fail else 200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):  # quiet
        pass


print(f"mock slack on :{args.port} (fail={args.fail}) -> {LOG}")
HTTPServer(("127.0.0.1", args.port), H).serve_forever()
