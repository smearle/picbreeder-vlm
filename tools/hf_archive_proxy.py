#!/usr/bin/env python3
"""Authenticated, transparent reverse-proxy for the PRIVATE HF archive dataset.

Unlike the `?archiveBase=.` local mirror (which serves files off local disk at
zero latency), this proxy forwards every request to the *real* huggingface.co
`resolve/main` endpoint — including the 302 hop to HF's LFS/CDN for the .webp
sheets — and only staples on your `Authorization: Bearer <token>` header so the
private repo answers instead of 404ing. So you measure genuine HF latency while
keeping the dataset private.

Usage:
    python tools/hf_archive_proxy.py [--port 8800] [--repo picbreeder-vlm/picbreeder-vlm-archive]
    # then open the gallery with:
    #   archive-gallery-sprite.html?archiveBase=http://localhost:8800

Token resolution order: $HF_TOKEN, $HUGGINGFACE_HUB_TOKEN,
~/.cache/huggingface/token, ~/.huggingface/token.
"""
import argparse
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
# Drop upstream's own CORS/Vary headers — we re-emit a single clean set below.
# Passing HF's `Access-Control-Allow-Origin: https://huggingface.co` through AND
# adding our own `*` yields two conflicting ACAO headers, which the browser
# rejects with an opaque "Failed to fetch" even on a 200.
DROP = HOP_BY_HOP | {
    "content-length", "vary",
    "access-control-allow-origin", "access-control-allow-credentials",
    "access-control-allow-methods", "access-control-allow-headers",
    "access-control-expose-headers", "access-control-max-age",
    # HF/CDN mark webp sheets cacheable (often long/immutable max-age). During dev we
    # overwrite files at the SAME resolve URL (e.g. grayscale -> color), so a cached copy
    # masks the new push. Strip caching and force no-store below so pushes show immediately.
    "cache-control", "etag", "expires", "last-modified", "age", "pragma",
}


def resolve_token() -> str:
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    for path in (Path.home() / ".cache/huggingface/token",
                 Path.home() / ".huggingface/token"):
        if path.is_file():
            tok = path.read_text().strip()
            if tok:
                return tok
    sys.exit("No HF token found (set $HF_TOKEN or log in via `huggingface-cli login`).")


def make_handler(upstream_base: str, token: str):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: the gallery fires many small fetches

        def _cors(self):
            # Single, unambiguous CORS set. "*" matches any origin including the
            # "null" origin a file:// page sends. We don't use credentials.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Expose-Headers", "*")
            self.send_header("Access-Control-Max-Age", "86400")
            # Never let the browser cache through the dev proxy (see DROP note).
            self.send_header("Cache-Control", "no-store")

        def do_OPTIONS(self):  # CORS preflight
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _proxy(self, body: bool):
            # Path is relative to the dataset's resolve/main root, mirroring ARCHIVE_BASE.
            url = upstream_base + self.path
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Bearer {token}")
            # urllib follows the 302 to the CDN automatically (real browser behavior).
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() in DROP:
                            continue
                        self.send_header(k, v)
                    self._cors()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if body:
                        self.wfile.write(data)
            except urllib.error.HTTPError as e:
                msg = e.read()
                self.send_response(e.code)
                self._cors()
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                if body:
                    self.wfile.write(msg)
            except Exception as e:  # noqa: BLE001
                self.send_error(502, f"upstream error: {e}")

        def do_GET(self):
            self._proxy(body=True)

        def do_HEAD(self):
            self._proxy(body=False)

        def log_message(self, fmt, *args):
            sys.stderr.write("[proxy] %s - %s\n" % (self.path, fmt % args))

    return Proxy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repo", default="picbreeder-vlm/picbreeder-vlm-archive")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    token = resolve_token()
    upstream = f"https://huggingface.co/datasets/{args.repo}/resolve/{args.revision}"
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(upstream, token))
    base = f"http://{args.host}:{args.port}"
    print(f"Proxying {base}  ->  {upstream}", file=sys.stderr)
    print(f"Open the gallery with:  ?archiveBase={base}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)


if __name__ == "__main__":
    main()
