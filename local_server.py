"""
local_server.py  --  local development server (NOT deployed to Vercel)
======================================================================
Serves the static dashboard in ``public/`` AND the ``/api/predict`` endpoint
from a single process, so you can test the whole app locally exactly as it will
behave on Vercel. It reuses the SAME ``service.predictor`` code the Vercel
function uses, so a prediction here is identical to a prediction in production.

Run:
    python local_server.py            # then open http://localhost:8000

This file is excluded from the Vercel deployment (see .vercelignore); Vercel
serves ``public/`` statically and runs ``api/predict.py`` as a function.
"""

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the exact request logic from the serverless function.
from api import predict as predict_fn  # noqa: E402

PORT = int(os.environ.get("PORT", "8000"))
PUBLIC_DIR = os.path.join(_ROOT, "public")


class DevHandler(SimpleHTTPRequestHandler):
    """Static files from public/, plus /api/predict routed to the predictor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def _json(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        if self.path.split("?")[0] == "/api/predict":
            self._json(204, {})
        else:
            super().do_OPTIONS() if hasattr(super(), "do_OPTIONS") else self._json(204, {})

    def do_GET(self):
        if self.path.split("?")[0] == "/api/predict":
            status, body = 200, {
                "service": "Coffee Line predictive-maintenance API (local)",
                "required": predict_fn.predictor.REQUIRED_FIELDS,
                "machine_types": sorted(
                    predict_fn.predictor.config.MACHINE_BASELINES.keys()),
            }
            self._json(status, body)
            return
        super().do_GET()   # serve static files

    def do_POST(self):
        if self.path.split("?")[0] != "/api/predict":
            self._json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        status, body = predict_fn._process(raw.decode("utf-8", errors="replace"))
        self._json(status, body)

    def log_message(self, fmt, *args):
        sys.stderr.write("[dev] " + (fmt % args) + "\n")


if __name__ == "__main__":
    os.chdir(PUBLIC_DIR)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DevHandler)
    print(f"Coffee Line dev server -> http://localhost:{PORT}")
    print(f"  static : {PUBLIC_DIR}")
    print(f"  api    : POST http://localhost:{PORT}/api/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()
