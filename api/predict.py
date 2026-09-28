"""
api/predict.py  --  Vercel Python serverless function
=====================================================
POST /api/predict

Thin HTTP layer over ``service.predictor``. It does NO machine learning of its
own: it parses the JSON body, hands it to the shared prediction service (which
loads the real trained model bundle), and returns the result. The model is
loaded once per warm instance and cached -- never retrained per request.

Request body (JSON). The six fields from the brief are required; the rest are
optional and fall back to healthy defaults:

    {
      "machine_type": "Grinder",          // optional (default "Grinder")
      "temperature": 75,                   // required
      "vibration": 4.2,                    // required
      "pressure": 8.5,                     // required
      "rotational_speed": 1500,            // required
      "operating_hours": 4200,             // required
      "machine_load": 72,                  // required (percent, 0-100)
      "motor_current": 30,                 // optional (default: type baseline)
      "hours_since_maintenance": 0,        // optional (default 0)
      "maintenance_count": 0,              // optional (default 0)
      "previous_failures": 0               // optional (default 0)
    }

Response (JSON): failure_probability, health_score, risk_level, recommendation
(plus a "details" object the dashboard uses for its charts).

Errors are returned as clean JSON with an "error" message and never leak stack
traces or filesystem paths to the client (full detail is logged server-side).
"""

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

# Make the project root importable (config, src/, service/ live one level up).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from service import predictor  # noqa: E402

MAX_BODY_BYTES = 64 * 1024   # a prediction request is tiny; reject anything huge


def _process(raw_body):
    """
    Core request logic, shared by the Vercel handler and the local dev server.

    Returns (status_code, response_dict). Never raises: every failure is turned
    into a clean JSON error. Client-fixable problems (bad JSON / bad values) map
    to 400; a missing/broken model maps to 503; anything else is a 500 with a
    generic message (the real cause is logged, not returned).
    """
    try:
        payload = json.loads(raw_body or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 400, {"error": "Request body must be valid JSON."}

    try:
        return 200, predictor.predict(payload)
    except predictor.PredictionError as exc:
        return 400, {"error": str(exc)}
    except predictor.ModelUnavailableError as exc:
        # Configuration/deployment problem -- safe to show the short message.
        print(f"[predict] model unavailable: {exc}", file=sys.stderr)
        return 503, {"error": "Prediction service is temporarily unavailable. "
                              "The model could not be loaded."}
    except Exception:   # noqa: BLE001 -- last-resort guard
        # Log the full traceback server-side; return nothing sensitive.
        print("[predict] unexpected error:\n" + traceback.format_exc(),
              file=sys.stderr)
        return 500, {"error": "Internal error while generating the prediction."}


class handler(BaseHTTPRequestHandler):
    """Vercel invokes this per request (name ``handler`` is the convention)."""

    def _send(self, status, body_dict):
        payload = json.dumps(body_dict).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # CORS: allow the dashboard (same origin in prod, any origin for local
        # testing / API reuse). No credentials are used.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):   # CORS preflight
        self._send(204, {})

    def do_GET(self):
        """A friendly note so hitting the URL in a browser isn't a dead end."""
        self._send(200, {
            "service": "Coffee Line predictive-maintenance API",
            "usage": "POST JSON to this endpoint. See 'required'.",
            "required": predictor.REQUIRED_FIELDS,
            "optional": ["machine_type", "motor_current",
                         "hours_since_maintenance", "maintenance_count",
                         "previous_failures"],
            "machine_types": sorted(predictor.config.MACHINE_BASELINES.keys()),
        })

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._send(413, {"error": "Request body too large."})
            return
        raw = self.rfile.read(length) if length else b""
        status, body = _process(raw.decode("utf-8", errors="replace"))
        self._send(status, body)

    # Keep the serverless logs quiet/clean (default logs every request line).
    def log_message(self, *args):
        pass

