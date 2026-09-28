"""
service/predictor.py
====================
Single-snapshot prediction service for the Coffee Line web API.

It reuses, unchanged, the artifacts the rest of the project already produced:

  * ``models/model.joblib``      -- the calibrated model + threshold + metrics
  * ``config.MACHINE_BASELINES`` -- the per-machine-type "what normal looks like"
  * ``src/health.py``            -- the transparent 0-100 health score
  * ``src/maintenance_rules.py`` -- the plain-language recommendation

So the web API can never score a machine differently from how the model was
trained and evaluated.

Why a separate service from ``src/scoring.py``?
  The Streamlit dashboard scores a whole time-series *history* (many hourly
  rows per machine), which is what the trend features were designed for. A web
  form sends only ONE reading. For a single steady-state snapshot the correct
  value of every "change since N hours ago" and "rolling std" feature is 0, and
  a "rolling mean" is just the reading itself -- that is exactly how we build
  them below. Deviation z-scores, interactions and utilisation features are
  computed identically to ``src/feature_engineering.py``. The probability we
  return is the real model's output on that feature vector -- never a
  placeholder or a random number.

Kept deliberately dependency-light (numpy + scikit-learn + joblib only, no
pandas / xgboost / shap) so it fits comfortably inside a Vercel Python
serverless function.
"""

import os
import sys

import numpy as np

# Make ``config`` and the ``src`` package importable no matter where the
# serverless runtime sets the working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config                                   # noqa: E402
from src import health                          # noqa: E402  (imports only config)
from src import maintenance_rules               # noqa: E402  (imports only config)

MODEL_PATH = os.path.join(config.MODELS_DIR, "model.joblib")

# The six sensor inputs the web form asks for, plus the optional fields the
# trained model also needs. The form maps "rotational_speed" -> the model's
# "rpm" and sends machine_load as a percentage (0-100); we convert both below.
DEFAULT_MACHINE_TYPE = "Grinder"

# Numeric fields we accept, with (min, max) sanity bounds for validation.
# Bounds are generous on purpose -- they catch nonsense (negative rpm, a load
# of 5000%) without rejecting a legitimately abnormal-but-real reading.
INPUT_BOUNDS = {
    "temperature": (-50.0, 2000.0),
    "vibration": (0.0, 500.0),
    "pressure": (0.0, 200.0),
    "rotational_speed": (0.0, 100000.0),
    "operating_hours": (0.0, 1_000_000.0),
    "machine_load": (0.0, 100.0),          # percent
    "motor_current": (0.0, 5000.0),
    "hours_since_maintenance": (0.0, 1_000_000.0),
    "maintenance_count": (0.0, 100000.0),
    "previous_failures": (0.0, 100000.0),
}

# Required vs optional. Optional fields fall back to a healthy default so a
# minimal 6-field request (the shape in the brief) still scores.
REQUIRED_FIELDS = [
    "temperature", "vibration", "pressure",
    "rotational_speed", "operating_hours", "machine_load",
]


class PredictionError(ValueError):
    """Raised for user-fixable problems (bad input). Maps to HTTP 400."""


class ModelUnavailableError(RuntimeError):
    """Raised when the model bundle cannot be loaded. Maps to HTTP 503."""


# ---------------------------------------------------------------------------
# Model loading (module-level singleton: loaded once per warm serverless
# instance, NOT on every request, and never retrained).
# ---------------------------------------------------------------------------
_BUNDLE = None


def get_bundle():
    """Load and cache the model bundle. Raises ModelUnavailableError on failure."""
    global _BUNDLE
    if _BUNDLE is None:
        import joblib   # local import keeps cold-start import graph minimal
        if not os.path.exists(MODEL_PATH):
            raise ModelUnavailableError(
                "Trained model file is missing from the deployment."
            )
        try:
            _BUNDLE = joblib.load(MODEL_PATH)
        except Exception as exc:   # noqa: BLE001 -- surfaced as a clean 503
            raise ModelUnavailableError(
                f"Trained model could not be loaded ({type(exc).__name__})."
            ) from exc
    return _BUNDLE

# ---------------------------------------------------------------------------
# Input parsing + validation
# ---------------------------------------------------------------------------
def _coerce_number(name, value):
    """Turn an incoming JSON value into a float, or raise a clear error."""
    if value is None or value == "":
        raise PredictionError(f"'{name}' is required.")
    if isinstance(value, bool):   # bool is a subclass of int -- reject it explicitly
        raise PredictionError(f"'{name}' must be a number, not a boolean.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise PredictionError(f"'{name}' must be a number (got {value!r}).")


def parse_payload(payload):
    """
    Validate and normalise a raw request body (a dict) into a clean input dict.

    Returns a dict with every field the model needs, with optional fields
    filled from healthy defaults. Raises PredictionError on anything the caller
    can fix.
    """
    if not isinstance(payload, dict):
        raise PredictionError("Request body must be a JSON object.")

    # 1. Machine type (drives the per-type baselines used by the z-scores).
    machine_type = payload.get("machine_type", DEFAULT_MACHINE_TYPE)
    if machine_type not in config.MACHINE_BASELINES:
        allowed = ", ".join(sorted(config.MACHINE_BASELINES))
        raise PredictionError(
            f"Unknown machine_type '{machine_type}'. Allowed: {allowed}."
        )
    base = config.MACHINE_BASELINES[machine_type]

    # 2. Required numeric sensor fields.
    clean = {"machine_type": machine_type}
    for name in REQUIRED_FIELDS:
        if name not in payload:
            raise PredictionError(f"'{name}' is required.")
        clean[name] = _coerce_number(name, payload[name])

    # 3. Optional fields -> healthy defaults if absent.
    #    motor_current defaults to this machine type's baseline, so its z-score
    #    is 0 (a healthy motor) when the caller doesn't measure it.
    clean["motor_current"] = _coerce_number(
        "motor_current", payload.get("motor_current", base["motor_current"])
    )
    for name in ("hours_since_maintenance", "maintenance_count", "previous_failures"):
        clean[name] = _coerce_number(name, payload.get(name, 0))

    # 4. Range checks.
    for name, val in clean.items():
        if name == "machine_type":
            continue
        low, high = INPUT_BOUNDS[name]
        if not (low <= val <= high):
            raise PredictionError(
                f"'{name}' = {val} is out of the accepted range [{low}, {high}]."
            )

    return clean

# ---------------------------------------------------------------------------
# Feature construction (mirrors src/feature_engineering.py for ONE snapshot)
# ---------------------------------------------------------------------------
def build_feature_row(clean):
    """
    Turn a validated input dict into the model's feature dict.

    Every formula here matches src/feature_engineering.py exactly. The only
    difference is the single-snapshot handling of trend features, documented in
    the module docstring: no history -> changes and rolling std are 0, and a
    rolling mean is the reading itself.

    NOTE on machine_load: the model was trained with load as a FRACTION in
    [0.4, 1.0] (see src/simulate.py). The web form sends a percentage (0-100),
    so we convert to a fraction before it touches any model feature.
    """
    base = config.MACHINE_BASELINES[clean["machine_type"]]

    temperature = clean["temperature"]
    vibration = clean["vibration"]
    pressure = clean["pressure"]
    rpm = clean["rotational_speed"]                # form name -> model's "rpm"
    motor_current = clean["motor_current"]
    load = clean["machine_load"] / 100.0           # percent -> training fraction

    def z(reading, sensor):
        return (reading - base[sensor]) / base[sensor + "_noise"]

    feat = {
        # rates of change -- no history in a single snapshot => 0
        "temperature_change_1h": 0.0,
        "temperature_change_6h": 0.0,
        "vibration_change_1h": 0.0,
        "pressure_change_1h": 0.0,
        # trailing rolling stats -- one reading => mean is the reading, std is 0
        "temperature_mean_6h": temperature,
        "temperature_std_6h": 0.0,
        "vibration_mean_6h": vibration,
        "vibration_std_6h": 0.0,
        "machine_load_mean_6h": load,
        "machine_load_std_6h": 0.0,
        # deviation from a healthy machine of the same type (identical formula)
        "temperature_zscore": z(temperature, "temperature"),
        "vibration_zscore": z(vibration, "vibration"),
        "pressure_zscore": z(pressure, "pressure"),
        "rpm_zscore": z(rpm, "rpm"),
        "motor_current_zscore": z(motor_current, "motor_current"),
        # utilisation / maintenance history
        "operating_hours": clean["operating_hours"],
        "machine_load": load,
        "hours_since_maintenance": clean["hours_since_maintenance"],
        "maintenance_count": clean["maintenance_count"],
        "previous_failures": clean["previous_failures"],
        # interactions
        "temp_x_load": temperature * load,
        "vibration_x_rpm": vibration * rpm,
        "pressure_x_load": pressure * load,
    }
    return feat


def _risk_level(probability, threshold):
    """
    Map a calibrated probability to a 4-band risk label.

    The bands are anchored on the model's own recall-favouring alert threshold
    (below it = the model would NOT raise an alert), then escalate from there.
    """
    if probability < threshold:
        return "Low"
    if probability < 0.40:
        return "Medium"
    if probability < 0.70:
        return "High"
    return "Critical"

# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------
def predict(payload):
    """
    Score one machine snapshot end to end.

    `payload` is the parsed JSON request body (a dict). Returns a JSON-safe
    dict. Raises PredictionError (400) for bad input or ModelUnavailableError
    (503) if the model can't be loaded. Any other exception is a real bug and
    should surface as a 500.
    """
    import warnings

    clean = parse_payload(payload)
    bundle = get_bundle()
    feat = build_feature_row(clean)

    # Order features exactly as the model was trained (the bundle is the source
    # of truth), then score. We pass a plain numpy array and silence sklearn's
    # "no feature names" notice -- the column ORDER is what matters and it is
    # guaranteed correct here.
    order = bundle["feature_columns"]
    X = np.array([[feat[col] for col in order]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probability = float(bundle["model"].predict_proba(X)[0, 1])

    threshold = float(bundle["threshold"])
    score = health.health_score(feat)
    rec = maintenance_rules.recommend(feat, top_driver=None)
    recommendation = f"{rec['headline']} {rec['action']}".strip()

    metrics = bundle.get("metrics", {}) or {}
    return {
        # --- the contract from the brief (top level, exact keys) ---
        "failure_probability": round(probability, 4),
        "health_score": int(round(score)),
        "risk_level": _risk_level(probability, threshold),
        "recommendation": recommendation,
        # --- extra context the dashboard uses for its visualisations ---
        "details": {
            "machine_type": clean["machine_type"],
            "alert": bool(probability >= threshold),
            "alert_threshold": round(threshold, 4),
            "health_band": health.health_label(score),
            "urgent": bool(rec["urgent"]),
            "worst_sensor": rec["worst_sensor"],
            "sensor_zscores": {
                "temperature": round(feat["temperature_zscore"], 3),
                "vibration": round(feat["vibration_zscore"], 3),
                "pressure": round(feat["pressure_zscore"], 3),
                "rotational_speed": round(feat["rpm_zscore"], 3),
                "motor_current": round(feat["motor_current_zscore"], 3),
            },
            "model": {
                "name": bundle.get("model_name", "model"),
                "recall": round(float(metrics.get("Recall", 0.0)), 3),
                "precision": round(float(metrics.get("Precision", 0.0)), 3),
                "pr_auc": round(float(metrics.get("PR_AUC", 0.0)), 3),
            },
            "inputs": {k: v for k, v in clean.items()},
        },
    }


# Quick local smoke test:  python -m service.predictor
if __name__ == "__main__":
    import json

    samples = [
        {"machine_type": "Grinder", "temperature": 55, "vibration": 4.0,
         "pressure": 1.2, "rotational_speed": 1450, "operating_hours": 1000,
         "machine_load": 70},
        {"machine_type": "Grinder", "temperature": 78, "vibration": 9.5,
         "pressure": 1.4, "rotational_speed": 1360, "operating_hours": 4200,
         "machine_load": 92, "hours_since_maintenance": 480},
        # the exact example from the brief (no machine_type -> default)
        {"temperature": 75, "vibration": 4.2, "pressure": 8.5,
         "rotational_speed": 1500, "operating_hours": 4200, "machine_load": 72},
    ]
    for s in samples:
        print(json.dumps(predict(s), indent=2))
        print("-" * 60)




