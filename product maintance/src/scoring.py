"""
scoring.py  --  Phase 6/7 glue: turn raw data + saved model into a live view
============================================================================

The dashboard should not know the messy details of "load CSV, build features,
run the model, compute health, rank priorities". It just wants a tidy table of
"here is every machine right now". That is what this file provides.

Pipeline in one place:
    raw sensor CSV  ->  causal features  ->  calibrated P(failure)
                    ->  health score + band  ->  recommendation  ->  priority

Everything here reuses the SAME feature code and the SAME saved model the
training script produced, so the dashboard can never accidentally score
machines differently from how the model was evaluated.
"""

import numpy as np
import pandas as pd
import joblib
import os

import config
from src import feature_engineering as fe
from src import health
from src import maintenance_rules
from src import priority


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_bundle():
    """Load the saved model bundle (calibrated model, threshold, features...)."""
    path = os.path.join(config.MODELS_DIR, "model.joblib")
    return joblib.load(path)


def load_raw():
    """Load the simulated sensor history.

    On a fresh deployment (e.g. Streamlit Community Cloud) the raw CSV is not
    stored in git -- it is large and fully reproducible -- so it may be absent.
    If so, we regenerate it once from the seeded simulator. Because the
    simulator uses a fixed seed (config.SEED), the rebuilt file is identical to
    the one the model was trained on, so the scores stay consistent.
    """
    if not os.path.exists(config.RAW_DATA_PATH):
        from src import simulate          # imported lazily: only needed if missing
        simulate.simulate()               # writes config.RAW_DATA_PATH
    return pd.read_csv(config.RAW_DATA_PATH)


# ---------------------------------------------------------------------------
# Validating a user-supplied dataset before we trust it
# ---------------------------------------------------------------------------
# The columns an uploaded CSV MUST have for us to score it. These are the raw
# inputs the feature builder needs -- the label columns (failure_event /
# failure_within_window) are NOT required, because scoring only predicts.
REQUIRED_RAW_COLUMNS = [
    "machine_id", "machine_type", "operating_hours",
    "temperature", "vibration", "pressure", "rpm", "motor_current",
    "machine_load", "hours_since_maintenance",
    "maintenance_count", "previous_failures",
]


def validate_raw(df):
    """
    Check a user-uploaded DataFrame is something we can actually score.

    Returns a list of human-readable problem strings. An EMPTY list means the
    data is good to go. We keep the messages plain so a non-expert can fix the
    file themselves.
    """
    problems = []

    # 1. Are all the required columns present?
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing required column(s): " + ", ".join(missing))
        # Without the columns we can't check anything else, so stop here.
        return problems

    # 2. Are the machine types ones the model actually knows about?
    #    (The z-score features look up a baseline per type in config.)
    known_types = set(config.MACHINE_BASELINES.keys())
    found_types = set(df["machine_type"].dropna().unique())
    unknown = sorted(found_types - known_types)
    if unknown:
        problems.append(
            "Unknown machine_type(s): " + ", ".join(unknown) +
            ". Allowed types are: " + ", ".join(sorted(known_types)) + "."
        )

    # 3. Do the sensor columns contain numbers?
    numeric_cols = ["operating_hours", "temperature", "vibration", "pressure",
                    "rpm", "motor_current", "machine_load",
                    "hours_since_maintenance", "maintenance_count",
                    "previous_failures"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Column '{col}' must contain numbers only.")

    # 4. Is there enough history? The features look back up to 6 hours (a
    #    6-hour rolling window AND a 6-hour change), so a machine needs at
    #    least 7 hourly readings before even one row can be scored -- fewer
    #    than that and every row gets dropped as "warm-up".
    if not df.empty:
        longest = df.groupby("machine_id").size().max()
        if longest < 7:
            problems.append(
                "Each machine needs at least 7 hourly readings so the "
                "rolling-window and rate-of-change features can be computed "
                f"(the most any machine has here is {int(longest)})."
            )

    return problems


def make_template_csv():
    """
    Build a tiny example CSV (as a string) showing the exact columns and a
    couple of healthy example rows. Handy as a "fill this in" starting point
    for someone bringing their own data. Values come from the healthy
    baselines in config, so the example is realistic.
    """
    example_rows = []
    # Two example machine types so the format is obvious. 30 hourly rows each:
    # comfortably more than the 7-row minimum, so the file scores right away.
    for machine_id, mtype in [("RST-01", "Roaster"), ("GRD-01", "Grinder")]:
        base = config.MACHINE_BASELINES[mtype]
        for hour in range(30):
            example_rows.append({
                "machine_id": machine_id,
                "machine_type": mtype,
                "operating_hours": hour,
                "temperature": base["temperature"],
                "vibration": base["vibration"],
                "pressure": base["pressure"],
                "rpm": base["rpm"],
                "motor_current": base["motor_current"],
                "machine_load": 0.8,
                "hours_since_maintenance": hour,
                "maintenance_count": 0,
                "previous_failures": 0,
            })
    template = pd.DataFrame(example_rows, columns=REQUIRED_RAW_COLUMNS)
    return template.to_csv(index=False)




# ---------------------------------------------------------------------------
# Scoring every row
# ---------------------------------------------------------------------------
def score_history(raw_df, bundle):
    """
    Build features for the whole history and attach the model's probability to
    every row. Returns the feature DataFrame plus a 'failure_probability' and a
    0/1 'failure_prediction' column (using the saved recall-favouring threshold).
    """
    feat = fe.build_feature_matrix(raw_df)
    X = feat[bundle["feature_columns"]]
    feat = feat.copy()
    feat["failure_probability"] = bundle["model"].predict_proba(X)[:, 1]
    feat["failure_prediction"] = (
        feat["failure_probability"] >= bundle["threshold"]
    ).astype(int)
    return feat


# ---------------------------------------------------------------------------
# SHAP explanations (why did the model say that?)
# ---------------------------------------------------------------------------
def compute_shap(base_model, X_rows, feature_columns):
    """
    Explain a few rows with SHAP on the raw tree model.

    Returns (shap_matrix, top_features):
      * shap_matrix: numpy array [n_rows, n_features] of SHAP values for the
        "failure" class -- positive value = pushed the prediction toward failure.
      * top_features: list of the single most-influential feature name per row.

    SHAP's output shape differs a little between versions / model types, so we
    normalise it here and keep the dashboard code simple.
    """
    import shap   # imported lazily -- it's only needed on the detail page

    explainer = shap.TreeExplainer(base_model)
    raw = explainer.shap_values(X_rows)

    # Normalise to a 2-D array of "failure class" contributions.
    if isinstance(raw, list):
        # older API: list [class0, class1]
        values = np.array(raw[1])
    else:
        values = np.array(raw)
        if values.ndim == 3:
            # newer API: shape (n_rows, n_features, n_classes) -> take failure class
            values = values[:, :, 1]

    top_features = []
    for i in range(values.shape[0]):
        # the feature that pushed HARDEST toward "failure" for this row
        j = int(np.argmax(values[i]))
        top_features.append(feature_columns[j])

    return values, top_features


# ---------------------------------------------------------------------------
# "Right now" snapshot: one row per machine
# ---------------------------------------------------------------------------
def latest_snapshot(scored_history, bundle, explain=True, as_of_hour=None):
    """
    Take the most recent scored reading for each machine and enrich it with
    health score/band, a maintenance recommendation, and (optionally) the
    model's top SHAP driver. Returns a DataFrame, one row per machine.

    `as_of_hour` lets the dashboard pretend "now" is any past operating hour:
    we use each machine's latest reading AT OR BEFORE that hour. Left as None,
    we use the very last reading (the true end of the simulation).
    """
    history = scored_history
    if as_of_hour is not None:
        history = history[history["operating_hours"] <= as_of_hour]

    # Most recent operating hour per machine.
    latest_idx = history.groupby("machine_id")["operating_hours"].idxmax()
    snap = history.loc[latest_idx].copy().reset_index(drop=True)

    # Health score + band (rule-based, independent of the ML probability).
    snap["health_score"] = snap.apply(health.health_score, axis=1)
    snap["health_band"] = snap["health_score"].apply(health.health_label)
    snap["health_colour"] = snap["health_score"].apply(
        lambda s: health.health_band(s)["colour"]
    )

    # Optional SHAP top-driver per machine (helps the recommendation wording).
    top_drivers = [None] * len(snap)
    if explain:
        try:
            X_rows = snap[bundle["feature_columns"]]
            _, top_drivers = compute_shap(
                bundle["base_model"], X_rows, bundle["feature_columns"]
            )
        except Exception:
            # If SHAP isn't available for any reason, we still work -- the
            # recommendation just falls back to the most-deviated sensor.
            top_drivers = [None] * len(snap)
    snap["top_driver"] = top_drivers

    # Plain-language recommendation per machine.
    headlines, actions, urgents = [], [], []
    for i in range(len(snap)):
        row = snap.iloc[i]
        rec = maintenance_rules.recommend(row, top_driver=row["top_driver"])
        headlines.append(rec["headline"])
        actions.append(rec["action"])
        urgents.append(rec["urgent"])
    snap["rec_headline"] = headlines
    snap["rec_action"] = actions
    snap["rec_urgent"] = urgents

    return snap


def priority_table(snapshot_df):
    """
    Rank the snapshot by expected production units lost (see src/priority.py).
    Returns a DataFrame sorted most-urgent first.
    """
    rows = snapshot_df[
        ["machine_id", "machine_type", "failure_probability",
         "health_score", "health_band"]
    ].to_dict("records")
    ranked = priority.build_priority_table(rows)
    return pd.DataFrame(ranked)


# Quick end-to-end smoke test:  python -m src.scoring
if __name__ == "__main__":
    bundle = load_bundle()
    raw = load_raw()
    scored = score_history(raw, bundle)
    snap = latest_snapshot(scored, bundle, explain=True)
    print(f"Scored {len(scored):,} rows; {len(snap)} machines in snapshot.\n")

    cols = ["machine_id", "machine_type", "failure_probability",
            "health_score", "health_band", "rec_headline"]
    print(snap[cols].round(3).to_string(index=False))

    print("\nPriority queue (fix these first):")
    pt = priority_table(snap)
    show = ["priority_rank", "machine_id", "failure_probability",
            "expected_units_lost"]
    print(pt[show].round(3).to_string(index=False))
