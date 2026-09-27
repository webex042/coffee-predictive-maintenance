"""
feature_engineering.py  --  Phase 3: causal, past-only features
================================================================

We turn the raw sensor readings into features a model can learn from.

THE GOLDEN RULE (this is what prevents data leakage):
  Every feature for a given hour may look at that hour and EARLIER hours
  of the SAME machine -- never the future, never another machine.

That is why every calculation below is done with `groupby("machine_id")`
and uses `.diff()` / trailing `.rolling()` windows (never centered windows).

Each function is a small, pure function: give it a DataFrame, it returns a
new DataFrame with extra columns. That makes them easy to unit-test and lets
the dashboard build features the EXACT same way as training.

We never feed `failure_event` or `failure_within_window` into the features --
those are the answer, not an input.
"""

import numpy as np
import pandas as pd

import config

# The raw sensors we build features from.
SENSORS = ["temperature", "vibration", "pressure", "rpm", "motor_current"]


def _sorted(df):
    """Always work in time order, per machine. Return a copy so we never mutate."""
    return df.sort_values(["machine_id", "operating_hours"]).copy()


# ---------------------------------------------------------------------------
# 1. Rates of change ("is it moving, and how fast?")
# ---------------------------------------------------------------------------
def add_rate_features(df):
    """Change since 1 hour ago and 6 hours ago. Uses past values only."""
    df = _sorted(df)
    g = df.groupby("machine_id")
    df["temperature_change_1h"] = g["temperature"].diff(1)
    df["temperature_change_6h"] = g["temperature"].diff(6)
    df["vibration_change_1h"] = g["vibration"].diff(1)
    df["pressure_change_1h"] = g["pressure"].diff(1)
    return df


# ---------------------------------------------------------------------------
# 2. Trailing rolling statistics ("what has the last 6 hours looked like?")
# ---------------------------------------------------------------------------
def add_rolling_features(df, window=6):
    """
    Rolling mean and std over a TRAILING window (current hour + previous 5).
    A trailing window is causal: it never peeks at future readings.
    """
    df = _sorted(df)
    for col in ["temperature", "vibration", "machine_load"]:
        roll = df.groupby("machine_id")[col].rolling(window, min_periods=window)
        # reset_index(level=0, drop=True) lines the result back up with df's rows
        df[f"{col}_mean_{window}h"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_std_{window}h"] = roll.std().reset_index(level=0, drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Deviation from nominal ("how far from normal is this reading?")
# ---------------------------------------------------------------------------
def add_deviation_features(df):
    """
    Z-score of each sensor versus its machine TYPE's healthy baseline
    (from config): (reading - baseline) / baseline_noise.

    This is a strong, interpretable signal: it says "this vibration is 3
    standard deviations above where a healthy machine of this type sits",
    and it works the same across a hot roaster and a cool conveyor.
    """
    df = _sorted(df)
    for s in SENSORS:
        base = df["machine_type"].map(lambda t: config.MACHINE_BASELINES[t][s])
        noise = df["machine_type"].map(lambda t: config.MACHINE_BASELINES[t][s + "_noise"])
        df[f"{s}_zscore"] = (df[s] - base) / noise
    return df


# ---------------------------------------------------------------------------
# 4. Interactions ("two things bad at once is worse")
# ---------------------------------------------------------------------------
def add_interaction_features(df):
    """Simple products that capture 'high load AND high X' style stress."""
    df = _sorted(df)
    df["temp_x_load"] = df["temperature"] * df["machine_load"]
    df["vibration_x_rpm"] = df["vibration"] * df["rpm"]
    df["pressure_x_load"] = df["pressure"] * df["machine_load"]
    return df


# ---------------------------------------------------------------------------
# The exact list of columns the model is trained on.
# (Kept in one place so training and the dashboard always agree.)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # rates of change
    "temperature_change_1h", "temperature_change_6h",
    "vibration_change_1h", "pressure_change_1h",
    # trailing rolling stats
    "temperature_mean_6h", "temperature_std_6h",
    "vibration_mean_6h", "vibration_std_6h",
    "machine_load_mean_6h", "machine_load_std_6h",
    # deviation from a healthy machine of the same type
    "temperature_zscore", "vibration_zscore", "pressure_zscore",
    "rpm_zscore", "motor_current_zscore",
    # utilisation / maintenance history
    "operating_hours", "machine_load", "hours_since_maintenance",
    "maintenance_count", "previous_failures",
    # interactions
    "temp_x_load", "vibration_x_rpm", "pressure_x_load",
]

LABEL_COLUMN = "failure_within_window"


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------
def add_all_features(df):
    """Run every feature step in order and return the enriched DataFrame."""
    df = add_rate_features(df)
    df = add_rolling_features(df)
    df = add_deviation_features(df)
    df = add_interaction_features(df)
    return df


def build_feature_matrix(df):
    """
    Build all features and drop the first few hours of each machine, where the
    rolling/rate windows don't have enough history yet (they are NaN). We drop
    rather than fill so we never invent data.

    Returns the full DataFrame (features + label + id/time columns kept for
    splitting and for the dashboard).
    """
    df = add_all_features(df)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    dropped = before - len(df)
    print(f"Feature matrix: {len(df):,} rows "
          f"({dropped} warm-up rows dropped for missing history).")
    return df
