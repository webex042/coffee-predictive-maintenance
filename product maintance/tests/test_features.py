"""
Unit tests for the feature functions.

The most important one is `test_no_lookahead`: it PROVES the features are
causal. If a feature accidentally used future data, computing it on a longer
history would change its value for an early row -- this test would catch that.

Run with:  python -m pytest -q
"""

import numpy as np
import pandas as pd

from src import feature_engineering as fe


def _fake_machine(n_hours, seed=0):
    """Build a small, single-machine DataFrame with reproducible readings."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "machine_id": ["RST-01"] * n_hours,
        "machine_type": ["Roaster"] * n_hours,
        "operating_hours": np.arange(1, n_hours + 1),
        "temperature": 210 + rng.normal(0, 3, n_hours).cumsum() * 0.1,
        "vibration": 2.5 + rng.normal(0, 0.3, n_hours),
        "pressure": 1.5 + rng.normal(0, 0.1, n_hours),
        "rpm": 30 + rng.normal(0, 1, n_hours),
        "motor_current": 45 + rng.normal(0, 2, n_hours),
        "machine_load": np.clip(0.7 + rng.normal(0, 0.05, n_hours), 0.4, 1.0),
        # maintenance/utilisation columns carried straight through from raw data
        "hours_since_maintenance": np.arange(1, n_hours + 1),
        "maintenance_count": np.zeros(n_hours, dtype=int),
        "previous_failures": np.zeros(n_hours, dtype=int),
    })


def test_no_lookahead():
    """
    Features for the early hours must NOT change when we append future hours.
    We compute features on 30 hours and on the first 60 hours of the SAME data,
    then check the overlapping early rows match exactly.
    """
    full = _fake_machine(60, seed=42)
    short = full.iloc[:30].copy()

    f_full = fe.add_all_features(full)
    f_short = fe.add_all_features(short)

    # Compare hours 7..30 (after the 6-hour warm-up) on the feature columns.
    a = f_full[f_full["operating_hours"].between(7, 30)][fe.FEATURE_COLUMNS].reset_index(drop=True)
    b = f_short[f_short["operating_hours"].between(7, 30)][fe.FEATURE_COLUMNS].reset_index(drop=True)

    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_rate_feature_is_simple_difference():
    """temperature_change_1h should equal this hour's temp minus last hour's."""
    df = _fake_machine(10, seed=1)
    out = fe.add_rate_features(df)
    expected = df["temperature"].iloc[5] - df["temperature"].iloc[4]
    assert np.isclose(out["temperature_change_1h"].iloc[5], expected)


def test_labels_are_not_features():
    """The answer columns must never sneak into the model's inputs."""
    assert "failure_event" not in fe.FEATURE_COLUMNS
    assert "failure_within_window" not in fe.FEATURE_COLUMNS


def test_zscore_zero_at_baseline():
    """A reading exactly at the baseline should have z-score ~0."""
    df = _fake_machine(8, seed=2)
    df["vibration"] = 2.5  # exactly the Roaster baseline from config
    out = fe.add_deviation_features(df)
    assert np.allclose(out["vibration_zscore"], 0.0)
