"""
simulate.py  --  Phase 1: the physics-based sensor simulator (the core)
=======================================================================

This is the most important file in the project. It invents ~6 months of
hourly sensor data for a coffee production line, in a way that is
PHYSICALLY PLAUSIBLE rather than random:

  * Each machine has a "normal" operating point (from config.py).
  * Sensors wobble hour-to-hour as an autocorrelated process (today's
    reading is close to yesterday's) plus a daily ambient-temperature cycle.
  * Each machine slowly DEGRADES through a hidden fault mode. The tell-tale
    signs (rising vibration, creeping temperature, unstable RPM) appear
    GRADUALLY and get stronger as failure approaches.
  * When degradation crosses a noisy threshold the machine FAILS, then gets
    "repaired" (health reset) and the cycle starts again.

The label we want to predict is `failure_within_window`: will this machine
fail within the next 24 operating hours? Crucially, the warning signs are
visible in the data BEFORE that label turns 1 -- that is what makes the
problem learnable but not trivial.

Run it with:  python -m src.simulate
"""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Tuning knobs for how strong each fault's symptoms are.
# Values are "fraction of the baseline reading added when wear = 1 (at failure)".
# Bigger = easier to detect. These were tuned so a quick model beats the
# baseline but does NOT score a perfect ~1.0 (see the check at the bottom).
# ---------------------------------------------------------------------------
FAULT_MODES = ["bearing", "overheat", "pressure"]

# Daily ambient temperature cycle (feeds machine temperature a little).
AMBIENT_MEAN = 22.0   # average outside/plant temperature (deg C)
AMBIENT_AMP = 5.0     # how much it swings between night and mid-afternoon

# How "sticky" the normal wobble is (0 = white noise, 1 = never changes).
AR_PHI = 0.7

# One degradation cycle lasts this many hours (drawn randomly per cycle).
CYCLE_MIN_HOURS = 250
CYCLE_MAX_HOURS = 550

# Fraction of each cycle that stays healthy before degradation begins.
DWELL_MIN = 0.25
DWELL_MAX = 0.40

# How sharply wear accelerates near the end (higher = more sudden).
WEAR_POWER = 2.5


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------
def ambient_temperature(hour_index):
    """Daily temperature cycle. Peaks in mid-afternoon (hour 15), dips at night."""
    hour_of_day = hour_index % 24
    return AMBIENT_MEAN + AMBIENT_AMP * np.cos(2 * np.pi * (hour_of_day - 15) / 24)


def machine_load(hour_index, rng):
    """
    How hard the machine is working (0.4 - 1.0). Busier during the day shift.
    A little random noise is added so it is not perfectly smooth.
    """
    hour_of_day = hour_index % 24
    daily = 0.7 + 0.15 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
    value = daily + rng.normal(0, 0.03)
    return float(np.clip(value, 0.4, 1.0))


def wear_curve(progress, dwell):
    """
    Turn 'how far through the cycle we are' (0..1) into a wear level (0..~1).

    The machine stays healthy for the first `dwell` fraction, then wear grows
    with an accelerating curve so it climbs slowly at first and fast near the end.
    """
    if progress <= dwell:
        return 0.0
    adjusted = (progress - dwell) / (1.0 - dwell)   # rescale to 0..1
    return adjusted ** WEAR_POWER


# ---------------------------------------------------------------------------
# Apply a fault's symptoms to one hour's readings
# ---------------------------------------------------------------------------
def apply_degradation(mode, wear, load, readings, base, rng):
    """
    Given the current fault `mode` and `wear` level (0..1), nudge the healthy
    `readings` dict to show that fault's symptoms. `base` is the machine type's
    baseline dict. Effects grow with wear, so they are tiny early and obvious
    near failure. Each fault has a DIFFERENT signature (this is what the model
    learns).
    """
    if mode == "bearing":
        # Vibration is the first and biggest sign; temperature, current and
        # RPM instability follow.
        readings["vibration"] *= (1 + 1.5 * wear)
        readings["temperature"] += base["temperature"] * 0.06 * wear
        readings["motor_current"] *= (1 + 0.25 * wear)
        readings["rpm"] -= base["rpm"] * 0.03 * wear                     # slight slowdown
        readings["rpm"] += rng.normal(0, base["rpm"] * 0.02 * wear)      # instability

    elif mode == "overheat":
        # A cooling fault: temperature climbs, and climbs MORE under high load.
        readings["temperature"] += base["temperature"] * 0.15 * wear * (0.5 + load)
        readings["motor_current"] *= (1 + 0.10 * wear)
        readings["vibration"] *= (1 + 0.15 * wear)

    elif mode == "pressure":
        # Pressure drifts up under load and its variance (noisiness) grows.
        readings["pressure"] *= (1 + 0.40 * wear * load)
        readings["pressure"] += rng.normal(0, base["pressure"] * 0.30 * wear)
        readings["temperature"] += base["temperature"] * 0.04 * wear

    return readings


# ---------------------------------------------------------------------------
# Simulate the full history of ONE machine
# ---------------------------------------------------------------------------
def simulate_one_machine(machine_id, machine_type, rng):
    """Return a DataFrame: one row per hour for this machine's whole history."""
    base = config.MACHINE_BASELINES[machine_type]
    sensors = ["temperature", "vibration", "pressure", "rpm", "motor_current"]

    # Start each sensor at its baseline. We update these hour by hour (AR(1)).
    ar_state = {s: base[s] for s in sensors}

    # Degradation-cycle bookkeeping.
    def new_cycle():
        return (
            int(rng.integers(CYCLE_MIN_HOURS, CYCLE_MAX_HOURS)),   # target length
            rng.choice(FAULT_MODES),                               # fault mode
            float(rng.uniform(DWELL_MIN, DWELL_MAX)),              # healthy dwell
            float(np.clip(rng.normal(0.95, 0.04), 0.80, 1.05)),    # failure threshold
        )

    cycle_len, mode, dwell, threshold = new_cycle()
    hours_into_cycle = 0
    hours_since_maint = 0
    maintenance_count = 0
    previous_failures = 0

    rows = []
    for t in range(config.TOTAL_HOURS):
        load = machine_load(t, rng)
        ambient = ambient_temperature(t)

        # --- 1. Healthy reading: autocorrelated wobble around the baseline ---
        readings = {}
        for s in sensors:
            target = base[s]
            noise = rng.normal(0, base[s + "_noise"])
            ar_state[s] = target + AR_PHI * (ar_state[s] - target) + noise
            readings[s] = ar_state[s]
        # Ambient temperature nudges machine temperature up/down a little.
        readings["temperature"] += 0.3 * (ambient - AMBIENT_MEAN)

        # --- 2. Degradation: how worn is the machine right now? ---
        progress = hours_into_cycle / cycle_len
        wear = wear_curve(progress, dwell)
        readings = apply_degradation(mode, wear, load, readings, base, rng)

        # --- 3. Does it fail this hour? (noisy threshold on wear) ---
        failure_event = 1 if wear >= threshold else 0

        # --- 4. Record the row ---
        rows.append({
            "machine_id": machine_id,
            "machine_type": machine_type,
            "temperature": readings["temperature"],
            "vibration": readings["vibration"],
            "pressure": readings["pressure"],
            "rpm": readings["rpm"],
            "motor_current": readings["motor_current"],
            "machine_load": load,
            "operating_hours": t + 1,                 # total age, never resets
            "hours_since_maintenance": hours_since_maint,
            "maintenance_count": maintenance_count,
            "previous_failures": previous_failures,
            "failure_event": failure_event,
        })

        # --- 5. Advance time / handle repair ---
        if failure_event == 1:
            # Maintenance restores health and starts a fresh cycle.
            maintenance_count += 1
            previous_failures += 1
            hours_since_maint = 0
            hours_into_cycle = 0
            cycle_len, mode, dwell, threshold = new_cycle()
        else:
            hours_into_cycle += 1
            hours_since_maint += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forward-looking label: failure in the NEXT 24 operating hours?
# ---------------------------------------------------------------------------
def add_failure_window_label(df, window=config.PREDICTION_WINDOW_HOURS):
    """
    For each row, set failure_within_window = 1 if a failure_event happens in
    the next `window` hours for the SAME machine. We do this per machine so one
    machine's failure never leaks into another's rows.

    The 24 rows immediately BEFORE each failure become the positive examples.
    """
    df = df.sort_values(["machine_id", "operating_hours"]).reset_index(drop=True)
    df["failure_within_window"] = 0

    for machine_id, group in df.groupby("machine_id"):
        idx = group.index.to_numpy()
        fe = group["failure_event"].to_numpy()
        label = np.zeros(len(group), dtype=int)
        fail_positions = np.where(fe == 1)[0]
        for fp in fail_positions:
            start = max(0, fp - window)
            label[start:fp] = 1          # the `window` rows before the failure
        df.loc[idx, "failure_within_window"] = label

    return df


# ---------------------------------------------------------------------------
# Build the whole dataset
# ---------------------------------------------------------------------------
def simulate():
    """Simulate every machine, add timestamps + label, and save the CSV."""
    rng = np.random.default_rng(config.SEED)   # one seeded generator = reproducible

    frames = []
    for machine in config.MACHINES:
        frames.append(
            simulate_one_machine(machine["machine_id"], machine["machine_type"], rng)
        )
    df = pd.concat(frames, ignore_index=True)

    # Give every row a real timestamp (nice for plotting).
    start = pd.Timestamp(config.START_DATE)
    df["timestamp"] = df.groupby("machine_id")["operating_hours"].transform(
        lambda h: start + pd.to_timedelta(h - 1, unit="h")
    )

    df = add_failure_window_label(df)

    # Put columns in a friendly order.
    cols = [
        "timestamp", "machine_id", "machine_type",
        "temperature", "vibration", "pressure", "rpm", "motor_current",
        "machine_load", "operating_hours", "hours_since_maintenance",
        "maintenance_count", "previous_failures",
        "failure_event", "failure_within_window",
    ]
    df = df[cols]

    import os
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# Quick sanity checks (the Phase 1 "done when" gate)
# ---------------------------------------------------------------------------
def print_summary(df):
    """Print the numbers that tell us the simulation is realistic."""
    n = len(df)
    pos = int(df["failure_within_window"].sum())
    rate = 100 * pos / n
    n_failures = int(df["failure_event"].sum())

    print("\n===== SIMULATION SUMMARY =====")
    print(f"Rows (machine-hours):        {n:,}")
    print(f"Machines:                    {df['machine_id'].nunique()}")
    print(f"Failure events:              {n_failures}")
    print(f"Positive windows (label=1):  {pos:,}  ({rate:.2f}% of rows)")
    print(f"  -> target is 3-8%. {'OK' if 3 <= rate <= 8 else 'ADJUST tuning knobs'}")
    print("Failures per machine:")
    per = df.groupby("machine_id")["failure_event"].sum().astype(int)
    print(per.to_string())


def quick_leakability_check(df):
    """
    A fast logistic regression on RAW sensors. It should clearly beat the
    majority-class baseline (proving there is signal) but NOT score ~1.0
    (proving the signal is realistically hard, not synthetic-perfect).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("\n(scikit-learn not installed yet -- skipping the quick model check.)")
        return

    feats = ["temperature", "vibration", "pressure", "rpm", "motor_current",
             "machine_load", "hours_since_maintenance"]
    # Simple chronological split: first 70% train, last 30% test (per machine).
    df = df.sort_values(["machine_id", "operating_hours"])
    train = df.groupby("machine_id").head(int(config.TOTAL_HOURS * 0.7))
    test = df.drop(train.index)

    scaler = StandardScaler().fit(train[feats])
    X_tr, X_te = scaler.transform(train[feats]), scaler.transform(test[feats])
    y_tr, y_te = train["failure_within_window"], test["failure_within_window"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]

    baseline = y_te.mean()   # PR-AUC of random guessing = positive rate
    print("\n===== QUICK MODEL CHECK (raw sensors, logistic regression) =====")
    print(f"Test positive rate (PR-AUC baseline): {baseline:.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_te, prob):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_te, prob):.3f}")
    print("  -> want PR-AUC clearly above baseline, but ROC-AUC well below ~1.0.")


if __name__ == "__main__":
    print("Simulating sensor data... (this takes a few seconds)")
    data = simulate()
    print(f"Saved -> {config.RAW_DATA_PATH}")
    print_summary(data)
    quick_leakability_check(data)
