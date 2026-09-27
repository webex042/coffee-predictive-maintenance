"""
health.py  --  Phase 6: a transparent 0-100 "health score"
===========================================================

Why a second number when we already have the ML model?

  * The ML model outputs P(failure in next 24h). That is powerful but it is a
    black box to a plant operator -- "0.31" doesn't obviously mean anything.
  * The HEALTH SCORE is the opposite: a simple, hand-written 0-100 gauge that
    anyone can audit. Higher = healthier. It is built ONLY from how far the
    machine's sensors have drifted from normal, plus how overdue maintenance is.

Keeping them separate is deliberate and honest: the health score is a
rule-based summary you can explain in one sentence, and the ML probability is
the learned prediction. The dashboard shows both. We never feed the ML
probability into the health score (or vice-versa) -- that would blur the line
between "what the sensors say right now" and "what the model predicts".

The score maps onto the colour BANDS defined in config.HEALTH_BANDS
(Healthy / Monitor / Warning / Critical).
"""

import config

# Which sensor deviations feed the score, and how much each one hurts.
# We use the per-machine-type z-scores built in feature_engineering.py: a
# z-score is "how many noise-widths above normal this reading is". We only
# penalise deviations in the DAMAGING direction (too hot, too much vibration,
# too much pressure) -- running slightly cool is not a health problem.
_DEVIATION_SENSORS = ["vibration_zscore", "temperature_zscore", "pressure_zscore"]

# Tuning knobs (placeholders, easy to read):
#   each sigma of average deviation removes this many health points ...
_POINTS_PER_SIGMA = 13.0
#   ... but the sensor part can never remove more than this many points.
_MAX_DEVIATION_PENALTY = 70.0
#   overdue maintenance can remove up to this many points on top.
_MAX_MAINTENANCE_PENALTY = 15.0


def _clip(value, low, high):
    """Keep a number inside [low, high]. (Plain helper, no numpy needed.)"""
    return max(low, min(high, value))


def deviation_penalty(row):
    """
    Points lost because the sensors have drifted from normal.

    We take the z-scores for vibration / temperature / pressure, ignore any
    that are below zero (cooler / calmer than normal is fine), average the rest,
    and turn that average deviation into a penalty. A machine sitting ~5 sigma
    hot-and-shaking will hit the cap; a perfectly normal machine loses nothing.
    """
    positive_deviations = []
    for sensor in _DEVIATION_SENSORS:
        z = row.get(sensor, 0.0)
        positive_deviations.append(max(0.0, float(z)))

    average_sigma = sum(positive_deviations) / len(positive_deviations)
    penalty = average_sigma * _POINTS_PER_SIGMA
    return _clip(penalty, 0.0, _MAX_DEVIATION_PENALTY)


def maintenance_penalty(row):
    """
    Points lost because the machine is overdue for maintenance.

    Grows linearly with hours-since-maintenance up to the "overdue" mark in
    config, then flattens out at the cap. This is a gentle nudge, not the main
    driver -- sensor drift matters far more than the calendar.
    """
    hours = float(row.get("hours_since_maintenance", 0.0))
    fraction_overdue = hours / config.RULE_HOURS_SINCE_MAINT_OVERDUE
    penalty = fraction_overdue * _MAX_MAINTENANCE_PENALTY
    return _clip(penalty, 0.0, _MAX_MAINTENANCE_PENALTY)


def health_score(row):
    """
    The headline number: 100 (perfect) down to 0 (very unhealthy).

    score = 100 - sensor-drift penalty - overdue-maintenance penalty

    `row` is one machine-hour of data (a pandas Series or a plain dict) that
    already has the z-score features from feature_engineering.py.
    """
    score = 100.0 - deviation_penalty(row) - maintenance_penalty(row)
    return _clip(score, 0.0, 100.0)


def health_band(score):
    """
    Turn a 0-100 score into its band dict {name, min_score, colour}.

    HEALTH_BANDS is sorted high-to-low, so we return the first band whose
    min_score the value clears. The last band has min_score 0, so this always
    returns something.
    """
    for band in config.HEALTH_BANDS:
        if score >= band["min_score"]:
            return band
    return config.HEALTH_BANDS[-1]   # safety net (shouldn't be reached)


def health_label(score):
    """Convenience: just the band NAME (e.g. 'Warning') for a score."""
    return health_band(score)["name"]


# A tiny self-test so you can eyeball the behaviour with:  python -m src.health
if __name__ == "__main__":
    print("Perfectly normal machine:")
    normal = {"vibration_zscore": 0.0, "temperature_zscore": 0.0,
              "pressure_zscore": 0.0, "hours_since_maintenance": 10}
    s = health_score(normal)
    print(f"  score={s:.1f} -> {health_label(s)}")

    print("Badly degraded machine (hot + shaking, overdue):")
    bad = {"vibration_zscore": 4.0, "temperature_zscore": 3.0,
           "pressure_zscore": 1.0, "hours_since_maintenance": 500}
    s = health_score(bad)
    print(f"  score={s:.1f} -> {health_label(s)}")
