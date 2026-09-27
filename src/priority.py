"""
priority.py  --  Phase 6: which machine should we fix FIRST?
============================================================

A plant has limited maintenance crews. If three machines look risky, which one
costs us the most to ignore? This file answers that by combining two things:

    risk    = P(failure in next 24h)          <- from the ML model
    impact  = how much production we lose if it fails
            = units_per_hour * downtime_hours  <- from config.PRODUCTION_IMPACT

We rank machines by the EXPECTED number of product units lost:

    expected_units_lost = risk * impact

A machine with a 20% failure chance that would cost 10,000 units if it went
down (2,000 expected units) outranks a 90% machine that only costs 800 units
(720 expected units). That is the whole idea: chase expected cost, not just
probability. All impact numbers are clearly-labelled ESTIMATES in the config.
"""

import config


def impact_units(machine_type):
    """
    Worst-case units lost to one unplanned failure of this machine type:
    units_per_hour * downtime_hours. Pulled straight from config.
    """
    impact = config.PRODUCTION_IMPACT.get(machine_type)
    if impact is None:
        return 0.0
    return float(impact["units_per_hour"] * impact["downtime_hours"])


def expected_units_lost(failure_probability, machine_type):
    """Risk-weighted production loss = P(failure) * worst-case units lost."""
    return float(failure_probability) * impact_units(machine_type)


def build_priority_table(snapshot_rows):
    """
    Rank machines most-urgent first.

    `snapshot_rows` is a list of dicts, one per machine, each with at least:
        machine_id, machine_type, failure_probability
    (usually the latest reading per machine -- see src/scoring.py).

    Returns a new list of dicts sorted by expected_units_lost (descending),
    each enriched with impact_units, expected_units_lost and a 1-based rank.
    """
    enriched = []
    for row in snapshot_rows:
        prob = float(row.get("failure_probability", 0.0))
        mtype = row.get("machine_type", "")
        item = dict(row)   # copy so we don't mutate the caller's data
        item["impact_units"] = impact_units(mtype)
        item["expected_units_lost"] = expected_units_lost(prob, mtype)
        enriched.append(item)

    # Highest expected loss first.
    enriched.sort(key=lambda r: r["expected_units_lost"], reverse=True)

    # Add a human-friendly rank (1 = fix this first).
    for rank, item in enumerate(enriched, start=1):
        item["priority_rank"] = rank

    return enriched


# Quick manual check:  python -m src.priority
if __name__ == "__main__":
    demo = [
        {"machine_id": "RST-01", "machine_type": "Roaster",    "failure_probability": 0.20},
        {"machine_id": "CNV-01", "machine_type": "Conveyor",   "failure_probability": 0.90},
        {"machine_id": "DRY-01", "machine_type": "SprayDryer", "failure_probability": 0.35},
    ]
    for item in build_priority_table(demo):
        print(f"#{item['priority_rank']} {item['machine_id']:<7} "
              f"P(fail)={item['failure_probability']:.2f}  "
              f"expected units lost={item['expected_units_lost']:,.0f}")
