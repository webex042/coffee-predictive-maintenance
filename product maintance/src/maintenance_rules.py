"""
maintenance_rules.py  --  Phase 6: plain-language recommendations
=================================================================

The ML model says "how likely" a failure is; SHAP says "which features pushed
the prediction up". This file does the last, human-facing step: turn the
machine's current state into a SENTENCE a technician can act on, e.g.

    "Vibration is clearly abnormal (3.4 sigma above normal). Inspect bearings
     for wear, imbalance or misalignment."

How it works (deliberately simple and auditable):
  * Each sensor has a known failure story (see _SENSOR_ADVICE below).
  * We look at the per-type z-scores (how far each sensor has drifted).
  * The sensor with the biggest damaging deviation, if it clears the
    "inspect" threshold in config, drives the main recommendation.
  * We optionally take a hint from SHAP (the model's top driver) so the words
    line up with what the model actually reacted to.

All thresholds live in config.py and are clearly-labelled PLACEHOLDERS.
"""

import config

# Map each z-score feature to (friendly sensor name, what it usually means).
# This is the whole "knowledge base" -- easy to read, easy to extend.
_SENSOR_ADVICE = {
    "vibration_zscore": (
        "vibration",
        "Inspect bearings for wear, and check for imbalance or shaft misalignment.",
    ),
    "temperature_zscore": (
        "temperature",
        "Check cooling/airflow and lubrication; a rising temperature often means "
        "friction or a cooling fault.",
    ),
    "pressure_zscore": (
        "pressure",
        "Inspect seals, valves and lines for a pressure fault or blockage.",
    ),
    "motor_current_zscore": (
        "motor current",
        "Check the motor and drivetrain for extra load, binding or electrical issues.",
    ),
}

# SHAP feature names can be derived features (e.g. 'vibration_change_1h' or
# 'vibration_x_rpm'). This maps a driver back to its underlying sensor so the
# advice still makes sense.
_DRIVER_TO_SENSOR = {
    "vibration": "vibration_zscore",
    "temperature": "temperature_zscore",
    "pressure": "pressure_zscore",
    "motor_current": "motor_current_zscore",
    "rpm": "vibration_zscore",   # rpm instability shows up mechanically as vibration
}


def _severity_word(z):
    """Describe a z-score in words, using the config thresholds."""
    if z >= config.RULE_Z_HIGH:
        return "clearly abnormal"
    if z >= config.RULE_Z_ELEVATED:
        return "slightly elevated"
    return "normal"


def _sensor_from_driver(driver_name):
    """
    Given a SHAP driver feature name like 'vibration_change_1h', find which
    underlying sensor it belongs to. Returns a z-score column name or None.
    """
    if not driver_name:
        return None
    for keyword, zscore_col in _DRIVER_TO_SENSOR.items():
        if keyword in driver_name:
            return zscore_col
    return None


def recommend(row, top_driver=None):
    """
    Produce a recommendation for one machine-hour `row`.

    Returns a dict:
        {
          "headline":   short status line,
          "action":     what to physically go and do,
          "worst_sensor": which sensor drove it (friendly name),
          "worst_z":    that sensor's z-score,
          "urgent":     True if something clears the "inspect" threshold,
        }

    `top_driver` (optional) is the model's top SHAP feature name; if given and
    it points at a sensor, we prefer that sensor so the wording matches the
    model's reasoning. Otherwise we simply pick the most-deviated sensor.
    """
    # 1. Score every known sensor by how far it has drifted (damaging side only).
    deviations = {}
    for zscore_col in _SENSOR_ADVICE:
        deviations[zscore_col] = max(0.0, float(row.get(zscore_col, 0.0)))

    # 2. Decide which sensor to talk about.
    driver_sensor = _sensor_from_driver(top_driver)
    if driver_sensor and driver_sensor in deviations:
        worst_col = driver_sensor
    else:
        # pick the sensor with the largest damaging deviation
        worst_col = max(deviations, key=deviations.get)

    worst_z = deviations[worst_col]
    friendly_name, action_text = _SENSOR_ADVICE[worst_col]
    severity = _severity_word(worst_z)

    # 3. Build the words.
    if worst_z >= config.RULE_Z_HIGH:
        headline = (f"{friendly_name.capitalize()} is {severity} "
                    f"({worst_z:.1f} sigma above normal).")
        action = action_text
        urgent = True
    elif worst_z >= config.RULE_Z_ELEVATED:
        headline = (f"{friendly_name.capitalize()} is {severity} "
                    f"({worst_z:.1f} sigma above normal).")
        action = "Keep monitoring. " + action_text
        urgent = False
    else:
        headline = "All sensors are within normal range."
        action = "No action needed. Continue routine monitoring."
        urgent = False

    # 4. Overdue-maintenance nudge (added on top, doesn't override the above).
    hours = float(row.get("hours_since_maintenance", 0.0))
    if hours >= config.RULE_HOURS_SINCE_MAINT_OVERDUE:
        action += (f" Also note this machine is overdue for maintenance "
                   f"({int(hours)}h since last service).")

    return {
        "headline": headline,
        "action": action,
        "worst_sensor": friendly_name,
        "worst_z": worst_z,
        "urgent": urgent,
    }


# Quick manual check:  python -m src.maintenance_rules
if __name__ == "__main__":
    example = {"vibration_zscore": 3.4, "temperature_zscore": 0.5,
               "pressure_zscore": 0.2, "motor_current_zscore": 1.0,
               "hours_since_maintenance": 120}
    rec = recommend(example)
    print(rec["headline"])
    print(rec["action"])
