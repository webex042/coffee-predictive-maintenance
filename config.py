"""
config.py
=========
Central settings for the whole project.

Everything that another file might need to agree on lives here:
the random seed, the list of machines, each machine type's "normal"
sensor values, the prediction window, and the thresholds used by the
health score and maintenance rules.

Keeping it all in one place means the simulator, the model training,
and the dashboard all read the SAME numbers. If you want to change how
the factory behaves, you change it here and nowhere else.

NOTE: All equipment numbers below are made-up placeholders for a
simulated factory. They are NOT real Nestle / NESCAFE specifications.
"""

# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------
# A "seed" makes random numbers come out the same every time we run the code.
# That is what lets someone else reproduce our exact dataset and results.
SEED = 42

# ---------------------------------------------------------------------------
# 2. Simulation length
# ---------------------------------------------------------------------------
# We generate one sensor reading per machine, per hour.
# 180 days * 24 hours = 4320 readings per machine (~6 months of history).
SIMULATION_DAYS = 180
HOURS_PER_DAY = 24
TOTAL_HOURS = SIMULATION_DAYS * HOURS_PER_DAY

# The simulation clock starts here (just for nice-looking timestamps).
START_DATE = "2024-01-01 00:00:00"

# ---------------------------------------------------------------------------
# 3. Prediction window
# ---------------------------------------------------------------------------
# The model answers the question:
#   "Will this machine fail within the next 24 operating hours?"
PREDICTION_WINDOW_HOURS = 24

# ---------------------------------------------------------------------------
# 4. The machine roster
# ---------------------------------------------------------------------------
# 13 machines spread across 6 areas of a coffee production line.
# Each machine has a short ID (used everywhere) and a type.
MACHINES = [
    {"machine_id": "RST-01", "machine_type": "Roaster"},
    {"machine_id": "RST-02", "machine_type": "Roaster"},
    {"machine_id": "GRD-01", "machine_type": "Grinder"},
    {"machine_id": "GRD-02", "machine_type": "Grinder"},
    {"machine_id": "EXT-01", "machine_type": "Extraction"},
    {"machine_id": "EXT-02", "machine_type": "Extraction"},
    {"machine_id": "DRY-01", "machine_type": "SprayDryer"},
    {"machine_id": "DRY-02", "machine_type": "SprayDryer"},
    {"machine_id": "CNV-01", "machine_type": "Conveyor"},
    {"machine_id": "CNV-02", "machine_type": "Conveyor"},
    {"machine_id": "PKG-01", "machine_type": "Packaging"},
    {"machine_id": "PKG-02", "machine_type": "Packaging"},
    {"machine_id": "PKG-03", "machine_type": "Packaging"},
]

# ---------------------------------------------------------------------------
# 5. Type-specific baselines ("what normal looks like")
# ---------------------------------------------------------------------------
# For each machine TYPE we store the nominal (normal, healthy) value of each
# sensor, plus how much it naturally wobbles ("noise"). A roaster runs hot and
# slow; a grinder spins fast; a conveyor is cool and steady. These differences
# are what make each machine type look distinct in the data.
#
# Units (all invented for the simulation):
#   temperature   -> degrees C
#   vibration     -> mm/s
#   pressure      -> bar
#   rpm           -> revolutions per minute
#   motor_current -> amps
#
# "*_noise" is the standard deviation of the small random wobble added each hour.
MACHINE_BASELINES = {
    "Roaster": {
        "temperature": 210.0, "temperature_noise": 3.0,
        "vibration": 2.5, "vibration_noise": 0.3,
        "pressure": 1.5, "pressure_noise": 0.1,
        "rpm": 30.0, "rpm_noise": 1.0,
        "motor_current": 45.0, "motor_current_noise": 2.0,
    },
    "Grinder": {
        "temperature": 55.0, "temperature_noise": 2.0,
        "vibration": 4.0, "vibration_noise": 0.5,
        "pressure": 1.2, "pressure_noise": 0.1,
        "rpm": 1450.0, "rpm_noise": 20.0,
        "motor_current": 30.0, "motor_current_noise": 1.5,
    },
    "Extraction": {
        "temperature": 95.0, "temperature_noise": 2.5,
        "vibration": 3.0, "vibration_noise": 0.4,
        "pressure": 9.0, "pressure_noise": 0.4,
        "rpm": 120.0, "rpm_noise": 4.0,
        "motor_current": 38.0, "motor_current_noise": 2.0,
    },
    "SprayDryer": {
        "temperature": 180.0, "temperature_noise": 4.0,
        "vibration": 2.0, "vibration_noise": 0.3,
        "pressure": 3.5, "pressure_noise": 0.2,
        "rpm": 8000.0, "rpm_noise": 100.0,
        "motor_current": 60.0, "motor_current_noise": 3.0,
    },
    "Conveyor": {
        "temperature": 35.0, "temperature_noise": 1.5,
        "vibration": 1.5, "vibration_noise": 0.25,
        "pressure": 1.0, "pressure_noise": 0.05,
        "rpm": 200.0, "rpm_noise": 5.0,
        "motor_current": 15.0, "motor_current_noise": 1.0,
    },
    "Packaging": {
        "temperature": 40.0, "temperature_noise": 1.5,
        "vibration": 2.2, "vibration_noise": 0.35,
        "pressure": 2.0, "pressure_noise": 0.15,
        "rpm": 600.0, "rpm_noise": 10.0,
        "motor_current": 20.0, "motor_current_noise": 1.2,
    },
}

# ---------------------------------------------------------------------------
# 6. Health-score bands (used by the dashboard colours)
# ---------------------------------------------------------------------------
# The health score is a 0-100 number. These bands turn it into a label/colour.
HEALTH_BANDS = [
    {"name": "Healthy", "min_score": 80, "colour": "#2e7d32"},   # green
    {"name": "Monitor", "min_score": 60, "colour": "#f9a825"},   # amber
    {"name": "Warning", "min_score": 40, "colour": "#ef6c00"},   # orange
    {"name": "Critical", "min_score": 0, "colour": "#c62828"},   # red
]

# ---------------------------------------------------------------------------
# 7. Production-impact estimates (Phase 6)
# ---------------------------------------------------------------------------
# Rough "if this machine goes down" numbers, per machine type.
# units_per_hour  -> how many product units the machine handles per hour
# downtime_hours  -> typical hours lost to an unplanned failure + repair
# These are ESTIMATES / placeholders, clearly labelled as such in the UI.
PRODUCTION_IMPACT = {
    "Roaster":    {"units_per_hour": 1200, "downtime_hours": 8},
    "Grinder":    {"units_per_hour": 1500, "downtime_hours": 5},
    "Extraction": {"units_per_hour": 1000, "downtime_hours": 10},
    "SprayDryer": {"units_per_hour": 900,  "downtime_hours": 12},
    "Conveyor":   {"units_per_hour": 2000, "downtime_hours": 3},
    "Packaging":  {"units_per_hour": 1800, "downtime_hours": 4},
}

# ---------------------------------------------------------------------------
# 7b. Maintenance-rule thresholds (Phase 6 decision layer)
# ---------------------------------------------------------------------------
# The rule engine (src/maintenance_rules.py) turns the machine's current sensor
# state into a plain-language recommendation. It compares each sensor's
# deviation-from-normal (a "z-score": how many noise-widths above baseline the
# reading sits) against these cut-offs. All numbers are PLACEHOLDERS chosen to
# look sensible in the demo -- a real plant would tune them per machine.
#
#   z-score >= ELEVATED  -> worth a mention / keep an eye on it
#   z-score >= HIGH      -> recommend an actual inspection
RULE_Z_ELEVATED = 1.5    # sigma above normal: "slightly elevated"
RULE_Z_HIGH = 2.5        # sigma above normal: "clearly abnormal -> inspect"

# How many hours since the last maintenance we treat as "overdue" (placeholder).
RULE_HOURS_SINCE_MAINT_OVERDUE = 400

# ---------------------------------------------------------------------------
# 8. File paths
# ---------------------------------------------------------------------------
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sensor_data.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# The disclaimer shown in the README and the dashboard footer.
DISCLAIMER = (
    "Unaffiliated portfolio prototype using fully SIMULATED data. "
    "This is NOT an official Nestle / NESCAFE system and uses no real "
    "logos, branding, or proprietary specifications."
)
