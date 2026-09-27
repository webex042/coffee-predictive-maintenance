# Predictive Maintenance for a Coffee Production Line (Simulated)

> **Disclaimer:** Unaffiliated portfolio prototype using fully **simulated** data.
> This is **NOT** an official Nestlé / NESCAFÉ system and uses no real logos,
> branding, or proprietary specifications. All equipment values are invented
> placeholders for demonstration only.

Predict machine failures on a simulated coffee production line **24 operating
hours before they happen**, so maintenance can be scheduled instead of reacting
to breakdowns.

---

## The problem

Unplanned equipment failure is expensive: lost production, rushed repairs, and
wasted product. If we can spot the early warning signs in sensor data (rising
vibration, creeping temperature, unstable RPM) we can flag a machine *before* it
fails and fix it on our own schedule.

## The solution

1. A **physics-based simulator** generates ~6 months of hourly sensor readings
   for 13 machines. Machines slowly degrade, occasionally fail, get repaired,
   and degrade again — so the data contains realistic, *learnable* failure
   precursors (not random noise, and not trivially perfect signals).
2. **Causal (past-only) feature engineering** turns raw sensors into trends,
   rolling statistics, and deviation-from-normal scores — with **no data leakage**.
3. Three models (Logistic Regression → Random Forest → XGBoost) are trained with
   **time-aware / grouped splits** and evaluated honestly (PR-AUC, recall, F1).
4. **SHAP** explains *why* each machine is flagged.
5. A **Streamlit dashboard** shows the line status, per-machine detail, trends,
   explanations, maintenance recommendations, a priority queue, and impact
   estimates.

## Key capabilities

- Line overview with per-machine health colours
- Per-machine sensor trends (24 / 48 / 72 h)
- Failure-probability + independent 0–100 health score
- Plain-language "why is this machine at risk?" explanation (SHAP)
- Rule-based maintenance recommendations
- Priority queue weighted by production impact
- Estimated units-lost-if-it-fails

## How to run it

```bash
# 1. Create and activate a virtual environment (Python 3.13)
py -3.13 -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows PowerShell/cmd

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the data -> train the model -> launch the dashboard
python -m src.simulate            # writes data/raw/sensor_data.csv
python -m src.train_model         # writes models/
streamlit run dashboard/app.py
```

## Project structure

```
product-maintenance/
├── config.py                 # all shared settings (seed, machines, thresholds)
├── requirements.txt          # pinned dependencies
├── data/
│   ├── raw/                  # simulated sensor_data.csv (generated)
│   └── processed/            # feature matrix (generated)
├── src/
│   ├── simulate.py           # Phase 1: the physics-based simulator (the core)
│   ├── feature_engineering.py# Phase 3: causal, past-only features
│   ├── train_model.py        # Phase 4: training + honest evaluation
│   ├── health.py             # Phase 6: transparent 0–100 health score
│   ├── maintenance_rules.py  # Phase 6: rule-based recommendations
│   └── priority.py           # Phase 6: priority queue + impact estimate
├── notebooks/                # 01 EDA, 02 features, 03 modeling, 04 SHAP
├── models/                   # saved model + scaler + feature list
├── dashboard/app.py          # Streamlit dashboard
├── tests/                    # unit tests for feature functions + leakage checks
└── screenshots/              # dashboard screenshots for this README
```

## Honesty / no-leakage checks

This project deliberately proves its own evaluation can fail:

- **Shuffled-label test** — train on randomly shuffled labels; a real pipeline
  collapses to the baseline rate. If it doesn't, there's a leak.
- **Random split vs time split** — a large gap between them quantifies leakage.
  We rely on the time-aware split.

_Status: work in progress — phases are being built up incrementally._
