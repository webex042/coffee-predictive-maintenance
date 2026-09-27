<<<<<<< HEAD
# Coffee Line — Predictive Maintenance System

Coffee Line is a machine learning project that predicts whether a machine in a simulated coffee production line is likely to fail within the next **24 operating hours**.

The project starts with simulated sensor data and takes it through the complete ML pipeline — feature engineering, model training, calibration, evaluation, scoring, and visualization. The final results are available through a **Streamlit dashboard**, where each machine can be inspected individually and the model's predictions can be explained using **SHAP**.

The data is generated locally using a physics-based simulator, so the project does not depend on any external dataset or industrial equipment.

## What the project does

* Predicts machine failure within the next 24 hours
* Uses temperature, vibration, pressure, RPM, and machine load
* Creates historical features without using future information
* Compares Logistic Regression, Random Forest, and XGBoost
* Uses a time-based train/test split instead of a random split
* Calibrates the final model's probabilities
* Generates a 0–100 machine health score
* Generates maintenance recommendations
* Ranks machines based on expected production loss
* Uses SHAP to explain individual predictions
* Provides an interactive Streamlit dashboard

## Pipeline

```text
Sensor Simulation
       ↓
Feature Engineering
       ↓
Time-based Model Training
       ↓
Probability Calibration
       ↓
Failure Prediction
       ↓
Health Score + Maintenance Recommendation
       ↓
Priority Ranking
       ↓
Streamlit Dashboard
```

The simulator generates **6 months of hourly data for 13 machines across 6 machine types**. Sensor readings include normal operating behavior, machine wear, environmental effects, and different fault conditions.

The feature engineering step creates **23 features** using historical sensor values. These include:

* 1-hour and 6-hour rate of change
* 6-hour rolling mean and standard deviation
* Deviation from machine-type baselines
* Sensor interaction features

The features are calculated using only information available up to the prediction time. Unit tests are included to check that future data does not leak into the feature calculations.

## Model Training

Three models are evaluated:

* Logistic Regression
* Random Forest
* XGBoost

The data is split chronologically, with a **24-hour purge gap** between training and testing data. This is important because the target represents failure within the following 24 hours.

The model selection process uses **PR-AUC**, which is more useful for this project because machine failures are relatively rare.

The selected model is then probability-calibrated using isotonic calibration. The final alert threshold is selected using **F-beta (β = 2)** to give more importance to recall.

The trained model and its metadata are stored in:

```text
models/model.joblib
```

## Machine Scoring

After prediction, the system adds a few additional layers on top of the ML output.

### Failure Probability

The calibrated probability that the machine will fail within the next 24 hours.

### Health Score

A separate **0–100 health score** based on sensor deviations and maintenance-related penalties. This score does not directly use the ML prediction.

### Maintenance Recommendation

A simple recommendation based on the machine's sensor condition and the main factors contributing to the prediction.

### Maintenance Priority

Machines are ranked using expected production loss:

```text
Expected Loss =
Failure Probability × Production Impact
```

This makes the priority list more useful than simply sorting machines by failure probability.

## Dashboard

The project includes a Streamlit dashboard designed as a small production control room.

### Line Overview

Shows the current condition of all machines and highlights machines with elevated failure risk.

### Machine Detail

For an individual machine, the dashboard shows:

* Current sensor readings
* Failure probability
* Health score
* Maintenance recommendation
* SHAP feature contributions
* 72-hour sensor trends
* 72-hour failure-probability trends

### Priority Queue

Displays machines ordered by expected production loss so that higher-impact risks can be reviewed first.

### CSV Upload

The dashboard also supports compatible user-provided CSV files using the same feature-engineering and scoring pipeline.

## Tech Stack

| Area             | Technology                  |
| ---------------- | --------------------------- |
| Language         | Python 3.13                 |
| Data Processing  | Pandas, NumPy               |
| Machine Learning | Scikit-learn, XGBoost       |
| Explainability   | SHAP                        |
| Dashboard        | Streamlit                   |
| Visualization    | Plotly, Matplotlib, Seaborn |
| Model Storage    | Joblib                      |
| Testing          | Pytest                      |

## Project Structure

```text
coffee-line/
│
├── config.py
├── requirements.txt
│
├── src/
│   ├── simulate.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── health.py
│   ├── maintenance_rules.py
│   ├── priority.py
│   └── scoring.py
│
├── dashboard/
│   ├── app.py
│   └── theme.py
│
├── tests/
│   └── test_features.py
│
├── models/
│   └── model.joblib
│
├── screenshots/
├── notebooks/
├── data/
└── .streamlit/
```

## Getting Started

### 1. Create a virtual environment

```bash
py -3.13 -m venv .venv
```

Activate it:

```bash
source .venv/Scripts/activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the data

```bash
python -m src.simulate
```

### 4. Train the model

```bash
python -m src.train_model
```

### 5. Start the dashboard

```bash
streamlit run dashboard/app.py
```

A trained model is already included in `models/model.joblib`, so you can also start directly with:

```bash
streamlit run dashboard/app.py
```

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

The tests check things such as feature causality, rate calculations, label leakage, and baseline z-scores.

The training script also includes additional checks for data leakage and the difference between random and time-based evaluation.

## Future Improvements

The current project uses simulated data. A real deployment could replace the simulator with data from an industrial historian or live sensor stream.

Possible next steps include:

* Real-time sensor ingestion
* Scheduled model scoring
* API-based predictions
* Model versioning
* Feature and prediction drift monitoring
* Integration with a maintenance management system

## Data

All sensor data and machine failures in this project are **synthetic** and generated by `src/simulate.py`.

The production-impact values used for maintenance prioritization are illustrative values defined in `config.py`.

The project is intended to demonstrate the complete workflow of building and evaluating a predictive-maintenance system rather than representing a real coffee production facility.

<img width="1109" height="1325" alt="architecture" src="https://github.com/user-attachments/assets/0cd2b825-8750-4bb4-90ed-4b66da8eb5fb" />

=======
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
>>>>>>> dbbd19b (Deployable project repo for Streamlit Community Cloud)
