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

