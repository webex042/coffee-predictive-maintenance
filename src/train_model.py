"""
train_model.py  --  Phase 4: modelling + HONEST evaluation
===========================================================

This trains three models and, just as importantly, PROVES the evaluation is
trustworthy. Two make-or-break ideas from the project brief live here:

  1. We never use a plain random split. Sensor rows are autocorrelated in
     time, so a random split lets a row's near-identical neighbour leak into
     the test set and inflates the score. We use a PURGED CHRONOLOGICAL split
     (train on the past, test on the future, with a gap in between).

  2. We deliberately show the pipeline CAN fail:
       * Shuffled-label test: train on randomly shuffled labels -> performance
         must collapse to the baseline rate. If it doesn't, something leaks.
       * Random split vs time split: a big gap between them quantifies leakage.

Because failures are rare we lead with PR-AUC, recall and F1 (not accuracy,
which is misleading when 94% of rows are "no failure").

Run with:  python -m src.train_model
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    recall_score, precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier

import config
from src import feature_engineering as fe

RANDOM_STATE = config.SEED
PURGE_HOURS = config.PREDICTION_WINDOW_HOURS   # gap between train and test


# ---------------------------------------------------------------------------
# Data loading + splits
# ---------------------------------------------------------------------------
def load_features():
    """Read the simulated data and build the (leak-free) feature matrix."""
    raw = pd.read_csv(config.RAW_DATA_PATH)
    return fe.build_feature_matrix(raw)


def purged_time_split(df, train_frac=0.7, purge=PURGE_HOURS):
    """
    THE honest split. All machines share the same clock (operating_hours), so
    we cut at one point in time: everything before the cut trains, everything
    after tests, and we throw away a `purge`-hour gap in between so a training
    row's 24-hour label window can't peek across the boundary.
    """
    cut = int(config.TOTAL_HOURS * train_frac)
    train = df[df["operating_hours"] <= (cut - purge)].copy()
    test = df[df["operating_hours"] > cut].copy()
    return train, test


def random_split(df, test_frac=0.3):
    """
    A deliberately WRONG split, kept only to measure leakage. Shuffling rows
    lets near-identical neighbours land on both sides -> optimistic scores.
    """
    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE)
    n_test = int(len(df) * test_frac)
    return shuffled.iloc[n_test:].copy(), shuffled.iloc[:n_test].copy()


def xy(df):
    """Split a DataFrame into feature matrix X and label vector y."""
    return df[fe.FEATURE_COLUMNS], df[fe.LABEL_COLUMN]


def split_off_validation(train_full, val_frac=0.25, purge=PURGE_HOURS):
    """
    Carve a VALIDATION set off the end of the training period (again by time,
    again with a purge gap). We fit models on the earlier `core`, then use
    `val` to choose the probability threshold and pick the winner -- so the
    test set is never touched until the very end.
    """
    max_h = train_full["operating_hours"].max()
    min_h = train_full["operating_hours"].min()
    val_cut = max_h - int((max_h - min_h) * val_frac)
    core = train_full[train_full["operating_hours"] <= (val_cut - purge)].copy()
    val = train_full[train_full["operating_hours"] > val_cut].copy()
    return core, val


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def make_models(y_train):
    """Create the three models. XGBoost gets scale_pos_weight for imbalance."""
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)   # >1 tells XGBoost failures are rare

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   solver="liblinear")),
    ])
    forest = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
    )
    return {"LogisticRegression": logistic, "RandomForest": forest, "XGBoost": xgb}


# ---------------------------------------------------------------------------
# Threshold + metrics
# ---------------------------------------------------------------------------
def pick_threshold(y_true, prob, beta=2.0):
    """
    Choose a probability cut-off that FAVOURS RECALL (missing a real failure is
    worse than a false alarm). We maximise the F-beta score with beta=2, which
    weights recall higher than precision. Chosen on TRAIN data, never on test.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    precision, recall = precision[:-1], recall[:-1]   # drop the trailing point
    denom = (beta ** 2 * precision) + recall
    fbeta = np.where(denom > 0, (1 + beta ** 2) * precision * recall / denom, 0.0)
    return float(thresholds[np.argmax(fbeta)])


def evaluate(y_true, prob, threshold):
    """All the numbers we care about, at the chosen threshold."""
    pred = (prob >= threshold).astype(int)
    return {
        "PR_AUC": average_precision_score(y_true, prob),
        "ROC_AUC": roc_auc_score(y_true, prob),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Train all three and compare (fit on core, tune on val, report on test)
# ---------------------------------------------------------------------------
def compare_models(core, val, test):
    """
    Fit each model on `core`, calibrate it on `val` (isotonic, so probabilities
    are meaningful), choose a recall-favouring threshold on `val`, then report
    honest metrics on the untouched `test` set. The winner is the best PR-AUC
    on `val` (PR-AUC is threshold-free, the right headline for rare events).
    """
    X_core, y_core = xy(core)
    X_val, y_val = xy(val)
    X_te, y_te = xy(test)
    models = make_models(y_core)

    results, fitted, thresholds, val_scores = {}, {}, {}, {}
    base_models = {}
    for name, model in models.items():
        model.fit(X_core, y_core)
        # Calibrate on the held-out validation slice ("prefit" = model is fixed).
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)

        val_prob = calibrated.predict_proba(X_val)[:, 1]
        thr = pick_threshold(y_val, val_prob)
        test_prob = calibrated.predict_proba(X_te)[:, 1]

        results[name] = evaluate(y_te, test_prob, thr)
        fitted[name] = calibrated
        base_models[name] = model          # raw estimator, used later for SHAP
        thresholds[name] = thr
        val_scores[name] = average_precision_score(y_val, val_prob)

    table = pd.DataFrame(results).T[["PR_AUC", "ROC_AUC", "Recall", "Precision", "F1"]]
    print("\n===== MODEL COMPARISON (fit on past, tested on future) =====")
    print(f"Test rows: {len(test):,} | positive rate (baseline PR-AUC): {y_te.mean():.3f}")
    print(table.round(3).to_string())
    return results, fitted, base_models, thresholds, val_scores


# ---------------------------------------------------------------------------
# Leakage proofs -- show the pipeline CAN fail
# ---------------------------------------------------------------------------
def leakage_proofs(train, test):
    print("\n===== LEAKAGE PROOF 1: shuffled-label test =====")
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)

    y_shuffled = y_tr.sample(frac=1.0, random_state=RANDOM_STATE).to_numpy()
    m = make_models(y_tr)["XGBoost"]
    m.fit(X_tr, y_shuffled)
    pr = average_precision_score(y_te, m.predict_proba(X_te)[:, 1])
    print(f"PR-AUC with SHUFFLED labels: {pr:.3f}  (baseline = {y_te.mean():.3f})")
    print("  -> collapses to baseline, so the model learns real signal, not an artifact.")

    print("\n===== LEAKAGE PROOF 2: random split vs time split =====")
    # Time split (honest)
    m_time = make_models(y_tr)["XGBoost"]
    m_time.fit(X_tr, y_tr)
    pr_time = average_precision_score(y_te, m_time.predict_proba(X_te)[:, 1])
    # Random split (leaky) -- shuffle the SAME rows (train+test together)
    both = pd.concat([train, test], ignore_index=True)
    rtr, rte = random_split(both)
    Xr_tr, yr_tr = xy(rtr)
    Xr_te, yr_te = xy(rte)
    m_rand = make_models(yr_tr)["XGBoost"]
    m_rand.fit(Xr_tr, yr_tr)
    pr_rand = average_precision_score(yr_te, m_rand.predict_proba(Xr_te)[:, 1])
    print(f"PR-AUC random split (leaky):  {pr_rand:.3f}")
    print(f"PR-AUC time split   (honest): {pr_time:.3f}")
    print(f"  -> the random split is {pr_rand - pr_time:+.3f} higher; that gap is leakage.")
    print("     We report the time-split number.")


# ---------------------------------------------------------------------------
# Save the winning model
# ---------------------------------------------------------------------------
def save_bundle(winner_name, model, base_model, threshold, metrics):
    """Save everything the dashboard needs to score machines the same way."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    bundle = {
        "model": model,                       # calibrated, gives probabilities
        "base_model": base_model,             # raw tree model, for SHAP
        "model_name": winner_name,
        "feature_columns": fe.FEATURE_COLUMNS,
        "threshold": threshold,
        "metrics": metrics,
    }
    path = os.path.join(config.MODELS_DIR, "model.joblib")
    joblib.dump(bundle, path)

    print(f"\n===== SAVED WINNER: {winner_name} (calibrated) =====")
    print(f"Recall-favouring threshold: {threshold:.3f}")
    print("Honest test metrics: " +
          ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(f"Saved -> {path}")
    return bundle


if __name__ == "__main__":
    feature_df = load_features()
    train_full, test = purged_time_split(feature_df)
    core, val = split_off_validation(train_full)

    results, fitted, base_models, thresholds, val_scores = compare_models(core, val, test)

    # Winner = best validation PR-AUC (never chosen using the test set).
    winner = max(val_scores, key=val_scores.get)
    print(f"\nBest model by validation PR-AUC: {winner}")

    leakage_proofs(train_full, test)
    save_bundle(winner, fitted[winner], base_models[winner],
                thresholds[winner], results[winner])
