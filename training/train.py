"""Train same-week and next-week Decision Tree + Random Forest models.

Usage (from burnout-ai/):
  py training/train.py

Or from repo root:
  npm run train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SAME_WEEK_FEATURES = [
    "stress_score",
    "academic_workload_score",
    "study_time_score",
    "sleep_hours_score",
]
NEXT_WEEK_FEATURES = [
    *SAME_WEEK_FEATURES,
    "stress_trend",
    "workload_trend",
    "study_trend",
    "sleep_trend",
    "mfbi_score",
]
LABEL = "burnout_risk_level"
RISK_ORDER = ["Low", "Moderate", "High"]


def _metrics(y_true, y_pred, labels=RISK_ORDER) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
    }


def _train_pair(X: pd.DataFrame, y: pd.Series, name: str, feature_names: list[str]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    decision_tree = DecisionTreeClassifier(max_depth=8, random_state=42)
    random_forest = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    dt_pred = decision_tree.predict(X_test)
    rf_pred = random_forest.predict(X_test)

    dt_metrics = _metrics(y_test, dt_pred)
    rf_metrics = _metrics(y_test, rf_pred)
    rf_importance = {
        feature: round(float(score), 4)
        for feature, score in zip(
            feature_names, random_forest.feature_importances_.tolist()
        )
    }

    prefix = name
    joblib.dump(decision_tree, MODELS_DIR / f"{prefix}_decision_tree.pkl")
    joblib.dump(random_forest, MODELS_DIR / f"{prefix}_random_forest.pkl")
    joblib.dump(feature_names, MODELS_DIR / f"{prefix}_features.pkl")

    # Keep legacy model.pkl as same-week RF for older callers.
    if name == "same_week":
        joblib.dump(random_forest, MODELS_DIR / "model.pkl")

    print(f"\n=== {name} ===")
    print(f"Decision Tree accuracy: {dt_metrics['accuracy']}")
    print(f"Random Forest accuracy: {rf_metrics['accuracy']}")

    return {
        "samples": int(len(X)),
        "features": feature_names,
        "decision_tree": dt_metrics,
        "random_forest": {
            **rf_metrics,
            "feature_importance": rf_importance,
        },
    }


def main():
    same_week_path = DATASET_DIR / "training_dataset.csv"
    next_week_path = DATASET_DIR / "next_week_dataset.csv"

    if not same_week_path.exists():
        raise FileNotFoundError(f"Missing dataset: {same_week_path}")
    if not next_week_path.exists():
        raise FileNotFoundError(f"Missing dataset: {next_week_path}")

    same_df = pd.read_csv(same_week_path)
    next_df = pd.read_csv(next_week_path)

    for col in SAME_WEEK_FEATURES + [LABEL]:
        if col not in same_df.columns:
            raise ValueError(f"training_dataset.csv missing column: {col}")

    for col in NEXT_WEEK_FEATURES + [LABEL]:
        if col not in next_df.columns:
            raise ValueError(f"next_week_dataset.csv missing column: {col}")

    # Drop incomplete rows
    same_df = same_df.dropna(subset=SAME_WEEK_FEATURES + [LABEL]).copy()
    next_df = next_df.dropna(subset=NEXT_WEEK_FEATURES + [LABEL]).copy()

    # Keep only supported labels
    same_df = same_df[same_df[LABEL].isin(RISK_ORDER)]
    next_df = next_df[next_df[LABEL].isin(RISK_ORDER)]

    evaluation = {
        "generated_by": "burnout-ai/training/train.py",
        "label": LABEL,
        "classes": RISK_ORDER,
        "same_week": _train_pair(
            same_df[SAME_WEEK_FEATURES],
            same_df[LABEL],
            "same_week",
            SAME_WEEK_FEATURES,
        ),
        "next_week": _train_pair(
            next_df[NEXT_WEEK_FEATURES],
            next_df[LABEL],
            "next_week",
            NEXT_WEEK_FEATURES,
        ),
        "notes": {
            "same_week": "Predicts current-week burnout risk from four questionnaire scores.",
            "next_week": (
                "Predicts next-week burnout risk from scores, week-over-week trends, "
                "and MFBI. Dataset contains next-week labels only — not two-week-ahead "
                "labels. Week-2 outlook in the API is a trend-based projection."
            ),
        },
    }

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
