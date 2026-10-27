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


def _real_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop synthetic balance copies when a source column is present."""
    if "source" not in df.columns:
        return df
    source = df["source"].astype(str)
    return df[source.eq("real") | source.str.startswith("real")].copy()


def _can_holdout(y: pd.Series) -> bool:
    """A stratified 20% split needs a test row and a train row in every class."""
    if len(y) < 5:
        return False
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return False
    try:
        train_test_split(y, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        return False
    return True


def _train_pair(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    feature_names: list[str],
    class_weight: str | None = "balanced",
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=8, random_state=42, class_weight=class_weight
    )
    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
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
        "holdout_rows": int(len(X_test)),
        "features": feature_names,
        "decision_tree": dt_metrics,
        "random_forest": {
            **rf_metrics,
            "feature_importance": rf_importance,
        },
    }


def _train_all_pairs(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    feature_names: list[str],
    class_weight: str | None = None,
):
    """Fit on every pair when a stratified hold-out is not possible."""
    decision_tree = DecisionTreeClassifier(
        max_depth=8, random_state=42, class_weight=class_weight
    )
    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
    )
    decision_tree.fit(X, y)
    random_forest.fit(X, y)

    joblib.dump(decision_tree, MODELS_DIR / f"{name}_decision_tree.pkl")
    joblib.dump(random_forest, MODELS_DIR / f"{name}_random_forest.pkl")
    joblib.dump(feature_names, MODELS_DIR / f"{name}_features.pkl")

    print(f"\n=== {name} ===")
    print(f"Trained on all {len(X)} pairs.")
    print("No hold-out: too few pairs for a stratified 20% split.")

    return {
        "samples": int(len(X)),
        "holdout_rows": 0,
        "features": feature_names,
        "note": (
            "Trained on every real consecutive-week pair. "
            "Too few pairs for a stratified 20% hold-out, so there is no test matrix."
        ),
    }


def main():
    same_week_path = DATASET_DIR / "training_dataset.csv"
    next_week_path = DATASET_DIR / "next_week_dataset.csv"

    if not same_week_path.exists():
        raise FileNotFoundError(f"Missing dataset: {same_week_path}")
    if not next_week_path.exists():
        raise FileNotFoundError(f"Missing dataset: {next_week_path}")

    same_df = _real_rows(pd.read_csv(same_week_path))
    next_df = _real_rows(pd.read_csv(next_week_path))

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

    same_week = _train_pair(
        same_df[SAME_WEEK_FEATURES],
        same_df[LABEL],
        "same_week",
        SAME_WEEK_FEATURES,
    )
    # Week 1 alone has no next-week label. Train that model only when
    # Week 1 to Week 2 pairs exist.
    next_week = None
    if _can_holdout(next_df[LABEL]):
        next_week = _train_pair(
            next_df[NEXT_WEEK_FEATURES],
            next_df[LABEL],
            "next_week",
            NEXT_WEEK_FEATURES,
            class_weight=None,
        )
    elif len(next_df) >= 2 and next_df[LABEL].nunique() >= 2:
        next_week = _train_all_pairs(
            next_df[NEXT_WEEK_FEATURES],
            next_df[LABEL],
            "next_week",
            NEXT_WEEK_FEATURES,
            class_weight=None,
        )
    else:
        # Do not delete a next-week model trained on an earlier Week 2 set.
        # A new pair file replaces it on the next successful train.
        kept = all(
            (MODELS_DIR / name).exists()
            for name in (
                "next_week_decision_tree.pkl",
                "next_week_random_forest.pkl",
                "next_week_features.pkl",
            )
        )
        print("\n=== next_week ===")
        print("Skipped: consecutive-week pairs are not available yet.")
        print(
            "Kept the existing next-week model files."
            if kept
            else "No next-week model files are on disk."
        )
    evaluation = {
        "generated_by": "burnout-ai/training/train.py",
        "label": LABEL,
        "classes": RISK_ORDER,
        "same_week": same_week,
        "next_week": next_week,
        "week1_week2_holdout": None
        if next_week is None or "decision_tree" not in next_week
        else {
            "pairs": next_week["samples"],
            "holdout_rows": next_week["holdout_rows"],
            "real_records": int(len(same_df)),
            "decision_tree": next_week["decision_tree"],
            "random_forest": {
                key: value
                for key, value in next_week["random_forest"].items()
                if key != "feature_importance"
            },
        },
        "notes": {
            "same_week": "Predicts current-week burnout risk from four questionnaire scores. Real monitoring rows only.",
            "next_week": (
                "Predicts the next week's risk from the current week's scores, trends, and MFBI. "
                "Pairs are real consecutive weeks: Week 1 to Week 2, and Week 2 to Week 3. "
                "No synthetic rows. The API week-2 outlook is still a trend-based projection, "
                "not a trained two-week-ahead model."
            ),
        },
    }

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
