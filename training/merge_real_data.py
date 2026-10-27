"""Merge real weekly monitoring rows into training CSVs.

Reads dataset/real_weekly_monitoring.csv and syncs:
- training_dataset.csv (same-week rows for Week 1 and Week 2 only)
- next_week_dataset.csv (Week 1 to Week 2 pairs with trends)

Usage (from burnout-ai/):
  py training/merge_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mfbi.calculator import classify_mfbi_score  # noqa: E402
DATASET_DIR = ROOT / "dataset"
REAL_PATH = DATASET_DIR / "real_weekly_monitoring.csv"
TRAIN_PATH = DATASET_DIR / "training_dataset.csv"
NEXT_PATH = DATASET_DIR / "next_week_dataset.csv"

RISK_ORDER = ["Low", "Moderate", "High"]
TRAIN_COLS = [
    "source",
    "student_id",
    "term_id",
    "week_number",
    "stress_score",
    "academic_workload_score",
    "study_time_score",
    "sleep_hours_score",
    "mfbi_score",
    "burnout_risk_level",
]
NEXT_COLS = [
    "source",
    "stress_score",
    "academic_workload_score",
    "study_time_score",
    "sleep_hours_score",
    "stress_trend",
    "workload_trend",
    "study_trend",
    "sleep_trend",
    "mfbi_score",
    "burnout_risk_level",
    "source_week_number",
    "next_week_number",
]


def _load_real() -> pd.DataFrame:
    if not REAL_PATH.exists():
        raise FileNotFoundError(f"Missing {REAL_PATH}")

    real = pd.read_csv(REAL_PATH)
    required = [
        "student_id",
        "term_id",
        "week_number",
        "stress_score",
        "academic_workload_score",
        "study_time_score",
        "sleep_hours_score",
        "mfbi_score",
        "burnout_risk_level",
    ]
    missing = [col for col in required if col not in real.columns]
    if missing:
        raise ValueError(f"real_weekly_monitoring.csv missing columns: {missing}")

    real = real.dropna(subset=required).copy()
    real["week_number"] = pd.to_numeric(real["week_number"], errors="coerce")
    real = real[real["week_number"].isin([1, 2])].copy()
    real["week_number"] = real["week_number"].astype(int)
    # The label is always the band of the stored MFBI score.
    real["burnout_risk_level"] = real["mfbi_score"].astype(float).map(classify_mfbi_score)
    real = real[real["burnout_risk_level"].isin(RISK_ORDER)].copy()
    real = real.drop_duplicates(["student_id", "term_id", "week_number"], keep="last")
    real["source"] = "real"
    return real


def _build_next_week_rows(real: pd.DataFrame) -> pd.DataFrame:
    """One row per student and term: week N inputs, week N+1 MFBI band as the label.

    Inputs come only from week N and the week before it. Nothing from week N+1
    is used except the label.
    """
    rows: list[dict] = []

    grouped = real.sort_values(["student_id", "term_id", "week_number"]).groupby(
        ["student_id", "term_id"], sort=False
    )

    for _, student_df in grouped:
        by_week = {int(row["week_number"]): row for _, row in student_df.iterrows()}
        for week, current in by_week.items():
            target = by_week.get(week + 1)
            if target is None:
                continue
            prior = by_week.get(week - 1)

            def trend(column: str) -> float:
                if prior is None:
                    return 0.0
                return float(current[column]) - float(prior[column])

            rows.append(
                {
                    "source": "real",
                    "stress_score": float(current["stress_score"]),
                    "academic_workload_score": float(current["academic_workload_score"]),
                    "study_time_score": float(current["study_time_score"]),
                    "sleep_hours_score": float(current["sleep_hours_score"]),
                    "stress_trend": trend("stress_score"),
                    "workload_trend": trend("academic_workload_score"),
                    "study_trend": trend("study_time_score"),
                    "sleep_trend": trend("sleep_hours_score"),
                    "mfbi_score": float(current["mfbi_score"]),
                    "burnout_risk_level": classify_mfbi_score(float(target["mfbi_score"])),
                    "source_week_number": week,
                    "next_week_number": week + 1,
                }
            )

    return pd.DataFrame(rows, columns=NEXT_COLS)


def _replace_real_rows(existing: pd.DataFrame) -> pd.DataFrame:
    # The published evaluation must follow the real monitoring records only.
    # Synthetic and balance rows are dropped so they cannot change the metrics.
    if existing.empty:
        return existing
    return existing.iloc[0:0].copy()


def merge() -> dict:
    real = _load_real()
    next_real = _build_next_week_rows(real)

    train_existing = pd.read_csv(TRAIN_PATH)
    next_existing = pd.read_csv(NEXT_PATH)

    train_merged = pd.concat(
        [_replace_real_rows(train_existing), real[TRAIN_COLS]],
        ignore_index=True,
    )
    next_merged = pd.concat(
        [_replace_real_rows(next_existing), next_real],
        ignore_index=True,
    )

    train_merged.to_csv(TRAIN_PATH, index=False)
    next_merged.to_csv(NEXT_PATH, index=False)

    return {
        "real_rows": len(real),
        "next_week_pairs": len(next_real),
        "training_total": len(train_merged),
        "next_week_total": len(next_merged),
        "real_label_counts": real["burnout_risk_level"].value_counts().to_dict(),
        "next_label_counts": next_real["burnout_risk_level"].value_counts().to_dict()
        if len(next_real)
        else {},
    }


def main() -> None:
    stats = merge()
    print("Merged real data into training CSVs:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\nNext: npm run train")


if __name__ == "__main__":
    main()
