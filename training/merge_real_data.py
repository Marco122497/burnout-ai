"""Merge real weekly monitoring rows into training CSVs.

Reads dataset/real_weekly_monitoring.csv and syncs:
- training_dataset.csv (same-week rows)
- next_week_dataset.csv (consecutive-week pairs with trends)

Usage (from burnout-ai/):
  py training/merge_real_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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
    real = real[real["burnout_risk_level"].isin(RISK_ORDER)].copy()
    real["source"] = "real"
    return real


def _build_next_week_rows(real: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = real.sort_values(["student_id", "term_id", "week_number"]).groupby(
        ["student_id", "term_id"], sort=False
    )

    for _, student_df in grouped:
        weeks = student_df.reset_index(drop=True)
        for i in range(len(weeks) - 1):
            current = weeks.iloc[i]
            nxt = weeks.iloc[i + 1]

            # Only use directly consecutive weeks (e.g. 2 -> 3, not 2 -> 4).
            if int(nxt["week_number"]) != int(current["week_number"]) + 1:
                continue

            prev = weeks.iloc[i - 1] if i > 0 else None
            if prev is not None and int(current["week_number"]) != int(prev["week_number"]) + 1:
                prev = None

            rows.append(
                {
                    "source": "real",
                    "stress_score": float(current["stress_score"]),
                    "academic_workload_score": float(current["academic_workload_score"]),
                    "study_time_score": float(current["study_time_score"]),
                    "sleep_hours_score": float(current["sleep_hours_score"]),
                    "stress_trend": float(current["stress_score"] - prev["stress_score"])
                    if prev is not None
                    else 0.0,
                    "workload_trend": float(
                        current["academic_workload_score"] - prev["academic_workload_score"]
                    )
                    if prev is not None
                    else 0.0,
                    "study_trend": float(current["study_time_score"] - prev["study_time_score"])
                    if prev is not None
                    else 0.0,
                    "sleep_trend": float(
                        current["sleep_hours_score"] - prev["sleep_hours_score"]
                    )
                    if prev is not None
                    else 0.0,
                    "mfbi_score": float(current["mfbi_score"]),
                    "burnout_risk_level": nxt["burnout_risk_level"],
                    "next_week_number": int(nxt["week_number"]),
                }
            )

    return pd.DataFrame(rows, columns=NEXT_COLS)


def _replace_real_rows(existing: pd.DataFrame) -> pd.DataFrame:
    if "source" not in existing.columns:
        return existing
    return existing[existing["source"] != "real"].copy()


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
