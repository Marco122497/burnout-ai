"""Export weekly monitoring rows from Supabase into real_weekly_monitoring.csv.

Requires burnout-ai/.env or repo-root .env.local with:
  SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_ROLE_KEY

Usage (from burnout-ai/):
  py training/export_real_data.py
  py training/export_real_data.py && py training/merge_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Allow `from app.database import ...` when run as a script from repo root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "dataset"
OUT_PATH = DATASET_DIR / "real_weekly_monitoring.csv"

COLUMNS = [
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


def export_rows() -> pd.DataFrame:
    from app.database import supabase

    response = (
        supabase.table("weekly_monitoring")
        .select(
            "student_id, term_id, week_number, stress_score, "
            "academic_workload_score, study_time_score, sleep_hours_score, "
            "mfbi_results(mfbi_score, burnout_risk_level)"
        )
        .order("student_id")
        .order("week_number")
        .execute()
    )

    rows: list[dict] = []
    for row in response.data or []:
        mfbi = row.get("mfbi_results")
        if isinstance(mfbi, list):
            mfbi = mfbi[0] if mfbi else None
        if not mfbi or mfbi.get("mfbi_score") is None or not mfbi.get("burnout_risk_level"):
            continue

        rows.append(
            {
                "source": "real",
                "student_id": row["student_id"],
                "term_id": row["term_id"],
                "week_number": row["week_number"],
                "stress_score": float(row["stress_score"]),
                "academic_workload_score": float(row["academic_workload_score"]),
                "study_time_score": float(row["study_time_score"]),
                "sleep_hours_score": float(row["sleep_hours_score"]),
                "mfbi_score": float(mfbi["mfbi_score"]),
                "burnout_risk_level": mfbi["burnout_risk_level"],
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    df = export_rows()
    if df.empty:
        print("No completed weekly monitoring rows with MFBI results found.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Exported {len(df)} rows to {OUT_PATH}")
    print(df["burnout_risk_level"].value_counts().to_dict())


if __name__ == "__main__":
    main()
