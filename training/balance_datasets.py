"""Balance training CSVs by oversampling minority burnout risk classes.

Creates synthetic variants of Low/High rows (small noise) until each class
matches the majority class count. Backs up originals as *.bak.csv.

Usage (from burnout-ai/):
  py training/balance_datasets.py

Or from repo root after this script exists:
  py burnout-ai/training/balance_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RNG = np.random.default_rng(42)

RISK_ORDER = ["Low", "Moderate", "High"]

TRAIN_NUMERIC = [
    "stress_score",
    "academic_workload_score",
    "study_time_score",
    "sleep_hours_score",
    "mfbi_score",
]
NEXT_NUMERIC = [
    "stress_score",
    "academic_workload_score",
    "study_time_score",
    "sleep_hours_score",
    "stress_trend",
    "workload_trend",
    "study_trend",
    "sleep_trend",
    "mfbi_score",
]

# Clip synthetic values to realistic questionnaire / MFBI ranges.
CLIP = {
    "stress_score": (0.0, 40.0),
    "academic_workload_score": (0.0, 10.0),
    "study_time_score": (0.0, 16.0),
    "sleep_hours_score": (0.0, 100.0),
    "mfbi_score": (0.0, 1.0),
    "stress_trend": (-40.0, 40.0),
    "workload_trend": (-10.0, 10.0),
    "study_trend": (-16.0, 16.0),
    "sleep_trend": (-100.0, 100.0),
}

# Relative noise scale per feature (fraction of class std, with floor).
NOISE_SCALE = 0.15


def _backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"Backup: {bak.name}")
    else:
        print(f"Backup exists: {bak.name}")
    return bak


def _synthesize(class_df: pd.DataFrame, numeric_cols: list[str], n_extra: int) -> pd.DataFrame:
    if n_extra <= 0 or class_df.empty:
        return class_df.iloc[0:0].copy()

    base = class_df.sample(n=n_extra, replace=True, random_state=42).copy()
    std = class_df[numeric_cols].std(numeric_only=True).fillna(0.0)

    for col in numeric_cols:
        scale = max(float(std.get(col, 0.0)) * NOISE_SCALE, 1e-3)
        noise = RNG.normal(0.0, scale, size=len(base))
        values = base[col].astype(float).to_numpy() + noise
        lo, hi = CLIP.get(col, (None, None))
        if lo is not None and hi is not None:
            values = np.clip(values, lo, hi)
        # Keep mfbi and scores to 4 decimals like existing CSVs.
        if col in {"mfbi_score"}:
            values = np.round(values, 4)
        elif col.endswith("_trend"):
            values = np.round(values, 4)
        else:
            values = np.round(values, 2)
        base[col] = values

    if "source" in base.columns:
        base["source"] = "synthetic_balance"
    if "student_id" in base.columns:
        base["student_id"] = [f"synthetic-{i:05d}" for i in range(n_extra)]
    if "term_id" in base.columns:
        base["term_id"] = base["term_id"].fillna(1).astype(int)
    if "week_number" in base.columns:
        # Keep week in a plausible 1–20 range.
        weeks = base["week_number"].fillna(1).astype(float).to_numpy()
        weeks = np.clip(np.round(weeks + RNG.integers(-1, 2, size=n_extra)), 1, 20)
        base["week_number"] = weeks.astype(int)
    if "next_week_number" in base.columns:
        weeks = base["next_week_number"].fillna(2).astype(float).to_numpy()
        weeks = np.clip(np.round(weeks + RNG.integers(-1, 2, size=n_extra)), 2, 21)
        base["next_week_number"] = weeks

    return base


def balance_csv(path: Path, numeric_cols: list[str], label_col: str = "burnout_risk_level") -> None:
    print(f"\n=== Balancing {path.name} ===")
    _backup(path)
    df = pd.read_csv(path)
    df = df.dropna(subset=numeric_cols + [label_col]).copy()
    df = df[df[label_col].isin(RISK_ORDER)].copy()

    counts = df[label_col].value_counts()
    print("Before:", counts.to_dict())
    target = int(counts.max())

    pieces = [df]
    for label in RISK_ORDER:
        class_df = df[df[label_col] == label]
        need = target - len(class_df)
        if need > 0:
            syn = _synthesize(class_df, numeric_cols, need)
            pieces.append(syn)
            print(f"  +{need} synthetic rows for {label}")

    balanced = pd.concat(pieces, ignore_index=True)
    # Shuffle for training friendliness
    balanced = balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
    balanced.to_csv(path, index=False)
    print("After:", balanced[label_col].value_counts().to_dict())
    print(f"Saved {path} ({len(balanced)} rows)")


def main() -> None:
    train_path = DATASET_DIR / "training_dataset.csv"
    next_path = DATASET_DIR / "next_week_dataset.csv"
    if not train_path.exists() or not next_path.exists():
        raise FileNotFoundError("Missing training_dataset.csv or next_week_dataset.csv")

    balance_csv(train_path, TRAIN_NUMERIC)
    balance_csv(next_path, NEXT_NUMERIC)
    print("\nDone. Run: npm run train")


if __name__ == "__main__":
    main()
