"""
MFBI calculator — same formula as lib/student/mfbi.ts.

MFBI = (SL_n + AW_n + ST_n + SH_n) / 4

Normalization (0–1), higher = greater burnout risk:
- Stress (PSS-10): 0–40
- Academic workload: 0–10
- Study time: 0–25 hours/week
- Sleep risk: 0–100 (already risk-oriented)
"""

from __future__ import annotations

STUDY_TIME_SCORE_MAX = 25.0


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def round4(value: float) -> float:
    return round(value * 10000) / 10000


def round_mfbi_score(score: float) -> float:
    return round(score * 100) / 100


def classify_mfbi_score(score: float) -> str:
    """0.00–0.39 Low · 0.40–0.69 Moderate · 0.70–1.00 High."""
    normalized = round_mfbi_score(score)
    if normalized <= 0.39:
        return "Low"
    if normalized <= 0.69:
        return "Moderate"
    return "High"


def compute_mfbi(
    stress_score: float,
    academic_workload: float,
    study_time: float,
    sleep_risk: float,
) -> dict:
    normalized_stress = clamp01(stress_score / 40.0)
    normalized_academic_workload = clamp01(academic_workload / 10.0)
    normalized_study_time = clamp01(study_time / STUDY_TIME_SCORE_MAX)
    normalized_sleep_hours = clamp01(sleep_risk / 100.0)

    mfbi_score = round4(
        (
            normalized_stress
            + normalized_academic_workload
            + normalized_study_time
            + normalized_sleep_hours
        )
        / 4.0
    )

    return {
        "normalized_stress": round4(normalized_stress),
        "normalized_academic_workload": round4(normalized_academic_workload),
        "normalized_study_time": round4(normalized_study_time),
        "normalized_sleep_hours": round4(normalized_sleep_hours),
        "mfbi_score": mfbi_score,
        "burnout_risk_level": classify_mfbi_score(mfbi_score),
    }


def qualitative_factor_label(normalized: float) -> str:
    return classify_mfbi_score(normalized)


def pss_stress_label(stress_score: float) -> str:
    if stress_score <= 13:
        return "Low"
    if stress_score <= 26:
        return "Moderate"
    return "High"
