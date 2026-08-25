from __future__ import annotations

RISK_RANK = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 2}


def risk_rank(level: str | None) -> int:
    if not level:
        return 0
    return RISK_RANK.get(level, 0)


def classify_trend(history_levels: list[str], mfbi_scores: list[float]) -> str:
    """Classify longitudinal movement using available weekly levels/MFBI."""
    if len(history_levels) < 2 and len(mfbi_scores) < 2:
        return "insufficient_history"

    if len(mfbi_scores) >= 2:
        delta = mfbi_scores[-1] - mfbi_scores[-2]
        if delta >= 0.08:
            return "increasing"
        if delta <= -0.08:
            return "decreasing"

    if len(history_levels) >= 2:
        a = risk_rank(history_levels[-2])
        b = risk_rank(history_levels[-1])
        if b > a:
            return "increasing"
        if b < a:
            return "decreasing"

    return "stable"


def project_week2(
    next_week_risk: str,
    trend: str,
    stress_trend: float,
    workload_trend: float,
    sleep_trend: float,
) -> dict:
    """
    Trend-based week-2 outlook.

    The training dataset labels next-week risk only. This projection is NOT a
    trained two-week-ahead model — it extrapolates direction from trends.
    """
    score = risk_rank(next_week_risk)

    pressure = 0
    if stress_trend > 0:
        pressure += 1
    if workload_trend > 0:
        pressure += 1
    if sleep_trend > 0:  # higher sleep risk = worse
        pressure += 1

    if trend == "increasing" or pressure >= 2:
        score = min(2, score + 1)
        basis = "trend_escalation"
    elif trend == "decreasing" and pressure == 0:
        score = max(0, score - 1)
        basis = "trend_improvement"
    else:
        basis = "hold_next_week"

    level = ["Low", "Moderate", "High"][score]
    return {
        "risk_level": level,
        "basis": basis,
        "is_ml_prediction": False,
        "note": (
            "Week-2 outlook is a trend-based early-warning projection, "
            "not a trained two-week-ahead model."
        ),
    }


def build_warning_message(
    current_risk: str,
    next_week_risk: str | None,
    week2_risk: str | None,
    trend: str,
) -> str | None:
    if not next_week_risk:
        return None

    escalating = risk_rank(next_week_risk) > risk_rank(current_risk) or (
        week2_risk is not None and risk_rank(week2_risk) > risk_rank(current_risk)
    )
    high_ahead = next_week_risk == "High" or week2_risk == "High"

    if escalating or high_ahead or trend == "increasing":
        return (
            "Your recent academic stress and workload patterns indicate an "
            "increasing burnout risk. Consider monitoring your workload, "
            "study habits, and sleep. This is an early-warning indicator, "
            "not a medical diagnosis."
        )

    if current_risk == "High":
        return (
            "Your current burnout risk is elevated. Review guidance "
            "recommendations and consider contacting the Guidance Office."
        )

    return None
