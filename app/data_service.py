from app.database import supabase


def get_student_history(student_id: str):
    response = (
        supabase.table("weekly_monitoring")
        .select(
            "student_id, week_number, stress_score, academic_workload_score, "
            "study_time_score, sleep_hours_score, "
            "mfbi_results(mfbi_score, burnout_risk_level)"
        )
        .eq("student_id", student_id)
        .order("week_number")
        .execute()
    )

    rows = []
    for row in response.data or []:
        mfbi = row.get("mfbi_results")
        if isinstance(mfbi, list):
            mfbi = mfbi[0] if mfbi else None

        rows.append(
            {
                "student_id": row["student_id"],
                "week": row["week_number"],
                "stress_score": float(row["stress_score"]),
                "academic_workload_score": float(row["academic_workload_score"]),
                "study_time_score": float(row["study_time_score"]),
                "sleep_hours_score": float(row["sleep_hours_score"]),
                "mfbi_score": float(mfbi["mfbi_score"]) if mfbi and mfbi.get("mfbi_score") is not None else None,
                "burnout_risk_level": (
                    mfbi.get("burnout_risk_level") if mfbi else None
                ),
                # Legacy aliases
                "stress_level": float(row["stress_score"]),
                "academic_workload": float(row["academic_workload_score"]),
                "study_time": float(row["study_time_score"]),
                "sleep_hours": float(row["sleep_hours_score"]),
            }
        )

    return rows


def build_features_from_history(history: list[dict]) -> dict:
    """Build same-week / next-week feature dict from ordered weekly history."""
    latest = history[-1]
    previous = history[-2] if len(history) >= 2 else None

    features = {
        "stress_score": latest["stress_score"],
        "academic_workload_score": latest["academic_workload_score"],
        "study_time_score": latest["study_time_score"],
        "sleep_hours_score": latest["sleep_hours_score"],
        "mfbi_score": latest.get("mfbi_score"),
        "stress_trend": 0.0,
        "workload_trend": 0.0,
        "study_trend": 0.0,
        "sleep_trend": 0.0,
        "has_prior_week": previous is not None,
    }

    if previous:
        features["stress_trend"] = (
            latest["stress_score"] - previous["stress_score"]
        )
        features["workload_trend"] = (
            latest["academic_workload_score"]
            - previous["academic_workload_score"]
        )
        features["study_trend"] = (
            latest["study_time_score"] - previous["study_time_score"]
        )
        features["sleep_trend"] = (
            latest["sleep_hours_score"] - previous["sleep_hours_score"]
        )

    return features
