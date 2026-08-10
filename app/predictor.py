from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SAME_WEEK_FEATURES = joblib.load(MODELS_DIR / "same_week_features.pkl")
NEXT_WEEK_FEATURES = joblib.load(MODELS_DIR / "next_week_features.pkl")

same_week_decision_tree = joblib.load(MODELS_DIR / "same_week_decision_tree.pkl")
same_week_random_forest = joblib.load(MODELS_DIR / "same_week_random_forest.pkl")
next_week_decision_tree = joblib.load(MODELS_DIR / "next_week_decision_tree.pkl")
next_week_random_forest = joblib.load(MODELS_DIR / "next_week_random_forest.pkl")


def _confidence(model, frame: pd.DataFrame) -> float:
    probabilities = model.predict_proba(frame)[0]
    return round(float(max(probabilities)) * 100, 2)


def _risk_score(model, frame: pd.DataFrame) -> float:
    """Weighted risk score in [0, 1] from class probabilities."""
    classes = list(model.classes_)
    probs = model.predict_proba(frame)[0]
    weights = {"Low": 0.2, "Moderate": 0.55, "High": 0.9, "Severe": 0.95}
    score = 0.0
    for cls, prob in zip(classes, probs):
        score += float(prob) * weights.get(str(cls), 0.5)
    return round(score, 4)


def _run_pair(decision_tree, random_forest, features: list[str], values: dict):
    missing = [name for name in features if values.get(name) is None]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    frame = pd.DataFrame([{name: values[name] for name in features}])

    dt_prediction = str(decision_tree.predict(frame)[0])
    rf_prediction = str(random_forest.predict(frame)[0])
    dt_confidence = _confidence(decision_tree, frame)
    rf_confidence = _confidence(random_forest, frame)
    risk_score = _risk_score(random_forest, frame)

    selected_model = (
        "Random Forest"
        if rf_confidence >= dt_confidence
        else "Decision Tree"
    )
    final_prediction = (
        rf_prediction if selected_model == "Random Forest" else dt_prediction
    )

    return {
        "decision_tree_prediction": dt_prediction,
        "decision_tree_confidence": dt_confidence,
        "random_forest_prediction": rf_prediction,
        "random_forest_confidence": rf_confidence,
        "final_prediction": final_prediction,
        "risk_level": final_prediction,
        "risk_score": risk_score,
        "selected_model": selected_model,
    }


def predict_same_week(features: dict) -> dict:
    result = _run_pair(
        same_week_decision_tree,
        same_week_random_forest,
        SAME_WEEK_FEATURES,
        features,
    )
    result["model_version"] = "burnout-ai-same-week-v1"
    result["horizon"] = "same_week"
    result["remarks"] = (
        f"Same-week assessment; selected {result['selected_model']}"
    )
    return result


def predict_next_week(features: dict) -> dict:
    result = _run_pair(
        next_week_decision_tree,
        next_week_random_forest,
        NEXT_WEEK_FEATURES,
        features,
    )
    result["model_version"] = "burnout-ai-next-week-v1"
    result["horizon"] = "next_week"
    result["remarks"] = (
        f"Next-week early detection; selected {result['selected_model']}"
    )
    return result


def models_ready() -> bool:
    required = [
        "same_week_decision_tree.pkl",
        "same_week_random_forest.pkl",
        "same_week_features.pkl",
        "next_week_decision_tree.pkl",
        "next_week_random_forest.pkl",
        "next_week_features.pkl",
    ]
    return all((MODELS_DIR / name).exists() for name in required)


def predict_burnout(
    stress_level,
    academic_workload,
    study_time,
    sleep_hours,
):
    """Backward-compatible helper used by older callers. """
    result = predict_same_week(
        {
            "stress_score": stress_level,
            "academic_workload_score": academic_workload,
            "study_time_score": study_time,
            "sleep_hours_score": sleep_hours,
        }
    )
    return {
        "risk": result["final_prediction"],
        "confidence": result["random_forest_confidence"],
        "risk_score": result["risk_score"],
    }
