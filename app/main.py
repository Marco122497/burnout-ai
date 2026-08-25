import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.data_service import build_features_from_history, get_student_history
from app.predictor import models_ready, predict_next_week, predict_same_week
from app.schemas import EarlyWarningRequest, PredictionRequest
from services.early_warning import (
    build_warning_message,
    classify_trend,
    project_week2,
)

app = FastAPI(title="Academic Burnout AI")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _features_from_request(request: PredictionRequest | EarlyWarningRequest) -> dict:
    if request.features is not None:
        payload = request.features.model_dump()
        has_trends = all(
            payload.get(key) is not None
            for key in (
                "stress_trend",
                "workload_trend",
                "study_trend",
                "sleep_trend",
            )
        )
        return {
            **payload,
            "stress_trend": payload.get("stress_trend") if payload.get("stress_trend") is not None else 0.0,
            "workload_trend": payload.get("workload_trend") if payload.get("workload_trend") is not None else 0.0,
            "study_trend": payload.get("study_trend") if payload.get("study_trend") is not None else 0.0,
            "sleep_trend": payload.get("sleep_trend") if payload.get("sleep_trend") is not None else 0.0,
            "has_prior_week": has_trends,
        }

    history = get_student_history(request.student_id)
    if not history:
        raise HTTPException(status_code=404, detail="No weekly data found")
    return build_features_from_history(history)


def _history_from_request(request: PredictionRequest | EarlyWarningRequest):
    levels = list(request.history_levels or [])
    mfbi = list(request.history_mfbi or [])

    if request.student_id and (not levels or not mfbi):
        history = get_student_history(request.student_id)
        if history:
            if not levels:
                levels = [
                    row.get("burnout_risk_level") or "Moderate"
                    for row in history
                ]
            if not mfbi:
                mfbi = [
                    float(row["mfbi_score"])
                    for row in history
                    if row.get("mfbi_score") is not None
                ]

    return levels, mfbi


def _can_run_next_week(features: dict) -> bool:
    return bool(
        features.get("has_prior_week")
        and features.get("mfbi_score") is not None
    )


def _enrich(prediction: dict, features: dict) -> dict:
    return {
        **prediction,
        "mfbi": features.get("mfbi_score"),
        "risk_level": prediction.get("risk_level")
        or prediction.get("final_prediction"),
    }


@app.get("/")
def home():
    return {
        "message": "Burnout AI is running",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "early_warning": "POST /predict/early-warning",
            "metrics": "GET /metrics",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    ready = models_ready()
    return {
        "status": "ok" if ready else "degraded",
        "models_ready": ready,
        "service": "burnout-ai",
    }


@app.get("/metrics")
def metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run npm run train first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(request: PredictionRequest):
    if not models_ready():
        raise HTTPException(status_code=503, detail="Model files missing")

    try:
        features = _features_from_request(request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    mode = request.mode
    if mode == "auto":
        mode = "next_week" if _can_run_next_week(features) else "same_week"

    try:
        same_week = _enrich(predict_same_week(features), features)
        next_week = None
        if _can_run_next_week(features):
            next_week = _enrich(predict_next_week(features), features)

        if mode == "next_week":
            if features.get("mfbi_score") is None:
                raise HTTPException(
                    status_code=400,
                    detail="next_week mode requires mfbi_score",
                )
            if not next_week:
                raise HTTPException(
                    status_code=400,
                    detail="next_week mode requires prior-week trends",
                )
            prediction = next_week
        else:
            prediction = same_week
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "success": True,
        "student_id": request.student_id,
        "mode": mode,
        "mfbi": features.get("mfbi_score"),
        "risk_level": prediction["risk_level"],
        "risk_score": prediction["risk_score"],
        "random_forest_prediction": prediction["random_forest_prediction"],
        "decision_tree_prediction": prediction["decision_tree_prediction"],
        "prediction": prediction,
        "same_week": same_week,
        "next_week": next_week,
    }


@app.post("/predict/early-warning")
def predict_early_warning(request: EarlyWarningRequest):
    if not models_ready():
        raise HTTPException(status_code=503, detail="Model files missing")

    try:
        features = _features_from_request(request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    history_levels, history_mfbi = _history_from_request(request)
    trend = classify_trend(history_levels, history_mfbi)

    same_week = _enrich(predict_same_week(features), features)
    next_week = None
    week2 = None

    if _can_run_next_week(features):
        next_week = _enrich(predict_next_week(features), features)
        week2 = project_week2(
            next_week_risk=next_week["risk_level"],
            trend=trend,
            stress_trend=float(features.get("stress_trend") or 0),
            workload_trend=float(features.get("workload_trend") or 0),
            sleep_trend=float(features.get("sleep_trend") or 0),
        )

    warning = build_warning_message(
        current_risk=same_week["risk_level"],
        next_week_risk=next_week["risk_level"] if next_week else None,
        week2_risk=week2["risk_level"] if week2 else None,
        trend=trend,
    )

    return {
        "success": True,
        "student_id": request.student_id,
        "mfbi": features.get("mfbi_score"),
        "current": {
            "risk_level": same_week["risk_level"],
            "risk_score": same_week["risk_score"],
            "mfbi": features.get("mfbi_score"),
            "random_forest_prediction": same_week["random_forest_prediction"],
            "decision_tree_prediction": same_week["decision_tree_prediction"],
            "selected_model": same_week["selected_model"],
            "model_version": same_week["model_version"],
            "prediction": same_week,
        },
        "early_warning": {
            "trend": trend,
            "next_week": next_week,
            "week2_projection": week2,
            "warning_message": warning,
            "has_ml_next_week": next_week is not None,
        },
        # Convenience aliases matching dashboard needs
        "risk_level": same_week["risk_level"],
        "risk_score": same_week["risk_score"],
        "random_forest_prediction": same_week["random_forest_prediction"],
        "decision_tree_prediction": same_week["decision_tree_prediction"],
    }
