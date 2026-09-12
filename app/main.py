import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from app.data_service import build_features_from_history, get_student_history
from app.predictor import models_ready, predict_next_week, predict_same_week
from app.schemas import EarlyWarningRequest, PredictionRequest, RecommendationRequest
from mfbi.calculator import compute_mfbi
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
            "recommendation": "POST /api/burnout/recommendation",
            "rag_search": "GET /api/rag/search",
            "metrics": "GET /metrics",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    ready = models_ready()
    openai_ok = False
    llm_model = None
    try:
        from rag.embeddings import LLM_MODEL, openai_configured

        openai_ok = openai_configured()
        llm_model = LLM_MODEL if openai_ok else None
    except Exception:
        openai_ok = False
    return {
        "status": "ok" if ready else "degraded",
        "models_ready": ready,
        "openai_configured": openai_ok,
        "llm_model": llm_model,
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


@app.get("/api/rag/search")
def rag_search(q: str = Query(..., min_length=3, description="Standalone RAG test query")):
    """Test retrieval only. Does not call the LLM and does not predict burnout."""
    try:
        from rag.embeddings import openai_configured
        from rag.retrieval import retrieve_for_test_query
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG is not available: {exc}") from exc
    if not openai_configured():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    try:
        results = retrieve_for_test_query(q)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "query": q,
        "matches": [
            {
                "title": row.get("title"),
                "category": row.get("category"),
                "source": row.get("source"),
                "chunk_number": row.get("chunk_number"),
                "similarity": row.get("similarity"),
                "content": row.get("content"),
            }
            for row in results
        ],
    }


@app.post("/api/burnout/recommendation")
def burnout_recommendation(request: RecommendationRequest):
    """
    Recommendation layer: verify MFBI, optionally confirm ML, then RAG + LLM.

    RAG never calculates burnout risk. The stored/requested ML prediction is kept.
    """
    try:
        scores = request.resolved_scores()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mfbi = compute_mfbi(
        scores["stress_score"],
        scores["academic_workload_score"],
        scores["study_time_score"],
        scores["sleep_hours_score"],
    )
    mfbi_score = (
        float(request.mfbi_score)
        if request.mfbi_score is not None
        else float(mfbi["mfbi_score"])
    )

    prediction = None
    if models_ready():
        try:
            prediction = predict_same_week(
                {
                    **scores,
                    "mfbi_score": mfbi_score,
                    "stress_trend": 0.0,
                    "workload_trend": 0.0,
                    "study_trend": 0.0,
                    "sleep_trend": 0.0,
                }
            )
        except Exception:
            prediction = None

    # Official risk remains the ML result already computed for this student.
    risk_level = (
        request.risk_level
        or (prediction.get("risk_level") if prediction else None)
        or mfbi["burnout_risk_level"]
    )
    prediction_model = (
        request.prediction_model
        or request.selected_model
        or (prediction.get("selected_model") if prediction else None)
        or "Random Forest"
    )

    try:
        from rag.pipeline import build_personalized_recommendation

        generated = build_personalized_recommendation(
            stress_score=scores["stress_score"],
            academic_workload=scores["academic_workload_score"],
            sleep_hours_score=scores["sleep_hours_score"],
            study_time=scores["study_time_score"],
            mfbi_score=mfbi_score,
            risk_level=risk_level,
            prediction_model=prediction_model,
        )
    except Exception as exc:
        generated = {
            "assessment_summary": (
                "The personalized AI recommendation is temporarily unavailable. "
                f"Your current MFBI score is {mfbi_score:.2f} with a predicted "
                f"burnout-related risk of {risk_level}. This is an educational "
                "early-warning result, not a medical diagnosis."
            ),
            "contributing_factors": [
                "Current academic demands from weekly monitoring"
            ],
            "recommended_actions": [
                "Based on your current results, consider prioritizing sleep, managing your workload, and taking regular study breaks.",
                "Break large academic tasks into smaller steps and finish required work before optional extras.",
                "Use shorter focused study sessions instead of one long late-night session.",
            ],
            "human_support": (
                "Consider talking with your instructor, academic adviser, or guidance counselor "
                "if these difficulties continue or become hard to manage. Please refer to your "
                "institution's official guidance office or verified student support directory."
            ),
            "sources": [],
            "llm_model": None,
            "used_fallback": True,
            "fallback_reason": f"rag_unavailable:{exc.__class__.__name__}",
            "retrieved_categories": [],
            "retrieved_chunks": [],
            "factor_labels": {},
        }

    return {
        "success": True,
        "mfbi_score": round(mfbi_score, 2),
        "mfbi_verified": mfbi,
        "risk_level": risk_level,
        "prediction_model": prediction_model,
        "prediction": prediction,
        "recommendation": {
            "assessment_summary": generated["assessment_summary"],
            "contributing_factors": generated["contributing_factors"],
            "recommended_actions": generated["recommended_actions"],
            "human_support": generated["human_support"],
            "sources": generated["sources"],
        },
        "audit": {
            "llm_model": generated.get("llm_model"),
            "used_fallback": bool(generated.get("used_fallback")),
            "fallback_reason": generated.get("fallback_reason"),
            "retrieved_categories": generated.get("retrieved_categories") or [],
            "retrieved_chunks": generated.get("retrieved_chunks") or [],
            "factor_labels": generated.get("factor_labels") or {},
        },
    }
