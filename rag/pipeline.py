"""MFBI + ML stay upstream. This module only retrieves knowledge and generates text."""

from __future__ import annotations

from rag.generator import fallback_recommendation, generate_recommendation
from rag.retrieval import (
    factor_labels,
    format_context,
    load_system_rules,
    retrieve_chunks,
)


def build_personalized_recommendation(
    *,
    stress_score: float,
    academic_workload: float,
    sleep_hours_score: float,
    study_time: float,
    mfbi_score: float,
    risk_level: str,
    prediction_model: str,
) -> dict:
    labels = factor_labels(
        stress_score, academic_workload, sleep_hours_score, study_time
    )
    chunks: list[dict] = []
    try:
        chunks = retrieve_chunks(
            stress_score,
            academic_workload,
            sleep_hours_score,
            study_time,
            mfbi_score,
            risk_level,
        )
        rules_text = load_system_rules()
        generated = generate_recommendation(
            stress_score=stress_score,
            academic_workload=academic_workload,
            sleep_hours_score=sleep_hours_score,
            study_time=study_time,
            mfbi_score=mfbi_score,
            risk_level=risk_level,
            prediction_model=prediction_model,
            labels=labels,
            chunks=chunks,
            rules_text=rules_text,
        )
    except Exception as exc:
        generated = fallback_recommendation(
            mfbi_score,
            risk_level,
            labels,
            f"rag_error:{exc.__class__.__name__}",
            chunks,
        )

    retrieved = [
        {
            "id": chunk.get("id"),
            "title": chunk.get("title"),
            "category": chunk.get("category"),
            "source": chunk.get("source"),
            "chunk_number": chunk.get("chunk_number"),
            "similarity": chunk.get("similarity"),
        }
        for chunk in chunks
    ]
    categories = []
    for chunk in retrieved:
        category = chunk.get("category")
        if category and category not in categories:
            categories.append(category)

    return {
        **generated,
        "retrieved_categories": categories,
        "retrieved_chunks": retrieved,
        "retrieval_context": format_context(chunks) if chunks else "",
        "factor_labels": labels,
    }
