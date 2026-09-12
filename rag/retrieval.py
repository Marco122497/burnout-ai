"""
Retrieve evidence-based knowledge for a student's monitoring situation.

RAG does not calculate MFBI and does not predict burnout risk.
It only finds relevant chunks after MFBI + ML have already run.
"""

from __future__ import annotations

from mfbi.calculator import classify_mfbi_score, pss_stress_label
from rag.embeddings import embed_query

from app.database import supabase

SKIP_CONTENT_MARKERS = (
    "PERSONALIZATION RULES",
    "WHAT THE SYSTEM SHOULD NOT",
    "PROHIBITED OUTPUT",
    "DECISION RESPONSIBILITY",
    "CORE SYSTEM PURPOSE",
    "REQUIRED INPUTS",
    "RETRIEVAL KEYWORDS",
)


def _is_student_facing_chunk(row: dict) -> bool:
    if row.get("category") == "System Rules":
        return False
    content = str(row.get("content") or "").upper()
    if "STUDENT-FACING GUIDANCE" in content:
        return True
    return not any(marker in content for marker in SKIP_CONTENT_MARKERS)

CATEGORY_BY_FACTOR = {
    "stress": "Stress",
    "workload": "Academic Workload",
    "sleep": "Sleep",
    "study": "Study Time",
}


def factor_labels(
    stress_score: float,
    academic_workload: float,
    sleep_hours_score: float,
    study_time: float,
) -> dict[str, str]:
    return {
        "stress": pss_stress_label(stress_score),
        "workload": classify_mfbi_score(academic_workload / 10.0),
        "sleep": classify_mfbi_score(sleep_hours_score / 100.0),
        "study": classify_mfbi_score(study_time / 25.0),
    }


def build_retrieval_query(
    stress_score: float,
    academic_workload: float,
    sleep_hours_score: float,
    study_time: float,
    mfbi_score: float,
    risk_level: str,
) -> str:
    labels = factor_labels(
        stress_score, academic_workload, sleep_hours_score, study_time
    )
    parts = []
    if labels["stress"] != "Low":
        parts.append(f"{labels['stress'].lower()} stress")
    if labels["workload"] != "Low":
        parts.append(f"{labels['workload'].lower()} academic workload")
    if labels["sleep"] != "Low":
        parts.append("insufficient or disrupted sleep")
    if labels["study"] != "Low":
        parts.append(f"{labels['study'].lower()} study time")
    if not parts:
        parts.append("generally manageable academic demands")

    return (
        "College student experiencing "
        + ", ".join(parts)
        + f". MFBI {mfbi_score:.2f}, predicted burnout-related risk {risk_level}. "
        "Provide evidence-based strategies for managing these factors and "
        "preventing worsening burnout-related risk. Include study-rest balance, "
        "workload planning, and when to contact school support."
    )


def _unique_chunks(rows: list[dict]) -> list[dict]:
    seen: set[int] = set()
    unique: list[dict] = []
    for row in rows:
        chunk_id = row.get("id")
        if chunk_id in seen:
            continue
        if chunk_id is not None:
            seen.add(chunk_id)
        unique.append(row)
    return unique


def _search(embedding: list[float], match_count: int = 8, threshold: float = 0.18) -> list[dict]:
    response = supabase.rpc(
        "match_rag_documents",
        {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": match_count,
        },
    ).execute()
    return list(response.data or [])


def _search_category(embedding: list[float], category: str, match_count: int = 2) -> list[dict]:
    response = supabase.rpc(
        "match_rag_documents_by_category",
        {
            "query_embedding": embedding,
            "filter_category": category,
            "match_count": match_count,
        },
    ).execute()
    return list(response.data or [])


def retrieve_chunks(
    stress_score: float,
    academic_workload: float,
    sleep_hours_score: float,
    study_time: float,
    mfbi_score: float,
    risk_level: str,
    top_k: int = 6,
) -> list[dict]:
    query = build_retrieval_query(
        stress_score,
        academic_workload,
        sleep_hours_score,
        study_time,
        mfbi_score,
        risk_level,
    )
    embedding = embed_query(query)
    rows = _search(embedding, match_count=max(top_k, 8))

    labels = factor_labels(
        stress_score, academic_workload, sleep_hours_score, study_time
    )
    elevated = [
        factor
        for factor, label in labels.items()
        if label in {"Moderate", "High", "Severe"}
    ]
    if not elevated:
        elevated = list(labels.keys())

    for factor in elevated:
        category = CATEGORY_BY_FACTOR[factor]
        rows.extend(_search_category(embedding, category, match_count=2))

    if risk_level in {"High", "Severe"} or any(
        labels[factor] in {"High", "Severe"} for factor in labels
    ):
        rows.extend(_search_category(embedding, "Student Support", match_count=1))

    ranked = sorted(rows, key=lambda item: float(item.get("similarity") or 0), reverse=True)
    filtered = [
        row
        for row in _unique_chunks(ranked)
        if _is_student_facing_chunk(row)
    ]
    filtered.sort(
        key=lambda row: (
            0
            if "STUDENT-FACING GUIDANCE" in str(row.get("content") or "").upper()
            else 1,
            -float(row.get("similarity") or 0),
        )
    )
    return filtered[:top_k]


def load_system_rules() -> str:
    response = (
        supabase.table("rag_documents")
        .select("content, chunk_number")
        .eq("category", "System Rules")
        .order("chunk_number")
        .execute()
    )
    chunks = [row["content"] for row in (response.data or []) if row.get("content")]
    return "\n\n".join(chunks)


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        similarity = float(chunk.get("similarity") or 0)
        blocks.append(
            f"[{index}] Category: {chunk.get('category')}\n"
            f"Title: {chunk.get('title')}\n"
            f"Source: {chunk.get('source')}\n"
            f"Similarity: {similarity:.3f}\n"
            f"{chunk.get('content')}"
        )
    return "\n\n---\n\n".join(blocks)


def retrieve_for_test_query(question: str, match_count: int = 5) -> list[dict]:
    embedding = embed_query(question)
    return _search(embedding, match_count=match_count, threshold=0.15)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test RAG retrieval independently.")
    parser.add_argument("query", nargs="?", help="Free-text test question")
    parser.add_argument("--stress", type=float, default=32)
    parser.add_argument("--workload", type=float, default=8.5)
    parser.add_argument("--sleep", type=float, default=70)
    parser.add_argument("--study", type=float, default=20)
    parser.add_argument("--mfbi", type=float, default=0.78)
    parser.add_argument("--risk", default="High")
    args = parser.parse_args()

    if args.query:
        results = retrieve_for_test_query(args.query)
        print(json.dumps(results, indent=2, default=str))
    else:
        results = retrieve_chunks(
            args.stress, args.workload, args.sleep, args.study, args.mfbi, args.risk
        )
        print(build_retrieval_query(
            args.stress, args.workload, args.sleep, args.study, args.mfbi, args.risk
        ))
        print(json.dumps(results, indent=2, default=str))
