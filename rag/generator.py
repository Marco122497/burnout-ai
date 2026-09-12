"""LLM recommendation generator. Does not classify burnout risk."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from rag.embeddings import LLM_MODEL, llm_client, openai_configured

SYSTEM_PROMPT = """You are a warm academic support assistant talking to one college student.

Write like a caring adviser: clear, kind, and practical. Use "you."
Keep the student's MFBI score and machine-learning risk exactly as given.
Use only retrieved knowledge. Do not diagnose. Do not invent contacts, medicines, or sources.

Focus recommended actions on Moderate and High factors only.
If a factor is High, it is the priority: mention it first and give it more than one tip.
If a factor is Low, mention it in one sentence only. Do not give it a list of tips.

Avoid robotic phrases such as "classified as," "factors included in your weekly assessment,"
"official school channels," or "sleep-related risk."
Say things a student would actually feel: tired, overloaded, tense, behind on rest.

Return JSON only with this shape:
{
  "assessment_summary": "string",
  "contributing_factors": ["string"],
  "recommended_actions": ["string", "string", "string"],
  "human_support": "string",
  "sources": ["string"]
}
"""


def contributing_factor_phrases(labels: dict[str, str]) -> list[str]:
    stories = {
        ("stress", "Moderate"): "Stress is building, and school may feel heavier than usual.",
        ("stress", "High"): "Stress is high, and things may feel hard to keep up with.",
        ("stress", "Severe"): "Stress feels overwhelming right now.",
        ("workload", "Moderate"): "Your classwork is starting to pile up.",
        ("workload", "High"): "Your schoolwork load looks heavy this week.",
        ("workload", "Severe"): "Your schoolwork load looks very hard to manage.",
        ("sleep", "Moderate"): "Your rest is a bit off, and sleep may not be as steady as you need.",
        ("sleep", "High"): "Sleep looks like the hardest part — you may not be getting enough rest.",
        ("sleep", "Severe"): "Sleep looks seriously stretched, and recovery is getting squeezed.",
        ("study", "Moderate"): "You are studying quite a bit, and it may be crowding the rest of your day.",
        ("study", "High"): "Study hours look very long, and rest may be getting squeezed.",
        ("study", "Severe"): "Study time looks excessive, with little room left to recover.",
    }
    severity = {"Severe": 3, "High": 2, "Moderate": 1}
    scored: list[tuple[int, str]] = []
    for key in ("stress", "workload", "sleep", "study"):
        level = labels.get(key)
        text = stories.get((key, level or ""))
        if text:
            scored.append((severity.get(level or "", 0), text))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored]


STUDENT_ACTIONS_BY_CATEGORY = {
    "Stress": [
        "Pause and name the one deadline or class that feels heaviest this week.",
        "Do one small next step on that task in under an hour instead of trying to finish everything tonight.",
        "Give yourself a real break, a meal, and a chance to breathe between classes.",
    ],
    "Academic Workload": [
        "Write this week's due dates in one simple list so you can see what actually needs you first.",
        "Start the biggest assignment with a small piece rather than opening every task at once.",
        "Leave a little unplanned time in your week. Packed days make everything feel worse.",
    ],
    "Sleep": [
        "Try for about 7 hours of sleep and a regular bedtime, even if one assignment is still unfinished.",
        "Move one study block earlier in the day so you are not working in bed late at night.",
        "Skip late caffeine if it keeps you awake, and keep your bed for rest rather than homework.",
    ],
    "Study Time": [
        "Study in a shorter focused block with one clear goal, then take a real break.",
        "Test yourself with a few questions instead of rereading for hours.",
        "If late-night studying is stealing sleep, shorten one session this week rather than adding more hours.",
    ],
    "Student Support": [
        "If this still feels too heavy, talk with a teacher you trust, your adviser, or the Guidance Office.",
    ],
}


def fallback_recommendation(
    mfbi_score: float,
    risk_level: str,
    labels: dict[str, str],
    reason: str = "llm_unavailable",
    chunks: list[dict] | None = None,
) -> dict[str, Any]:
    factors = contributing_factor_phrases(labels) or [
        "Current academic demands from weekly monitoring"
    ]
    sources: list[str] = []
    retrieved_categories: list[str] = []
    if chunks:
        for chunk in chunks:
            title = str(chunk.get("title") or "").strip()
            category = str(chunk.get("category") or "").strip()
            if title and title not in sources and category != "System Rules":
                sources.append(title)
            if category and category not in retrieved_categories:
                retrieved_categories.append(category)

    actions = _student_actions_for_labels(
        labels, retrieved_categories, risk_level, chunks or []
    )
    grounded = bool(chunks)
    lead = {
        "Low": f"This week looks manageable (MFBI {mfbi_score:.2f}).",
        "Moderate": f"This week looks a bit heavy (MFBI {mfbi_score:.2f}).",
        "High": f"This week looks quite hard to carry (MFBI {mfbi_score:.2f}).",
        "Severe": f"This week looks very hard to carry (MFBI {mfbi_score:.2f}).",
    }.get(risk_level, f"This week's burnout-related risk is {risk_level} (MFBI {mfbi_score:.2f}).")
    extra = " " + " ".join(factors[:2]) if factors else ""
    low_note = _low_factor_sentence(labels)
    if low_note:
        extra = f"{extra} {low_note}".rstrip()

    support = (
        "You do not have to handle a hard week alone. If this still feels too heavy, "
        "talk with a teacher you trust, your adviser, or the Guidance Office. "
        "Use the school's official information to find them — this app will not invent a phone number."
    )
    if risk_level in {"Low"}:
        support = (
            "If you want extra support anyway, a teacher, adviser, or the Guidance Office "
            "can still help you plan. Use the school's official information to reach them."
        )

    return {
        "assessment_summary": (
            f"{lead}{extra} This is school well-being guidance, not a medical diagnosis."
        ),
        "contributing_factors": factors,
        "recommended_actions": actions,
        "human_support": support,
        "sources": sources,
        "used_fallback": not grounded,
        "fallback_reason": None if grounded else reason,
        "llm_model": None,
    }


FACTOR_TO_CATEGORY = {
    "stress": "Stress",
    "workload": "Academic Workload",
    "sleep": "Sleep",
    "study": "Study Time",
}
LOW_FACTOR_SENTENCES = {
    "stress": "Stress looks manageable this week.",
    "workload": "Your classwork load looks manageable this week.",
    "sleep": "Your sleep looks all right this week.",
    "study": "Your study time looks manageable this week.",
}


def _focused_factors(labels: dict[str, str]) -> list[tuple[int, str, str]]:
    severity = {"Severe": 3, "High": 2, "Moderate": 1}
    scored: list[tuple[int, str, str]] = []
    for key, category in FACTOR_TO_CATEGORY.items():
        level = labels.get(key) or "Low"
        if level in severity:
            scored.append((severity[level], category, level))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _low_factor_sentence(labels: dict[str, str]) -> str:
    lows = [
        key
        for key in ("stress", "workload", "sleep", "study")
        if labels.get(key) == "Low"
    ]
    if not lows:
        return ""
    if len(lows) == 1:
        return LOW_FACTOR_SENTENCES[lows[0]]
    if set(lows) == {"workload", "study"}:
        return "Your classwork and study time look manageable this week."
    return "The other areas look manageable this week."


def _action_slot_categories(focused: list[tuple[int, str, str]], total: int = 4) -> list[str]:
    slots: list[str] = []
    for _, category, _ in focused:
        slots.append(category)
    for _, category, level in focused:
        if level in {"High", "Severe"}:
            slots.append(category)
    for _, category, level in focused:
        if level == "Moderate":
            slots.append(category)
    while focused and len(slots) < total:
        slots.append(focused[0][1])
    return slots[:total]


def _student_actions_for_labels(
    labels: dict[str, str],
    retrieved_categories: list[str],
    risk_level: str,
    chunks: list[dict] | None = None,
) -> list[str]:
    _ = retrieved_categories
    by_category = _extract_student_facing_by_category(chunks or [])
    focused = _focused_factors(labels)
    if not focused:
        return [
            "Keep the habits that are working: a little planning, a real break, and enough rest."
        ]

    level_by_category = {category: level for _, category, level in focused}

    def take_from(category: str, actions: list[str]) -> str | None:
        templates = list(STUDENT_ACTIONS_BY_CATEGORY.get(category) or [])
        extracted = list(by_category.get(category) or [])
        if level_by_category.get(category) in {"High", "Severe"}:
            candidates = templates + extracted
        else:
            candidates = extracted + templates
        for action in candidates:
            if _is_referral_line(action) and risk_level not in {"High", "Severe"}:
                continue
            if any(_too_similar(action, existing) for existing in actions):
                continue
            return action
        return None

    actions: list[str] = []
    for category in _action_slot_categories(focused):
        action = take_from(category, actions)
        if action:
            actions.append(action)

    if risk_level in {"High", "Severe"}:
        support = STUDENT_ACTIONS_BY_CATEGORY["Student Support"][0]
        if support not in actions and not any(_too_similar(support, item) for item in actions):
            if len(actions) >= 4:
                actions[-1] = support
            else:
                actions.append(support)

    return actions[:4]


def _extract_student_facing_by_category(chunks: list[dict]) -> dict[str, list[str]]:
    skip_starts = (
        "write to the student",
        "do not diagnose",
        "the system",
        "recommend ",
    )
    by_category: dict[str, list[str]] = {}
    for chunk in chunks:
        category = str(chunk.get("category") or "").strip()
        if not category or category == "System Rules":
            continue
        content = str(chunk.get("content") or "")
        marker = "STUDENT-FACING GUIDANCE"
        idx = content.upper().find(marker)
        if idx == -1:
            continue
        body = content[idx + len(marker) :]
        for raw in body.splitlines():
            line = raw.strip().lstrip("-*").strip()
            if re.match(r"^\d+\.", line) or line.upper().startswith("RETRIEVAL KEYWORDS"):
                break
            if len(line) < 28 or len(line) > 220:
                continue
            if line.lower().startswith(skip_starts):
                continue
            bucket = by_category.setdefault(category, [])
            if line not in bucket:
                bucket.append(line)
    return by_category


def _too_similar(left: str, right: str) -> bool:
    words_left = set(re.findall(r"[a-z]{4,}", left.lower()))
    words_right = set(re.findall(r"[a-z]{4,}", right.lower()))
    if not words_left or not words_right:
        return left.strip().lower() == right.strip().lower()
    overlap = len(words_left & words_right) / min(len(words_left), len(words_right))
    return overlap >= 0.55


def _is_referral_line(line: str) -> bool:
    low = line.lower()
    return any(
        token in low
        for token in (
            "guidance office",
            "academic adviser",
            "do not have to handle a hard week",
            "official school",
        )
    )


def _looks_robotic(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            "classified as",
            "factors included",
            "sleep-related risk",
            "official school channels",
            "weekly assessment",
        )
    ) or bool(
        re.match(r"^(moderate|high|severe|low)\s+(stress|sleep|workload|study)", low)
    )


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


PRIORITY_ACTION_HINTS = {
    "Sleep": r"sleep|bedtime|7 hours|wake time|\brest\b",
    "Stress": r"stress|tense|deadline|pressure|break",
    "Academic Workload": r"due date|assignment|workload|task|list",
    "Study Time": r"study|break|reread|focused",
}


def _priority_action_covers(category: str, action: str) -> bool:
    pattern = PRIORITY_ACTION_HINTS.get(category)
    if not pattern:
        return True
    return bool(re.search(pattern, action, re.I))


def _apply_llm_output(
    parsed: dict[str, Any],
    fallback: dict[str, Any],
    labels: dict[str, str],
    source_titles: list[str],
) -> dict[str, Any]:
    summary = str(parsed.get("assessment_summary") or "").strip()
    if not summary or _looks_robotic(summary):
        summary = fallback["assessment_summary"]
    else:
        low_note = _low_factor_sentence(labels)
        if low_note and "manageable this week" not in summary.lower() and "all right this week" not in summary.lower():
            summary = f"{summary.rstrip('.')} {low_note}"

    factors = [
        item
        for item in _as_string_list(parsed.get("contributing_factors"))
        if not _looks_robotic(item)
    ] or fallback["contributing_factors"]

    actions = [
        item
        for item in _as_string_list(parsed.get("recommended_actions"))
        if not _looks_robotic(item)
    ]
    actions = [item for item in actions if item]
    if len(actions) < 2:
        actions = list(fallback["recommended_actions"])

    focused = _focused_factors(labels)
    if focused:
        _, priority_category, priority_level = focused[0]
        if priority_level in {"High", "Severe"} and actions:
            if not _priority_action_covers(priority_category, actions[0]):
                priority_tip = next(
                    (
                        item
                        for item in fallback["recommended_actions"]
                        if _priority_action_covers(priority_category, item)
                    ),
                    None,
                )
                if priority_tip:
                    actions = [priority_tip, *[item for item in actions if item != priority_tip]]

    support = str(parsed.get("human_support") or "").strip()
    if not support or _looks_robotic(support):
        support = fallback["human_support"]

    sources = _as_string_list(parsed.get("sources")) or source_titles

    return {
        "assessment_summary": summary,
        "contributing_factors": factors[:4],
        "recommended_actions": actions[:4],
        "human_support": support,
        "sources": sources,
        "used_fallback": False,
        "fallback_reason": None,
        "llm_model": os.getenv("OPENAI_LLM_MODEL", LLM_MODEL),
    }


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    return json.loads(cleaned)


def generate_recommendation(
    *,
    stress_score: float,
    academic_workload: float,
    sleep_hours_score: float,
    study_time: float,
    mfbi_score: float,
    risk_level: str,
    prediction_model: str,
    labels: dict[str, str],
    chunks: list[dict],
    rules_text: str,
) -> dict[str, Any]:
    if not chunks:
        return fallback_recommendation(mfbi_score, risk_level, labels, "retrieval_empty")
    if not openai_configured():
        return fallback_recommendation(
            mfbi_score, risk_level, labels, "openai_not_configured", chunks
        )

    context_blocks = []
    source_titles: list[str] = []
    for chunk in chunks:
        title = str(chunk.get("title") or "Knowledge chunk")
        if title not in source_titles and chunk.get("category") != "System Rules":
            source_titles.append(title)
        context_blocks.append(
            f"Category: {chunk.get('category')}\n"
            f"Title: {title}\n"
            f"{chunk.get('content')}"
        )

    focused = _focused_factors(labels)
    priority = ", ".join(f"{category} ({level})" for _, category, level in focused) or "none"
    low_note = _low_factor_sentence(labels) or "none"

    user_prompt = f"""Student monitoring results (do not change these values):
- Stress: {labels.get('stress')} (PSS-style score {stress_score})
- Academic workload: {labels.get('workload')} (score {academic_workload})
- Sleep/rest: {labels.get('sleep')} (sleep-risk score {sleep_hours_score}; higher means poorer rest)
- Study time: {labels.get('study')} (weekly study-time score {study_time})
- MFBI: {mfbi_score:.2f}
- Predicted burnout-related risk: {risk_level}
- Prediction model: {prediction_model}

Priority factors to write tips for, highest first: {priority}
Low factors (one sentence only, no tip list): {low_note}

System rules:
{rules_text or "Follow the academic-support rules. Do not diagnose. Do not invent contacts."}

Retrieved evidence-based knowledge:
{chr(10).join(context_blocks)}

Write personalized student recommendations grounded in the retrieved knowledge.
Prefer STUDENT-FACING GUIDANCE bullets. Ignore personalization-rule lists and retrieval keywords.
Focus tips on Moderate and High factors only. The highest-priority factor must come first, with more than one tip.
If a factor is Low, mention it in one sentence only and do not give it a list of tips.
Do not change the MFBI score or the predicted burnout-related risk.
If a requested phone number or named person is not present above, do not invent one.
"""

    try:
        client = llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = _extract_json(content)
    except Exception as exc:  # pragma: no cover - network/API failures
        return fallback_recommendation(
            mfbi_score, risk_level, labels, f"llm_error:{exc.__class__.__name__}", chunks
        )

    fallback = fallback_recommendation(
        mfbi_score, risk_level, labels, "llm_sanitize", chunks
    )
    return _apply_llm_output(parsed, fallback, labels, source_titles)
