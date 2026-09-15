"""LLM recommendation generator. Does not classify burnout risk."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from rag.embeddings import LLM_MODEL, llm_client, openai_configured

SYSTEM_PROMPT = """You are a caring college adviser sitting with one tired student.

Use simple, basic words. Write like you are talking to a first-year student.
Short sentences. Easy words. Use "you."
Do not use hard or deep words such as: overwhelmed, strategies, implementing, utilize,
appraisal, cognitive, self-efficacy, rumination, arousal, subsequently, furthermore,
prioritize, facilitate, intervention, or "consider reaching out."
Prefer: tired, worried, tense, heavy, pile up, rest, break, list, small step, hard week.

Keep the MFBI score and machine-learning risk exactly as given.
Ground every tip in retrieved STUDENT-FACING GUIDANCE / HOW THIS MAY FEEL.
Stress tips should be about how the week feels in the mind and body (worry, control, calm), not a second homework list.
Do not diagnose. Do not invent contacts, medicines, or sources.

Focus tips on Moderate and High factors only.
If a factor is High, mention it first and give it more than one tip.
If a factor is Low, one sentence only — no tip list for it.

assessment_summary: 2–4 short sentences. Name the week honestly, then what is hardest.
contributing_factors: full easy sentences about how the week feels. Never labels.
recommended_actions: 3 or 4 this-week steps. Simple advice, not a flowchart.
Do not start every tip with "If you...". Just tell them what to try.
human_support: one kind, simple paragraph. Do not say "I know."

Good contributing_factors:
- "Quizzes, projects, and readings are piling up, and the week feels too full."
- "Nights look too short, so you may come to class already tired."

Bad contributing_factors:
- "Academic workload"
- "Sleep/rest"
- "You might be feeling a bit overwhelmed"
- "classified as High risk"

Good actions:
- "Try to sleep about 7 hours, even if one homework is not done."
- "Write this week's due dates on one page, then work on the soonest one for the next hour."

Bad actions:
- "Consider implementing time-management strategies."
- "If these feelings continue, consider reaching out."

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
        ("stress", "Moderate"): "This week has felt a bit heavy, like you are waiting for the next problem.",
        ("stress", "High"): "You have felt tense and worried, and it is hard to feel in control.",
        ("stress", "Severe"): "The worry feels very heavy right now, and it may be hard to catch your breath.",
        ("workload", "Moderate"): "Your classwork is starting to pile up.",
        ("workload", "High"): "Quizzes, projects, and readings are piling up, and the week feels too full.",
        ("workload", "Severe"): "There is so much schoolwork this week that it looks very hard to carry alone.",
        ("sleep", "Moderate"): "Your rest is a bit off, so days may start before you feel ready.",
        ("sleep", "High"): "Nights look too short, so you may wake up still tired.",
        ("sleep", "Severe"): "Sleep looks far too short, and your body is not getting a real rest.",
        ("study", "Moderate"): "You have been studying quite a bit, and it may be taking time from the rest of your day.",
        ("study", "High"): "Long study hours are taking time from rest, and more sitting may not help.",
        ("study", "Severe"): "Study time looks so long that there is little time left to rest.",
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
        "Name the one worry sitting heaviest, so it is not going round and round in your head.",
        "Pick one thing you can still do today — one deadline, one class, or one message — then give it one honest hour.",
        "Take two slow breaths, a meal, or a short walk before you open another file. Calm down a little, then take one small next step.",
    ],
    "Academic Workload": [
        "Write this week's quizzes, projects, and readings on one page with due dates, then circle what is due first.",
        "Open only the soonest file for the next hour. Jumping between five unfinished tasks makes the pile feel bigger.",
        "Leave a little empty time for sleep and a meal. A packed day makes small delays feel much worse.",
    ],
    "Sleep": [
        "Try to sleep about 7 hours and keep a bedtime you can keep, even if one homework is not done.",
        "Move one study block earlier so you are not working in bed after midnight.",
        "Skip late coffee or energy drinks, dim the phone, and keep your bed for sleep, not homework.",
    ],
    "Study Time": [
        "Sit for 40 to 50 minutes with one written goal, then stand up. More hours are not the fix this week.",
        "Test yourself with a few questions or say one idea out loud. Stop rereading for hours.",
        "Make tonight's last study shorter if it is stealing sleep, and do that work earlier tomorrow.",
    ],
    "Student Support": [
        "Tell a teacher you trust, your adviser, or someone in Guidance what felt hardest this week. You do not have to sort it alone.",
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
        "Schoolwork from this week's form"
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
        "You do not have to do this hard week alone. A teacher you already talk to, "
        "your adviser, or someone in the Guidance Office can help you sort what you must do. "
        "Start with one honest sentence about what felt hardest. Use the school's own pages "
        "to find them — this app will not make up a phone number."
    )
    if risk_level in {"Low"}:
        support = (
            "This week looks more doable, and you can still ask a teacher, adviser, or the Guidance "
            "Office to help you plan if you want. Use the school's own information to reach them."
        )

    return {
        "assessment_summary": (
            f"{lead}{extra} This is school well-being help, not a medical diagnosis."
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
    "stress": "The worry from this week looks okay.",
    "workload": "Your classwork load looks okay this week.",
    "sleep": "Your sleep looks okay this week.",
    "study": "Your study time looks okay this week.",
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
        return "Your classwork and study time look okay this week."
    return "The other areas look okay this week."


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
        "write like a caring",
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
        in_actions = True
        for raw in body.splitlines():
            line = raw.strip().lstrip("-*").strip()
            heading = line.upper()
            if re.match(r"^\d+\.", line) or heading.startswith("RETRIEVAL KEYWORDS"):
                break
            if heading.startswith("HOW THIS MAY FEEL"):
                in_actions = False
                continue
            if heading.startswith("THINGS YOU CAN TRY") or heading.startswith("WHAT YOU CAN TRY"):
                in_actions = True
                continue
            if not in_actions:
                continue
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


def _looks_like_factor_label(text: str) -> bool:
    compact = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    return compact in {
        "stress",
        "academic workload",
        "workload",
        "sleep",
        "sleep rest",
        "study time",
        "study",
        "rest",
    }


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
            "consider reaching out",
            "feelings of being overwhelmed",
            "overwhelmed",
            "if these difficulties",
            "consider implementing",
            "time-management strategies",
            "strategies",
            "utilize",
            "self-efficacy",
            "appraisal",
            "cognitive",
            "rumination",
            "i know this week",
            "it's okay to seek support",
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
    "Stress": r"stress|tense|feeling|control|cope|breath|deadline|pressure|break",
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
        if low_note and "okay this week" not in summary.lower() and "looks okay" not in summary.lower():
            summary = f"{summary.rstrip('.')} {low_note}"

    factors = [
        item
        for item in _as_string_list(parsed.get("contributing_factors"))
        if not _looks_robotic(item) and not _looks_like_factor_label(item)
    ] or fallback["contributing_factors"]

    actions = [
        item
        for item in _as_string_list(parsed.get("recommended_actions"))
        if not _looks_robotic(item)
    ]
    actions = [item for item in actions if item]
    if_you_heavy = sum(1 for item in actions if re.match(r"^if you\b", item.strip(), re.I))
    if len(actions) < 2 or (len(actions) >= 3 and if_you_heavy >= 2):
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
    if not support or _looks_robotic(support) or len(support) < 40:
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
    use_llm: bool = True,
) -> dict[str, Any]:
    if not chunks:
        return fallback_recommendation(mfbi_score, risk_level, labels, "retrieval_empty")
    if not use_llm:
        return fallback_recommendation(
            mfbi_score, risk_level, labels, "llm_disabled_by_admin", chunks
        )
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

Write to the student as a caring adviser using HOW THIS MAY FEEL and STUDENT-FACING GUIDANCE.
Use simple, basic words. Short sentences. No hard or deep words.
contributing_factors must be full easy sentences about how the week feels.
Never copy the labels "Academic workload", "Sleep/rest", "Study time", or "Stress".
Never use: overwhelmed, strategies, implementing, utilize, appraisal, cognitive.
Each recommended_action should be one kind, simple thing they can do this week.
Do not start tips with "If you...". Speak directly: "Try to sleep about 7 hours..." / "Write the due dates..."
human_support should sound like a person, not a policy notice. Do not say "I know."
Do not change the MFBI score or the predicted burnout-related risk.
If a requested phone number or named person is not present above, do not invent one.
"""

    try:
        client = llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.4,
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
