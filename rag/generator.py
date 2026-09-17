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

Match tips to the four weekly questionnaires the student filled out: PSS-10, Academic Workload, Study Time, and Sleep Hours. Use the question wording from the form.
Keep the next-week predicted score and risk exactly as given. Do not use this week's MFBI as next week's prediction.
Ground every tip in retrieved STUDENT-FACING GUIDANCE / HOW THIS MAY FEEL / THINGS YOU CAN TRY NEXT WEEK.
Copy the advice voice of those documents. Do not sound like a research paper.
Stress tips should be about how next week may feel in the mind and body (worry, control, calm), not a second homework list.
Do not diagnose. Do not invent contacts, medicines, or sources.

Focus tips on Moderate and High factors only.
If a factor is High, mention it first and give it more than one tip.
If a factor is Low, one sentence only — no tip list for it.

assessment_summary: 2–4 short sentences. Start with how next week looks, using the next-week predicted score.
contributing_factors: full easy sentences about how next week may feel. Never labels.
recommended_actions: 3 or 4 next-week steps. Simple advice, not a flowchart.
Do not start every tip with "If you...". Just tell them what to try.
human_support: one kind, simple paragraph. Do not say "I know."

Good contributing_factors:
- "Quizzes, projects, and readings may pile up, and next week may feel too full."
- "Nights look too short, so you may come to class already tired."

Bad contributing_factors:
- "Academic workload"
- "Sleep/rest"
- "You might be feeling a bit overwhelmed"
- "classified as High risk"

Good actions:
- "Try to sleep about 7 hours, even if one homework is not done."
- "Write next week's due dates on one page, then work on the soonest one for the next hour."

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
        ("stress", "Moderate"): "Next week may feel a bit heavy, like you are waiting for the next unexpected problem.",
        ("stress", "High"): "You may feel nervous and stressed, and it is hard to feel in control of the important things.",
        ("stress", "Severe"): "Difficulties may feel piled so high that it is hard to cope with all the things you have to do.",
        ("workload", "Moderate"): "Your academic workload is starting to feel heavy.",
        ("workload", "High"): "Overlapping deadlines, plus quizzes, projects, and readings, may make next week too full.",
        ("workload", "Severe"): "The volume of academic requirements may look very hard to keep up with next week.",
        ("sleep", "Moderate"): "Your sleep schedule is a bit off, so you may not feel rested upon waking.",
        ("sleep", "High"): "Short nights and lack of sleep may affect your academics next week.",
        ("sleep", "Severe"): "Sleep hours look far too short, and your body is not getting a real rest.",
        ("study", "Moderate"): "You have been studying quite a bit, and reviewing lessons may be taking time from the rest of your day.",
        ("study", "High"): "Long study hours, assignments, and focused sessions may take time from rest.",
        ("study", "Severe"): "Study time may look so long that there is little time left to rest.",
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
        "Write next week's quizzes, projects, and readings on one page with due dates, then circle what is due first.",
        "Open only the soonest file for the next hour. Jumping between five unfinished tasks makes the pile feel bigger.",
        "Leave a little empty time for sleep and a meal. A packed day makes small delays feel much worse.",
    ],
    "Sleep": [
        "Try to sleep about 7 hours and keep a bedtime you can keep, even if one homework is not done.",
        "Move one study block earlier so you are not working in bed after midnight.",
        "Skip late coffee or energy drinks, dim the phone, and keep your bed for sleep, not homework.",
    ],
    "Study Time": [
        "Sit for 40 to 50 minutes with one written goal, then stand up. More hours are not the fix next week.",
        "Test yourself with a few questions or say one idea out loud. Stop rereading for hours.",
        "Make tonight's last study shorter if it is stealing sleep, and do that work earlier tomorrow.",
    ],
    "Student Support": [
        "Tell a teacher you trust, your adviser, or someone in Guidance what feels hardest going into next week. You do not have to sort it alone.",
    ],
}


def _replace_this_week(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        week = "week's" if token.lower().endswith("'s") else "week"
        prefix = "Next" if token[0].isupper() else "next"
        return f"{prefix} {week}"

    return re.sub(r"\bthis week's\b|\bthis week\b", repl, text, flags=re.I)


def _frame_as_next_week(text: str, score: float | None) -> str:
    out = _replace_this_week(text)
    if score is None:
        return out
    predicted = f"{float(score):.2f}"
    out = re.sub(r"\(MFBI\s*\d+\.\d+\)", f"(predicted {predicted})", out, flags=re.I)
    return re.sub(r"\bMFBI\s+\d+\.\d+", f"predicted {predicted}", out, flags=re.I)


def fallback_recommendation(
    mfbi_score: float,
    risk_level: str,
    labels: dict[str, str],
    reason: str = "llm_unavailable",
    chunks: list[dict] | None = None,
    *,
    outlook_score: float | None = None,
    outlook_risk: str | None = None,
) -> dict[str, Any]:
    display_score = float(outlook_score) if outlook_score is not None else mfbi_score
    display_risk = outlook_risk or risk_level
    factors = contributing_factor_phrases(labels) or [
        "Schoolwork from your latest form may follow you into next week"
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
        labels, retrieved_categories, display_risk, chunks or []
    )
    grounded = bool(chunks)
    # OpenAI off / unavailable still returns next-week advice from scores + retrieved docs.
    lead = {
        "Low": f"Next week looks okay (predicted {display_score:.2f}).",
        "Moderate": f"Next week looks a bit heavy (predicted {display_score:.2f}).",
        "High": f"Next week looks quite hard to carry (predicted {display_score:.2f}).",
        "Severe": f"Next week looks very hard to carry (predicted {display_score:.2f}).",
    }.get(display_risk, f"Next week's predicted burnout-related risk is {display_risk} ({display_score:.2f}).")
    extra = " " + " ".join(factors[:2]) if factors else ""
    low_note = _low_factor_sentence(labels)
    if low_note:
        extra = f"{extra} {low_note}".rstrip()

    support = (
        "You do not have to do next week alone. A teacher you already talk to, "
        "your adviser, or someone in the Guidance Office can help you sort what you must do "
        "before the next form. Start with one honest sentence about what feels hardest."
    )
    if display_risk in {"Low"}:
        support = (
            "Next week looks more doable, and you can still ask a teacher, adviser, or the Guidance "
            "Office to help you plan before the next form if you want."
        )

    summary = (
        f"{lead}{extra} Use these tips going into next week and the next monitoring form. "
        "This is school well-being help, not a medical diagnosis."
    )
    # Mark as fallback whenever GPT wording was not used (admin off, no key, errors, etc.).
    no_llm = reason in {
        "llm_disabled_by_admin",
        "openai_not_configured",
        "llm_unavailable",
        "retrieval_empty",
    } or reason.startswith("rag_error") or reason.startswith("llm_")
    used_fallback = (not grounded) or no_llm

    return {
        "assessment_summary": _frame_as_next_week(summary, display_score),
        "contributing_factors": [
            _frame_as_next_week(item, display_score) for item in factors
        ],
        "recommended_actions": [
            _frame_as_next_week(item, display_score) for item in actions
        ],
        "human_support": _frame_as_next_week(support, display_score),
        "sources": sources,
        "used_fallback": used_fallback,
        "fallback_reason": reason if used_fallback else None,
        "llm_model": None,
    }


FACTOR_TO_CATEGORY = {
    "stress": "Stress",
    "workload": "Academic Workload",
    "sleep": "Sleep",
    "study": "Study Time",
}
LOW_FACTOR_SENTENCES = {
    "stress": "The worry looking into next week looks okay.",
    "workload": "Your classwork load looks okay next week.",
    "sleep": "Your sleep looks okay next week.",
    "study": "Your study time looks okay next week.",
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
        return "Your classwork and study time look okay next week."
    return "The other areas look okay next week."


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
        "match advice",
        "do not start every tip",
        "use the school's own pages",
        "this app will not make up",
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
    outlook_score: float | None = None,
) -> dict[str, Any]:
    summary = str(parsed.get("assessment_summary") or "").strip()
    if not summary or _looks_robotic(summary):
        summary = fallback["assessment_summary"]
    else:
        low_note = _low_factor_sentence(labels)
        if low_note and "okay this week" not in summary.lower() and "okay next week" not in summary.lower() and "looks okay" not in summary.lower():
            summary = f"{summary.rstrip('.')} {low_note}"
        summary = _frame_as_next_week(summary, outlook_score)

    factors = [
        _frame_as_next_week(item, outlook_score)
        for item in _as_string_list(parsed.get("contributing_factors"))
        if not _looks_robotic(item) and not _looks_like_factor_label(item)
    ] or fallback["contributing_factors"]

    actions = [
        _frame_as_next_week(item, outlook_score)
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
    else:
        support = _frame_as_next_week(support, outlook_score)

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
    outlook_score: float | None = None,
    outlook_risk: str | None = None,
) -> dict[str, Any]:
    display_score = float(outlook_score) if outlook_score is not None else mfbi_score
    display_risk = outlook_risk or risk_level
    fallback_kwargs = {
        "outlook_score": display_score,
        "outlook_risk": display_risk,
    }
    if not chunks:
        return fallback_recommendation(
            mfbi_score, risk_level, labels, "retrieval_empty", **fallback_kwargs
        )
    if not use_llm:
        return fallback_recommendation(
            mfbi_score, risk_level, labels, "llm_disabled_by_admin", chunks, **fallback_kwargs
        )
    if not openai_configured():
        return fallback_recommendation(
            mfbi_score, risk_level, labels, "openai_not_configured", chunks, **fallback_kwargs
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
- This week's MFBI (already measured; do not describe this as next week): {mfbi_score:.2f} ({risk_level})
- Next-week predicted score: {display_score:.2f}
- Next-week predicted burnout-related risk: {display_risk}
- Prediction model: {prediction_model}

Priority factors to write tips for, highest first: {priority}
Low factors (one sentence only, no tip list): {low_note}

System rules:
{rules_text or "Follow the academic-support rules. Do not diagnose. Do not invent contacts."}

Retrieved evidence-based knowledge:
{chr(10).join(context_blocks)}

Write to the student as a caring adviser using HOW THIS MAY FEEL and STUDENT-FACING GUIDANCE.
Use simple, basic words. Short sentences. No hard or deep words.
Write about NEXT WEEK. Do not say this week's MFBI is next week's prediction.
Match tips to the four questionnaires the student filled out:
PSS-10 (unexpected upset, control, nervous and stressed, could not cope, difficulties piling up),
Academic Workload (how heavy this week, overlapping deadlines, assignments taking more time, volume of requirements, quizzes/projects/readings),
Study Time (hours studied, assignments and projects, reviewing lessons, focused sessions),
Sleep Hours (hours slept per night, consistent schedule, rested upon waking, lack of sleep affecting academics).
assessment_summary must start with how next week looks, using the next-week predicted score {display_score:.2f}.
contributing_factors must be full easy sentences about how next week may feel.
Never copy the labels "Academic workload", "Sleep/rest", "Study time", or "Stress".
Never use: overwhelmed, strategies, implementing, utilize, appraisal, cognitive.
Each recommended_action should be one kind, simple thing they can do next week.
Do not start tips with "If you...". Speak directly: "Try to sleep about 7 hours..." / "Write the due dates..."
human_support should sound like a person, not a policy notice. Do not say "I know."
Do not change the next-week predicted score or risk.
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
            mfbi_score,
            risk_level,
            labels,
            f"llm_error:{exc.__class__.__name__}",
            chunks,
            **fallback_kwargs,
        )

    fallback = fallback_recommendation(
        mfbi_score, risk_level, labels, "llm_sanitize", chunks, **fallback_kwargs
    )
    return _apply_llm_output(
        parsed, fallback, labels, source_titles, outlook_score=display_score
    )
