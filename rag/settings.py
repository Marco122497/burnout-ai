"""App settings stored in Supabase. RAG does not classify burnout risk."""

from __future__ import annotations

from app.database import supabase

OPENAI_LLM_SETTING_KEY = "openai_llm_enabled"


def openai_llm_enabled() -> bool:
    """Guidance Settings toggle. Default on if the row is missing."""
    try:
        response = (
            supabase.table("app_settings")
            .select("value")
            .eq("key", OPENAI_LLM_SETTING_KEY)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return True
        value = str(rows[0].get("value") or "true").strip().lower()
        return value not in {"0", "false", "off", "no"}
    except Exception:
        return True
