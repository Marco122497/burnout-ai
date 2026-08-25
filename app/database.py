from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv()  # also allow process env / repo-root .env.local overrides

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Set SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) and "
        "SUPABASE_SERVICE_ROLE_KEY in burnout-ai/.env"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
