# Burnout AI service

FastAPI microservice for academic burnout early detection using trained Decision Tree and Random Forest models.

## Setup

```bash
cd burnout-ai
py -m venv venv
# Windows
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create `burnout-ai/.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM_MODEL=gpt-4o-mini
```

Do not commit this file. The service-role key and OpenAI key stay on the FastAPI server, never in the Next.js browser bundle.

## RAG knowledge base

1. Copy your knowledge files from `burnout_rag_knowledge_base/` into `burnout-ai/rag/documents/` (already done if you are using this repo).
   Current files:
   - `01_stress_management.txt`
   - `02_academic_workload.txt`
   - `03_sleep_management.txt`
   - `04_study_time_and_learning.txt`
   - `05_student_support_and_referral.txt`
   - `06_system_rag_rules.txt`
2. In the Supabase SQL editor, run `supabase/phase12-rag.sql`, then `supabase/phase13-rag-local-embeddings.sql` (384-d local embeddings).
3. Embeddings run locally (`BAAI/bge-small-en-v1.5` via fastembed). Keep `EMBEDDING_PROVIDER=local`. Ingest does not need OpenAI credits.
4. After retrieval, FastAPI calls the OpenAI LLM (`gpt-4o-mini` by default) to write the student-facing recommendation from those chunks. If the LLM is missing or out of credits, retrieved knowledge still produces a template recommendation.
5. Ingest chunks + embeddings:

```bash
npm run rag:ingest
```

6. Test retrieval only:

```bash
cd burnout-ai
python -m rag.retrieval "What should a student do when experiencing high stress?"
```

RAG never calculates burnout risk. The pipeline is MFBI → Random Forest / Decision Tree → RAG retrieval → LLM recommendations.

If the LLM or vector search fails, the API still returns a predefined fallback recommendation.

## Run with Next.js

From the repo root, Next.js alone (uses remote AI via `AI_API_URL`):

```bash
npm run dev
```

Optional: Next.js + local FastAPI together:

```bash
npm run dev:with-ai
```

AI only (local):

```bash
npm run ai
```

## Train models

```bash
npm run train
```

Trains same-week and next-week Decision Tree + Random Forest models, writes pickles to `models/`, and saves evaluation metrics to `models/metrics.json`.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service + model readiness |
| GET | `/metrics` | Trained evaluation metrics |
| POST | `/predict` | Current / next-week prediction |
| POST | `/predict/early-warning` | Current risk + next-week ML + week-2 trend projection |
| POST | `/api/burnout/recommendation` | Verify MFBI, confirm ML, RAG retrieval, LLM recommendation |
| GET | `/api/rag/search?q=` | Retrieval-only test (no LLM, no risk classification) |
| GET | `/docs` | OpenAPI docs |

### Early warning notes

- **Next week** uses the trained next-week model (scores + trends + MFBI) when prior-week data exists.
- **Week 2** is a **trend-based projection**, not a trained two-week-ahead model (dataset only labels next week).

## Next.js env

```env
AI_API_URL=https://burnout-ai-1.onrender.com
```

For local uvicorn instead: `AI_API_URL=http://127.0.0.1:8000`.

`BURNOUT_AI_URL` is also accepted as an alias.
