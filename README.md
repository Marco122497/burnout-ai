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
```

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
