# AlertIQ — Hybrid Log & Anomaly Classification System

**A cost-aware log/alert triage pipeline that combines a fast classical ML classifier with LLM escalation — routing 80%+ of traffic through a near-free model and reserving the expensive LLM only for what actually needs it.**

**Stack:** Python 3.10+ · FastAPI 0.115 · scikit-learn 1.5 · Groq (Llama 3.1) · MIT License

---

## The Problem

When you deploy an application to the cloud, it can generate **thousands of log lines, alerts, and tickets** per day — `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `SECURITY_ALERT`. Feeding every single one of them to an LLM for triage works, but it doesn't scale economically: LLM calls are billed per token, and most log lines (a routine login, a scheduled backup completing) are trivially easy to classify.

**AlertIQ** solves this with a **hybrid routing architecture**: a cheap classical ML model handles the vast majority of "obvious" cases in milliseconds, and only genuinely uncertain or high-severity events are escalated to an LLM for a second opinion — cutting inference cost by 10–100x while keeping the safety net of LLM-grade reasoning where it matters.

---

## Key Features

- **🧠 Two-tier classification** — TF-IDF + Logistic Regression as the fast/cheap path, Groq-hosted Llama 3.1 as the smart/expensive path
- **🎯 Confidence-based routing** — logs below a configurable confidence threshold, or belonging to always-review severities (e.g. `CRITICAL`), are automatically escalated
- **🔁 Self-improving feedback loop** — every LLM escalation is logged and folded back into the training set on retrain, so the classical model gradually learns the patterns it used to be unsure about
- **💰 Built-in cost dashboard** — live "$ per 1,000 events" metric split by classical vs. LLM routing, so the economics of the system are always visible
- **🛡️ Zero-dependency demo mode** — if no LLM API key is configured, a deterministic rule-based mock stands in for the LLM so the entire pipeline runs end-to-end out of the box
- **📊 Live dashboard UI** — a single-page frontend showing routing split, per-log classification results, model confidence, and LLM reasoning

---

## How It Works

```
                        ┌──────────────────────────┐
                        │ Raw log / ticket / alert │
                        └──────────────────────────┘
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │ Classical Classifier         │
                      │ TF-IDF + Logistic Regression │
                      └──────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │ Confident enough AND not │
                        │ CRITICAL severity?       │
                        └──────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────┴───────────────┐
                       │   YES                     NO │
                       ▼                              ▼
          ┌────────────────────────┐      ┌──────────────────────┐
          │ Return classical label │      │ Escalate to LLM      │
          │ ~$0.02 / 1000 events   │      │ (Groq · Llama 3.1)   │
          │                        │      │ ~$2.50 / 1000 events │
          └────────────────────────┘      └──────────────────────┘
                                                      │
                                                      ▼
                                     ┌─────────────────────────────────┐
                                     │ Label + reasoning returned,     │
                                     │ written to feedback store (CSV) │
                                     └─────────────────────────────────┘
                                                      │
                                                      ▼
                                      ┌──────────────────────────────┐
                                      │ Retrain: folds feedback back │
                                      │ into the classical model     │
                                      └──────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | **FastAPI** (async, auto-generated OpenAPI docs at `/docs`) |
| Classical ML | **scikit-learn** — `TfidfVectorizer` + `LogisticRegression` |
| LLM Escalation | **Groq API** (OpenAI-compatible), model: `llama-3.1-8b-instant` |
| Model persistence | **joblib** |
| Data handling | **pandas** |
| Frontend | Vanilla HTML/CSS/JS single-page dashboard (no build step) |
| Config | `python-dotenv` + a single `config.py` source of truth |

---

## Project Structure

```
.
├── backend/
│   ├── api/
│   │   ├── routes_logs.py       # POST /api/classify
│   │   ├── routes_stats.py      # GET  /api/stats
│   │   └── routes_feedback.py   # POST /api/retrain
│   ├── core/
│   │   ├── classifier.py        # Classical model wrapper (load + predict)
│   │   ├── confidence.py        # Pure routing decision function
│   │   ├── llm_escalation.py    # Groq call + mock fallback
│   │   ├── cost_tracker.py      # $ / 1,000 events calculations
│   │   └── feedback_store.py    # Persists LLM labels + in-memory stats
│   ├── ml/
│   │   ├── train_classifier.py       # Trains + evaluates + saves the model
│   │   ├── generate_sample_data.py   # Synthetic dataset generator (demo only)
│   │   ├── data/sample_logs.csv      # Base training data
│   │   └── artifacts/                # classifier.joblib, vectorizer.joblib, label_encoder.joblib
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response contracts
│   ├── config.py                # Central config (thresholds, cost model, LLM settings)
│   ├── main.py                  # FastAPI app entrypoint
│   └── requirements.txt
├── frontend/
│   └── index.html               # Dashboard UI (routing split, metrics, results table)
└── README.md
```

---

## Getting Started

### 1. Clone & set up the backend

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 2. (Optional) Configure the LLM

Copy the example env file and add a Groq API key if you have one:

```bash
cp backend/.env.example backend/.env
```

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
CONFIDENCE_THRESHOLD=0.45
```

> **No API key? No problem.** If `GROQ_API_KEY` is left empty, the system automatically falls back to a deterministic rule-based mock LLM, so the entire pipeline — including escalation and reasoning — still runs end-to-end for demos and local testing.

### 3. Train the classical model

```bash
python -m backend.ml.train_classifier
```

This generates a synthetic labeled dataset (if one doesn't already exist), trains the TF-IDF + Logistic Regression pipeline, prints a held-out evaluation report, and saves the model artifacts to `backend/ml/artifacts/`.

### 4. Run the backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**.

### 5. Open the frontend

Open `frontend/index.html` directly in a browser (or serve it with any static server) and paste log lines from `sample.txt` — or use the built-in **"Load sample logs"** button — then click **Classify batch**.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/classify` | Classifies a batch of raw log/ticket lines, routes each through the classical model or LLM, and returns per-item results plus batch cost/escalation metrics |
| `GET` | `/api/stats` | Returns cumulative processed/escalated counts, escalation rate, label & routing distributions, and average cost per 1,000 events |
| `POST` | `/api/retrain` | Retrains the classical model on base data + accumulated LLM feedback, then hot-reloads it into memory |
| `GET` | `/` | Health check — reports whether the classifier is loaded and ready |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {"raw_text": "User priya logged in successfully from 192.168.1.5"},
      {"raw_text": "Production database payments-db is DOWN - all connections refused"}
    ]
  }'
```

---

## Configuration

All tunables live in `backend/config.py` and can be overridden via `backend/.env`:

| Variable | Default | Purpose |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.75` (0.45 in `.env.example`) | Classical model confidence below this triggers LLM escalation |
| `GROQ_API_KEY` | *(empty)* | Enables real LLM calls; falls back to mock mode if unset |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model used for escalated classification |
| `COST_PER_1000_CLASSICAL` | `0.02` | Assumed $ cost per 1,000 classical-model inferences |
| `COST_PER_1000_LLM` | `2.50` | Assumed $ cost per 1,000 LLM inferences |

The severity `CRITICAL` is **always** escalated to the LLM regardless of confidence — a deliberate design choice, since a false negative on a production-down alert is far more expensive than an extra LLM call.

---

## The Feedback Loop

This is the core design idea that makes the system *hybrid* rather than just "two models bolted together":

1. Every log the classical model was unsure about (or that hit an always-escalate severity) gets a ground-truth-quality label from the LLM.
2. That `(text, label)` pair is appended to `backend/ml/data/feedback_logs.csv`.
3. Calling `/api/retrain` merges this feedback into the base training set and retrains the classical model from scratch.
4. Over time, patterns the classifier used to escalate become patterns it's confident about — **shrinking the escalation rate, and therefore the LLM bill, the longer the system runs.**

---

## Roadmap

- [ ] Swap in-memory stats store for Redis/Postgres for multi-instance deployments
- [ ] Move retraining off the request thread into a scheduled/async job
- [ ] Add authentication + role-based access to the dashboard
- [ ] Support additional LLM providers (OpenAI, Gemini) as pluggable backends
- [ ] Real-time log ingestion via a message queue instead of manual batch paste

---

## License

MIT — free to use, modify, and build on.
