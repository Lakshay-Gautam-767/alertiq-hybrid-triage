"""
Central configuration for the Hybrid Log/Anomaly Classification System.
Change thresholds / cost assumptions / LLM provider settings here — nowhere else.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Classical ML classifier settings
# ---------------------------------------------------------------------------
MODEL_DIR = BASE_DIR / "ml" / "artifacts"
MODEL_PATH = MODEL_DIR / "classifier.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"

TRAIN_DATA_PATH = BASE_DIR / "ml" / "data" / "sample_logs.csv"
FEEDBACK_DATA_PATH = BASE_DIR / "ml" / "data" / "feedback_logs.csv"

# ---------------------------------------------------------------------------
# Confidence thresholding — the heart of the "hybrid" design
# ---------------------------------------------------------------------------
# If the classical model's top confidence is BELOW this, escalate to LLM.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

# Even if confidence is high, ALWAYS escalate these severities for a second
# opinion (e.g. anything that could mean production is on fire).
ALWAYS_ESCALATE_SEVERITIES = {"CRITICAL"}

# ---------------------------------------------------------------------------
# LLM escalation settings (Groq - OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# If no API key is set, the system falls back to a rule-based mock LLM so
# the demo still works end-to-end without any credentials.
USE_MOCK_LLM = GROQ_API_KEY == ""

# ---------------------------------------------------------------------------
# Cost model — used purely to show the "cost per 1,000 events" dashboard
# metric that clients actually care about. Tune these to real pricing.
# ---------------------------------------------------------------------------
COST_PER_1000_CLASSICAL = float(os.getenv("COST_PER_1000_CLASSICAL", 0.02))   # CPU inference
COST_PER_1000_LLM = float(os.getenv("COST_PER_1000_LLM", 2.50))               # LLM API calls

# ---------------------------------------------------------------------------
# CORS (frontend index.html is opened as a static file / different origin)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = ["*"]
