"""
LLM escalation layer. Only called for logs the classical model is unsure
about (or that are always-escalate severities like CRITICAL).

If GROQ_API_KEY is not set, falls back to a deterministic mock so the
whole pipeline runs end-to-end without any API keys — useful for demos
and for client environments still procuring API access.
"""
import json
import requests

from backend import config

VALID_LABELS = ["INFO", "WARNING", "ERROR", "CRITICAL", "SECURITY_ALERT"]

SYSTEM_PROMPT = (
    "You are a senior SRE triaging a log/ticket/alert line that a lightweight "
    "classifier was NOT confident about. Classify it into exactly one of: "
    f"{', '.join(VALID_LABELS)}. Respond ONLY as compact JSON with keys "
    '"label" and "reasoning" (reasoning must be one short sentence).'
)


def _call_groq(text: str) -> dict:
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 150,
    }
    resp = requests.post(config.GROQ_BASE_URL, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def _mock_llm(text: str) -> dict:
    """Rule-based stand-in for a real LLM call — used when no API key is set."""
    lower = text.lower()
    if any(k in lower for k in ["breach", "unauthorized", "injection", "attack", "exploit"]):
        label = "SECURITY_ALERT"
    elif any(k in lower for k in ["down", "crash", "outage", "fatal", "panic"]):
        label = "CRITICAL"
    elif any(k in lower for k in ["fail", "exception", "error", "timeout"]):
        label = "ERROR"
    elif any(k in lower for k in ["slow", "retry", "deprecated", "warn"]):
        label = "WARNING"
    else:
        label = "INFO"
    return {
        "label": label,
        "reasoning": "[MOCK LLM — set GROQ_API_KEY in .env for real inference]",
    }


def escalate(text: str) -> dict:
    """Returns {"label": str, "reasoning": str}. Never raises — falls back to mock on error."""
    if config.USE_MOCK_LLM:
        return _mock_llm(text)
    try:
        result = _call_groq(text)
        if result.get("label") not in VALID_LABELS:
            result["label"] = "ERROR"
        return result
    except Exception as e:
        fallback = _mock_llm(text)
        fallback["reasoning"] = f"[LLM call failed, used mock fallback: {e}]"
        return fallback
