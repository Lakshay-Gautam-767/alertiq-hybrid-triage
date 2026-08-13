"""
Routing decision: should this log go to the classical model's answer,
or get escalated to the LLM for a second opinion?

This is deliberately kept as a standalone, pure function — easy to unit
test and easy to tune without touching classifier or LLM code.
"""
from backend import config


def should_escalate(predicted_label: str, confidence: float) -> bool:
    if predicted_label in config.ALWAYS_ESCALATE_SEVERITIES:
        return True
    if confidence < config.CONFIDENCE_THRESHOLD:
        return True
    return False
