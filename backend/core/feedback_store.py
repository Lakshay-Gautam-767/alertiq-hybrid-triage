"""
Two responsibilities:
1. Persist every LLM-labeled example to a feedback CSV so the classical
   model can be retrained on cases it originally got wrong/unsure about.
2. Keep simple in-memory running stats for the dashboard endpoint.
   (For a real client deployment, swap this for Redis/Postgres — the
   interface below stays the same, only the storage backend changes.)
"""
import csv
from collections import Counter
from pathlib import Path
from threading import Lock

from backend import config

_lock = Lock()

_stats = {
    "total_processed": 0,
    "total_escalated": 0,
    "label_distribution": Counter(),
    "routing_distribution": Counter(),
}


def record_feedback(raw_text: str, label: str) -> None:
    config.FEEDBACK_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = config.FEEDBACK_DATA_PATH.exists()
    with _lock:
        with open(config.FEEDBACK_DATA_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["text", "label"])
            writer.writerow([raw_text, label])


def update_stats(label: str, routed_to: str) -> None:
    with _lock:
        _stats["total_processed"] += 1
        if routed_to == "llm":
            _stats["total_escalated"] += 1
        _stats["label_distribution"][label] += 1
        _stats["routing_distribution"][routed_to] += 1


def get_stats() -> dict:
    with _lock:
        total = _stats["total_processed"]
        escalated = _stats["total_escalated"]
        return {
            "total_processed": total,
            "total_escalated": escalated,
            "escalation_rate": round(escalated / total, 4) if total else 0.0,
            "label_distribution": dict(_stats["label_distribution"]),
            "routing_distribution": dict(_stats["routing_distribution"]),
        }


def count_feedback_rows() -> int:
    if not config.FEEDBACK_DATA_PATH.exists():
        return 0
    with open(config.FEEDBACK_DATA_PATH, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header
