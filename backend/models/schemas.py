from typing import List, Optional
from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    raw_text: str = Field(..., description="Single raw log line / ticket / alert text")


class ClassifyRequest(BaseModel):
    logs: List[LogEntry]


class ClassificationResult(BaseModel):
    raw_text: str
    predicted_label: str
    confidence: float
    routed_to: str            # "classical_ml" | "llm"
    final_label: str          # label after any LLM override
    severity: str
    reasoning: Optional[str] = None   # only populated when routed_to == "llm"


class ClassifyResponse(BaseModel):
    results: List[ClassificationResult]
    total: int
    escalated: int
    escalation_rate: float
    estimated_cost_usd: float


class StatsResponse(BaseModel):
    total_processed: int
    total_escalated: int
    escalation_rate: float
    avg_cost_per_1000_events: float
    label_distribution: dict
    routing_distribution: dict


class FeedbackRetrainResponse(BaseModel):
    status: str
    new_training_samples: int
    message: str
