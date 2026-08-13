from fastapi import APIRouter, HTTPException

from backend.models.schemas import ClassifyRequest, ClassifyResponse, ClassificationResult
from backend.core.classifier import classifier
from backend.core.confidence import should_escalate
from backend.core.llm_escalation import escalate
from backend.core.cost_tracker import estimate_cost
from backend.core import feedback_store

router = APIRouter(prefix="/api", tags=["classification"])

SEVERITY_RANK = ["INFO", "WARNING", "ERROR", "CRITICAL", "SECURITY_ALERT"]


@router.post("/classify", response_model=ClassifyResponse)
def classify_logs(payload: ClassifyRequest):
    if not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python -m backend.ml.train_classifier",
        )

    if len(payload.logs) == 0:
        raise HTTPException(status_code=400, detail="No logs provided")

    results = []
    escalated_count = 0

    for entry in payload.logs:
        text = entry.raw_text.strip()
        if not text:
            continue

        predicted_label, confidence = classifier.predict(text)
        escalate_flag = should_escalate(predicted_label, confidence)

        if escalate_flag:
            llm_result = escalate(text)
            final_label = llm_result["label"]
            reasoning = llm_result.get("reasoning")
            routed_to = "llm"
            escalated_count += 1
            # persist for the retrain feedback loop
            feedback_store.record_feedback(text, final_label)
        else:
            final_label = predicted_label
            reasoning = None
            routed_to = "classical_ml"

        feedback_store.update_stats(final_label, routed_to)

        results.append(ClassificationResult(
            raw_text=text,
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            routed_to=routed_to,
            final_label=final_label,
            severity=final_label,
            reasoning=reasoning,
        ))

    total = len(results)
    cost = estimate_cost(total, escalated_count)

    return ClassifyResponse(
        results=results,
        total=total,
        escalated=escalated_count,
        escalation_rate=round(escalated_count / total, 4) if total else 0.0,
        estimated_cost_usd=cost,
    )
