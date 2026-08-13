from fastapi import APIRouter

from backend.models.schemas import StatsResponse
from backend.core import feedback_store
from backend.core.cost_tracker import cost_per_1000

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    stats = feedback_store.get_stats()
    avg_cost = cost_per_1000(stats["total_processed"], stats["total_escalated"])
    return StatsResponse(
        total_processed=stats["total_processed"],
        total_escalated=stats["total_escalated"],
        escalation_rate=stats["escalation_rate"],
        avg_cost_per_1000_events=avg_cost,
        label_distribution=stats["label_distribution"],
        routing_distribution=stats["routing_distribution"],
    )
