"""
Translates raw counts into the metric clients actually care about:
$ cost per 1,000 events, split by how many hit the cheap classical path
vs. the expensive LLM path.
"""
from backend import config


def estimate_cost(total: int, escalated: int) -> float:
    classical_count = total - escalated
    cost = (
        (classical_count / 1000) * config.COST_PER_1000_CLASSICAL
        + (escalated / 1000) * config.COST_PER_1000_LLM
    )
    return round(cost, 6)


def cost_per_1000(total: int, escalated: int) -> float:
    if total == 0:
        return 0.0
    cost = estimate_cost(total, escalated)
    return round((cost / total) * 1000, 4)
