from __future__ import annotations

from typing import Any, Dict


SOURCE_WEIGHTS = {
    "dataset": 0.30,
    "gemini": 0.40,
    "rag": 0.20,
    "business_rules": 0.10,
}


def fuse_candidate_scores(components: Dict[str, float | None]) -> Dict[str, Any]:
    available = {
        name: float(score)
        for name, score in components.items()
        if name in SOURCE_WEIGHTS and score is not None
    }
    if not available:
        return {
            "final_score": None,
            "recommendation": "Insufficient evidence",
            "used_weights": {},
            "components": components,
        }

    total_weight = sum(SOURCE_WEIGHTS[name] for name in available)
    normalized_weights = {
        name: SOURCE_WEIGHTS[name] / total_weight for name in available
    }
    final_score = sum(
        available[name] * normalized_weights[name] for name in available
    )
    recommendation = (
        "Advance" if final_score >= 65
        else "Review" if final_score >= 45
        else "Hold"
    )
    return {
        "final_score": round(final_score, 1),
        "recommendation": recommendation,
        "used_weights": normalized_weights,
        "components": components,
    }
