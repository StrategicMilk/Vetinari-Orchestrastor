"""Model scoring logic extracted from DynamicModelRouter.

Provides standalone scoring functions that can be imported and used
without instantiating the full router.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vetinari.models.dynamic_model_router import ModelInfo, ModelProvider, TaskType

logger = logging.getLogger(__name__)


def score_model(
    model: "ModelInfo",
    task_type: "TaskType",
    task_description: str,
    preferred_models: List[str],
    prefer_local: bool = True,
    ponder_engine=None,
) -> float:
    """Score a model for a given task with a minimum floor of 0.1.

    Args:
        model: The ModelInfo to score.
        task_type: TaskType enum value for the current task.
        task_description: Human-readable description used by PonderEngine.
        preferred_models: Ordered list of preferred model IDs (index 0 = most preferred).
        prefer_local: When True, local/LMStudio models receive a provider bonus.
        ponder_engine: Optional PonderEngine instance for blended scoring.

    Returns:
        Float score >= 0.1.
    """
    ponder_score = _ponder_score(ponder_engine, model, task_description)
    if ponder_score is not None:
        internal = _internal_score(model, task_type, task_description, preferred_models, prefer_local)
        raw = 0.50 * ponder_score + 0.50 * internal
    else:
        raw = _internal_score(model, task_type, task_description, preferred_models, prefer_local)

    return max(raw, 0.1)


def _ponder_score(ponder_engine, model: "ModelInfo", task_description: str) -> Optional[float]:
    """Ask PonderEngine for a score; returns None if unavailable or failed."""
    if ponder_engine is None:
        return None
    try:
        return ponder_engine.score_model(model.id, task_description)
    except Exception as e:
        logger.debug(f"PonderEngine scoring failed for {model.id}: {e}")
        return None


def _internal_score(
    model: "ModelInfo",
    task_type: "TaskType",
    task_description: str,
    preferred_models: Optional[List[str]],
    prefer_local: bool = True,
) -> float:
    """Internal scoring algorithm (original DynamicModelRouter algorithm)."""
    # Import here to avoid circular imports at module level
    from vetinari.models.dynamic_model_router import ModelProvider

    score = 0.0

    # Capability match (40%)
    capability_score = model.capabilities.matches_task(task_type)
    score += 0.40 * capability_score

    # Preference match (20%)
    if preferred_models and model.id in preferred_models:
        pref_index = preferred_models.index(model.id)
        preference_score = 1.0 - (pref_index * 0.3)
        score += 0.20 * preference_score

    # Performance (20%) - incorporates Thompson Sampling bonus when available
    if model.total_uses > 0:
        perf_score = model.success_rate * (1.0 - min(model.avg_latency_ms / 60000, 1.0))
        score += 0.20 * perf_score
    else:
        score += 0.10  # Neutral for unknown performance

    # Thompson Sampling bonus (up to +0.10)
    try:
        from vetinari.learning.model_selector import get_thompson_selector
        ts = get_thompson_selector()
        task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
        arm = ts._arms.get(f"{model.id}:{task_type_str}")
        if arm is not None and (arm.alpha + arm.beta) > 2:
            ts_bonus = arm.mean * 0.10
            score += ts_bonus
    except Exception:
        pass

    # Provider preference (10%)
    if prefer_local:
        if model.provider in (ModelProvider.LOCAL, ModelProvider.LMSTUDIO):
            score += 0.10
        elif model.provider == ModelProvider.OTHER:
            score += 0.05
    else:
        score += 0.10

    # Context length fit (10%)
    if model.context_length >= 8192:
        score += 0.10
    elif model.context_length >= 4096:
        score += 0.05

    return score


def calculate_confidence(scored: list) -> float:
    """Calculate selection confidence from a scored list of (model, score) tuples."""
    if len(scored) < 2:
        return 0.5

    best_score = scored[0][1]
    second_score = scored[1][1]

    if best_score == 0:
        return 0.1

    gap = best_score - second_score
    confidence = min(1.0, gap * 2 + 0.3)
    return confidence


def generate_reasoning(model: "ModelInfo", task_type: "TaskType", score: float) -> str:
    """Generate human-readable reasoning for a model selection."""
    from vetinari.models.dynamic_model_router import ModelProvider, TaskType

    reasons = []

    caps = model.capabilities
    if task_type == TaskType.CODING and caps.code_gen:
        reasons.append("excellent code generation")
    elif task_type == TaskType.REASONING and caps.reasoning:
        reasons.append("strong reasoning capabilities")
    elif task_type == TaskType.DOCUMENTATION and caps.docs:
        reasons.append("good documentation skills")

    if model.total_uses > 10:
        reasons.append(f"proven track record ({model.total_uses} uses)")
    if model.avg_latency_ms > 0 and model.avg_latency_ms < 5000:
        reasons.append(f"fast response ({model.avg_latency_ms:.0f}ms)")

    if model.provider in (ModelProvider.LOCAL, ModelProvider.LMSTUDIO):
        reasons.append("local model (no API costs)")

    if not reasons:
        reasons.append("best available match")

    return f"Selected {model.id}: {', '.join(reasons)}"
