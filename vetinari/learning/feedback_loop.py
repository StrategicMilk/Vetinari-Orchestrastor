"""
Feedback Loop - Vetinari Self-Improvement Subsystem

Closes the learning loop: execution outcomes → model performance updates.
Feeds quality scores back into ModelPerformance table and DynamicModelRouter.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Closes the feedback loop between execution outcomes and model selection.

    After each task completes, this component:
    1. Fetches the quality score for the task
    2. Updates ModelPerformance in the memory store (running EMA)
    3. Updates the DynamicModelRouter performance cache
    4. Optionally updates the Ponder scoring system
    """

    EMA_ALPHA = 0.3  # Exponential moving average weight for recent performance

    def __init__(self):
        self._memory = None
        self._router = None

    def _get_memory(self):
        if self._memory is None:
            try:
                from vetinari.memory import get_memory_store
                self._memory = get_memory_store()
            except Exception:
                pass
        return self._memory

    def _get_router(self):
        if self._router is None:
            try:
                from vetinari.dynamic_model_router import get_model_router
                self._router = get_model_router()
            except Exception:
                pass
        return self._router

    def record_outcome(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        quality_score: float,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
    ) -> None:
        """
        Record a task outcome and propagate it to all performance tracking systems.

        Args:
            task_id: The task identifier.
            model_id: The model used.
            task_type: Type of task (coding, research, etc.).
            quality_score: Overall quality score 0.0-1.0.
            latency_ms: Execution latency in milliseconds.
            cost_usd: Estimated cost in USD.
            success: Whether the task succeeded.
        """
        logger.debug(
            f"[FeedbackLoop] Recording outcome: task={task_id} model={model_id} "
            f"type={task_type} quality={quality_score:.2f} success={success}"
        )

        # 1. Update memory store ModelPerformance
        self._update_memory_performance(model_id, task_type, quality_score, latency_ms, success)

        # 2. Update DynamicModelRouter performance cache
        self._update_router_cache(model_id, task_type, quality_score, latency_ms, success)

        # 3. Update SubtaskMemory outcome with quality score
        self._update_subtask_quality(task_id, quality_score, success)

    def _update_memory_performance(
        self, model_id: str, task_type: str, quality: float,
        latency_ms: int, success: bool
    ) -> None:
        """Update the ModelPerformance running averages in the memory store."""
        mem = self._get_memory()
        if mem is None:
            return
        try:
            existing = mem.get_model_performance(model_id, task_type) or {}
            old_rate = existing.get("success_rate", 0.7)
            old_latency = existing.get("avg_latency", latency_ms or 1000)
            old_uses = existing.get("total_uses", 0)

            # Exponential moving average update
            new_rate = (1 - self.EMA_ALPHA) * old_rate + self.EMA_ALPHA * (1.0 if success else 0.0)
            new_latency = (1 - self.EMA_ALPHA) * old_latency + self.EMA_ALPHA * (latency_ms or old_latency)
            # Incorporate quality score into success rate
            new_rate = (new_rate + quality) / 2

            # Pass dict-form update (new signature)
            mem.update_model_performance(model_id, task_type, {
                "success_rate": round(new_rate, 4),
                "avg_latency": int(new_latency),
                "total_uses": old_uses + 1,
            })
        except Exception as e:
            logger.debug(f"Memory performance update failed: {e}")

    def _update_router_cache(
        self, model_id: str, task_type: str, quality: float,
        latency_ms: int, success: bool
    ) -> None:
        """Update the DynamicModelRouter's in-memory performance cache."""
        router = self._get_router()
        if router is None:
            return
        try:
            cache_key = f"{model_id}:{task_type}"
            existing = router._performance_cache.get(cache_key, {})
            old_rate = existing.get("success_rate", 0.7)
            old_latency = existing.get("avg_latency_ms", latency_ms or 1000)

            new_rate = (1 - self.EMA_ALPHA) * old_rate + self.EMA_ALPHA * float(success)
            new_rate = (new_rate + quality) / 2
            new_latency = (1 - self.EMA_ALPHA) * old_latency + self.EMA_ALPHA * (latency_ms or old_latency)

            router._performance_cache[cache_key] = {
                "success_rate": round(new_rate, 4),
                "avg_latency_ms": int(new_latency),
                "quality_score": round(quality, 4),
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"Router cache update failed: {e}")

    def _update_subtask_quality(self, task_id: str, quality: float, success: bool) -> None:
        """Annotate the SubtaskMemory record with quality score."""
        mem = self._get_memory()
        if mem is None:
            return
        try:
            mem.update_subtask_quality(task_id, quality_score=quality, succeeded=success)
        except Exception as e:
            logger.debug(f"Subtask quality update failed: {e}")


# Singleton
_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop
