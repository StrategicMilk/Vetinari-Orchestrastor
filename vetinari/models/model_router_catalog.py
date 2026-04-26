"""Catalog mixin for DynamicModelRouter.

Provides model registration, performance tracking, health checks, and
query methods as a mixin so ``DynamicModelRouter`` can focus on selection logic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from vetinari.models.model_router_types import RouterModelInfo, TaskType

logger = logging.getLogger(__name__)

__all__ = ["ModelRouterCatalogMixin"]


class ModelRouterCatalogMixin:
    """Mixin providing catalog management for DynamicModelRouter.

    Requires the host class to define:
    - ``self.models: dict[str, RouterModelInfo]``
    - ``self._performance_cache: dict[str, dict[str, Any]]``
    - ``self._selection_history`` (deque or list)
    - ``self._health_check_callback: Callable | None``
    - ``self.max_latency_ms: float``
    """

    def register_model(self, model: RouterModelInfo) -> None:
        """Register a model in the router.

        Args:
            model: The RouterModelInfo instance to register.
        """
        self.models[model.id] = model
        logger.debug("Registered model: %s", model.id)

    def register_models_from_pool(self, models: list[dict[str, Any]]) -> None:
        """Register models from a model pool (list of dicts).

        Args:
            models: List of model dictionaries to register.
        """
        for m in models:
            model_info = RouterModelInfo.from_dict(m)
            self.register_model(model_info)
        logger.info("Registered %s models from pool", len(models))

    def set_health_check_callback(self, callback: Callable) -> None:
        """Set a callback for health checking models.

        Args:
            callback: Callable that accepts a model ID and returns health status.
        """
        self._health_check_callback = callback

    def update_model_performance(
        self,
        model_id: str,
        latency_ms: float,
        success: bool,
        task_type: TaskType | None = None,
    ) -> None:
        """Update performance metrics for a model.

        Args:
            model_id: The model to update.
            latency_ms: Observed latency in milliseconds.
            success: Whether the inference succeeded.
            task_type: Optional task type for cache keying.
        """
        if model_id not in self.models:
            return

        model = self.models[model_id]

        total = model.total_uses + 1
        model.avg_latency_ms = (model.avg_latency_ms * model.total_uses + latency_ms) / total
        model.success_rate = (model.success_rate * model.total_uses + (1 if success else 0)) / total
        model.total_uses = total
        model.last_checked = datetime.now(timezone.utc).isoformat()

        cache_key = f"{model_id}:{task_type.value if task_type else 'general'}"
        self._performance_cache[cache_key] = {
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": time.time(),
        }

    def get_performance_cache(self, cache_key: str) -> dict[str, Any]:
        """Retrieve cached performance data for a model/task-type key.

        Args:
            cache_key: Cache key in the form ``model_id:task_type``.

        Returns:
            Dict with ``success_rate``, ``avg_latency_ms``, etc., or empty dict.
        """
        return dict(self._performance_cache.get(cache_key, {}))

    def update_performance_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        """Merge external performance data into the cache.

        Called by the FeedbackLoop to propagate EMA-smoothed metrics from
        learning subsystems into the router's selection cache.

        Args:
            cache_key: Cache key in the form ``model_id:task_type``.
            data: Performance fields to merge (``success_rate``,
                ``avg_latency_ms``, ``quality_score``, etc.).
        """
        existing = self._performance_cache.get(cache_key, {})
        existing.update(data)
        self._performance_cache[cache_key] = existing

    def get_model_by_id(self, model_id: str) -> RouterModelInfo | None:
        """Return the model registered under ``model_id``, or None.

        Args:
            model_id: Model identifier to look up.

        Returns:
            ``RouterModelInfo`` if found, ``None`` otherwise.
        """
        return self.models.get(model_id)

    def get_available_models(self) -> list[RouterModelInfo]:
        """Return all currently available models.

        Returns:
            List of ``RouterModelInfo`` objects where ``is_available`` is True.
        """
        return [m for m in self.models.values() if m.is_available]

    def get_models_by_capability(self, capability: str) -> list[RouterModelInfo]:
        """Return all available models with a specific capability.

        Args:
            capability: One of ``"code_gen"``, ``"reasoning"``, or ``"docs"``.

        Returns:
            List of available ``RouterModelInfo`` objects that have the capability.
        """
        results = []
        for model in self.models.values():
            if not model.is_available:
                continue
            caps = model.capabilities
            if (
                (capability == "code_gen" and caps.code_gen)
                or (capability == "reasoning" and caps.reasoning)
                or (capability == "docs" and caps.docs)
            ):
                results.append(model)
        return results

    def check_model_health(self, model_id: str) -> bool:
        """Check if a model is healthy.

        Args:
            model_id: The model identifier to health-check.

        Returns:
            True if the model is healthy, False otherwise.
        """
        if model_id not in self.models:
            return False

        model = self.models[model_id]

        if self._health_check_callback:
            try:
                return self._health_check_callback(model_id)
            except Exception as exc:
                logger.error("Health check failed for %s: %s", model_id, exc)

        if model.avg_latency_ms > self.max_latency_ms * 2:
            return False

        return not model.success_rate < 0.5

    def get_routing_stats(self) -> dict[str, Any]:
        """Return routing statistics.

        Returns:
            A dict with keys: ``total_selections`` (number of routing decisions
            made), ``models_used`` (per-model selection counts), ``available_models``
            (count of currently available models), and ``total_models`` (total
            registered models including unavailable ones).
        """
        total_selections = len(self._selection_history)

        model_counts: dict[str, int] = {}
        for sel in self._selection_history:
            model_id = sel["selected_model"]
            model_counts[model_id] = model_counts.get(model_id, 0) + 1

        return {
            "total_selections": total_selections,
            "models_used": model_counts,
            "available_models": len(self.get_available_models()),
            "total_models": len(self.models),
        }
