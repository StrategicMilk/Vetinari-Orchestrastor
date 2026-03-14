"""SLM/LLM Hybrid Model Router (C3).

==================================
Routes tasks to appropriately-sized models based on complexity heuristics.

Queries LM Studio ``/v1/models`` to discover available models, classifies
them by parameter count into tiers (small / medium / large), and routes
tasks to the cheapest tier that meets the quality threshold.

Tier classification:
  small  — ≤7B params   (routine extraction, formatting, simple Q&A)
  medium — 8B-32B params (code review, summarisation, moderate reasoning)
  large  — >32B params   (architecture, deep analysis, complex generation)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model size tiers."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class ModelInfo:
    """Discovered model metadata."""

    model_id: str
    tier: ModelTier = ModelTier.MEDIUM
    param_count_b: float = 0.0  # billions
    context_length: int = 4096
    is_coder: bool = False
    is_vision: bool = False
    last_seen: float = field(default_factory=time.monotonic)


# ── Task complexity heuristics ────────────────────────────────────────

_COMPLEXITY_KEYWORDS: dict[str, list[str]] = {
    "high": [
        "architect",
        "design system",
        "security audit",
        "deep analysis",
        "refactor entire",
        "complex",
        "multi-step",
        "ontological",
        "contrarian",
        "risk assessment",
        "lateral thinking",
    ],
    "medium": [
        "review",
        "summarize",
        "test generation",
        "code review",
        "document",
        "implement",
        "build",
        "database",
        "devops",
    ],
    "low": [
        "format",
        "extract",
        "simple",
        "list",
        "convert",
        "rename",
        "classify",
        "tag",
        "label",
        "count",
    ],
}

# Maps task complexity → minimum model tier
_COMPLEXITY_TO_TIER: dict[str, ModelTier] = {
    "high": ModelTier.LARGE,
    "medium": ModelTier.MEDIUM,
    "low": ModelTier.SMALL,
}

# Maps agent type → default tier (can be overridden by task complexity)
_AGENT_DEFAULT_TIER: dict[str, ModelTier] = {
    "PLANNER": ModelTier.LARGE,
    "CONSOLIDATED_RESEARCHER": ModelTier.MEDIUM,
    "CONSOLIDATED_ORACLE": ModelTier.LARGE,
    "BUILDER": ModelTier.MEDIUM,
    "QUALITY": ModelTier.MEDIUM,
    "OPERATIONS": ModelTier.MEDIUM,
}


def estimate_complexity(description: str) -> str:
    """Estimate task complexity from description keywords.

    Returns ``"high"``, ``"medium"``, or ``"low"``.

    Returns:
        The result string.
    """
    desc_lower = description.lower()
    for level in ("high", "medium", "low"):
        keywords = _COMPLEXITY_KEYWORDS[level]
        if any(kw in desc_lower for kw in keywords):
            return level
    return "medium"  # default


def _extract_param_count(model_id: str) -> float:
    """Try to extract parameter count in billions from model ID string."""
    # Match patterns like "7b", "14B", "72b", "30b-a3b", "1.5b"
    match = re.search(r"(\d+(?:\.\d+)?)[bB]\b", model_id)
    if match:
        return float(match.group(1))
    return 0.0


def _classify_tier(param_count_b: float) -> ModelTier:
    """Classify a model into a tier by parameter count."""
    if param_count_b <= 0:
        return ModelTier.MEDIUM  # unknown size → default
    if param_count_b <= 7:
        return ModelTier.SMALL
    if param_count_b <= 32:
        return ModelTier.MEDIUM
    return ModelTier.LARGE


class ModelRouter:
    """Routes tasks to appropriately-sized models.

    Usage::

        router = ModelRouter()
        model_id = router.select_model("BUILDER", "implement login feature")
    """

    def __init__(
        self,
        lmstudio_host: str | None = None,
        refresh_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
    ):
        self._host = lmstudio_host or os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")  # noqa: VET041
        self._refresh_interval = refresh_interval
        self._enabled = enabled
        self._models: dict[str, ModelInfo] = {}
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()
        self._tier_preference: dict[ModelTier, list[str]] = {
            ModelTier.SMALL: [],
            ModelTier.MEDIUM: [],
            ModelTier.LARGE: [],
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def available_models(self) -> dict[str, ModelInfo]:
        self._maybe_refresh()
        return dict(self._models)

    def select_model(
        self,
        agent_type: str,
        task_description: str = "",
        preferred_tier: ModelTier | None = None,
    ) -> str | None:
        """Select the best model for an agent + task combination.

        Args:
            agent_type: The agent type string (e.g. ``"BUILDER"``).
            task_description: Task description for complexity estimation.
            preferred_tier: Force a specific tier (overrides heuristics).

        Returns:
            A model ID string, or None if no models are available.
        """
        if not self._enabled:
            return None

        self._maybe_refresh()

        # Determine the desired tier
        if preferred_tier:
            tier = preferred_tier
        else:
            complexity = estimate_complexity(task_description)
            complexity_tier = _COMPLEXITY_TO_TIER.get(complexity, ModelTier.MEDIUM)
            agent_tier = _AGENT_DEFAULT_TIER.get(agent_type, ModelTier.MEDIUM)
            # Use the higher of the two
            tier_order = [ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]
            tier = max(complexity_tier, agent_tier, key=lambda t: tier_order.index(t))

        # Find best model in the desired tier, falling back to adjacent tiers
        model_id = self._find_model_in_tier(tier, agent_type)
        if model_id:
            logger.info(
                "ModelRouter selected %s (tier=%s) for %s",
                model_id,
                tier.value,
                agent_type,
            )
            return model_id

        # Fallback: try any available model
        if self._models:
            fallback = next(iter(self._models))
            logger.warning(
                "ModelRouter falling back to %s (no %s tier model available)",
                fallback,
                tier.value,
            )
            return fallback

        return None

    def get_model_tier(self, model_id: str) -> ModelTier:
        """Get the tier of a specific model.

        Returns:
            The ModelTier result.
        """
        info = self._models.get(model_id)
        if info:
            return info.tier
        # Try to infer from name
        param_count = _extract_param_count(model_id)
        return _classify_tier(param_count)

    def get_summary(self) -> dict[str, Any]:
        """Dashboard-friendly summary.

        Returns:
            The result string.
        """
        self._maybe_refresh()
        tiers: dict[str, list[str]] = {"small": [], "medium": [], "large": []}
        for mid, info in self._models.items():
            tiers[info.tier.value].append(mid)
        return {
            "enabled": self._enabled,
            "total_models": len(self._models),
            "tiers": tiers,
            "host": self._host,
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _find_model_in_tier(self, tier: ModelTier, agent_type: str) -> str | None:
        """Find a model in the given tier, preferring coder models for BUILDER/QUALITY."""
        candidates = [(mid, info) for mid, info in self._models.items() if info.tier == tier]
        if not candidates:
            # Try adjacent tiers (prefer larger)
            tier_order = [ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]
            idx = tier_order.index(tier)
            for offset in (1, -1, 2, -2):
                adj_idx = idx + offset
                if 0 <= adj_idx < len(tier_order):
                    adj_tier = tier_order[adj_idx]
                    candidates = [(mid, info) for mid, info in self._models.items() if info.tier == adj_tier]
                    if candidates:
                        break

        if not candidates:
            return None

        # Prefer coder models for code-heavy agents
        prefer_coder = agent_type in ("BUILDER", "QUALITY")
        if prefer_coder:
            coder_candidates = [(mid, info) for mid, info in candidates if info.is_coder]
            if coder_candidates:
                return coder_candidates[0][0]

        return candidates[0][0]

    def _maybe_refresh(self) -> None:
        """Refresh model list if stale."""
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval:
            return
        with self._lock:
            # Double-check after acquiring lock
            if now - self._last_refresh < self._refresh_interval:
                return
            self._refresh_models()
            self._last_refresh = now

    def _refresh_models(self) -> None:
        """Query LM Studio /v1/models and update the model registry."""
        try:
            import requests

            url = f"{self._host}/v1/models"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            models_list = data.get("data", [])
            new_models: dict[str, ModelInfo] = {}

            for m in models_list:
                model_id = m.get("id", "")
                if not model_id:
                    continue
                param_count = _extract_param_count(model_id)
                tier = _classify_tier(param_count)
                is_coder = "coder" in model_id.lower() or "code" in model_id.lower()
                is_vision = "vl" in model_id.lower() or "vision" in model_id.lower()

                new_models[model_id] = ModelInfo(
                    model_id=model_id,
                    tier=tier,
                    param_count_b=param_count,
                    is_coder=is_coder,
                    is_vision=is_vision,
                )

            self._models = new_models
            logger.info(
                "ModelRouter refreshed: %d models (S=%d, M=%d, L=%d)",
                len(new_models),
                sum(1 for m in new_models.values() if m.tier == ModelTier.SMALL),
                sum(1 for m in new_models.values() if m.tier == ModelTier.MEDIUM),
                sum(1 for m in new_models.values() if m.tier == ModelTier.LARGE),
            )
        except Exception as e:
            logger.debug("ModelRouter refresh failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────

_model_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Get or create the global model router.

    Returns:
        The ModelRouter result.
    """
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router
