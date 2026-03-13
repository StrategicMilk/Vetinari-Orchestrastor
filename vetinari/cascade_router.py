"""Cascade Model Router — vetinari.cascade_router.

Implements a cost-optimised routing strategy that starts inference with the
cheapest/smallest model and escalates to larger models when the response
confidence is below a configurable threshold.

Strategy
--------
1. Try cheapest model in the cascade chain.
2. Evaluate confidence of the response (heuristic or model-provided score).
3. If confidence >= threshold  →  return response (done, cheap).
4. If confidence <  threshold  →  escalate to next model in chain.
5. Repeat until the chain is exhausted or confidence is satisfied.

The caller always gets *a* response — the best one seen across the chain.

Integration with DynamicModelRouter
------------------------------------
``CascadeRouter`` can be used as a standalone router or wired into
``DynamicModelRouter`` as a routing strategy::

    from vetinari.cascade_router import CascadeRouter, get_cascade_router
    from vetinari.dynamic_model_router import DynamicModelRouter, TaskType

    cr = get_cascade_router()
    cr.add_tier("small-model", cost_per_1k=0.001, priority=0)
    cr.add_tier("medium-model", cost_per_1k=0.005, priority=1)
    cr.add_tier("large-model", cost_per_1k=0.015, priority=2)

    response, used_model = cr.route(request, adapter_fn=my_adapter)

Configuration via environment
------------------------------
- ``CASCADE_CONFIDENCE_THRESHOLD``  — minimum confidence to accept (default 0.7)
- ``CASCADE_MAX_ESCALATIONS``       — max number of escalation steps (default 2)
- ``CASCADE_ENABLED``               — set to "0" to disable cascading
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = float(os.environ.get("CASCADE_CONFIDENCE_THRESHOLD", "0.7"))
_MAX_ESCALATIONS = int(os.environ.get("CASCADE_MAX_ESCALATIONS", "2"))
_CASCADE_ENABLED = os.environ.get("CASCADE_ENABLED", "1").lower() not in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CascadeTier:
    """A single model tier in the cascade chain."""

    model_id: str
    cost_per_1k_tokens: float = 0.0
    priority: int = 0  # lower = tried first
    max_tokens_override: int | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeResult:
    """Result of a cascade routing attempt."""

    response: Any  # InferenceResponse
    model_id: str
    confidence: float
    escalation_count: int = 0
    total_latency_ms: float = 0.0
    tiers_tried: list[str] = field(default_factory=list)
    cost_saved_vs_largest: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "confidence": self.confidence,
            "escalation_count": self.escalation_count,
            "total_latency_ms": self.total_latency_ms,
            "tiers_tried": self.tiers_tried,
            "cost_saved_vs_largest": self.cost_saved_vs_largest,
        }


# ---------------------------------------------------------------------------
# Confidence estimators
# ---------------------------------------------------------------------------


def _heuristic_confidence(response_text: str) -> float:
    """Estimate confidence from response text heuristics.

    Signals of LOW confidence (reduce score):
    - Very short responses (< 20 chars)
    - Uncertainty language ("I'm not sure", "I don't know", "unclear")
    - Refusal phrases ("I cannot", "I'm unable")
    - Trailing incomplete sentences

    Returns float in [0.0, 1.0].
    """
    if not response_text:
        return 0.0

    text = response_text.strip()
    score = 1.0

    # Very short response
    if len(text) < 20:
        score -= 0.4

    # Uncertainty language
    uncertainty_patterns = [
        r"\bi('m| am) not sure\b",
        r"\bi don'?t know\b",
        r"\bit('s| is) unclear\b",
        r"\buncertain\b",
        r"\bI cannot (determine|say|tell)\b",
        r"\bI'?m unable to\b",
        r"\bI lack (the |)information\b",
        r"\bcannot (answer|provide|help)\b",
    ]
    for pattern in uncertainty_patterns:
        if re.search(pattern, text, re.I):
            score -= 0.2

    # Refusal phrases
    refusal_patterns = [
        r"\bI can'?t (help|do|provide)\b",
        r"\bI (am|'m) not able to\b",
        r"\bThis (is|seems) beyond\b",
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, text, re.I):
            score -= 0.15

    # Trailing incomplete sentence (ends mid-sentence without punctuation)
    if text and text[-1] not in ".!?\"'`":
        score -= 0.05

    # Repetition (same sentence repeated 3+ times)
    sentences = re.split(r"[.!?]+", text)
    if len(sentences) > 3:
        unique = len({s.strip().lower() for s in sentences if s.strip()})
        if unique < len(sentences) * 0.5:
            score -= 0.2

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# CascadeRouter
# ---------------------------------------------------------------------------


class CascadeRouter:
    """Cost-optimising cascade router.

    Tries the cheapest model first; escalates to more capable models
    only when confidence is below the threshold.

    Thread-safe. Can be used standalone or integrated into DynamicModelRouter.
    """

    def __init__(
        self,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        max_escalations: int = _MAX_ESCALATIONS,
        enabled: bool = _CASCADE_ENABLED,
        confidence_estimator: Callable[[str], float] | None = None,
    ):
        """Args:.

        confidence_threshold: Accept response if confidence >= this value.
        max_escalations: Maximum number of escalation steps after first attempt.
        enabled: If False, always use the cheapest tier (no escalation).
        confidence_estimator: Custom function(response_text) -> float [0,1].
                              Defaults to built-in heuristic estimator.
        """
        self.confidence_threshold = confidence_threshold
        self.max_escalations = max_escalations
        self.enabled = enabled
        self._estimate_confidence = confidence_estimator or _heuristic_confidence

        self._tiers: list[CascadeTier] = []
        self._lock = threading.RLock()

        # Stats
        self._stats = {
            "total_requests": 0,
            "escalations": 0,
            "accepted_at_tier": {},  # tier_index -> count
            "total_cost_usd": 0.0,
        }

        logger.info(
            "CascadeRouter initialized (threshold=%.2f, max_escalations=%d, enabled=%s)",
            confidence_threshold,
            max_escalations,
            enabled,
        )

    def add_tier(  # noqa: D417
        self, model_id: str, cost_per_1k_tokens: float = 0.0, priority: int | None = None, **kwargs
    ) -> CascadeRouter:
        """Add a model tier to the cascade chain.

        Args:
            model_id: Model identifier.
            cost_per_1k_tokens: Cost for cost-saving calculations.
            priority: Explicit priority (lower = tried first). Defaults to
                      current number of tiers (appended at end).

        Returns:
            self (fluent interface).
        """
        with self._lock:
            if priority is None:
                priority = len(self._tiers)
            tier = CascadeTier(
                model_id=model_id,
                cost_per_1k_tokens=cost_per_1k_tokens,
                priority=priority,
                **{k: v for k, v in kwargs.items() if k in ("max_tokens_override", "tags", "metadata")},
            )
            self._tiers.append(tier)
            self._tiers.sort(key=lambda t: t.priority)
            logger.debug(
                "CascadeRouter: added tier %s (priority=%d, cost=%.4f)", model_id, priority, cost_per_1k_tokens
            )
        return self

    def get_tiers(self) -> list[CascadeTier]:
        """Return ordered list of tiers (cheapest first)."""
        with self._lock:
            return list(self._tiers)

    def route(
        self,
        request: Any,
        adapter_fn: Callable[[Any], Any],
        context: dict[str, Any] | None = None,
    ) -> CascadeResult:
        """Route a request through the cascade chain.

        Args:
            request: InferenceRequest (or any object with .model_id and .max_tokens).
            adapter_fn: Callable(request) -> InferenceResponse. Called for each tier.
            context: Optional context dict for logging/metadata.

        Returns:
            CascadeResult with the accepted response and routing metadata.
        """
        with self._lock:
            tiers = list(self._tiers)

        if not tiers:
            raise ValueError("CascadeRouter has no tiers configured. Call add_tier() first.")

        start_total = time.monotonic()
        tiers_tried: list[str] = []
        best_response = None
        best_confidence = 0.0
        best_model = tiers[0].model_id
        escalation_count = 0
        largest_tier_cost = tiers[-1].cost_per_1k_tokens if tiers else 0.0

        max_tiers = 1 + self.max_escalations if self.enabled else 1

        for i, tier in enumerate(tiers[:max_tiers]):
            # Clone request with this tier's model_id
            tier_request = self._apply_tier(request, tier)
            tiers_tried.append(tier.model_id)

            try:
                response = adapter_fn(tier_request)
            except Exception as exc:
                logger.warning("CascadeRouter: tier %s failed: %s", tier.model_id, exc)
                escalation_count += 1
                continue

            output_text = getattr(response, "output", "") or ""
            status = getattr(response, "status", "ok")

            if status != "ok":
                logger.debug("CascadeRouter: tier %s returned status=%s, escalating", tier.model_id, status)
                escalation_count += 1
                if best_response is None:
                    best_response = response
                    best_model = tier.model_id
                continue

            confidence = self._estimate_confidence(output_text)
            logger.debug(
                "CascadeRouter: tier %s confidence=%.3f (threshold=%.2f)",
                tier.model_id,
                confidence,
                self.confidence_threshold,
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_response = response
                best_model = tier.model_id

            if not self.enabled or confidence >= self.confidence_threshold:
                # Accepted — no escalation needed
                self._record_accepted(i)
                break

            # Escalate
            if i < max_tiers - 1:
                logger.info(
                    "CascadeRouter: confidence %.3f < %.2f, escalating %s -> %s",
                    confidence,
                    self.confidence_threshold,
                    tier.model_id,
                    tiers[i + 1].model_id if i + 1 < len(tiers) else "end",
                )
                escalation_count += 1
                self._record_accepted(None)
            else:
                # Exhausted chain
                self._record_accepted(i)

        total_latency = (time.monotonic() - start_total) * 1000

        # Calculate cost saved vs always using largest tier
        used_tier_cost = next((t.cost_per_1k_tokens for t in tiers if t.model_id == best_model), 0.0)
        tokens = getattr(best_response, "tokens_used", 0) or 0
        cost_saved = max(0.0, (largest_tier_cost - used_tier_cost) * tokens / 1000.0)

        with self._lock:
            self._stats["total_requests"] += 1
            self._stats["escalations"] += escalation_count
            self._stats["total_cost_usd"] += used_tier_cost * tokens / 1000.0

        if best_response is None:
            # Absolute fallback — should not happen
            from vetinari.adapters.base import InferenceResponse

            best_response = InferenceResponse(
                model_id=tiers[0].model_id,
                output="",
                latency_ms=int(total_latency),
                tokens_used=0,
                status="error",
                error="All cascade tiers failed",
            )

        return CascadeResult(
            response=best_response,
            model_id=best_model,
            confidence=best_confidence,
            escalation_count=escalation_count,
            total_latency_ms=total_latency,
            tiers_tried=tiers_tried,
            cost_saved_vs_largest=cost_saved,
        )

    def _apply_tier(self, request: Any, tier: CascadeTier) -> Any:
        """Return a copy of the request with the tier's model_id applied."""
        import copy

        new_req = copy.copy(request)
        new_req.model_id = tier.model_id
        if tier.max_tokens_override is not None:
            new_req.max_tokens = tier.max_tokens_override
        return new_req

    def _record_accepted(self, tier_index: int | None) -> None:
        if tier_index is not None:
            key = str(tier_index)
            with self._lock:
                self._stats["accepted_at_tier"][key] = self._stats["accepted_at_tier"].get(key, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["tiers"] = [
                {"model_id": t.model_id, "priority": t.priority, "cost_per_1k": t.cost_per_1k_tokens}
                for t in self._tiers
            ]
            stats["confidence_threshold"] = self.confidence_threshold
            stats["max_escalations"] = self.max_escalations
            stats["enabled"] = self.enabled
            if stats["total_requests"] > 0:
                stats["escalation_rate"] = stats["escalations"] / stats["total_requests"]
            else:
                stats["escalation_rate"] = 0.0
            return stats

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        with self._lock:
            self._stats = {
                "total_requests": 0,
                "escalations": 0,
                "accepted_at_tier": {},
                "total_cost_usd": 0.0,
            }


# ---------------------------------------------------------------------------
# Integration helper: wire into DynamicModelRouter
# ---------------------------------------------------------------------------


def build_cascade_from_router(
    dynamic_router: Any,
    task_type: Any,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    max_escalations: int = _MAX_ESCALATIONS,
) -> CascadeRouter:
    """Build a CascadeRouter from the models registered in a DynamicModelRouter,.

    ordered by cost (cheapest first).

    Args:
        dynamic_router: DynamicModelRouter instance.
        task_type: TaskType used to filter/sort candidate models.
        confidence_threshold: Confidence threshold for escalation.
        max_escalations: Max escalation steps.

    Returns:
        Configured CascadeRouter.
    """
    cr = CascadeRouter(
        confidence_threshold=confidence_threshold,
        max_escalations=max_escalations,
    )

    # Get all available models, sorted by cost
    models = dynamic_router.get_available_models()
    # Sort by cost_per_1k or latency as proxy for "cheapness"
    models_sorted = sorted(
        models,
        key=lambda m: (
            getattr(m, "metadata", {}).get("cost_per_1k_tokens", 0.0) if hasattr(m, "metadata") else 0.0,
            getattr(m, "avg_latency_ms", 0.0),
        ),
    )

    for priority, model in enumerate(models_sorted):
        cost = model.metadata.get("cost_per_1k_tokens", 0.0) if hasattr(model, "metadata") and model.metadata else 0.0
        cr.add_tier(model.id, cost_per_1k_tokens=cost, priority=priority)

    return cr


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_cascade_router: CascadeRouter | None = None
_cr_lock = threading.Lock()


def get_cascade_router(
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    max_escalations: int = _MAX_ESCALATIONS,
) -> CascadeRouter:
    """Get or create the global CascadeRouter instance."""
    global _cascade_router
    if _cascade_router is None:
        with _cr_lock:
            if _cascade_router is None:
                _cascade_router = CascadeRouter(
                    confidence_threshold=confidence_threshold,
                    max_escalations=max_escalations,
                )
    return _cascade_router


def reset_cascade_router() -> None:
    """Reset the global CascadeRouter (useful for testing)."""
    global _cascade_router
    with _cr_lock:
        _cascade_router = None
