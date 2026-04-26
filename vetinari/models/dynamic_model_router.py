"""Dynamic Model Router for Vetinari.

Provides intelligent model selection based on task requirements, capability
matching, performance history, latency, cost, and availability.  Supports
both local and cloud models, configurable ``RoutingPolicy``, and an optional
``PonderEngine`` scoring backend via dependency injection.

This is step 2 of the model-selection pipeline:
Discovery -> **Routing** -> Inference.

All type definitions (``ModelCapabilities``, ``ModelInfo``, ``ModelSelection``,
``RoutingPolicy``, ``TaskType`` alias) live in
``vetinari.models.model_router_types`` and are re-exported from here for
backward compatibility.

Pure scoring helpers (``assess_difficulty``, ``parse_model_size_b``,
``calculate_confidence``, ``generate_reasoning``, ``infer_task_type``) live in
``vetinari.models.model_router_scoring`` and are re-exported from here.
"""

from __future__ import annotations

import logging
import random
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from vetinari.models.model_router_catalog import ModelRouterCatalogMixin
from vetinari.models.model_router_scoring import (  # re-exported for callers
    assess_difficulty,
    assess_warm_model_bonus,
    calculate_confidence,
    generate_reasoning,
    infer_task_type,
    parse_model_size_b,
)
from vetinari.models.model_router_types import (  # re-exported for callers
    _TASK_TYPE_COMPAT,
    ModelCapabilities,
    ModelInfo,
    ModelSelection,
    ModelStatus,
    TaskType,
    parse_task_type,
)
from vetinari.types import ModelProvider  # canonical enums

if TYPE_CHECKING:
    from vetinari.awareness.confidence import ConfidenceResult, UnknownSituationProtocol
    from vetinari.models.best_of_n import BestOfNSelector

logger = logging.getLogger(__name__)

# ── Public re-exports (backward compat) ──────────────────────────────────────
__all__ = [
    "_TASK_TYPE_COMPAT",
    "DynamicModelRouter",
    "ModelCapabilities",
    "ModelInfo",
    "ModelSelection",
    "ModelStatus",
    "RoutingPolicy",
    "TaskType",
    "assess_difficulty",
    "get_dynamic_router",
    "get_model_router",
    "infer_task_type",
    "init_model_router",
    "parse_model_size_b",
    "parse_task_type",
]


# =====================================================================
# DynamicModelRouter
# =====================================================================


class DynamicModelRouter(ModelRouterCatalogMixin):
    """Dynamic model routing based on task requirements and model capabilities.

    Features:
    - Task-type aware model selection
    - Performance-based routing
    - Latency optimization
    - Cost optimization (for cloud models)
    - Fallback handling
    - Model health checking
    - Configurable RoutingPolicy (merged from ModelRelay)
    - Optional PonderEngine scoring backend (dependency injection)

    Catalog methods (registration, queries, health, stats, performance) are
    inherited from ``ModelRouterCatalogMixin`` in ``model_router_catalog``.
    """

    def __init__(
        self,
        prefer_local: bool = True,
        max_latency_ms: float = 60000,
        max_memory_gb: float = 64,
        ponder_engine: Any | None = None,
    ):
        """Initialize the model router.

        Args:
            prefer_local: Prefer local models over cloud when possible.
            max_latency_ms: Maximum acceptable latency in milliseconds.
            max_memory_gb: Maximum memory budget in gigabytes.
            ponder_engine: Optional PonderEngine instance for scoring.
        """
        self.prefer_local = prefer_local
        self.max_latency_ms = max_latency_ms
        self.max_memory_gb = max_memory_gb
        self._ponder_engine = ponder_engine

        try:
            from vetinari.config.model_config import load_model_config

            self._model_config = load_model_config()
        except Exception:
            self._model_config: dict[str, Any] = {}

        self._task_defaults: dict[str, str] = self._model_config.get("task_defaults", {})
        self._model_tiers: list[dict[str, Any]] = self._model_config.get("model_tiers", [])

        # Model registry: keyed by model_id
        self.models: dict[str, ModelInfo] = {}

        # Performance tracking: keyed by "model_id:task_type"
        self._performance_cache: dict[str, dict[str, Any]] = {}

        # Selection history (capped to avoid unbounded growth)
        self._selection_history: deque[dict[str, Any]] = deque(maxlen=500)

        self._health_check_callback: Callable | None = None

        logger.info("DynamicModelRouter initialized (prefer_local=%s)", prefer_local)

    # ------------------------------------------------------------------
    # PonderEngine integration
    # ------------------------------------------------------------------

    def set_ponder_engine(self, engine: Any) -> None:
        """Inject a PonderEngine instance for scoring.

        Args:
            engine: PonderEngine instance with a ``score_models()`` method.
        """
        self._ponder_engine = engine

    def _ponder_score(self, model: ModelInfo, task_description: str) -> float | None:
        """Use PonderEngine (if available) to score a model for a task.

        Args:
            model: The model to score.
            task_description: Free-text task description passed to PonderEngine.

        Returns:
            Score in [0, 1] or None if PonderEngine is not configured.
        """
        if self._ponder_engine is None:
            return None

        try:
            model_dict = {
                "id": model.id,
                "name": model.name,
                "context_len": model.context_length,
                "memory_gb": model.memory_gb,
                "tags": model.capabilities.tags,
                "capabilities": model.capabilities.tags,
            }
            ranking = self._ponder_engine.score_models([model_dict], task_description, top_n=1)
            if ranking.rankings:
                return ranking.rankings[0].total_score
        except Exception as exc:
            logger.warning("PonderEngine scoring failed for %s — using internal scoring: %s", model.id, exc)

        return None

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    # Catalog methods (register_model, register_models_from_pool,
    # set_health_check_callback, update_model_performance,
    # get_performance_cache, update_performance_cache,
    # get_model_by_id, get_available_models, get_models_by_capability,
    # check_model_health, get_routing_stats) are inherited from
    # ModelRouterCatalogMixin.

    def select_model(
        self,
        task_type: TaskType,
        task_description: str = "",
        required_capabilities: list[str] | None = None,
        preferred_models: list[str] | None = None,
        context_length_needed: int | None = None,
        difficulty_score: float | None = None,
        agent_role: str = "",
    ) -> ModelSelection:
        """Select the best model for a given task.

        Args:
            task_type: Type of task to perform.
            task_description: Description of the task.
            required_capabilities: List of required capabilities.
            preferred_models: List of preferred model IDs (in order).
            context_length_needed: Required context length.
            difficulty_score: Optional pre-computed difficulty (0.0-1.0).
            agent_role: Agent role hint (``"inspector"`` prefers reasoning models).

        Returns:
            ModelSelection with chosen model and reasoning.
        """
        if difficulty_score is None and task_description:
            calibration = 0.0
            try:
                from vetinari.learning.difficulty_feedback import get_calibration_bias

                calibration = get_calibration_bias(task_type.value)
            except Exception:  # noqa: VET023 — calibration is optional; uncalibrated fallback is acceptable
                logger.debug(
                    "Calibration bias unavailable for %s — using uncalibrated difficulty heuristic",
                    task_type.value,
                )
            difficulty_score = assess_difficulty(task_description, task_type.value, calibration_bias=calibration)

        candidates = []

        available_vram_gb = self.max_memory_gb
        try:
            from vetinari.models.vram_manager import get_vram_manager

            available_vram_gb = min(get_vram_manager().get_free_vram_gb(), self.max_memory_gb)
            logger.debug("VRAM-aware routing: %.1f GB available", available_vram_gb)
        except Exception:
            logger.warning("VRAMManager unavailable, using max_memory_gb=%s", self.max_memory_gb)

        # Check if CPU offload is enabled (llama-cpp can split across VRAM+RAM)
        _cfg = getattr(self, "_model_config", {}) or {}
        _cpu_offload = _cfg.get("local_inference", {}).get("cpu_offload_enabled", True)

        for model in self.models.values():
            if not model.is_available:
                continue

            # VRAM filtering: llama-cpp (LOCAL) can offload layers to CPU RAM,
            # so large models are still candidates.  vLLM/NIM are GPU-only —
            # models must fit entirely in VRAM.  Decision: ADR-0084.
            if model.memory_gb > available_vram_gb:
                can_offload = model.provider == ModelProvider.LOCAL and _cpu_offload
                if not can_offload:
                    continue
            if model.avg_latency_ms > self.max_latency_ms and model.avg_latency_ms > 0:
                continue
            if context_length_needed and model.context_length < context_length_needed:
                continue
            if required_capabilities:
                caps = model.capabilities
                meets_requirements = not any(
                    (req == "code_gen" and not caps.code_gen)
                    or (req == "reasoning" and not caps.reasoning)
                    or (req == "docs" and not caps.docs)
                    for req in required_capabilities
                )
                if not meets_requirements:
                    continue
            candidates.append(model)

        if not candidates:
            return self._fallback_selection(task_type)

        task_key = task_type.value.lower()
        boosted_preferred = list(preferred_models or [])  # noqa: VET112 — Optional per func param
        default_id = self._task_defaults.get(task_key, "")
        if default_id and default_id not in boosted_preferred:
            boosted_preferred.append(default_id)

        scored = [(m, self._score_model(m, task_type, task_description, boosted_preferred)) for m in candidates]

        try:
            from vetinari.learning.cost_optimizer import get_cost_optimizer

            optimizer = get_cost_optimizer()
            cheapest = optimizer.select_cheapest_adequate(
                task_type=task_type.value,
                models=[m.id for m, _ in scored],
                min_quality=0.6,
            )
            if cheapest:
                scored = [(m, s + 0.05) if m.id == cheapest else (m, s) for m, s in scored]
        except Exception:
            logger.warning("CostOptimizer unavailable for scoring bonus")

        if difficulty_score is not None and difficulty_score > 0.7:
            scored = [
                (m, s + (0.08 if parse_model_size_b(m.id) >= 30 else 0.04 if parse_model_size_b(m.id) >= 14 else 0))
                for m, s in scored
            ]

        if agent_role == "inspector":
            scored = [(m, s + 0.06) if m.capabilities.reasoning else (m, s) for m, s in scored]

        scored.sort(key=lambda x: x[1], reverse=True)

        best_model, best_score = scored[0]
        alternatives = [m for m, _s in scored[1:4]]
        confidence = calculate_confidence(scored)
        reasoning_text = generate_reasoning(best_model, task_type, best_score)

        # Multi-signal confidence computation (Session 11 — awareness layer)
        confidence_result, unknown_situations = self._compute_awareness_confidence(
            best_model,
            task_type,
            confidence,
        )
        if confidence_result is not None:
            confidence = confidence_result.score

        try:
            from vetinari.audit import get_audit_logger

            get_audit_logger().log_decision(
                decision_type="model_selection",
                choice=best_model.id,
                reasoning=reasoning_text,
                alternatives=[m.id for m in alternatives],
                context={
                    "task_type": task_type.value,
                    "score": round(best_score, 3),
                    "confidence": round(confidence, 3),
                    "confidence_level": confidence_result.level.value if confidence_result else "unknown",
                },
            )
        except Exception:
            logger.warning("Audit logging failed during model selection", exc_info=True)

        self._selection_history.append({
            "task_type": task_type.value,
            "selected_model": best_model.id,
            "score": best_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return ModelSelection(
            model=best_model,
            score=best_score,
            reasoning=reasoning_text,
            alternatives=alternatives,
            confidence=confidence,
            confidence_result=confidence_result,
            unknown_situations=unknown_situations,
        )

    def _fallback_selection(self, task_type: TaskType) -> ModelSelection:
        """Return a fallback model selection when no candidates pass filters.

        Tries the task-type default first, then any available model at random.

        Args:
            task_type: The requested task type (used for default lookup).

        Returns:
            A low-confidence ModelSelection, or None when no models are registered.
        """
        task_key = task_type.value.lower()
        default_id = self._task_defaults.get(task_key, "")
        if default_id and default_id in self.models:
            default_model = self.models[default_id]
            if default_model.is_available:
                logger.info("Using task default model %s for %s", default_id, task_key)
                fallback_cr, fallback_protocols = self._compute_awareness_confidence(
                    default_model,
                    task_type,
                    0.5,
                )
                return ModelSelection(
                    model=default_model,
                    score=0.5,
                    reasoning=f"Task default: {default_id} configured for {task_key} tasks",
                    confidence=fallback_cr.score if fallback_cr else 0.5,
                    confidence_result=fallback_cr,
                    unknown_situations=fallback_protocols,
                )

        available = [m for m in self.models.values() if m.is_available]
        if not available:
            logger.warning("No models available — model selection cannot proceed")
            return None

        fallback = random.choice(available)  # noqa: S311 - deterministic randomness is non-cryptographic
        return ModelSelection(
            model=fallback,
            score=0.0,
            reasoning="Fallback: no models matched criteria",
            confidence=0.1,
        )

    def get_best_of_n_selector(
        self,
        generate_fn: Callable[[str], str],
    ) -> BestOfNSelector:
        """Return a BestOfNSelector wired to the provided generation function.

        Creates a new ``BestOfNSelector`` on each call so the caller can supply
        a fresh ``generate_fn`` that captures the current model and context.
        The selector is intentionally stateless (its only state is the callable),
        so there is no benefit to caching it on the router.

        Args:
            generate_fn: Callable that accepts a prompt string and returns a
                single candidate string.  Called N times per selection request.

        Returns:
            BestOfNSelector instance ready for use.
        """
        from vetinari.models.best_of_n import BestOfNSelector

        return BestOfNSelector(generate_fn=generate_fn)

    def _score_model(
        self, model: ModelInfo, task_type: TaskType, task_description: str, preferred_models: list[str]
    ) -> float:
        """Score a model for a given task, blending PonderEngine and internal scoring.

        Args:
            model: The model to score.
            task_type: The GoalCategory for this task.
            task_description: Free-text task description.
            preferred_models: Ordered list of preferred model IDs.

        Returns:
            Score in [0, 1+] (higher is better).
        """
        ponder_score = self._ponder_score(model, task_description)
        if ponder_score is not None:
            return 0.50 * ponder_score + 0.50 * self._internal_score(
                model, task_type, task_description, preferred_models
            )
        return self._internal_score(model, task_type, task_description, preferred_models)

    # Minimum Thompson observations before Thompson becomes the primary scorer.
    # Below this, rule-based scoring dominates with a small Thompson bonus.
    _THOMPSON_MATURITY_THRESHOLD = 20

    def _internal_score(
        self, model: ModelInfo, task_type: TaskType, task_description: str, preferred_models: list[str] | None
    ) -> float:
        """Internal scoring with Thompson Sampling as primary when data is mature.

        When a model+task_type arm has >= 20 observations, Thompson Sampling
        provides 70% of the score (bandit-first). Below that threshold,
        rule-based scoring dominates with a small Thompson bonus and an
        exploration bonus for undertested models.

        Args:
            model: The model to score.
            task_type: The GoalCategory for this task.
            task_description: Free-text task description (currently unused in base scoring).
            preferred_models: Ordered list of preferred model IDs for preference boost.

        Returns:
            Score in [0, 1+].
        """
        # -- Rule-based component --
        rule_score = 0.0
        rule_score += 0.40 * model.capabilities.matches_task(task_type)

        if preferred_models and model.id in preferred_models:
            pref_index = preferred_models.index(model.id)
            rule_score += 0.20 * max(0.0, 1.0 - pref_index * 0.3)

        if model.total_uses > 0:
            perf_score = model.success_rate * (1.0 - min(model.avg_latency_ms / 60000, 1.0))
            rule_score += 0.20 * perf_score
        else:
            rule_score += 0.10  # Neutral for unknown performance

        if self.prefer_local:
            if model.provider == ModelProvider.LOCAL:
                rule_score += 0.10
            elif model.provider == ModelProvider.OTHER:
                rule_score += 0.05
        else:
            rule_score += 0.10

        if model.context_length >= 8192:
            rule_score += 0.10
        elif model.context_length >= 4096:
            rule_score += 0.05

        # -- Thompson Sampling component --
        thompson_score = 0.0
        thompson_observations = 0
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            ts = get_thompson_selector()
            task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
            arm = ts._arms.get(f"{model.id}:{task_type_str}")
            if arm is not None:
                thompson_observations = arm.total_pulls
                if arm.alpha + arm.beta > 2:
                    thompson_score = arm.sample()
        except Exception:
            logger.warning("Thompson Sampling failed for %s — using rule-based scoring only", model.id)

        # -- Blend Thompson and rule-based based on data maturity --
        # When Thompson has enough observations, it becomes PRIMARY (0.70 weight)
        # because bandit-first selection outperforms rule-based baselines.
        if thompson_observations >= self._THOMPSON_MATURITY_THRESHOLD:
            score = 0.70 * thompson_score + 0.30 * rule_score
        else:
            score = rule_score
            if thompson_score > 0:
                score += thompson_score * 0.10
            # Exploration bonus for new/undertested models — decays linearly
            # as observations accumulate, ensuring new models get tested quickly
            # instead of being ignored for 300+ tasks behind established models.
            exploration_bonus = 0.15 * (1.0 - min(1.0, thompson_observations / 20))
            score += exploration_bonus

        # -- Additive adjustments (always applied) --
        # Warm model bonus: prefer models already loaded in VRAM to avoid
        # eviction churn and load latency.  Decision: ADR-0087.
        score += assess_warm_model_bonus(model.id)

        if self._model_tiers:
            best_tier = self._find_cheapest_adequate_tier(task_type.value.lower())
            if best_tier is not None:
                tier_max_gb = best_tier.get("max_params_b", 999) * 0.6
                if model.memory_gb <= tier_max_gb:
                    score += 0.05
                elif model.memory_gb > tier_max_gb * 2:
                    score -= 0.03

        return score

    def _calculate_confidence(self, scored: list[tuple[ModelInfo, float]]) -> float:
        """Delegate to the module-level ``calculate_confidence`` pure function.

        Exists as an instance method so tests and callers can access it via the
        router object without importing the scoring module directly.

        Args:
            scored: List of ``(ModelInfo, score)`` tuples sorted by score descending.

        Returns:
            Confidence value in [0.0, 1.0].
        """
        return calculate_confidence(scored)

    def _compute_awareness_confidence(
        self,
        model: ModelInfo,
        task_type: TaskType,
        gap_confidence: float,
    ) -> tuple[ConfidenceResult | None, list[UnknownSituationProtocol]]:
        """Compute multi-signal confidence for a model selection decision.

        Extracts Thompson arm data, quality history, and capability match
        to feed the awareness-layer ConfidenceComputer. Also detects and
        logs any unknown-situation protocols ("I don't know").

        Args:
            model: The selected model.
            task_type: The task type for this selection.
            gap_confidence: Gap-based confidence from ``calculate_confidence``.

        Returns:
            Tuple of (ConfidenceResult or None, list of unknown situation protocols).
        """
        try:
            from vetinari.awareness.confidence import get_confidence_computer
        except Exception:
            logger.warning(
                "Confidence computer unavailable for model %s/%s — falling back to gap confidence only",
                model.id,
                task_type,
                exc_info=True,
            )
            return None, []

        task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
        thompson_observations = 0
        thompson_mean = 0.5
        last_data_timestamp: str | None = None

        # Extract Thompson arm data for this model+task
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            ts = get_thompson_selector()
            arm = ts._arms.get(f"{model.id}:{task_type_str}")
            if arm is not None:
                thompson_observations = arm.total_pulls
                thompson_mean = arm.mean
                last_data_timestamp = getattr(arm, "last_updated", None)
        except Exception:
            logger.warning(
                "Thompson data unavailable for confidence computation on %s/%s",
                model.id,
                task_type_str,
                exc_info=True,
            )

        # Extract recent quality scores for variance computation
        quality_scores: list[float] | None = None
        try:
            from vetinari.learning.quality_scorer import get_quality_scorer

            qs = get_quality_scorer()
            history = getattr(qs, "_score_history", {})
            key = f"{model.id}:{task_type_str}"
            if key in history:
                quality_scores = list(history[key])[-20:]
        except Exception:
            logger.warning(
                "Quality score history unavailable for confidence computation on %s/%s",
                model.id,
                task_type_str,
                exc_info=True,
            )

        capability_match = model.capabilities.matches_task(task_type)
        computer = get_confidence_computer()

        result = computer.compute(
            model_id=model.id,
            task_type=task_type_str,
            capability_match_score=capability_match,
            thompson_observations=thompson_observations,
            thompson_mean=thompson_mean,
            success_rate=model.success_rate,
            total_uses=model.total_uses,
            quality_scores=quality_scores,
        )

        # "I don't know" protocol — detect and log unknown situations
        protocols = computer.detect_unknown_situations(
            model_id=model.id,
            task_type=task_type_str,
            thompson_observations=thompson_observations,
            last_data_timestamp=last_data_timestamp,
            thompson_mean=thompson_mean,
            capability_match_score=capability_match,
        )
        for protocol in protocols:
            logger.info(
                "[Awareness] %s — %s: %s",
                protocol.situation.value,
                protocol.message,
                protocol.action,
            )

        return result, protocols

    def _find_cheapest_adequate_tier(self, task_key: str) -> dict[str, Any] | None:
        """Find the lowest-tier (cheapest) model tier adequate for a task type.

        Tiers are sorted by tier number ascending (cheapest first). Returns
        the first tier whose ``preferred_for`` list includes the task key.

        Args:
            task_key: Lowercase task type string (e.g. ``"coding"``, ``"reasoning"``).

        Returns:
            Tier config dict, or None if no tier matches.
        """
        for tier in sorted(self._model_tiers, key=lambda t: t.get("tier", 999)):
            preferred = tier.get("preferred_for", [])
            if task_key in preferred or "general" in preferred:
                return tier
        return None


# =====================================================================
# Global singleton accessors
# =====================================================================

_model_router: DynamicModelRouter | None = None
_model_router_lock = threading.Lock()


def get_model_router() -> DynamicModelRouter:
    """Get or create the global DynamicModelRouter singleton.

    Returns:
        The singleton DynamicModelRouter instance.
    """
    global _model_router
    if _model_router is None:
        with _model_router_lock:
            if _model_router is None:
                _model_router = DynamicModelRouter()
    return _model_router


# Legacy alias used by assignment_pass.py and other callers
get_dynamic_router = get_model_router


def init_model_router(prefer_local: bool = True, **kwargs) -> DynamicModelRouter:
    """Initialize a new model router and inject PonderEngine if available.

    Args:
        prefer_local: Prefer local models over cloud when possible.
        **kwargs: Additional keyword arguments passed to DynamicModelRouter.

    Returns:
        The initialized DynamicModelRouter with PonderEngine wired in when
        the ponder module is importable.
    """
    global _model_router
    _model_router = DynamicModelRouter(prefer_local=prefer_local, **kwargs)  # noqa: VET111 - stateful fallback preserves legacy compatibility

    try:
        from vetinari.models.ponder import PonderEngine

        _model_router.set_ponder_engine(PonderEngine())
        logger.debug("PonderEngine injected into model router")
    except (ImportError, Exception):
        logger.warning("PonderEngine not available — model router using local scoring only")

    return _model_router
