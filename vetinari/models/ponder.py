"""Ponder module — scores available models against a task using benchmark data.

Combines four scoring dimensions: capability (benchmark-weighted or tag-based),
context window fit, memory efficiency, and name-based heuristics.  When a model
has stored benchmark scores the capability dimension is fully data-driven via
benchmarks.yaml.  Tag-based fallback is preserved for cold-start models.

Pipeline role: model selection input — called before every agent dispatch.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from vetinari.knowledge import get_benchmark_info

logger = logging.getLogger(__name__)


POLICY_SENSITIVE_KEYWORDS = [
    "harmful",
    "illegal",
    "attack",
    "exploit",
    "weapon",
    "bypass",
    " jailbreak",
    "darkweb",
    "malware",
    "phishing",
    "fraud",
    "scam",
    "explicit",
    "violence",
    "hate",
    "discriminat",
    "terroris",
]

ENABLE_PONDER_MODEL_DISCOVERY = os.environ.get("ENABLE_PONDER_MODEL_DISCOVERY", "true").lower() in ("1", "true", "yes")


def _env_unit_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid ponder weight override %s; using %.2f", name, default)
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


PONDER_CLOUD_WEIGHT = _env_unit_float("PONDER_CLOUD_WEIGHT", 0.20)


@dataclass
class ModelScore:
    """Model score."""

    model_id: str
    model_name: str
    total_score: float
    capability_score: float
    context_score: float
    memory_score: float
    heuristic_score: float
    policy_penalty: float
    reasoning: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"ModelScore(model_id={self.model_id!r},"
            f" total_score={self.total_score!r},"
            f" policy_penalty={self.policy_penalty!r})"
        )


@dataclass
class PonderRanking:
    """Ponder ranking."""

    task_id: str
    task_description: str
    rankings: list[ModelScore]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    phase: str = "unknown"

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"PonderRanking(task_id={self.task_id!r}, phase={self.phase!r}, rankings={len(self.rankings)!r})"


class PonderEngine:
    """Ponder engine."""

    def __init__(self) -> None:
        self.weights = {"capability": 0.40, "context": 0.20, "memory": 0.20, "heuristic": 0.20}
        self.policy_penalty = -1.0

    def _get_task_capability_requirements(self, task_description: str) -> dict[str, Any]:
        # Safely handle None task descriptions
        if task_description is None:
            task_description = ""
        task_lower = task_description.lower()

        requirements = {
            "reasoning": 0.5,
            "code": 0.5,
            "creative": 0.3,
            "analysis": 0.5,
            "instruction_following": 0.7,
            "context_needed": 4096,
            "policy_sensitive": False,
        }

        if any(kw in task_lower for kw in ["reason", "think", "analyze", "evaluate", "assess"]):
            requirements["reasoning"] = 0.9
            requirements["analysis"] = 0.9

        if any(kw in task_lower for kw in ["code", "implement", "build", "write", "create", "function"]):
            requirements["code"] = 0.9

        if any(kw in task_lower for kw in ["write", "story", "creative", "compose", "generate"]):
            requirements["creative"] = 0.8

        if any(kw in task_lower for kw in ["search", "find", "lookup", "research"]):
            requirements["context_needed"] = 8192

        if any(kw in task_lower for kw in POLICY_SENSITIVE_KEYWORDS):
            requirements["policy_sensitive"] = True

        return requirements

    def _calculate_capability_score(self, model: dict, requirements: dict) -> float:
        """Score a model's capability fit for the given task requirements.

        Uses benchmark data from benchmarks.yaml when the model has stored
        benchmark_scores — each score is weighted by its relevance to the
        dominant task type.  Falls back to tag-based keyword matching for
        models without benchmark data (cold-start).

        Args:
            model: Model dict; may include a ``benchmark_scores`` key mapping
                benchmark IDs to normalised scores (0-1).
            requirements: Task requirement dict from
                ``_get_task_capability_requirements()``.

        Returns:
            Capability score in [0, 1].
        """
        base_score = 0.5

        tags = [t.lower() for t in model.get("tags", [])]

        # Use .get() with defaults to avoid KeyError
        reasoning = requirements.get("reasoning", 0)
        code = requirements.get("code", 0)
        creative = requirements.get("creative", 0)
        analysis = requirements.get("analysis", 0)

        # Determine the dominant task type for benchmark weighting
        task_scores = {
            "coding": code,
            "reasoning": reasoning,
            "creative": creative,
            "analysis": analysis,
        }
        dominant_task = max(task_scores, key=lambda k: task_scores[k])
        dominant_requirement = task_scores[dominant_task]

        # Data-driven path: use stored benchmark scores weighted by task relevance.
        # Only applies when the model has benchmark data and the task is demanding.
        benchmark_scores: dict[str, float] = model.get("benchmark_scores", {})
        if benchmark_scores and dominant_requirement > 0.7:
            weighted_sum = 0.0
            weight_total = 0.0
            for benchmark_id, score in benchmark_scores.items():
                info = get_benchmark_info(benchmark_id)
                if not info:
                    continue
                task_weight = info.get("weight_for_task", {}).get(dominant_task, 0.0)
                if task_weight > 0:
                    weighted_sum += float(score) * task_weight
                    weight_total += task_weight
            if weight_total > 0:
                # Weighted mean benchmark score replaces tag-based estimate
                return min(1.0, base_score + weighted_sum / weight_total)

        # Tag-based fallback — used when no benchmark data is available
        if reasoning > 0.7:
            if any(t in tags for t in ["reasoning", "reason", "think", "advanced"]):
                base_score += 0.3
            elif "coder" in tags or "code" in tags:
                base_score += 0.1

        if code > 0.7:
            if "coder" in tags or "code" in tags:
                base_score += 0.4
            elif any(t in tags for t in ["programming", "dev"]):
                base_score += 0.3

        if creative > 0.7 and any(t in tags for t in ["creative", "writing", "story"]):
            base_score += 0.3

        if analysis > 0.7 and any(t in tags for t in ["analysis", "analyze", "research"]):
            base_score += 0.3

        return min(1.0, base_score)

    def _calculate_context_score(self, model: dict, requirements: dict) -> float:
        # Canonicalize context length across providers (context_len vs context_length)
        ctx_len = model.get("context_len", model.get("context_length", 8192))
        needed = requirements.get("context_needed", 8192)

        if needed <= 8192:
            return 1.0 if ctx_len >= 8192 else 0.7
        elif needed <= 32768:
            return 1.0 if ctx_len >= 32768 else 0.6
        elif needed <= 65536:
            return 1.0 if ctx_len >= 65536 else 0.5
        else:
            return 1.0 if ctx_len >= 131072 else 0.4

    def _calculate_memory_score(self, model: dict) -> float:
        quantization = model.get("quantization", "unknown")

        if quantization in ["q4_k_m", "q5_k_s", "q5_k_m", "q6_k"]:
            return 1.0
        elif quantization in ["q4_0", "q4_1", "q4_2", "q4_3"]:
            return 0.9
        elif quantization in ["q8_0", "q8_1"]:
            return 0.7
        elif quantization in ["f16", "f32"]:
            return 0.5
        else:
            return 0.7

    def _get_thompson_score(self, model_id: str, task_type: str = "general") -> float | None:
        """Return the Thompson arm mean for a model+task_type pair if well-observed.

        Only used when the arm has at least 10 pulls so the posterior is
        meaningful.  Returns None for cold-start models — callers fall back
        to keyword heuristics.

        Args:
            model_id: Model identifier string.
            task_type: Task type to look up (e.g., "coding", "general").

        Returns:
            Float in [0, 1] representing arm mean, or None if cold-start.
        """
        _MIN_THOMPSON_PULLS = 10
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            selector = get_thompson_selector()
            arm_key = f"{model_id}:{task_type}"
            with selector._lock:
                arm = selector._arms.get(arm_key)
                if arm is not None and arm.total_pulls >= _MIN_THOMPSON_PULLS:
                    return arm.mean
        except Exception:
            logger.warning(
                "Thompson selector unavailable for model %s — falling back to heuristic memory score",
                model_id,
                exc_info=True,
            )
        return None

    def _calculate_heuristic_score(self, model: dict, task_description: str) -> float:
        base_score = 0.6
        # Safely handle None task descriptions
        if task_description is None:
            task_description = ""
        task_lower = task_description.lower()
        model_name = model.get("id", "").lower()

        if "task" in task_lower and "small" in model_name:
            base_score += 0.2
        elif "complex" in task_lower or "difficult" in task_lower:
            if "70b" in model_name or "72b" in model_name:
                base_score += 0.3
            elif "34b" in model_name or "32b" in model_name:
                base_score += 0.2

        if any(t in model_name for t in ["fast", "speed", "turbo"]):
            base_score += 0.15

        return min(1.0, base_score)

    def _check_policy_sensitivity(self, model: dict, requirements: dict) -> float:
        if not requirements.get("policy_sensitive"):
            return 0.0

        tags = [t.lower() for t in model.get("tags", [])]
        model_name = model.get("id", "").lower()

        if any(t in tags for t in ["uncensored", "unfiltered", "dirty", "explicit"]):
            return 0.0

        if any(t in model_name for t in ["uncensored", "unfiltered"]):
            return 0.0

        return self.policy_penalty

    def score_models(self, available_models: list[dict], task_description: str, top_n: int = 3) -> PonderRanking:
        """Score models.

        Returns:
            The PonderRanking result.

        Args:
            available_models: The available models.
            task_description: The task description.
            top_n: The top n.
        """
        task_id = f"ponder_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        requirements = self._get_task_capability_requirements(task_description)

        scored_models = []

        for model in available_models:
            cap_score = self._calculate_capability_score(model, requirements)
            ctx_score = self._calculate_context_score(model, requirements)
            mem_score = self._calculate_memory_score(model)
            heur_score = self._calculate_heuristic_score(model, task_description)
            policy = self._check_policy_sensitivity(model, requirements)

            # Override memory score with Thompson arm mean when we have sufficient data
            model_id = model.get("id", "")
            _task_scores = {
                "coding": requirements.get("code", 0.5),
                "reasoning": requirements.get("reasoning", 0.5),
                "creative": requirements.get("creative", 0.3),
                "analysis": requirements.get("analysis", 0.5),
            }
            _effective_task_type = max(_task_scores, key=lambda k: _task_scores[k])
            thompson_score = self._get_thompson_score(model_id, _effective_task_type)
            if thompson_score is not None:
                mem_score = thompson_score

            weighted_sum = (
                cap_score * self.weights["capability"]
                + ctx_score * self.weights["context"]
                + mem_score * self.weights["memory"]
                + heur_score * self.weights["heuristic"]
            )
            # Policy penalty is multiplicative: a violation reduces the score to 30%
            # rather than subtracting 1.0 (which could produce negative totals).
            penalized_score = weighted_sum * 0.3 if policy < 0 else weighted_sum

            reasoning = []
            if cap_score > 0.7:
                reasoning.append(f"capability match: {cap_score:.2f}")
            if ctx_score > 0.8:
                reasoning.append(f"context fit: {ctx_score:.2f}")
            if mem_score > 0.8:
                reasoning.append(f"memory efficient: {mem_score:.2f}")
            if policy < 0:
                reasoning.append(f"policy penalty: {policy}")

            scored_models.append(
                ModelScore(
                    model_id=model_id or "unknown",
                    model_name=model_id or "unknown",
                    total_score=penalized_score,
                    capability_score=cap_score,
                    context_score=ctx_score,
                    memory_score=mem_score,
                    heuristic_score=heur_score,
                    policy_penalty=policy,
                    reasoning=", ".join(reasoning) if reasoning else "balanced profile",
                )
            )

        scored_models.sort(key=lambda x: x.total_score, reverse=True)

        return PonderRanking(task_id=task_id, task_description=task_description, rankings=scored_models[:top_n])

    def get_template_prompts(self) -> list[dict]:
        """Return ponder prompt templates (empty — template loading was removed).

        Returns:
            Empty list; template-file-based scoring has been superseded by
            the capability-scoring and Thompson-sampling methods on this class.
        """
        return []


def get_available_models() -> list[dict]:
    """Get available models.

    Returns:
        List of results.
    """
    try:
        from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

        adapter = LocalInferenceAdapter()
        models = adapter.list_loaded_models()

        if not models:
            return _get_last_known_good_models()

        return [
            {
                "id": m.get("id", m.get("model", "unknown")),
                "name": m.get("id", m.get("model", "unknown")),
                "context_length": m.get("context_length", 8192),
                "quantization": m.get("quantization", "q4_k_m"),
                "tags": m.get("tags", []),
            }
            for m in models
        ]
    except Exception:
        logger.error(
            "Could not get available models from ModelPool — returning last-known-good list",
            exc_info=True,
        )
        return _get_last_known_good_models()


def _get_last_known_good_models() -> list[dict]:
    """Return the last successfully discovered models from ModelPool.

    Falls back to an empty list if no discovery has succeeded yet, rather
    than returning hardcoded model names that may not exist on this system.

    Returns:
        List of model dicts from ModelPool._last_known_good, or empty list.
    """
    try:
        from vetinari.web.shared import get_orchestrator

        orch = get_orchestrator()
        pool = getattr(orch, "model_pool", None)
        if pool is not None:
            last_good = getattr(pool, "_last_known_good", None)
            if last_good:
                return list(last_good)
    except Exception:
        logger.warning("Could not retrieve last-known-good models from orchestrator — returning empty list")
    return []


def rank_models(task_description: str, top_n: int = 3) -> dict[str, Any]:
    """Score available local models against a task and return ranked results.

    Args:
        task_description: Natural-language description of the task to route.
        top_n: Maximum number of top-scoring models to include in the ranking.

    Returns:
        A dict with keys ``task_id``, ``task_description``, ``rankings`` (list
        of per-model score breakdowns including capability, context, memory, and
        heuristic scores), ``timestamp``, and ``phase`` set to ``"result"``.
    """
    engine = PonderEngine()
    models = get_available_models()

    ranking = engine.score_models(models, task_description, top_n)

    return {
        "task_id": ranking.task_id,
        "task_description": ranking.task_description,
        "rankings": [
            {
                "rank": i + 1,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "total_score": round(r.total_score, 3),
                "capability_score": round(r.capability_score, 3),
                "context_score": round(r.context_score, 3),
                "memory_score": round(r.memory_score, 3),
                "heuristic_score": round(r.heuristic_score, 3),
                "policy_penalty": r.policy_penalty,
                "reasoning": r.reasoning,
            }
            for i, r in enumerate(ranking.rankings)
        ],
        "timestamp": ranking.timestamp,
        "phase": "result",
    }


def get_cloud_models() -> list[dict]:
    """Get available cloud models from ModelPool.

    Returns:
        List of results.
    """
    try:
        from .model_pool import ModelPool

        config = {}
        pool = ModelPool(config)
        return pool.get_cloud_models()
    except Exception as e:
        logger.error("Error getting cloud models: %s", e)
        return []


def get_all_models_with_cloud() -> list[dict]:
    """Get all available models (local + cloud).

    Returns:
        List of results.
    """
    local_models = get_available_models()
    cloud_models = get_cloud_models()
    return local_models + cloud_models


def _get_model_discovery_candidates(task_description: str, models: list[dict]) -> dict[str, float]:
    """Get model relevance scores from ModelSearchEngine."""
    if not ENABLE_PONDER_MODEL_DISCOVERY:
        return {}

    try:
        from .model_discovery import ModelDiscovery

        search_engine = ModelDiscovery()
        candidates = search_engine.search_for_task(task_description, models)

        relevance = {}
        for candidate in candidates:
            model_id = candidate.id
            relevance[model_id] = candidate.final_score

        return relevance
    except Exception as e:
        logger.error("Error getting model search candidates: %s", e)
        return {}


def score_models_with_cloud(available_models: list[dict], task_description: str, top_n: int = 3) -> PonderRanking:
    """Score available models, optionally augmented by ModelDiscovery relevance scores.

    When the ``vetinari.models.model_discovery`` module is available and
    ``ENABLE_PONDER_MODEL_DISCOVERY`` is set, each model's score is boosted by a
    cloud-discovery relevance component (weighted by ``PONDER_CLOUD_WEIGHT``).
    When model discovery is unavailable or disabled, the function falls back to
    local-only scoring using capability, context, memory, and heuristic weights —
    no ``ImportError`` is raised and no false cloud-augmentation is claimed.

    Args:
        available_models: List of model dicts to rank.  Each dict must have at
            minimum an ``id`` key.
        task_description: Plain-English description of the task the selected
            model will execute.
        top_n: Maximum number of ranked results to return.

    Returns:
        A ``PonderRanking`` whose ``rankings`` list contains at most ``top_n``
        entries, sorted by descending total score.  The ``reasoning`` field on
        each entry indicates whether a cloud-discovery boost was applied.
    """
    engine = PonderEngine()

    search_relevance = _get_model_discovery_candidates(task_description, available_models)
    using_cloud = bool(search_relevance)
    if not using_cloud:
        task_preview = task_description[:80] if task_description else "(empty or None)"
        logger.debug(
            "score_models_with_cloud: model discovery unavailable or disabled — using local-only scoring for task: %s",
            task_preview,
        )

    requirements = engine._get_task_capability_requirements(task_description)
    scored_models = []

    for model in available_models:
        cap_score = engine._calculate_capability_score(model, requirements)
        ctx_score = engine._calculate_context_score(model, requirements)
        mem_score = engine._calculate_memory_score(model)
        heur_score = engine._calculate_heuristic_score(model, task_description)
        policy = engine._check_policy_sensitivity(model, requirements)

        base_total = (
            cap_score * engine.weights["capability"]
            + ctx_score * engine.weights["context"]
            + mem_score * engine.weights["memory"]
            + heur_score * engine.weights["heuristic"]
        ) + policy

        cloud_boost = 0.0
        model_id = model.get("id", "")
        if model_id in search_relevance:
            cloud_boost = search_relevance[model_id] * PONDER_CLOUD_WEIGHT

        total = base_total + cloud_boost

        reasoning = []
        if not using_cloud:
            reasoning.append("local-only scoring")
        if cap_score > 0.7:
            reasoning.append(f"capability: {cap_score:.2f}")
        if ctx_score > 0.8:
            reasoning.append(f"context: {ctx_score:.2f}")
        if cloud_boost > 0.1:
            reasoning.append(f"cloud boost: +{cloud_boost:.2f}")
        if policy < 0:
            reasoning.append(f"policy: {policy}")

        scored_models.append(
            ModelScore(
                model_id=model_id,
                model_name=model.get("name", model_id),
                total_score=total,
                capability_score=cap_score,
                context_score=ctx_score,
                memory_score=mem_score,
                heuristic_score=heur_score,
                policy_penalty=policy,
                reasoning=", ".join(reasoning) if reasoning else "balanced",
            )
        )

    scored_models.sort(key=lambda x: x.total_score, reverse=True)

    return PonderRanking(
        task_id=f"ponder_cloud_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        task_description=task_description,
        rankings=scored_models[:top_n],
    )


def ponder_project_for_plan(plan_id: str) -> dict[str, Any]:
    """Run project-wide ponder pass for all subtasks in a plan.

    Returns:
        A dict with keys ``plan_id``, ``total_subtasks``, ``updated_subtasks``
        (count of subtasks that were successfully scored and updated),
        ``errors`` (list of per-subtask error dicts), and ``success`` (False
        only when the plan is not found or has no subtasks).
    """
    from vetinari.planning import get_plan_manager

    plan_manager = get_plan_manager()
    from vetinari.planning.subtask_tree import subtask_tree

    plan = plan_manager.get_plan(plan_id)
    if not plan:
        return {"error": f"Plan {plan_id} not found", "success": False}

    all_subtasks = subtask_tree.get_all_subtasks(plan_id)

    if not all_subtasks:
        return {"error": "No subtasks found", "success": False}

    available_models = get_all_models_with_cloud()

    results = {
        "plan_id": plan_id,
        "total_subtasks": len(all_subtasks),
        "updated_subtasks": 0,
        "errors": [],
        "success": True,
    }

    for subtask in all_subtasks:
        try:
            ranking = score_models_with_cloud(available_models, subtask.description, top_n=3)

            ranking_data = [
                {
                    "rank": i + 1,
                    "model_id": r.model_id,
                    "model_name": r.model_name,
                    "total_score": round(r.total_score, 3),
                    "capability_score": round(r.capability_score, 3),
                    "context_score": round(r.context_score, 3),
                    "memory_score": round(r.memory_score, 3),
                    "heuristic_score": round(r.heuristic_score, 3),
                    "policy_penalty": r.policy_penalty,
                    "reasoning": r.reasoning,
                }
                for i, r in enumerate(ranking.rankings)
            ]

            scores = {r.model_id: r.total_score for r in ranking.rankings}

            subtask_tree.update_subtask(
                plan_id,
                subtask.subtask_id,
                {"ponder_ranking": ranking_data, "ponder_scores": scores, "ponder_used": True},
            )

            results["updated_subtasks"] += 1

        except Exception as e:
            results["errors"].append({"subtask_id": subtask.subtask_id, "error": str(e)})

    return results


def get_ponder_results_for_plan(plan_id: str) -> dict[str, Any]:
    """Get ponder results for all subtasks in a plan.

    Returns:
        A dict with keys ``plan_id``, ``total_subtasks``, ``subtasks_with_ponder``
        (count of subtasks that have stored rankings), and ``subtasks`` (list of
        subtask dicts containing ``ponder_ranking``, ``ponder_scores``, and
        ``ponder_used`` for each subtask that has been scored).
    """
    from vetinari.planning.subtask_tree import subtask_tree

    all_subtasks = subtask_tree.get_all_subtasks(plan_id)

    subtask_results = [
        {
            "subtask_id": subtask.subtask_id,
            "description": subtask.description,
            "agent_type": subtask.agent_type,
            "ponder_ranking": subtask.ponder_ranking,
            "ponder_scores": subtask.ponder_scores,
            "ponder_used": subtask.ponder_used,
        }
        for subtask in all_subtasks
        if subtask.ponder_ranking or subtask.ponder_scores
    ]

    return {
        "plan_id": plan_id,
        "total_subtasks": len(all_subtasks),
        "subtasks_with_ponder": len(subtask_results),
        "subtasks": subtask_results,
    }


def get_ponder_health() -> dict[str, Any]:
    """Check health/status of cloud providers.

    Returns:
        A dict with keys ``enable_model_discovery`` (whether ponder model
        discovery is active), ``cloud_weight`` (the configured weight given to
        cloud models during scoring), and ``providers`` (per-provider health
        data from ``ModelPool.get_cloud_provider_health()``).
    """
    from .model_pool import ModelPool

    cloud_health = ModelPool.get_cloud_provider_health()

    return {
        "enable_model_discovery": ENABLE_PONDER_MODEL_DISCOVERY,
        "cloud_weight": PONDER_CLOUD_WEIGHT,
        "providers": cloud_health,
    }


ponder_engine = PonderEngine()
