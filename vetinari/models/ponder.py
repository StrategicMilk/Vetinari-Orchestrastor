"""Ponder module."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
PONDER_CLOUD_WEIGHT = float(os.environ.get("PONDER_CLOUD_WEIGHT", "0.20"))


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


@dataclass
class PonderRanking:
    """Ponder ranking."""
    task_id: str
    task_description: str
    rankings: list[ModelScore]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    phase: str = "unknown"


class PonderEngine:
    """Ponder engine."""
    def __init__(self, template_version: str = "v1"):
        self.template_version = template_version
        self.templates = self._load_templates()
        self.weights = {"capability": 0.40, "context": 0.20, "memory": 0.20, "heuristic": 0.20}
        self.policy_penalty = -1.0

    def _load_templates(self) -> list[dict]:
        template_dir = Path(__file__).parent.parent / "templates" / self.template_version
        template_file = template_dir / "ponder.json"

        if not template_file.exists():
            return []

        with open(template_file, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("templates", [])

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
        base_score = 0.5

        tags = [t.lower() for t in model.get("tags", [])]

        # Use .get() with defaults to avoid KeyError
        reasoning = requirements.get("reasoning", 0)
        code = requirements.get("code", 0)
        creative = requirements.get("creative", 0)
        analysis = requirements.get("analysis", 0)

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
        ctx_len = model.get("context_len", model.get("context_length", 4096))
        needed = requirements.get("context_needed", 4096)

        if needed <= 4096:
            return 1.0 if ctx_len >= 4096 else 0.7
        elif needed <= 8192:
            return 1.0 if ctx_len >= 8192 else 0.6
        elif needed <= 32768:
            return 1.0 if ctx_len >= 32768 else 0.5
        else:
            return 1.0 if ctx_len >= 65536 else 0.4

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
        if not requirements.get("policy_sensitive", False):
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
        task_id = f"ponder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        requirements = self._get_task_capability_requirements(task_description)

        scored_models = []

        for model in available_models:
            cap_score = self._calculate_capability_score(model, requirements)
            ctx_score = self._calculate_context_score(model, requirements)
            mem_score = self._calculate_memory_score(model)
            heur_score = self._calculate_heuristic_score(model, task_description)
            policy = self._check_policy_sensitivity(model, requirements)

            total = (
                cap_score * self.weights["capability"]
                + ctx_score * self.weights["context"]
                + mem_score * self.weights["memory"]
                + heur_score * self.weights["heuristic"]
            ) + policy

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
                    model_id=model.get("id", "unknown"),
                    model_name=model.get("id", "unknown"),
                    total_score=total,
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
        return self.templates


def get_available_models() -> list[dict]:
    """Get available models.

    Returns:
        List of results.
    """
    try:
        from vetinari.adapters.lmstudio_adapter import LMStudioAdapter

        host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")  # noqa: VET041
        adapter = LMStudioAdapter(host=host)
        models = adapter.list_loaded_models()

        if not models:
            return _get_fallback_models()

        return [
            {
                "id": m.get("id", m.get("model", "unknown")),
                "name": m.get("id", m.get("model", "unknown")),
                "context_length": m.get("context_length", 4096),
                "quantization": m.get("quantization", "q4_k_m"),
                "tags": m.get("tags", []),
            }
            for m in models
        ]
    except Exception as e:
        logger.error("Error getting models: %s", e)
        return _get_fallback_models()


def _get_fallback_models() -> list[dict]:
    return [
        {
            "id": "qwen2.5-coder-14b-instruct",
            "context_length": 32768,
            "quantization": "q4_k_m",
            "tags": ["code", "reasoning"],
        },
        {
            "id": "qwen2.5-14b-instruct",
            "context_length": 32768,
            "quantization": "q4_k_m",
            "tags": ["general", "reasoning"],
        },
        {"id": "deepseek-coder-33b-instruct", "context_length": 16384, "quantization": "q4_k_m", "tags": ["code"]},
        {"id": "llama-3.1-8b-instruct", "context_length": 8192, "quantization": "q4_k_m", "tags": ["general"]},
        {"id": "phi-3.5-mini-instruct", "context_length": 4096, "quantization": "q4_0", "tags": ["fast", "small"]},
    ]


def rank_models(task_description: str, top_n: int = 3, template_version: str = "v1") -> dict[str, Any]:
    """Rank models.

    Returns:
        The result string.

    Args:
        task_description: The task description.
        top_n: The top n.
        template_version: The template version.
    """
    engine = PonderEngine(template_version=template_version)
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
    """Score models with cloud provider augmentation.

    Args:
        available_models: The available models.
        task_description: The task description.
        top_n: The top n.

    Returns:
        The PonderRanking result.
    """
    engine = PonderEngine()

    search_relevance = _get_model_discovery_candidates(task_description, available_models)

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
        task_id=f"ponder_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_description=task_description,
        rankings=scored_models[:top_n],
    )


def ponder_project_for_plan(plan_id: str) -> dict[str, Any]:
    """Run project-wide ponder pass for all subtasks in a plan.

    Returns:
        The result string.
    """
    from vetinari.planning.planning import plan_manager
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
        The result string.
    """
    from vetinari.planning.subtask_tree import subtask_tree

    all_subtasks = subtask_tree.get_all_subtasks(plan_id)

    subtask_results = []
    for subtask in all_subtasks:
        if subtask.ponder_ranking or subtask.ponder_scores:
            subtask_results.append(
                {
                    "subtask_id": subtask.subtask_id,
                    "description": subtask.description,
                    "agent_type": subtask.agent_type,
                    "ponder_ranking": subtask.ponder_ranking,
                    "ponder_scores": subtask.ponder_scores,
                    "ponder_used": subtask.ponder_used,
                }
            )

    return {
        "plan_id": plan_id,
        "total_subtasks": len(all_subtasks),
        "subtasks_with_ponder": len(subtask_results),
        "subtasks": subtask_results,
    }


def get_ponder_health() -> dict[str, Any]:
    """Check health/status of cloud providers.

    Returns:
        The result string.
    """
    from .model_pool import ModelPool

    cloud_health = ModelPool.get_cloud_provider_health()

    return {
        "enable_model_discovery": ENABLE_PONDER_MODEL_DISCOVERY,
        "cloud_weight": PONDER_CLOUD_WEIGHT,
        "providers": cloud_health,
    }


ponder_engine = PonderEngine()
