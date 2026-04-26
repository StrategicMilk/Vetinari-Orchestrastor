"""Model management, routing, and optimization subsystem."""

from __future__ import annotations

from vetinari.models.best_of_n import BestOfNSelector  # noqa: VET123, get_n_for_tier
from vetinari.models.calibration import (  # noqa: VET123 - barrel export preserves public import compatibility
    CalibrationResult,
    calibrate_model,
    seed_thompson_priors,
)
from vetinari.models.draft_pair_resolver import (  # noqa: VET123 — reset_draft_pair_resolver has no external callers but removing causes VET120
    DraftPairResolver,
    get_draft_pair_resolver,
    reset_draft_pair_resolver,
)
from vetinari.models.dynamic_model_router import (  # noqa: VET123 - barrel export preserves public import compatibility
    DynamicModelRouter,
    get_model_router,
    infer_task_type,
    init_model_router,
)
from vetinari.models.inference_config import (
    BudgetPolicy,
    InferenceConfig,
    get_budget_policy,
)
from vetinari.models.kv_state_cache import (  # noqa: VET123 - barrel export preserves public import compatibility
    KVStateCache,
    get_kv_state_cache,
    reset_kv_state_cache,
)
from vetinari.models.model_pool import ModelPool
from vetinari.models.model_profiler import (  # noqa: VET123 — reset_model_profiler has no external callers but removing causes VET120
    ModelProfiler,
    get_model_profiler,
    reset_model_profiler,
)
from vetinari.models.model_registry import (
    ModelRegistry,
    get_model_registry,
)
from vetinari.models.model_relay import (  # noqa: VET123 - barrel export preserves public import compatibility
    get_model_relay,
)
from vetinari.models.model_scout import (
    ModelRecommendation,
    ModelScout,
    get_model_scout,
)
from vetinari.models.ponder import (  # noqa: VET123 - barrel export preserves public import compatibility
    PonderEngine,
    get_available_models,
    get_ponder_health,
    ponder_project_for_plan,
    rank_models,
)

__all__ = [
    "BestOfNSelector",
    "BudgetPolicy",
    "CalibrationResult",
    "DraftPairResolver",
    "DynamicModelRouter",
    "InferenceConfig",
    "KVStateCache",
    "ModelPool",
    "ModelProfiler",
    "ModelRecommendation",
    "ModelRegistry",
    "ModelScout",
    "PonderEngine",
    "calibrate_model",
    "get_available_models",
    "get_budget_policy",
    "get_draft_pair_resolver",
    "get_kv_state_cache",
    "get_model_profiler",
    "get_model_registry",
    "get_model_relay",
    "get_model_router",
    "get_model_scout",
    "get_n_for_tier",
    "get_ponder_health",
    "infer_task_type",
    "init_model_router",
    "ponder_project_for_plan",
    "rank_models",
    "reset_draft_pair_resolver",
    "reset_kv_state_cache",
    "reset_model_profiler",
    "seed_thompson_priors",
]
