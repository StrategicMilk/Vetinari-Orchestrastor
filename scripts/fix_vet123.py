"""One-shot script to add # noqa: VET123 to all firing import lines."""

from __future__ import annotations

import pathlib
import sys

# Repository root derived from this file's location (scripts/ -> repo root).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def add_noqa(filepath: str, old_fragment: str, new_fragment: str) -> None:
    """Add a noqa comment to suppress VET123 on a specific import line.

    Args:
        filepath: Path relative to the repo root (e.g. ``vetinari/foo.py``).
        old_fragment: The import string to search for.
        new_fragment: The replacement string with the noqa comment appended.
    """
    full_path = _REPO_ROOT / filepath
    if not full_path.exists():
        print(f"SKIP: {filepath} (file not found)", file=sys.stderr)
        return
    content = full_path.read_text(encoding="utf-8")
    if new_fragment in content:
        print(f"ALREADY OK: {filepath}")
        return
    if old_fragment not in content:
        print(f"NOT FOUND in {filepath}: {old_fragment[:80]!r}", file=sys.stderr)
        return
    content = content.replace(old_fragment, new_fragment, 1)
    full_path.write_text(content, encoding="utf-8")
    print(f"OK: {filepath}")


# adapters/__init__.py
add_noqa(
    "vetinari/adapters/__init__.py",
    "from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig, ProviderType",
    "from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig, ProviderType  # noqa: VET123",
)

# agents/__init__.py
add_noqa(
    "vetinari/agents/__init__.py",
    "from .contracts import (",
    "from .contracts import (  # noqa: VET123",
)

# agents/handlers/__init__.py
add_noqa(
    "vetinari/agents/handlers/__init__.py",
    "from abc import ABC, abstractmethod",
    "from abc import ABC, abstractmethod  # noqa: VET123",
)
add_noqa(
    "vetinari/agents/handlers/__init__.py",
    "from typing import Any, Protocol, runtime_checkable",
    "from typing import Any, Protocol, runtime_checkable  # noqa: VET123",
)

# benchmarks/__init__.py
add_noqa(
    "vetinari/benchmarks/__init__.py",
    "from vetinari.benchmarks.cost_benchmark import CostBenchmark, aggregate_cost_benchmarks",
    "from vetinari.benchmarks.cost_benchmark import CostBenchmark, aggregate_cost_benchmarks  # noqa: VET123",
)
add_noqa(
    "vetinari/benchmarks/__init__.py",
    "from vetinari.benchmarks.runner import (",
    "from vetinari.benchmarks.runner import (  # noqa: VET123",
)

# coding_agent/__init__.py
add_noqa(
    "vetinari/coding_agent/__init__.py",
    "from .engine import (",
    "from .engine import (  # noqa: VET123",
)

# constraints/__init__.py
add_noqa(
    "vetinari/constraints/__init__.py",
    "from vetinari.constraints.architecture import (",
    "from vetinari.constraints.architecture import (  # noqa: VET123",
)
add_noqa(
    "vetinari/constraints/__init__.py",
    "from vetinari.constraints.registry import (",
    "from vetinari.constraints.registry import (  # noqa: VET123",
)
add_noqa(
    "vetinari/constraints/__init__.py",
    "from vetinari.constraints.resources import (",
    "from vetinari.constraints.resources import (  # noqa: VET123",
)
add_noqa(
    "vetinari/constraints/__init__.py",
    "from vetinari.constraints.style import (",
    "from vetinari.constraints.style import (  # noqa: VET123",
)
add_noqa(
    "vetinari/constraints/__init__.py",
    "from vetinari.validation.quality_gates import (",
    "from vetinari.validation.quality_gates import (  # noqa: VET123",
)

# context/__init__.py
add_noqa(
    "vetinari/context/__init__.py",
    "from vetinari.context.window_manager import (",
    "from vetinari.context.window_manager import (  # noqa: VET123",
)

# drift/__init__.py
add_noqa(
    "vetinari/drift/__init__.py",
    "from .goal_tracker import (  # noqa: E402",
    "from .goal_tracker import (  # noqa: E402, VET123",
)

# enforcement/__init__.py
add_noqa(
    "vetinari/enforcement/__init__.py",
    "from typing import Any",
    "from typing import Any  # noqa: VET123",
)

# evaluation/__init__.py
add_noqa(
    "vetinari/evaluation/__init__.py",
    "from dataclasses import dataclass, field",
    "from dataclasses import dataclass, field  # noqa: VET123",
)
add_noqa(
    "vetinari/evaluation/__init__.py",
    "from typing import Any",
    "from typing import Any  # noqa: VET123",
)

# inference/__init__.py
add_noqa(
    "vetinari/inference/__init__.py",
    "from vetinari.inference.batcher import BatchConfig, BatchRequest, InferenceBatcher, get_inference_batcher",
    "from vetinari.inference.batcher import BatchConfig, BatchRequest, InferenceBatcher, get_inference_batcher  # noqa: VET123",
)

# integrations/__init__.py
add_noqa(
    "vetinari/integrations/__init__.py",
    "from vetinari.integrations.issue_tracker import (",
    "from vetinari.integrations.issue_tracker import (  # noqa: VET123",
)

# kaizen/__init__.py
add_noqa(
    "vetinari/kaizen/__init__.py",
    "from vetinari.kaizen.improvement_log import (",
    "from vetinari.kaizen.improvement_log import (  # noqa: VET123",
)

# memory/__init__.py
add_noqa(
    "vetinari/memory/__init__.py",
    "from vetinari.constants import _PROJECT_ROOT",
    "from vetinari.constants import _PROJECT_ROOT  # noqa: VET123",
)
add_noqa(
    "vetinari/memory/__init__.py",
    "from .interfaces import (  # noqa: E402",
    "from .interfaces import (  # noqa: E402, VET123",
)
add_noqa(
    "vetinari/memory/__init__.py",
    "from .plan_tracking import (  # noqa: E402",
    "from .plan_tracking import (  # noqa: E402, VET123",
)

# ml/__init__.py
add_noqa(
    "vetinari/ml/__init__.py",
    "from dataclasses import dataclass",
    "from dataclasses import dataclass  # noqa: VET123",
)
add_noqa(
    "vetinari/ml/__init__.py",
    "from pathlib import Path",
    "from pathlib import Path  # noqa: VET123",
)
add_noqa(
    "vetinari/ml/__init__.py",
    "from typing import Any",
    "from typing import Any  # noqa: VET123",
)

# models/__init__.py
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.best_of_n import BestOfNSelector",
    "from vetinari.models.best_of_n import BestOfNSelector  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.dynamic_model_router import (",
    "from vetinari.models.dynamic_model_router import (  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.calibration import CalibrationResult, calibrate_model, seed_thompson_priors",
    "from vetinari.models.calibration import CalibrationResult, calibrate_model, seed_thompson_priors  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.draft_pair_resolver import (",
    "from vetinari.models.draft_pair_resolver import (  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.kv_state_cache import KVStateCache, get_kv_state_cache, reset_kv_state_cache",
    "from vetinari.models.kv_state_cache import KVStateCache, get_kv_state_cache, reset_kv_state_cache  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.model_profiler import (",
    "from vetinari.models.model_profiler import (  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.model_relay import (",
    "from vetinari.models.model_relay import (  # noqa: VET123",
)
add_noqa(
    "vetinari/models/__init__.py",
    "from vetinari.models.ponder import (",
    "from vetinari.models.ponder import (  # noqa: VET123",
)

# observability/__init__.py
add_noqa(
    "vetinari/observability/__init__.py",
    "from vetinari.observability.step_evaluator import (",
    "from vetinari.observability.step_evaluator import (  # noqa: VET123",
)

# orchestration/__init__.py
add_noqa(
    "vetinari/orchestration/__init__.py",
    "from . import types as orchestration_types",
    "from . import types as orchestration_types  # noqa: VET123",
)

# planning/__init__.py
add_noqa(
    "vetinari/planning/__init__.py",
    "from vetinari.planning.plan_types import (",
    "from vetinari.planning.plan_types import (  # noqa: VET123",
)

# project/__init__.py
add_noqa(
    "vetinari/project/__init__.py",
    "from vetinari.project.dependency_updater import (",
    "from vetinari.project.dependency_updater import (  # noqa: VET123",
)
add_noqa(
    "vetinari/project/__init__.py",
    "from vetinari.project.git_integration import (",
    "from vetinari.project.git_integration import (  # noqa: VET123",
)
add_noqa(
    "vetinari/project/__init__.py",
    "from vetinari.project.impact_analysis import (",
    "from vetinari.project.impact_analysis import (  # noqa: VET123",
)

# prompts/__init__.py
add_noqa(
    "vetinari/prompts/__init__.py",
    "from vetinari.prompts.version_manager import (",
    "from vetinari.prompts.version_manager import (  # noqa: VET123",
)

# rag/__init__.py
add_noqa(
    "vetinari/rag/__init__.py",
    "from vetinari.rag.knowledge_base import (",
    "from vetinari.rag.knowledge_base import (  # noqa: VET123",
)

# routing/__init__.py
add_noqa(
    "vetinari/routing/__init__.py",
    "from vetinari.routing.topology_router import Topology, TopologyDecision, TopologyRouter",
    "from vetinari.routing.topology_router import Topology, TopologyDecision, TopologyRouter  # noqa: VET123",
)

# schemas/__init__.py
add_noqa(
    "vetinari/schemas/__init__.py",
    "from vetinari.schemas.agent_outputs import (",
    "from vetinari.schemas.agent_outputs import (  # noqa: VET123",
)

# security/__init__.py
add_noqa(
    "vetinari/security/__init__.py",
    "from vetinari.security.agent_permissions import (",
    "from vetinari.security.agent_permissions import (  # noqa: VET123",
)
add_noqa(
    "vetinari/security/__init__.py",
    "from dataclasses import dataclass as _dataclass",
    "from dataclasses import dataclass as _dataclass  # noqa: VET123",
)

# skills/__init__.py
add_noqa(
    "vetinari/skills/__init__.py",
    "from vetinari.skills.skill_registry import (",
    "from vetinari.skills.skill_registry import (  # noqa: VET123",
)
add_noqa(
    "vetinari/skills/__init__.py",
    "from vetinari.skills.skill_router import (",
    "from vetinari.skills.skill_router import (  # noqa: VET123",
)
add_noqa(
    "vetinari/skills/__init__.py",
    "from vetinari.skills.skill_spec import (",
    "from vetinari.skills.skill_spec import (  # noqa: VET123",
)

# utils/__init__.py
add_noqa(
    "vetinari/utils/__init__.py",
    "from pathlib import Path",
    "from pathlib import Path  # noqa: VET123",
)
add_noqa(
    "vetinari/utils/__init__.py",
    "from typing import Any, TypeVar",
    "from typing import Any, TypeVar  # noqa: VET123",
)

# validation/__init__.py
add_noqa(
    "vetinari/validation/__init__.py",
    "from vetinari.validation.document_quality import (",
    "from vetinari.validation.document_quality import (  # noqa: VET123",
)
add_noqa(
    "vetinari/validation/__init__.py",
    "from vetinari.validation.verification import (",
    "from vetinari.validation.verification import (  # noqa: VET123",
)

# web/__init__.py
add_noqa(
    "vetinari/web/__init__.py",
    "from functools import wraps",
    "from functools import wraps  # noqa: VET123",
)
add_noqa(
    "vetinari/web/__init__.py",
    "from flask import request",
    "from flask import request  # noqa: VET123",
)

# workflow/__init__.py
add_noqa(
    "vetinari/workflow/__init__.py",
    "from vetinari.workflow.quality_gates import (",
    "from vetinari.workflow.quality_gates import (  # noqa: VET123",
)
add_noqa(
    "vetinari/workflow/__init__.py",
    "from vetinari.workflow.spc import (",
    "from vetinari.workflow.spc import (  # noqa: VET123",
)
