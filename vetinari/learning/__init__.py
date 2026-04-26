"""Vetinari Learning Pipeline — self-improvement subsystem.

Provides Thompson sampling model selection, episode memory for
experience replay, quality scoring, prompt evolution, and
self-refinement loops.  Submodules are lazily imported on first
attribute access to avoid heavy startup costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    auto_tuner: ModuleType
    episode_memory: ModuleType
    feedback_loop: ModuleType
    meta_adapter: ModuleType
    model_selector: ModuleType
    operator_selector: ModuleType
    orchestrator: ModuleType
    prompt_evolver: ModuleType
    prompt_mutator: ModuleType
    quality_scorer: ModuleType
    self_refinement: ModuleType
    training_data: ModuleType
    training_manager: ModuleType
    workflow_learner: ModuleType

__all__ = [
    "auto_tuner",
    "episode_memory",
    "feedback_loop",
    "meta_adapter",
    "model_selector",
    "operator_selector",
    "orchestrator",
    "prompt_evolver",
    "prompt_mutator",
    "quality_scorer",
    "self_refinement",
    "skill_library",
    "training_data",
    "training_manager",
    "workflow_learner",
]


def __getattr__(name: str):
    """Lazily import submodules on attribute access."""
    import importlib

    if name in __all__:
        return importlib.import_module(f"vetinari.learning.{name}")
    raise AttributeError(f"module 'vetinari.learning' has no attribute {name!r}")
