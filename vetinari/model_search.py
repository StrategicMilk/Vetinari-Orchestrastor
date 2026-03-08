"""Backward-compatibility shim for vetinari.model_search.

The implementation has moved to ``vetinari.model_discovery``.
Import from there for new code.
"""
import warnings

warnings.warn(
    "vetinari.model_search is deprecated; use vetinari.model_discovery instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vetinari.model_discovery import (  # noqa: F401, E402
    ModelCandidate,
    ModelDiscovery,
    ModelDiscovery as ModelSearchEngine,
    ModelSource,
    HuggingFaceAdapter,
    RedditAdapter,
    GitHubAdapter,
    PapersWithCodeAdapter,
    LiveModelSearchAdapter,
)

__all__ = [
    "ModelCandidate",
    "ModelDiscovery",
    "ModelSearchEngine",
    "ModelSource",
    "HuggingFaceAdapter",
    "RedditAdapter",
    "GitHubAdapter",
    "PapersWithCodeAdapter",
    "LiveModelSearchAdapter",
]
