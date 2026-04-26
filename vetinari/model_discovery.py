"""Unified model discovery facade.

This module preserves the public import surface for model search, repository
inspection, and managed downloads while implementation details live in focused
``model_discovery_*`` sibling modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vetinari.constants import get_user_dir
from vetinari.model_discovery_artifacts import (
    _DOWNLOAD_MARKER_SUFFIX,
    _GGUF_DOWNLOAD_SUFFIXES,
    _MAX_REPO_FILES,
    _MIN_FREE_SPACE_HEADROOM_BYTES,
    _NATIVE_DOWNLOAD_BACKENDS,
    _NATIVE_DOWNLOAD_FORMATS,
    _NATIVE_METADATA_FILENAMES,
    _ensure_free_space,
    _extract_lfs_metadata,
    _file_size_gb,
    _infer_file_quantization,
    _infer_model_family,
    _is_native_snapshot_file,
    _marker_path,
    _matches_artifact_filters,
    _matches_objective,
    _materialized_snapshot_files,
    _normalize_backend,
    _normalize_model_format,
    _read_download_marker,
    _read_snapshot_marker,
    _repo_storage_name,
    _resolve_destination,
    _resolve_snapshot_destination,
    _safe_hf_filename,
    _safe_snapshot_relative,
    _select_native_snapshot_files,
    _sha256_file,
    _snapshot_marker_path,
    _validate_existing_download,
    _validate_existing_snapshot,
    _validate_model_header,
    _validate_repo_id,
    _write_download_marker,
    _write_snapshot_marker,
)
from vetinari.model_discovery_cache import (
    _CACHE_TTL_DAYS,
    _load_download_state,
    _load_from_cache,
    _public_download_state,
    _save_to_cache,
    _write_download_state,
)
from vetinari.model_discovery_downloads import (
    _ACTIVE_DOWNLOAD_STATES,
    _DOWNLOAD_CANCEL_EVENTS,
    _DOWNLOAD_JOBS,
    _DOWNLOAD_LOCK,
    _DOWNLOAD_STATE_FILENAME,
    _ModelDiscoveryDownloads,
)
from vetinari.model_discovery_search import (
    _calculate_score,
    _candidate_matches_filters,
    _extract_keywords,
    _generate_rationale,
    _get_adapters,
    _local_candidate,
    _ModelDiscoverySearch,
)
from vetinari.model_discovery_types import (
    ModelCandidate,
    ModelSource,
    RepoModelFile,
    RepoModelSnapshot,
    RepoSnapshotFile,
)

logger = logging.getLogger(__name__)


class ModelDiscovery(_ModelDiscoverySearch, _ModelDiscoveryDownloads):
    """Unified model discovery interface for search and managed downloads.

    Compatible API surface:
    - ``search(query, local_models)``: main discovery entry point.
    - ``search_for_task(task_description, local_models)``: ModelSearchEngine compatibility alias.
    - ``refresh_all_caches()`` and ``get_cached_candidates()``: cache management helpers.
    - ``get_repo_files()``, ``download_model()``, and tracked download methods for catalog routes.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Create a discovery service with cache and source-adapter state.

        Args:
            cache_dir: Optional cache directory override. When omitted, the
                user model cache directory is used.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_user_dir() / "model_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_state_path = self.cache_dir / _DOWNLOAD_STATE_FILENAME
        HuggingFaceAdapter, RedditAdapter, GitHubAdapter, PapersWithCodeAdapter = _get_adapters()
        self.hf_adapter = HuggingFaceAdapter()
        self.reddit_adapter = RedditAdapter()
        self.github_adapter = GitHubAdapter()
        self.pwc_adapter = PapersWithCodeAdapter()


# model_search.ModelSearchEngine -> ModelDiscovery (same API)
ModelSearchEngine = ModelDiscovery

# live_model_search.LiveModelSearchAdapter -> ModelDiscovery
LiveModelSearchAdapter = ModelDiscovery

__all__ = [
    "_ACTIVE_DOWNLOAD_STATES",
    "_CACHE_TTL_DAYS",
    "_DOWNLOAD_CANCEL_EVENTS",
    "_DOWNLOAD_JOBS",
    "_DOWNLOAD_LOCK",
    "_DOWNLOAD_MARKER_SUFFIX",
    "_DOWNLOAD_STATE_FILENAME",
    "_GGUF_DOWNLOAD_SUFFIXES",
    "_MAX_REPO_FILES",
    "_MIN_FREE_SPACE_HEADROOM_BYTES",
    "_NATIVE_DOWNLOAD_BACKENDS",
    "_NATIVE_DOWNLOAD_FORMATS",
    "_NATIVE_METADATA_FILENAMES",
    "LiveModelSearchAdapter",
    "ModelCandidate",
    "ModelDiscovery",
    "ModelSearchEngine",
    "ModelSource",
    "RepoModelFile",
    "RepoModelSnapshot",
    "RepoSnapshotFile",
    "_calculate_score",
    "_candidate_matches_filters",
    "_ensure_free_space",
    "_extract_keywords",
    "_extract_lfs_metadata",
    "_file_size_gb",
    "_generate_rationale",
    "_get_adapters",
    "_infer_file_quantization",
    "_infer_model_family",
    "_is_native_snapshot_file",
    "_load_download_state",
    "_load_from_cache",
    "_local_candidate",
    "_marker_path",
    "_matches_artifact_filters",
    "_matches_objective",
    "_materialized_snapshot_files",
    "_normalize_backend",
    "_normalize_model_format",
    "_public_download_state",
    "_read_download_marker",
    "_read_snapshot_marker",
    "_repo_storage_name",
    "_resolve_destination",
    "_resolve_snapshot_destination",
    "_safe_hf_filename",
    "_safe_snapshot_relative",
    "_save_to_cache",
    "_select_native_snapshot_files",
    "_sha256_file",
    "_snapshot_marker_path",
    "_validate_existing_download",
    "_validate_existing_snapshot",
    "_validate_model_header",
    "_validate_repo_id",
    "_write_download_marker",
    "_write_download_state",
    "_write_snapshot_marker",
]
