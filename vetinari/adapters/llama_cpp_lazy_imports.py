"""Lazy-import getters for vetinari sub-packages used by the llama.cpp adapter.

These modules live in sub-packages that would create circular imports if
imported at module load time. Each getter is called at most once; the result
is cached in a module-level variable guarded by a threading.Lock (double-checked
locking pattern required by VET quality gates).

All callables returned by these getters are singleton factories from their
respective sub-packages — call the returned callable to get the live instance,
e.g. ``_get_semantic_cache_fn()()``.
"""

from __future__ import annotations

import threading
from typing import Any

# ── Semantic cache ────────────────────────────────────────────────────────────

_semantic_cache_fn = None
_semantic_cache_lock = threading.Lock()


def _get_semantic_cache_fn() -> Any:
    """Return the get_semantic_cache callable, importing it once on first call."""
    global _semantic_cache_fn
    if _semantic_cache_fn is None:
        with _semantic_cache_lock:
            if _semantic_cache_fn is None:
                from vetinari.optimization.semantic_cache import get_semantic_cache

                _semantic_cache_fn = get_semantic_cache
    return _semantic_cache_fn


# ── KV state cache ────────────────────────────────────────────────────────────

_kv_state_cache_fn = None
_hash_system_prompt_fn = None
_kv_state_cache_lock = threading.Lock()


def _get_kv_state_cache_fn() -> Any:
    """Return the get_kv_state_cache callable, importing it once on first call."""
    global _kv_state_cache_fn, _hash_system_prompt_fn
    if _kv_state_cache_fn is None:
        with _kv_state_cache_lock:
            if _kv_state_cache_fn is None:
                from vetinari.models.kv_state_cache import get_kv_state_cache, hash_system_prompt

                _kv_state_cache_fn = get_kv_state_cache
                _hash_system_prompt_fn = hash_system_prompt
    return _kv_state_cache_fn


def _get_hash_system_prompt_fn() -> Any:
    """Return the hash_system_prompt callable, importing it once on first call."""
    _get_kv_state_cache_fn()  # Ensures _hash_system_prompt_fn is populated
    return _hash_system_prompt_fn


# ── Draft pair resolver ───────────────────────────────────────────────────────

_draft_pair_resolver_fn = None
_draft_pair_resolver_lock = threading.Lock()


def _get_draft_pair_resolver_fn() -> Any:
    """Return the get_draft_pair_resolver callable, importing it once on first call."""
    global _draft_pair_resolver_fn
    if _draft_pair_resolver_fn is None:
        with _draft_pair_resolver_lock:
            if _draft_pair_resolver_fn is None:
                from vetinari.models.draft_pair_resolver import get_draft_pair_resolver

                _draft_pair_resolver_fn = get_draft_pair_resolver
    return _draft_pair_resolver_fn


# ── Model selector ────────────────────────────────────────────────────────────

_model_selector_fn = None
_model_selector_lock = threading.Lock()


def _get_model_selector_fn() -> Any:
    """Return the get_model_selector callable, importing it once on first call."""
    global _model_selector_fn
    if _model_selector_fn is None:
        with _model_selector_lock:
            if _model_selector_fn is None:
                from vetinari.learning.model_selector import get_thompson_selector

                _model_selector_fn = get_thompson_selector
    return _model_selector_fn


# ── Model profiler ────────────────────────────────────────────────────────────

_model_profiler_fn = None
_model_profiler_lock = threading.Lock()


def _get_model_profiler_fn() -> Any:
    """Return the get_model_profiler callable, importing it once on first call."""
    global _model_profiler_fn
    if _model_profiler_fn is None:
        with _model_profiler_lock:
            if _model_profiler_fn is None:
                from vetinari.models.model_profiler import get_model_profiler

                _model_profiler_fn = get_model_profiler
    return _model_profiler_fn


# ── Calibration functions ─────────────────────────────────────────────────────

_calibrate_model_fn = None
_seed_thompson_priors_fn = None
_load_cached_profile_fn = None
_save_profile_fn = None
_calibration_lock = threading.Lock()


def _get_calibration_fns() -> tuple[Any, Any, Any, Any]:
    """Return calibration callables (calibrate_model, seed_thompson_priors, _load_cached_profile, _save_profile).

    Imports from vetinari.models.calibration and vetinari.models.model_profiler
    once on first call and caches all four callables.

    Returns:
        Tuple of (calibrate_model, seed_thompson_priors, _load_cached_profile, _save_profile).
    """
    global _calibrate_model_fn, _seed_thompson_priors_fn, _load_cached_profile_fn, _save_profile_fn
    if _calibrate_model_fn is None:
        with _calibration_lock:
            if _calibrate_model_fn is None:
                from vetinari.models.calibration import calibrate_model, seed_thompson_priors
                from vetinari.models.model_profiler import _load_cached_profile, _save_profile

                _calibrate_model_fn = calibrate_model
                _seed_thompson_priors_fn = seed_thompson_priors
                _load_cached_profile_fn = _load_cached_profile
                _save_profile_fn = _save_profile
    return _calibrate_model_fn, _seed_thompson_priors_fn, _load_cached_profile_fn, _save_profile_fn


# ── Web shared config ─────────────────────────────────────────────────────────

_current_config_fn = None
_current_config_lock = threading.Lock()


def _get_current_config() -> Any:
    """Return the current_config dict from vetinari.web.shared, importing once on first call."""
    global _current_config_fn
    if _current_config_fn is None:
        with _current_config_lock:
            if _current_config_fn is None:
                from vetinari.web.shared import current_config

                _current_config_fn = current_config
    return _current_config_fn


# ── Inference config ──────────────────────────────────────────────────────────

_inference_config_fn = None
_inference_config_lock = threading.Lock()


def _get_inference_config_fn() -> Any:
    """Return the get_inference_config callable, importing it once on first call."""
    global _inference_config_fn
    if _inference_config_fn is None:
        with _inference_config_lock:
            if _inference_config_fn is None:
                from vetinari.config.inference_config import get_inference_config

                _inference_config_fn = get_inference_config
    return _inference_config_fn


# ── Inference batcher ─────────────────────────────────────────────────────────

_inference_batcher_fn = None
_batch_request_cls = None
_inference_batcher_lock = threading.Lock()


def _get_inference_batcher_fns() -> tuple[Any, Any]:
    """Return (get_inference_batcher, BatchRequest) from vetinari.inference, importing once.

    Returns:
        Tuple of (get_inference_batcher callable, BatchRequest class).
    """
    global _inference_batcher_fn, _batch_request_cls
    if _inference_batcher_fn is None:
        with _inference_batcher_lock:
            if _inference_batcher_fn is None:
                from vetinari.inference import BatchRequest, get_inference_batcher

                _inference_batcher_fn = get_inference_batcher
                _batch_request_cls = BatchRequest
    return _inference_batcher_fn, _batch_request_cls
