"""Shared state, locks, and helpers for web UI blueprints.

All global mutable state and cross-cutting helpers live here so that
both the main ``web_ui`` module and individual Flask Blueprints can
reference them without circular dependencies.
"""

from __future__ import annotations

import hmac
import logging
import os
import queue as _queue
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Load .env file if present (before reading env vars)
# ---------------------------------------------------------------------------
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip()
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:  # noqa: S110, VET022
        pass

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
current_config = {
    "host": os.environ.get("LM_STUDIO_HOST", "http://localhost:1234"),
    "config_path": "manifest/vetinari.yaml",
    "api_token": os.environ.get("LM_STUDIO_API_TOKEN") or os.environ.get("VETINARI_API_TOKEN", ""),
    "default_models": ["qwen3-coder-next", "qwen3-30b-a3b-gemini-pro-high-reasoning-2507-hi8"],
    "fallback_models": [
        "llama-3.2-1b-instruct",
        "qwen2.5-0.5b-instruct",
        "devstral-small-2505-deepseek-v3.2-speciale-distill",
    ],
    "uncensored_fallback_models": [
        "qwen3-vl-32b-gemini-heretic-uncensored-thinking",
        "glm-4.7-flash-uncensored-heretic-neo-code-imatrix-max",
    ],
    "memory_budget_gb": 48,
    "active_model_id": None,
}

# Global feature flag for external model discovery (default enabled)
ENABLE_EXTERNAL_DISCOVERY = str(os.environ.get("ENABLE_EXTERNAL_DISCOVERY", "true")).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Task cancellation registry  (thread-safe)
# ---------------------------------------------------------------------------
_cancel_flags: dict = {}  # project_id -> threading.Event
_cancel_flags_lock = threading.Lock()


def _register_project_task(project_id: str) -> threading.Event:
    flag = threading.Event()
    with _cancel_flags_lock:
        _cancel_flags[project_id] = flag
    return flag


def _cancel_project_task(project_id: str) -> bool:
    with _cancel_flags_lock:
        flag = _cancel_flags.get(project_id)
    if flag:
        flag.set()
        return True
    return False


# ---------------------------------------------------------------------------
# SSE stream registry  (thread-safe, project_id -> queue of messages)
# ---------------------------------------------------------------------------
_sse_streams: dict = {}  # project_id -> queue.Queue
_sse_streams_lock = threading.Lock()


def _get_sse_queue(project_id: str):
    with _sse_streams_lock:
        if project_id not in _sse_streams:
            _sse_streams[project_id] = _queue.Queue(maxsize=200)
        return _sse_streams[project_id]


def _push_sse_event(project_id: str, event_type: str, data: dict) -> None:
    """Push an SSE event to all listeners for a project."""
    import json as _json

    with _sse_streams_lock:
        q = _sse_streams.get(project_id)
    if q:
        try:  # noqa: SIM105
            q.put_nowait({"event": event_type, "data": _json.dumps(data)})
        except Exception:  # noqa: S110, VET022
            pass  # Queue full — drop event


def _cleanup_project_state(project_id: str) -> None:
    """Remove cancel flags and SSE queues for a project to prevent memory leaks."""
    with _cancel_flags_lock:
        _cancel_flags.pop(project_id, None)
    with _sse_streams_lock:
        q = _sse_streams.pop(project_id, None)
    if q:
        try:  # noqa: SIM105
            q.put_nowait(None)  # Sentinel to close any active SSE stream
        except Exception:  # noqa: S110, VET022
            pass


# ---------------------------------------------------------------------------
# Model discovery cache  (avoids blocking the UI on every request)
# ---------------------------------------------------------------------------
_models_cache: list = []
_models_cache_ts: float = 0.0
_MODELS_CACHE_TTL: float = 60.0  # seconds
_models_cache_lock = threading.Lock()


def _get_models_cached(force: bool = False) -> list:
    """Return models from cache if fresh, otherwise run discovery and cache result."""
    global _models_cache, _models_cache_ts
    now = time.time()
    with _models_cache_lock:
        if not force and _models_cache and (now - _models_cache_ts) < _MODELS_CACHE_TTL:
            return list(_models_cache)
    try:
        orb = get_orchestrator()
        orb.model_pool.discover_models()
        fresh = [
            {
                "id": m.get("id", ""),
                "name": m.get("name", m.get("id", "")),
                "capabilities": m.get("capabilities", []),
                "context_len": m.get("context_len", 0),
                "memory_gb": m.get("memory_gb", 0),
                "version": m.get("version", ""),
            }
            for m in orb.model_pool.models
        ]
        with _models_cache_lock:
            if fresh:
                _models_cache = fresh
                _models_cache_ts = now
            return fresh if fresh else list(_models_cache)
    except Exception as e:
        logger.warning(f"[web_ui] Model discovery failed: {e}")
        return _models_cache


# ---------------------------------------------------------------------------
# Orchestrator singleton
# ---------------------------------------------------------------------------
orchestrator = None


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from vetinari.orchestrator import Orchestrator

        api_token = current_config.get("api_token") or None
        orchestrator = Orchestrator(
            current_config["config_path"],
            current_config["host"],
            api_token=api_token,
        )
    return orchestrator


# ---------------------------------------------------------------------------
# Model search helpers
# ---------------------------------------------------------------------------
def refresh_model_cache():
    try:
        from vetinari.model_search import ModelSearchEngine

        search_engine = ModelSearchEngine()
        search_engine.refresh_all_caches()
        from datetime import datetime

        logger.info(f"Model cache refreshed at {datetime.now()}")
    except Exception as e:
        logger.error(f"Error refreshing model cache: {e}")


def trigger_light_search(project_id: str, task_description: str):
    try:
        from vetinari.model_search import ModelSearchEngine

        search_engine = ModelSearchEngine()

        lm_models = []
        try:
            from vetinari.model_pool import ModelPool

            model_pool = ModelPool(current_config, current_config.get("host", "http://localhost:1234"))
            model_pool.discover_models()
            lm_models = model_pool.list_models()
        except Exception:  # noqa: S110, VET022
            pass

        candidates = search_engine.search_for_task(task_description, lm_models)
        return candidates
    except Exception as e:
        logger.error(f"Light search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Admin / security helpers
# ---------------------------------------------------------------------------
def _is_admin_user() -> bool:
    """Check if the current request comes from an admin user."""
    from flask import request

    admin_token = os.environ.get("VETINARI_ADMIN_TOKEN", "")
    if admin_token:
        auth_header = request.headers.get("Authorization", "")
        req_token = request.headers.get("X-Admin-Token", "")
        provided = req_token or auth_header.replace("Bearer ", "")
        return hmac.compare_digest(provided, admin_token)
    remote = request.remote_addr or ""
    return remote in ("127.0.0.1", "::1", "localhost")


def _project_external_model_enabled(project_dir) -> bool:
    """Check if external model discovery is enabled for a specific project."""
    try:
        import yaml

        config_file = Path(project_dir) / "project.yaml"
        if config_file.exists():
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("enable_external_model_discovery", ENABLE_EXTERNAL_DISCOVERY)
    except Exception:  # noqa: S110, VET022
        pass
    return ENABLE_EXTERNAL_DISCOVERY
