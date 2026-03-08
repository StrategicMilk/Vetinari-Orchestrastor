"""Shared state, locks, and helpers for web UI blueprints.

All global mutable state and cross-cutting helpers live here so that
both the main ``web_ui`` module and individual Flask Blueprints can
import them without circular dependencies.
"""

import collections
import functools
import logging
import os
import queue as _queue
import secrets
import threading
import time
from pathlib import Path

from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation helpers (S8)
# ---------------------------------------------------------------------------
_MAX_INPUT_LENGTH = 10000


def validate_json_request() -> Tuple[Optional[dict], Optional[Any]]:
    """Validate that the incoming request contains well-formed JSON.

    Checks:
    - Content-Type is application/json
    - Body parses as valid JSON
    - String values are sanitized (null bytes stripped, length limited)

    Returns:
        (data, None) on success, or (None, error_response) on failure.
        The error_response is a Flask response tuple ready to return.
    """
    from flask import jsonify as _jsonify, request as _request

    content_type = _request.content_type or ""
    if "application/json" not in content_type:
        return None, (_jsonify({"error": "Content-Type must be application/json"}), 400)

    try:
        data = _request.get_json(force=False, silent=False)
    except Exception:
        return None, (_jsonify({"error": "Malformed JSON in request body"}), 400)

    if data is None:
        return None, (_jsonify({"error": "Malformed JSON in request body"}), 400)

    if isinstance(data, dict):
        data = _sanitize_dict(data)

    return data, None


def _sanitize_string(value: str) -> str:
    """Strip null bytes and limit length of a string value."""
    value = value.replace("\x00", "")
    if len(value) > _MAX_INPUT_LENGTH:
        value = value[:_MAX_INPUT_LENGTH]
    return value


def _sanitize_dict(d: dict) -> dict:
    """Recursively sanitize all string values in a dictionary."""
    sanitized = {}
    for key, value in d.items():
        if isinstance(key, str):
            key = _sanitize_string(key)
        if isinstance(value, str):
            sanitized[key] = _sanitize_string(value)
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = _sanitize_list(value)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_list(lst: list) -> list:
    """Recursively sanitize all string values in a list."""
    sanitized = []
    for item in lst:
        if isinstance(item, str):
            sanitized.append(_sanitize_string(item))
        elif isinstance(item, dict):
            sanitized.append(_sanitize_dict(item))
        elif isinstance(item, list):
            sanitized.append(_sanitize_list(item))
        else:
            sanitized.append(item)
    return sanitized


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
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
from vetinari.constants import (
    DEFAULT_LM_STUDIO_HOST,
    DEFAULT_API_TOKEN,
    ADMIN_TOKEN as _ADMIN_TOKEN,
    ENABLE_EXTERNAL_DISCOVERY,
)

current_config = {
    "host": DEFAULT_LM_STUDIO_HOST,
    "config_path": "manifest/vetinari.yaml",
    "api_token": DEFAULT_API_TOKEN or os.environ.get("VETINARI_API_TOKEN", ""),
    "default_models": ["qwen3-coder-next", "qwen3-30b-a3b-gemini-pro-high-reasoning-2507-hi8"],
    "fallback_models": ["llama-3.2-1b-instruct", "qwen2.5-0.5b-instruct", "devstral-small-2505-deepseek-v3.2-speciale-distill"],
    "uncensored_fallback_models": ["qwen3-vl-32b-gemini-heretic-uncensored-thinking", "glm-4.7-flash-uncensored-heretic-neo-code-imatrix-max"],
    "memory_budget_gb": 48,
    "active_model_id": None,
}

# ---------------------------------------------------------------------------
# Task cancellation registry  (thread-safe)
# ---------------------------------------------------------------------------
_cancel_flags: dict = {}  # project_id -> threading.Event
_cancel_flags_lock = threading.Lock()


def _register_project_task(project_id: str) -> "threading.Event":
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
        try:
            q.put_nowait({"event": event_type, "data": _json.dumps(data)})
        except Exception:
            pass  # Queue full — drop event


def _cleanup_project_state(project_id: str) -> None:
    """Remove cancel flags and SSE queues for a project to prevent memory leaks."""
    with _cancel_flags_lock:
        _cancel_flags.pop(project_id, None)
    with _sse_streams_lock:
        q = _sse_streams.pop(project_id, None)
    if q:
        try:
            q.put_nowait(None)  # Sentinel to close any active SSE stream
        except Exception:
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

            model_pool = ModelPool(
                current_config, current_config.get("host", "http://localhost:1234")
            )
            model_pool.discover_models()
            lm_models = model_pool.list_models()
        except Exception:
            pass

        candidates = search_engine.search_for_task(task_description, lm_models)
        return candidates
    except Exception as e:
        logger.error(f"Light search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Admin / security helpers
# ---------------------------------------------------------------------------

# Rate limiting: track failed auth attempts per IP
_auth_failures: dict = {}  # ip -> deque of timestamps
_auth_failures_lock = threading.Lock()
_AUTH_MAX_FAILURES = 5
_AUTH_WINDOW_SECONDS = 60


def _record_auth_failure(ip: str) -> None:
    """Record a failed authentication attempt for the given IP."""
    now = time.time()
    with _auth_failures_lock:
        if ip not in _auth_failures:
            _auth_failures[ip] = collections.deque()
        dq = _auth_failures[ip]
        dq.append(now)
        # Prune entries older than the window
        cutoff = now - _AUTH_WINDOW_SECONDS
        while dq and dq[0] < cutoff:
            dq.popleft()


def _is_rate_limited(ip: str) -> bool:
    """Check if an IP has exceeded the maximum allowed auth failures."""
    now = time.time()
    with _auth_failures_lock:
        dq = _auth_failures.get(ip)
        if not dq:
            return False
        cutoff = now - _AUTH_WINDOW_SECONDS
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq) >= _AUTH_MAX_FAILURES


def _is_admin_user() -> bool:
    """Check if the current request comes from an admin user.

    Uses ``secrets.compare_digest`` for timing-safe token comparison.
    No localhost bypass -- a valid token is always required when
    VETINARI_ADMIN_TOKEN is set.
    """
    from flask import request

    admin_token = _ADMIN_TOKEN
    if not admin_token:
        # No token configured -- deny all requests for safety
        return False

    remote_ip = request.remote_addr or "unknown"

    # Rate-limit check
    if _is_rate_limited(remote_ip):
        return False

    auth_header = request.headers.get("Authorization", "")
    req_token = request.headers.get("X-Admin-Token", "")
    provided = req_token or auth_header.replace("Bearer ", "")

    if not provided:
        _record_auth_failure(remote_ip)
        return False

    if secrets.compare_digest(provided, admin_token):
        return True

    _record_auth_failure(remote_ip)
    return False


def require_admin_token(f):
    """Decorator to require a valid admin token for an endpoint."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not _is_admin_user():
            from flask import jsonify
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


def _project_external_model_enabled(project_dir) -> bool:
    """Check if external model discovery is enabled for a specific project."""
    try:
        import yaml
        config_file = Path(project_dir) / "project.yaml"
        if config_file.exists():
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("enable_external_model_discovery", ENABLE_EXTERNAL_DISCOVERY)
    except Exception:
        pass
    return ENABLE_EXTERNAL_DISCOVERY
