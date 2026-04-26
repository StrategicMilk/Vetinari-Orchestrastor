"""Shared state, locks, and helpers for web UI handlers.

All global mutable state and cross-cutting helpers live here so that
both the main ``web_ui`` module and individual Litestar handlers can
reference them without circular dependencies.

# MUTABLE GLOBALS (consolidated list for auditability):
# - current_config: VetinariConfig — loaded at import time via get_config(),
#     read by all route handlers; updated only by config POST endpoint.
# - ENABLE_EXTERNAL_DISCOVERY: bool — snapshot of current_config at startup;
#     stale after startup; live value served via get_external_discovery_enabled().
# - _external_discovery_cache: dict — TTL cache for ENABLE_EXTERNAL_DISCOVERY;
#     written by get_external_discovery_enabled(), protected by _discovery_cache_lock.
# - _cancel_flags: dict[str, threading.Event] — project_id -> cancel signal;
#     written by _register_project_task(), read by background threads;
#     protected by _cancel_flags_lock.
# - _sse_streams: dict[str, queue.Queue] — project_id -> SSE event queue;
#     written by _get_sse_queue(), drained by SSE generator endpoints;
#     protected by _sse_streams_lock.
# - _sse_sequence_counters: dict[str, int] — monotonic sequence per project;
#     written inside _sse_streams_lock by _push_sse_event().
# - _sse_dropped_counts: dict[str, int] — dropped-event counter per project;
#     written inside _sse_streams_lock by _push_sse_event() on queue-full.
# - _models_cache: list — cached model list for the /models API;
#     written by the model-discovery path; protected by _models_cache_lock.
# - _models_cache_ts: float — monotonic timestamp of last _models_cache fill;
#     used for TTL expiry check inside _models_cache_lock.
"""

from __future__ import annotations

import logging
import os
import queue as _queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from vetinari.constants import SSE_PROJECT_QUEUE_SIZE
from vetinari.web.config import VetinariConfig, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Runtime configuration — VetinariConfig is the single source of truth.
# Supports attribute access (config.api_token) and dict-style access
# (config["api_token"], config.get("api_token")) for backward compatibility.
# ---------------------------------------------------------------------------
current_config: VetinariConfig = get_config()

# Global feature flag for external model discovery — read from config at startup.
# Live value is served by get_external_discovery_enabled() with a 60-second TTL.
ENABLE_EXTERNAL_DISCOVERY: bool = current_config.enable_external_discovery

_external_discovery_cache: dict[str, Any] = {"value": None, "expires": 0.0}
_DISCOVERY_TTL_SECONDS = 60  # Re-read config every 60 seconds
_discovery_cache_lock = threading.Lock()  # Protects _external_discovery_cache reads+writes


def get_external_discovery_enabled() -> bool:
    """Check if external discovery is enabled, with a 60-second TTL cache.

    Re-reads the runtime config periodically so that changes to
    ``enable_external_discovery`` take effect without a server restart.
    Thread-safe: lock held through check-and-update to prevent races.

    Returns:
        ``True`` when external model discovery is enabled, ``False`` otherwise
        or if config cannot be read.
    """
    now = time.monotonic()
    with _discovery_cache_lock:
        if _external_discovery_cache["expires"] > now:
            return _external_discovery_cache["value"]  # type: ignore[return-value]
        try:
            cfg = get_config()
            val: bool = cfg.enable_external_discovery
        except Exception:
            val = False
        _external_discovery_cache["value"] = val
        _external_discovery_cache["expires"] = now + _DISCOVERY_TTL_SECONDS
        return val


# ---------------------------------------------------------------------------
# Task cancellation registry  (thread-safe)
# ---------------------------------------------------------------------------
_cancel_flags: dict[str, threading.Event] = {}  # project_id -> threading.Event
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


def _is_project_actually_running(project_id: str) -> bool:
    """Check if a project has an active background task in the current process.

    The YAML status may say 'running' from a previous session that crashed.
    This checks the in-memory task registry to see if execution is genuinely
    happening right now.
    """
    with _cancel_flags_lock:
        flag = _cancel_flags.get(project_id)
    # If there's a registered cancel flag that hasn't been set, it's running
    return flag is not None and not flag.is_set()


# ---------------------------------------------------------------------------
# SSE stream registry  (thread-safe, project_id -> queue of messages)
# ---------------------------------------------------------------------------
_sse_streams: dict[str, _queue.Queue] = {}  # project_id -> queue.Queue
_sse_streams_lock = threading.Lock()
_sse_sequence_counters: dict[str, int] = {}  # project_id -> monotonic sequence number
_sse_dropped_counts: dict[str, int] = {}  # project_id -> count of dropped events


def _get_sse_queue(project_id: str) -> _queue.Queue:
    """Return the SSE event queue for a project, creating it if it does not exist.

    Args:
        project_id: The project identifier to get or create the SSE queue for.

    Returns:
        The queue.Queue instance for the given project.
    """
    with _sse_streams_lock:
        if project_id not in _sse_streams:
            _sse_streams[project_id] = _queue.Queue(maxsize=SSE_PROJECT_QUEUE_SIZE)
            _sse_sequence_counters[project_id] = 0
            _sse_dropped_counts[project_id] = 0
        return _sse_streams[project_id]


def _push_sse_event(project_id: str, event_type: str, data: dict) -> None:
    """Push an SSE event with a sequence number to listeners for a project.

    Each event is assigned a monotonically increasing sequence number so
    clients can detect gaps. Also writes an audit record to SQLite.
    """
    import json as _json

    payload_json = _json.dumps(data)
    with _sse_streams_lock:
        q = _sse_streams.get(project_id)
        seq = _sse_sequence_counters.get(project_id, 0) + 1
        _sse_sequence_counters[project_id] = seq

    if q is not None:
        event_payload = {"event": event_type, "data": payload_json, "id": str(seq)}
        try:
            q.put_nowait(event_payload)
        except _queue.Full:
            with _sse_streams_lock:
                _sse_dropped_counts[project_id] = _sse_dropped_counts.get(project_id, 0) + 1
                dropped = _sse_dropped_counts[project_id]
            logger.warning(
                "SSE queue full for project %s — dropped event %s (seq=%d, total_dropped=%d)",
                project_id,
                event_type,
                seq,
                dropped,
            )

    # Best-effort audit log — never blocks the caller on DB contention
    try:
        from vetinari.database import get_connection

        conn = get_connection()
        conn.execute(
            "INSERT INTO sse_event_log (project_id, event_type, payload_json, sequence_num) VALUES (?, ?, ?, ?)",
            (project_id, event_type, payload_json, seq),
        )
        conn.commit()
    except Exception as exc:
        logger.warning(
            "SSE event log write failed for project %s event %s: %s",
            project_id,
            event_type,
            exc,
        )


def _cleanup_project_state(project_id: str) -> None:
    """Remove cancel flags, SSE queues, and sequence/drop counters for a project.

    Must be called when a project's execution ends (completed, cancelled, or
    client-disconnected) to prevent memory leaks across all four shared state
    dicts: ``_cancel_flags``, ``_sse_streams``, ``_sse_sequence_counters``, and
    ``_sse_dropped_counts``.
    """
    with _cancel_flags_lock:
        _cancel_flags.pop(project_id, None)
    with _sse_streams_lock:
        q = _sse_streams.pop(project_id, None)
        _sse_sequence_counters.pop(project_id, None)
        _sse_dropped_counts.pop(project_id, None)
    if q:
        try:
            q.put_nowait(None)  # Sentinel to close any active SSE stream
        except Exception:  # Queue full — sentinel not delivered; stream will time out naturally
            logger.warning("SSE queue full when sending close sentinel for project %s", project_id)


# ---------------------------------------------------------------------------
# Validated config loading (Pydantic schema validation for YAML config files)
# ---------------------------------------------------------------------------

_CONFIG_DIR = PROJECT_ROOT / "config"


def _load_models_config_validated() -> dict[str, Any]:
    """Load and validate ``config/models.yaml``, returning a ModelPool-compatible config.

    Uses :class:`~vetinari.config.models_schema.ModelsConfig` to validate the
    static model registry.  Returns a dict with a ``"models"`` key containing
    the validated local model entries in the format expected by
    :class:`~vetinari.models.model_pool.ModelPool`.

    Returns:
        A dictionary with a ``"models"`` key containing validated local model
        dicts, or an empty dict when the YAML file is absent or fails
        schema validation.
    """
    models_yaml = _CONFIG_DIR / "models.yaml"
    if not models_yaml.exists():
        return {}
    try:
        from vetinari.config.models_schema import ModelsConfig

        validated = ModelsConfig.from_yaml_file(models_yaml)
        static_models = [
            {
                "id": m.model_id,
                "name": m.display_name or m.model_id,
                "capabilities": m.capabilities,
                "context_len": m.context_window,
                "memory_gb": m.memory_requirements_gb,
                "version": "",
                "endpoint": m.endpoint,
            }
            for m in validated.models
        ]
        logger.debug("Loaded %d validated models from models.yaml", len(static_models))
        return {"models": static_models}
    except Exception as exc:
        logger.warning("models.yaml failed schema validation — using defaults: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Task-type recommendation inference (shared with models_api.py)
# ---------------------------------------------------------------------------

# Maps capability tags from model discovery to task types the model suits.
_CAPABILITY_TO_TASKS: dict[str, list[str]] = {
    "coding": ["coding", "review", "documentation"],
    "reasoning": ["reasoning", "planning", "security"],
    "analysis": ["reasoning", "research", "security"],
    "vision": ["creative", "research"],
    "creative": ["creative", "documentation"],
    "fast": ["classification", "general"],
    "uncensored": ["creative", "research"],
    "classification": ["classification"],
    "general": ["general"],
}


def _infer_recommended_tasks(capabilities: list[str]) -> list[str]:
    """Derive task-type recommendations from a model's capability tags.

    Args:
        capabilities: List of capability tag strings (e.g. ["coding", "reasoning"]).

    Returns:
        Deduplicated, sorted list of task type strings this model is suited for.
    """
    tasks: set[str] = set()
    for cap in capabilities:
        tasks.update(_CAPABILITY_TO_TASKS.get(cap, []))
    if not tasks:
        tasks.add("general")
    return sorted(tasks)


# ---------------------------------------------------------------------------
# Model discovery cache  (avoids blocking the UI on every request)
# ---------------------------------------------------------------------------
_models_cache: list = []
_models_cache_ts: float = 0.0
_MODELS_CACHE_TTL: float = 60.0  # seconds
_models_cache_lock = threading.Lock()


def _get_models_cached(force: bool = False) -> list[dict[str, Any]]:
    """Return models from cache if fresh, otherwise run discovery and cache result.

    Discovers local models via llama-cpp-python adapter and cloud models from
    configured API tokens.  Falls back gracefully when local discovery fails.
    """
    global _models_cache, _models_cache_ts
    now = time.time()
    with _models_cache_lock:
        if not force and _models_cache and (now - _models_cache_ts) < _MODELS_CACHE_TTL:
            return list(_models_cache)

    from vetinari.models.model_pool import ModelPool

    # Start from runtime config; overlay schema-validated static models when available.
    pool_config = current_config.to_dict()
    validated_models = _load_models_config_validated()
    if validated_models:
        pool_config.update(validated_models)
    pool = ModelPool(pool_config)

    # Discover local models (handles failures gracefully internally)
    try:
        pool.discover_models()
    except Exception as e:
        logger.warning("[web_ui] Local model discovery failed: %s", e)

    # Merge local + cloud models, dedupe by id
    all_models = list(pool.models)
    try:
        cloud_models = pool.get_cloud_models()
        seen_ids = {m.get("id", "") for m in all_models}
        for cm in cloud_models:
            if cm.get("id", "") not in seen_ids:
                all_models.append(cm)
                seen_ids.add(cm.get("id", ""))
    except Exception as e:
        logger.warning("[web_ui] Cloud model lookup failed: %s", e)

    fresh = [
        {
            "id": m.get("id", ""),
            "name": m.get("name", m.get("id", "")),
            "capabilities": m.get("capabilities", []),
            "context_len": m.get("context_len", 0),
            "memory_gb": m.get("memory_gb", 0),
            "version": m.get("version", ""),
            "source": "cloud" if m.get("id", "").startswith("cloud:") else "local",
            "recommended_for": _infer_recommended_tasks(m.get("capabilities", [])),
        }
        for m in all_models
    ]
    with _models_cache_lock:
        if fresh:
            _models_cache = fresh
            _models_cache_ts = now
        elif _models_cache:
            # Discovery returned nothing — serve stale cache but do not suppress the log
            logger.warning(
                "[web_ui] Model discovery returned no results — serving %d stale cached models",
                len(_models_cache),
            )
        return fresh or list(_models_cache)


# ---------------------------------------------------------------------------
# Cloud API key detection
# ---------------------------------------------------------------------------

# Canonical map of provider name -> list of env var names that grant access.
# Checked in order; first present var wins for each provider.
_CLOUD_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "huggingface": ["HF_HUB_TOKEN"],
    "replicate": ["REPLICATE_API_TOKEN"],
}


def detect_api_keys() -> dict[str, str | None]:
    """Detect which cloud provider API keys are present in the environment.

    Checks each provider's env vars in order and returns the first found for
    each provider, or ``None`` when no key is set.  Never returns the actual
    key values — only the env-var name that is populated, or ``None``.

    Returns:
        Dict mapping provider name to the env-var name that is set (e.g.
        ``{"anthropic": "ANTHROPIC_API_KEY", "openai": None, ...}``).
        A ``None`` value means no key is present for that provider.
    """
    result: dict[str, str | None] = {}
    for provider, env_vars in _CLOUD_PROVIDER_ENV_VARS.items():
        found: str | None = None
        for var in env_vars:
            if os.environ.get(var):
                found = var
                break
        result[provider] = found
    return result


# ---------------------------------------------------------------------------
# Orchestrator singleton
# ---------------------------------------------------------------------------
orchestrator = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> Any:
    """Return the singleton Orchestrator instance, creating it on first call.

    Uses double-checked locking so that sys.path mutation and Orchestrator
    construction are thread-safe under concurrent request load.

    Returns:
        The Orchestrator singleton configured with the current config path and API token.
    """
    global orchestrator
    if orchestrator is None:
        with _orchestrator_lock:
            if orchestrator is None:
                import sys

                sys.path.insert(0, str(PROJECT_ROOT))
                from vetinari.orchestrator import Orchestrator

                api_token = current_config.api_token or None
                orchestrator = Orchestrator(
                    current_config.config_path,
                    api_token=api_token,
                )
    return orchestrator


def set_orchestrator(orch: Any) -> None:
    """Replace the singleton Orchestrator instance.

    Intended for testing and re-configuration after startup.  Passes the
    supplied orchestrator to all route handlers that call ``get_orchestrator()``.

    Args:
        orch: The Orchestrator instance to install as the singleton.
    """
    global orchestrator
    with _orchestrator_lock:
        orchestrator = orch


# ---------------------------------------------------------------------------
# Request validation helpers (consolidated from web/utils.py)
# ---------------------------------------------------------------------------

# Matches only safe identifier characters: alphanumeric, hyphens, underscores.
# Rejects path-traversal sequences (../, ..\, absolute paths).
_SAFE_PATH_PARAM_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def validate_path_param(value: str) -> str | None:
    r"""Validate a URL path parameter against traversal and injection sequences.

    Rejects any value containing ``..``, ``/``, ``\\``, null bytes, or any
    character outside ``[A-Za-z0-9_-]``.  Returns the validated value if safe,
    or ``None`` if the value is unsafe.

    Args:
        value: The raw path parameter string to validate (e.g. ``project_id``).

    Returns:
        The original ``value`` unchanged when it is safe, or ``None`` when
        the value contains potentially dangerous characters.
    """
    if not value or not _SAFE_PATH_PARAM_RE.match(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Admin / security helpers
# ---------------------------------------------------------------------------


def _project_external_model_enabled(project_dir) -> bool:
    """Check if external model discovery is enabled for a specific project."""
    try:
        import yaml

        config_file = Path(project_dir) / "project.yaml"
        if config_file.exists():
            with Path(config_file).open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("enable_external_model_discovery", get_external_discovery_enabled())
    except Exception:  # Missing or malformed project.yaml — fall back to global default
        logger.warning("Failed to read project.yaml for %s; using global default", project_dir, exc_info=True)
    return get_external_discovery_enabled()


def _derive_project_status(
    config_status: str,
    planned_tasks: list[dict[str, Any]],
    completed_tasks: set[str],
    project_id: str = "",
) -> str:
    """Derive a project's status from its execution state.

    Falls back to the config-provided status if it's an explicit value other
    than ``"unknown"``.  Otherwise inspects task completion to infer the
    real state.

    Args:
        config_status: The ``status`` value read from project.yaml.
        planned_tasks: List of planned task dicts from the config.
        completed_tasks: Set of task IDs that have completed output files.
        project_id: Project directory name, used to check if a background
            thread is actually running (detects stale "running" status).

    Returns:
        One of ``"completed"``, ``"in_progress"``, ``"pending"``, or the
        original *config_status* when it carries an explicit value.
    """
    from vetinari.types import StatusEnum

    if config_status and config_status != "unknown":
        if config_status == StatusEnum.RUNNING.value and not _is_project_actually_running(project_id):
            return "interrupted"
        return config_status

    if not planned_tasks:
        return "pending"

    task_ids = {t.get("id", "") for t in planned_tasks if t.get("id")}
    if not task_ids:
        return "pending"

    done = task_ids & completed_tasks
    if done == task_ids:
        return "completed"
    if done:
        return "in_progress"
    return "pending"
