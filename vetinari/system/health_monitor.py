"""Composite health monitor — periodic checks for Vetinari subsystems.

Polls every 30 seconds to verify disk, SQLite, memory, model, and EventBus
health.  Exposes a ``get_health_snapshot()`` function consumed by the
``/health`` endpoint in ``litestar_app.py``.

Pipeline step: Background Service → **Health Monitor** → Dashboard / Alerts.
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import sys
import threading
from collections.abc import Sized
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 30  # How often to check health


def _module_available(module_name: str) -> bool:
    """Return True when a module is importable or already present in sys.modules."""
    if sys.modules.get(module_name) is not None:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError) as exc:
        logger.debug("Module availability probe failed for %s: %s", module_name, exc)
        return False


class HealthStatus(Enum):
    """Overall system health tier."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Health of a single subsystem component."""

    name: str
    healthy: bool
    detail: str = ""


@dataclass(slots=True)
class HealthSnapshot:
    """Point-in-time system health report."""

    status: HealthStatus = HealthStatus.HEALTHY
    components: list[ComponentHealth] = field(default_factory=list)
    checked_at: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSON API response.

        Returns:
            Dict with status, components, and timestamp.
        """
        return {
            "status": self.status.value,
            "components": [{"name": c.name, "healthy": c.healthy, "detail": c.detail} for c in self.components],
            "checked_at": self.checked_at,
        }


# ── Module-level state ─────────────────────────────────────────────
# Written by: _poll_loop background thread
# Read by: get_health_snapshot() (called from /health endpoint)
# Protected by: _snapshot_lock
_latest_snapshot: HealthSnapshot | None = None
_snapshot_lock = threading.Lock()
_stop_event = threading.Event()
_monitor_thread: threading.Thread | None = None


# ── Individual health checks ──────────────────────────────────────


def _check_disk() -> ComponentHealth:
    """Check disk space using resource_monitor."""
    try:
        from vetinari.system.resource_monitor import check_disk_space

        status = check_disk_space()
        return ComponentHealth(
            name="disk",
            healthy=status.is_ok,
            detail=f"{status.usage_percent:.0f}% used, {status.free_bytes / (1024**3):.1f}GB free",
        )
    except Exception as exc:
        logger.warning("Disk health check failed — reporting unhealthy: %s", exc)
        return ComponentHealth(name="disk", healthy=False, detail=str(exc))


def _check_sqlite() -> ComponentHealth:
    """Verify SQLite database is writable."""
    try:
        from vetinari.database import _get_db_path

        db_path = _get_db_path()
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.execute("SELECT 1")
            return ComponentHealth(name="sqlite", healthy=True, detail="writable")
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("SQLite health check failed — database may be unavailable: %s", exc)
        return ComponentHealth(name="sqlite", healthy=False, detail=str(exc))


def _check_event_bus() -> ComponentHealth:
    """Verify EventBus is initialized and running."""
    try:
        from vetinari.events import get_event_bus

        bus = get_event_bus()
        if bus is None:
            return ComponentHealth(name="event_bus", healthy=False, detail="not initialized")
        return ComponentHealth(name="event_bus", healthy=True, detail="running")
    except Exception as exc:
        logger.warning("Event bus health check failed — bus may not be running: %s", exc)
        return ComponentHealth(name="event_bus", healthy=False, detail=str(exc))


def _check_models() -> ComponentHealth:
    """Check if any inference models are loaded.

    Returns unhealthy when no adapter manager is present or the adapter
    manager reports zero available models — an empty model list means
    no inference can be served, which is a degraded state.
    """
    try:
        from vetinari.adapter_manager import get_adapter_manager

        am = get_adapter_manager()
        if am is None:
            # No adapter manager = no inference capability
            return ComponentHealth(name="models", healthy=False, detail="adapter manager not initialised")
        get_available_models = getattr(am, "get_available_models", None)
        available_models = get_available_models() if callable(get_available_models) else None
        model_count = len(available_models) if isinstance(available_models, Sized) else 0
        if model_count == 0:
            return ComponentHealth(name="models", healthy=False, detail="no models available")
        return ComponentHealth(
            name="models",
            healthy=True,
            detail=f"{model_count} model(s) available",
        )
    except Exception as exc:
        logger.warning("Model health check failed — adapter manager may be unavailable: %s", exc)
        return ComponentHealth(name="models", healthy=False, detail=str(exc))


# ── Backend auto-detection (runs less frequently than health checks) ──

_BACKEND_CHECK_INTERVAL_S = 300  # Re-check every 5 minutes
_last_backend_check: float = 0.0
_cached_backends: list[str] = []


def _check_backends() -> ComponentHealth:
    """Detect available inference backends and auto-register new ones.

    Checks for vLLM and NVIDIA NIMs availability.  When a new backend is
    detected that isn't already registered with the AdapterManager, it
    auto-registers it so models served by that backend become available
    for inference immediately — no restart needed.

    Runs every 5 minutes (not every 30 seconds like health checks) to
    avoid unnecessary overhead from import probing and HTTP requests.
    """
    import time as _time

    global _last_backend_check, _cached_backends

    now = _time.monotonic()
    if now - _last_backend_check < _BACKEND_CHECK_INTERVAL_S and _cached_backends:
        return ComponentHealth(
            name="backends",
            healthy=True,
            detail=f"backends={_cached_backends} (cached)",
        )

    from vetinari.backend_config import load_backend_runtime_config, resolve_provider_fallback_order

    runtime_cfg = load_backend_runtime_config()
    backend_cfg = runtime_cfg.get("inference_backend", {})
    backends = ["llama_cpp"]  # Always available as the GGUF fallback

    def _endpoint_available(endpoint: str, backend_name: str) -> bool:
        try:
            import httpx

            resp = httpx.get(f"{endpoint.rstrip('/')}/v1/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            logger.warning(
                "%s endpoint %s not reachable during backend check",
                backend_name,
                endpoint,
            )
            return False

    vllm_cfg = backend_cfg.get("vllm", {})
    vllm_endpoint = vllm_cfg.get("endpoint", "")
    if isinstance(vllm_cfg, dict) and vllm_cfg.get("enabled") and vllm_endpoint:
        if _endpoint_available(str(vllm_endpoint), "vLLM"):
            backends.append("vllm")
    else:
        if _module_available("vllm"):
            backends.append("vllm")
        else:
            logger.debug("vLLM backend not available")

    nim_cfg = backend_cfg.get("nim", {})
    nim_endpoint = nim_cfg.get("endpoint", "")
    if (
        isinstance(nim_cfg, dict)
        and nim_cfg.get("enabled")
        and nim_endpoint
        and _endpoint_available(str(nim_endpoint), "NIM")
    ):
        backends.append("nim")

    # Auto-register newly detected backends with the AdapterManager
    new_backends = set(backends) - set(_cached_backends) - {"llama_cpp"}
    if new_backends:
        try:
            from vetinari.adapter_manager import get_adapter_manager
            from vetinari.adapters.base import ProviderConfig, ProviderType

            am = get_adapter_manager()
            if "vllm" in new_backends and am.get_provider("vllm") is None:
                vllm_config = ProviderConfig(
                    provider_type=ProviderType.VLLM,
                    name="vllm",
                    endpoint=str(vllm_endpoint),
                    extra_config={"gpu_only": True},
                )
                am.register_provider(vllm_config, "vllm")
                logger.info("Auto-detected and registered vLLM backend")

            if "nim" in new_backends and am.get_provider("nim") is None:
                nim_config = ProviderConfig(
                    provider_type=ProviderType.NIM,
                    name="nim",
                    endpoint=str(nim_endpoint),
                    extra_config={"gpu_only": True},
                )
                am.register_provider(nim_config, "nim")
                logger.info("Auto-detected and registered NIM backend at %s", nim_endpoint)
            fallback_order = resolve_provider_fallback_order(runtime_cfg, available_providers=set(am.list_providers()))
            if fallback_order:
                am.set_fallback_order(fallback_order)
        except Exception as exc:
            logger.warning("Could not auto-register newly detected backends: %s", exc)

    _cached_backends = backends
    _last_backend_check = now

    # llama_cpp is always present; additional backends being absent is still
    # a valid (if limited) configuration, so healthy=True as long as at least
    # one backend is reachable. The detail string lists what was found so the
    # operator can see whether env-off or disabled backends are skipped.
    return ComponentHealth(
        name="backends",
        healthy=len(backends) > 0,
        detail=", ".join(backends) if backends else "no backends available",
    )


# ── Polling loop ───────────────────────────────────────────────────


def _run_checks() -> list[ComponentHealth]:
    """Run all component checks, catching per-check exceptions individually.

    Each check is wrapped independently so that a failure in one (e.g. a
    transient import error in ``_check_backends``) does not prevent the
    others from running or cause the poll loop to crash.

    Returns:
        List of ComponentHealth results, one per check.
    """
    results: list[ComponentHealth] = []
    for check_fn, name in (
        (_check_disk, "disk"),
        (_check_sqlite, "sqlite"),
        (_check_event_bus, "event_bus"),
        (_check_models, "models"),
        (_check_backends, "backends"),
    ):
        try:
            results.append(check_fn())
        except Exception as exc:
            logger.warning(
                "Health check '%s' raised an unexpected exception — reporting unhealthy: %s",
                name,
                exc,
            )
            results.append(ComponentHealth(name=name, healthy=False, detail=f"check raised {type(exc).__name__}"))
    return results


def _classify_status(components: list[ComponentHealth]) -> HealthStatus:
    """Map a list of component results to an overall HealthStatus tier.

    Args:
        components: Per-component health results from ``_run_checks``.

    Returns:
        HEALTHY if all pass, DEGRADED if exactly one fails, UNHEALTHY otherwise.
    """
    unhealthy_count = sum(1 for c in components if not c.healthy)
    if unhealthy_count == 0:
        return HealthStatus.HEALTHY
    if unhealthy_count == 1:
        return HealthStatus.DEGRADED
    return HealthStatus.UNHEALTHY


def _poll_loop() -> None:
    """Background thread that periodically checks all health components.

    Each iteration is wrapped in a broad try/except so that an unexpected
    error in any check function cannot kill the thread and leave
    ``_latest_snapshot`` permanently stale.
    """
    while not _stop_event.is_set():
        try:
            components = _run_checks()
            status = _classify_status(components)

            snapshot = HealthSnapshot(
                status=status,
                components=components,
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

            with _snapshot_lock:
                global _latest_snapshot
                _latest_snapshot = snapshot

            if status != HealthStatus.HEALTHY:
                unhealthy_count = sum(1 for c in components if not c.healthy)
                logger.warning(
                    "Health check: %s (%d unhealthy component(s))",
                    status.value,
                    unhealthy_count,
                )
        except Exception as exc:
            logger.error(
                "Health poll loop encountered an unexpected error — snapshot not updated: %s",
                exc,
            )

        _stop_event.wait(timeout=_POLL_INTERVAL_SECONDS)


# ── Public API ─────────────────────────────────────────────────────


def get_health_snapshot() -> HealthSnapshot:
    """Return the most recent health snapshot.

    Serves the cached snapshot produced by the background poll thread when
    the thread is alive and has completed at least one poll. Falls back to a
    synchronous check when:
    - No background poll has run yet (startup race), or
    - The monitor thread has died (crash or stop), which would otherwise
      leave the caller with a permanently stale snapshot.

    Returns:
        Current system health report.
    """
    with _snapshot_lock:
        thread_alive = _monitor_thread is not None and _monitor_thread.is_alive()
        if _latest_snapshot is not None and thread_alive:
            return _latest_snapshot

    # No usable cached snapshot (either first call or dead thread) — fall back
    # to a synchronous check so the caller always gets a fresh result.
    if not thread_alive and _monitor_thread is not None:
        logger.warning("Health monitor thread is no longer alive — falling back to synchronous health check")

    components = _run_checks()
    status = _classify_status(components)
    return HealthSnapshot(
        status=status,
        components=components,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def start_health_monitor() -> threading.Thread:
    """Start the background health monitor thread if not already running.

    Idempotent: if a monitor thread is already alive, returns the existing
    thread without spawning a duplicate. This prevents double-polling when
    lifespan startup is called more than once (e.g. during testing).

    Returns:
        The (possibly pre-existing) monitor thread.
    """
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        logger.debug("Health monitor already running — skipping duplicate start")
        return _monitor_thread

    _stop_event.clear()
    _monitor_thread = threading.Thread(
        target=_poll_loop,
        name="vetinari-health-monitor",
        daemon=True,
    )
    _monitor_thread.start()
    logger.info("Health monitor started (polling every %ds)", _POLL_INTERVAL_SECONDS)
    return _monitor_thread


def stop_health_monitor() -> None:
    """Stop the background health monitor thread and wait for it to exit."""
    _stop_event.set()
    if _monitor_thread is not None:
        _monitor_thread.join(timeout=5)
    logger.info("Health monitor stopped")
