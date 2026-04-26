"""Health check, shutdown, GPU statistics, and system resource Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``system_hardware_api._register(bp)``.  Handles the composite health-check
dashboard endpoint, graceful server shutdown, NVIDIA GPU statistics via
pynvml, and CPU/RAM/disk metrics via psutil.

This is part of the Flask->Litestar migration (ADR-0066).  The URL paths
are identical to the Flask originals so existing clients require no changes.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Serialise nvmlInit/nvmlShutdown calls to prevent NVML handle corruption
# under concurrent GPU status requests.
_nvml_lock = threading.Lock()

try:
    from litestar import MediaType, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _gpu_info_locked(pynvml: Any) -> dict[str, Any]:
    """Collect GPU info under the NVML lock to prevent handle corruption.

    Args:
        pynvml: The imported pynvml module.

    Returns:
        Dict with GPU stats or error description.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.warning("api_system_gpu: NVML initialisation failed", exc_info=True)
        return {"gpu_available": False, "gpus": [], "error": "NVML initialisation failed"}

    gpus: list[dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_mb = round(mem.used / (1024 * 1024), 1)
            vram_total_mb = round(mem.total / (1024 * 1024), 1)
            vram_pct = round(100.0 * mem.used / mem.total, 1) if mem.total else 0.0
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent: int | None = int(util.gpu)
            except pynvml.NVMLError:
                utilization_percent = None
            try:
                temp: int | None = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                temp = None
            gpus.append({
                "name": name,
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "vram_used_percent": vram_pct,
                "utilization_percent": utilization_percent,
                "temperature_c": temp,
            })
    except pynvml.NVMLError as exc:
        logger.warning("api_system_gpu: NVML error reading device info: %s", exc)
        return {"gpu_available": False, "gpus": [], "error": "NVML device read failed"}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            logger.warning("api_system_gpu: nvmlShutdown failed (non-fatal)", exc_info=True)

    return {"gpu_available": True, "gpus": gpus}


def _collect_composite_health() -> dict[str, Any]:
    """Collect release-readiness health without exposing private diagnostics."""
    from datetime import datetime, timezone

    checks: dict[str, Any] = {}

    try:
        from vetinari.resilience import get_circuit_breaker_registry

        registry = get_circuit_breaker_registry()
        summary = registry.get_summary()
        all_closed = all(cb.get("state") == "closed" for cb in summary.values()) if summary else True
        checks["circuit_breakers"] = {"ok": all_closed, "breakers": summary}
    except Exception:
        logger.exception("Failed to read circuit breaker registry for health check")
        checks["circuit_breakers"] = {"ok": False, "error": "unavailable"}

    try:
        from vetinari.workflow.spc import get_spc_monitor

        spc = get_spc_monitor()
        spc_status = spc.get_status() if hasattr(spc, "get_status") else {}
        checks["spc"] = {"ok": True, "status": spc_status}
    except Exception:
        checks["spc"] = {"ok": False, "status": "unavailable"}

    try:
        from vetinari.workflow.andon import get_andon_system

        andon = get_andon_system()
        andon_state = andon.current_state() if hasattr(andon, "current_state") else "unknown"
        checks["andon"] = {"ok": True, "state": andon_state}
    except Exception:
        checks["andon"] = {"ok": False, "state": "unavailable"}

    try:
        from vetinari.learning.model_selector import get_thompson_selector

        selector = get_thompson_selector()
        arm_count = len(selector._arms) if hasattr(selector, "_arms") else 0
        checks["thompson"] = {"ok": True, "arm_count": arm_count}
    except Exception:
        checks["thompson"] = {"ok": False, "status": "unavailable"}

    try:
        from vetinari.memory.unified import get_unified_memory_store

        store = get_unified_memory_store()
        health = store.check_health() if hasattr(store, "check_health") else {"ok": True}
        checks["memory"] = health if isinstance(health, dict) else {"ok": True}
    except Exception:
        checks["memory"] = {"ok": False, "status": "unavailable"}

    try:
        from vetinari.analytics.anomaly import get_anomaly_detector

        detector = get_anomaly_detector()
        stats = detector.get_stats()
        checks["anomaly_detector"] = {"ok": True, "stats": stats}
    except Exception:
        checks["anomaly_detector"] = {"ok": False, "status": "unavailable"}

    try:
        from vetinari.analytics.forecasting import get_forecaster

        fc = get_forecaster()
        fc_stats = fc.get_stats()
        checks["forecaster"] = {"ok": True, "stats": fc_stats}
    except Exception:
        checks["forecaster"] = {"ok": False, "status": "unavailable"}

    checks["privacy"] = {
        "ok": True,
        "mode": "redacted-public-projection",
        "credential_presence": "redacted",
        "local_inference": "not_proven_by_health",
    }

    try:
        from vetinari.workflow.wiring import get_dispatch_status

        dispatch_status = get_dispatch_status()
        checks["dispatch"] = {
            "ok": not dispatch_status.get("andon_paused", False),
            "status": "paused" if dispatch_status.get("andon_paused", False) else "available",
            "capacity": "redacted",
        }
    except Exception:
        checks["dispatch"] = {"ok": False, "status": "unavailable"}

    failed_count = sum(1 for c in checks.values() if isinstance(c, dict) and not c.get("ok", True))
    overall = "degraded" if failed_count else "healthy"

    return {
        "status": overall,
        "liveness": "live",
        "readiness": "ready" if overall == "healthy" else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_system_hardware_handlers() -> list[Any]:
    """Create Litestar handlers for health, shutdown, GPU, and resource routes.

    Returns an empty list when Litestar is not installed so the application
    starts cleanly in environments where the optional dependency is absent.

    Returns:
        List of Litestar route handler objects, or empty list when unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    @get("/api/v1/health", media_type=MediaType.JSON)
    async def api_health() -> dict[str, Any]:
        """Composite health check — factory floor sensor dashboard.

        Returns a unified view of all subsystem health status. Each subsystem
        check is wrapped in error handling so a single failing subsystem
        does not crash the entire health endpoint.

        Returns:
            JSON object with overall status, individual checks, and timestamp.
        """
        return _collect_composite_health()

    @get("/ready", media_type=MediaType.JSON)
    async def readiness() -> Response:
        """Readiness check that fails nonzero for release-critical degradation.

        Returns:
            JSON response with 200 for healthy readiness or 503 otherwise.
        """
        health = _collect_composite_health()
        try:
            from vetinari.web.litestar_app import is_shutting_down

            shutting_down = is_shutting_down()
        except Exception:
            shutting_down = False

        if shutting_down:
            health["status"] = "degraded"
            health["readiness"] = "shutting_down"
            health.setdefault("checks", {})["shutdown"] = {"ok": False, "status": "in_progress"}

        status_code = 200 if health.get("status") == "healthy" else 503
        return Response(content=health, status_code=status_code, media_type=MediaType.JSON)

    @post("/api/v1/server/shutdown", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_server_shutdown(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Gracefully shut down the Vetinari server.

        Requires admin privileges. Sends SIGTERM to the current process after
        a short delay to allow the response to flush.  This endpoint accepts
        no body parameters; any non-empty body is rejected with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON acknowledgement with status.
        """
        from vetinari.web.responses import litestar_error_response

        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        import signal
        import time

        def _delayed_shutdown() -> None:
            """Give the response time to flush, then terminate."""
            time.sleep(0.5)
            logger.info("Server shutdown requested via dashboard")
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_delayed_shutdown, daemon=True).start()
        return {"status": "shutting_down", "message": "Server is shutting down..."}

    @get("/api/v1/system/gpu", media_type=MediaType.JSON)
    async def api_system_gpu() -> dict[str, Any]:
        """Return GPU statistics via pynvml with graceful fallback.

        Initialises and shuts down NVML per-request so no persistent handle is
        held between calls.  Falls back gracefully when pynvml is not installed
        or when NVML initialisation fails (e.g. no NVIDIA driver present).

        Returns:
            JSON object with ``gpu_available`` boolean and a ``gpus`` list.
            Each GPU entry includes name, VRAM usage, utilisation percent, and
            core temperature.  On failure ``error`` describes the problem.
        """
        try:
            import pynvml  # type: ignore[import]
        except ImportError:
            logger.debug("pynvml not installed — GPU info unavailable; install with: pip install pynvml")  # noqa: VET301 — user guidance string
            return {"gpu_available": False, "gpus": [], "error": "pynvml not installed"}

        try:
            with _nvml_lock:
                return _gpu_info_locked(pynvml)
        except Exception as exc:
            logger.warning(
                "api_system_gpu: unexpected error reading GPU info — returning unavailable: %s",
                exc,
            )
            return {"gpu_available": False, "gpus": [], "error": str(exc)}

    @get("/api/v1/system/resources", media_type=MediaType.JSON)
    async def api_system_resources() -> dict[str, Any]:
        """Return system resource statistics via psutil.

        Reads CPU, RAM, and disk usage.  Disk usage is measured against the
        models directory from the ``MODELS_DIR`` environment variable, falling
        back to ``"."`` when absent.  All numeric values are rounded to one
        decimal place.  Returns 503 when psutil is not installed or fails.

        Returns:
            JSON object with ``ram``, ``cpu``, and ``disk`` sub-objects,
            or a 503 response when psutil is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            import psutil  # type: ignore[import]
        except ImportError:
            logger.warning(
                "psutil not installed — system resource metrics unavailable; install with: pip install psutil"  # noqa: VET301 — user guidance string
            )
            return litestar_error_response(  # type: ignore[return-value]
                "System resource metrics unavailable — psutil not installed", 503
            )

        try:
            ram = psutil.virtual_memory()
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            cpu_count = psutil.cpu_count(logical=True) or 0
        except Exception as exc:
            logger.warning("api_system_resources: could not read CPU/RAM stats — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "System resource metrics unavailable", 503
            )

        # Read models dir from environment (replaces flask.current_app.config["MODELS_DIR"])
        models_dir = os.environ.get("MODELS_DIR", ".")
        try:
            disk = psutil.disk_usage(models_dir)
            disk_used_gb = round(disk.used / (1024**3), 1)
            disk_total_gb = round(disk.total / (1024**3), 1)
            disk_pct = round(disk.percent, 1)
        except (FileNotFoundError, PermissionError) as exc:
            logger.warning("api_system_resources: disk_usage failed for %s: %s", models_dir, exc)
            disk_used_gb = 0.0
            disk_total_gb = 0.0
            disk_pct = 0.0

        return {
            "ram": {
                "used_mb": round(ram.used / (1024 * 1024), 1),
                "total_mb": round(ram.total / (1024 * 1024), 1),
                "percent": round(ram.percent, 1),
            },
            "cpu": {
                "percent": cpu_pct,
                "cores": cpu_count,
            },
            "disk": {
                "used_gb": disk_used_gb,
                "total_gb": disk_total_gb,
                "percent": disk_pct,
                "path": "redacted",
                "path_redacted": True,
            },
        }

    @get("/api/v1/system/vram", media_type=MediaType.JSON)
    async def api_system_vram() -> dict[str, Any]:
        """Return VRAM budget, thermal throttle status, and phase recommendation.

        Returns:
            JSON with current VRAM state including thermal throttle flag,
            phase recommendation, and reservation summary.

        Raises:
            ServiceUnavailableException: If VRAM manager is unavailable or cannot
                be initialized.
        """
        try:
            from vetinari.models.vram_manager import get_vram_manager

            mgr = get_vram_manager()
            return {
                "thermal_throttled": mgr.is_thermal_throttled(),
                "phase_recommendation": mgr.get_phase_recommendation(),
                "status": mgr.get_status() if hasattr(mgr, "get_status") else {},
            }
        except Exception as exc:
            logger.warning(
                "VRAM manager unavailable — returning 503 so callers can distinguish failure from success",
                exc_info=True,
            )
            from litestar.exceptions import ServiceUnavailableException

            raise ServiceUnavailableException("VRAM manager unavailable") from exc

    @post("/api/v1/system/vram/phase", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_set_vram_phase(data: dict[str, Any]) -> Response:
        """Set the current inference phase for VRAM budget management.

        Accepts JSON ``{"phase": "loading"|"inference"|"idle"}``.

        Args:
            data: JSON body containing the ``phase`` field.

        Returns:
            JSON acknowledgement with new phase recommendation, or HTTP 400
            when phase is missing, or HTTP 500 when the VRAM manager fails.
        """
        from vetinari.web.responses import litestar_error_response

        phase = data.get("phase")
        if not isinstance(phase, str) or phase == "":
            return litestar_error_response("phase is required", 400)

        _VALID_PHASES = {"loading", "inference", "idle"}
        if phase not in _VALID_PHASES:
            return litestar_error_response(
                f"'phase' must be one of: loading, inference, idle — got {phase!r}",
                422,
            )

        try:
            from vetinari.models.vram_manager import get_vram_manager

            mgr = get_vram_manager()
            mgr.set_phase(phase)
            return Response(
                content={
                    "status": "ok",
                    "phase": phase,
                    "recommendation": mgr.get_phase_recommendation(),
                },
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning("Failed to set VRAM phase to %s", phase, exc_info=True)
            return litestar_error_response("Failed to set VRAM phase", 500)

    return [
        api_health,
        readiness,
        api_server_shutdown,
        api_system_gpu,
        api_system_resources,
        api_system_vram,
        api_set_vram_phase,
    ]
