"""Status, configuration, and path-validation Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``system_status_api._register(bp)``.  Handles application status, token
statistics, global search, runtime configuration updates, filesystem path
validation, directory browsing, batch queue stats, and the upgrade-check
endpoint.

This is part of the Flask->Litestar migration (ADR-0066).  The URL paths
are identical to the Flask originals so existing clients require no changes.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post, put
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_system_status_handlers() -> list[Any]:
    """Create Litestar handlers for status, config, path, and search routes.

    Returns an empty list when Litestar is not installed so the application
    starts cleanly in environments where the optional dependency is absent.

    Returns:
        List of Litestar route handler objects, or empty list when unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    @get("/api/v1/status", media_type=MediaType.JSON)
    async def api_status() -> dict[str, Any]:
        """Return current application configuration and runtime status.

        Returns:
            JSON object containing host, config path, token presence, model
            lists, memory budget, and the active model identifier.
        """
        from vetinari.types import StatusEnum
        from vetinari.web.responses import success_response
        from vetinari.web.shared import current_config as _cfg

        _has_cloud_keys = bool(_cfg.api_token)
        _privacy_mode = "local-only" if not _has_cloud_keys else "hybrid"

        # Redact absolute paths — operator-facing status must never leak
        # filesystem layout. Clients that need the real paths should call
        # the admin config endpoint (requires admin guard).
        return success_response({
            "status": StatusEnum.RUNNING.value,
            "models_dir": "<configured>",
            "config_path": "<configured>",
            "api_token": "***" if _has_cloud_keys else "",
            "default_models": _cfg.default_models,
            "fallback_models": _cfg.fallback_models,
            "uncensored_fallback_models": _cfg.uncensored_fallback_models,
            "memory_budget_gb": _cfg.memory_budget_gb,
            "gpu_layers": _cfg.local_gpu_layers,
            "active_model_id": _cfg.active_model_id,
            "privacy_mode": _privacy_mode,
            "data_locality": "All data stays on this machine. No cloud APIs configured."
            if not _has_cloud_keys
            else "Cloud API key detected. Model responses may transit external servers.",
        })

    @get("/api/v1/token-stats", media_type=MediaType.JSON)
    async def api_token_stats() -> dict[str, Any]:
        """Return token usage statistics from the analytics system.

        Tracks which collectors were reachable so callers can distinguish
        "no usage recorded" from "collectors were unavailable."  When both
        collectors fail, the response includes ``"degraded": True`` and an
        ``"unavailable_sources"`` list so operators know the zeros are not
        reliable.

        Returns:
            JSON object with ``total_tokens_used``, ``total_cost_usd``,
            ``by_model``, ``by_provider``, and ``session_requests`` fields.
            Also includes ``"degraded": bool`` and ``"unavailable_sources": list``
            to signal partial or total collector failure.
        """
        stats: dict[str, Any] = {
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "by_model": {},
            "by_provider": {},
            "session_requests": 0,
        }
        unavailable_sources: list[str] = []

        try:
            from vetinari.telemetry import get_telemetry_collector

            tel = get_telemetry_collector()
            if hasattr(tel, "get_summary"):
                summary = tel.get_summary()
                if summary:
                    stats.update(summary)
        except Exception:
            logger.warning(
                "Telemetry collector unavailable — token usage counts may be zero",
                exc_info=True,
            )
            unavailable_sources.append("telemetry")

        try:
            from vetinari.analytics.cost import get_cost_tracker

            cost_summary = get_cost_tracker().get_summary()
            if cost_summary:
                stats["total_cost_usd"] = cost_summary.get("total_cost_usd", 0.0)
                stats["by_model"] = cost_summary.get("by_model", {})
        except Exception:
            logger.warning(
                "Cost tracker unavailable — cost totals may be zero",
                exc_info=True,
            )
            unavailable_sources.append("cost_tracker")

        # Signal degradation when no collector was reachable so callers can
        # distinguish genuine zero-usage from "data sources offline".
        stats["unavailable_sources"] = unavailable_sources
        stats["degraded"] = len(unavailable_sources) >= 2

        return stats

    @get("/api/v1/search", media_type=MediaType.JSON)
    async def api_global_search(
        q: str | None = Parameter(query="q", default=None),
    ) -> dict[str, Any]:
        """Search across projects, task outputs, and the memory system.

        Combines project-level search (names, descriptions, task outputs) with
        memory-system full-text search so that log entries, agent discoveries,
        and other persisted knowledge are also discoverable.

        Args:
            q: The search query string (URL query parameter ``q``).

        Returns:
            JSON object with ``results`` (list of up to 20 matching items),
            ``query`` (the normalised search term), and ``total`` (full match
            count before the 20-item cap).
        """
        from vetinari.web import shared

        query = (q or "").strip().lower()  # noqa: VET112 - empty fallback preserves optional request metadata contract
        if not query:
            return {"results": [], "query": ""}

        results: list[dict[str, Any]] = []
        # Degradation warnings: populated when a source fails so callers can
        # distinguish "no matches" from "search source was unavailable".
        warnings: list[str] = []
        _project_scan_failed = False

        projects_dir = shared.PROJECT_ROOT / "projects"
        if projects_dir.exists():
            for proj_dir in projects_dir.iterdir():
                if not proj_dir.is_dir():
                    continue
                config_path = proj_dir / "project.yaml"
                if not config_path.exists():
                    continue
                try:
                    with pathlib.Path(config_path).open(encoding="utf-8") as f:
                        config = yaml.safe_load(f) or {}
                    name = config.get("project_name", proj_dir.name)
                    desc = config.get("description", "")
                    if query in name.lower() or query in desc.lower():
                        results.append({
                            "type": "project",
                            "id": proj_dir.name,
                            "name": name,
                            "description": desc[:100],
                            "status": config.get("status", "unknown"),
                        })
                    outputs_glob = (
                        (proj_dir / "outputs").glob("*/output.txt") if (proj_dir / "outputs").exists() else []
                    )
                    for task_dir in outputs_glob:
                        try:
                            content = task_dir.read_text(encoding="utf-8", errors="ignore")
                            if query in content.lower():
                                results.append({
                                    "type": "output",
                                    "project_id": proj_dir.name,
                                    "task_id": task_dir.parent.name,
                                    "preview": content[:150],
                                })
                        except Exception:
                            logger.warning("Failed to read output file during search", exc_info=True)
                except Exception:
                    logger.warning(
                        "Failed to scan project directory %s — project will be absent from results",
                        proj_dir.name,
                        exc_info=True,
                    )
                    _project_scan_failed = True

        if _project_scan_failed:
            warnings.append("some_project_scans_failed")

        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            memory_limit = max(1, 20 - len(results))
            memories = store.search(query, limit=memory_limit)
            results.extend(
                {
                    "type": "memory",
                    "id": mem.id,
                    "agent": mem.agent,
                    "entry_type": mem.entry_type.value if hasattr(mem.entry_type, "value") else str(mem.entry_type),
                    "preview": (mem.summary or mem.content)[:150],
                }
                for mem in memories
            )
        except Exception:
            logger.warning(
                "Memory store unavailable during global search — memory results omitted",
                exc_info=True,
            )
            warnings.append("memory_search_unavailable")

        return {"results": results[:20], "query": query, "total": len(results), "warnings": warnings}

    async def _apply_config_update(data: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial config update and invalidate the orchestrator cache.

        Args:
            data: Partial configuration fields to update.

        Returns:
            JSON dict with ``status: updated`` and the full updated config.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import current_config

        if not data:
            return litestar_error_response(  # type: ignore[return-value]
                "Request body must not be empty — provide at least one configuration key to update",
                422,
            )

        _CONFIG_KEYS = frozenset({
            "host",
            "config_path",
            "api_token",
            "models_dir",
            "hf_token",
            "memory_budget_gb",
            "gpu_layers",
        })
        if not _CONFIG_KEYS.intersection(data):
            return litestar_error_response(  # type: ignore[return-value]
                "Request body contains no recognised configuration keys",
                422,
            )

        if "host" in data:
            current_config.host = data["host"]
        if "config_path" in data:
            current_config.config_path = data["config_path"]
        if "api_token" in data:
            current_config.api_token = data["api_token"]
        if "models_dir" in data:
            current_config.models_dir = data["models_dir"]
        if "hf_token" in data:
            current_config.hf_token = data["hf_token"]
        if "memory_budget_gb" in data:
            current_config.memory_budget_gb = int(data["memory_budget_gb"])
        if "gpu_layers" in data:
            current_config.local_gpu_layers = int(data["gpu_layers"])

        shared.orchestrator = None
        shared._models_cache_ts = 0.0

        return {"status": "updated", "config": current_config.to_dict()}

    @post("/api/v1/config", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_config_post(data: dict[str, Any]) -> dict[str, Any]:
        """Update runtime configuration via POST.

        Accepts a JSON body with any subset of ``host``, ``config_path``,
        ``api_token``, ``models_dir``, ``hf_token``, ``memory_budget_gb``, and
        ``gpu_layers``.  After applying, the cached orchestrator is discarded and
        model discovery is invalidated.

        Args:
            data: Partial configuration update dict.

        Returns:
            JSON with ``status: updated`` and the full updated config dict.
        """
        return await _apply_config_update(data)

    @put("/api/v1/config", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_config_put(data: dict[str, Any]) -> dict[str, Any]:
        """Update runtime configuration via PUT.

        Identical to the POST variant — both methods are accepted for
        backward compatibility with the Flask endpoint that registered
        ``methods=["POST", "PUT"]``.

        Args:
            data: Partial configuration update dict.

        Returns:
            JSON with ``status: updated`` and the full updated config dict.
        """
        return await _apply_config_update(data)

    @post("/api/v1/validate-path", media_type=MediaType.JSON)
    async def api_validate_path(data: dict[str, Any]) -> Response:
        """Validate a filesystem path for model storage.

        Checks whether the path exists, is a directory, is readable, and
        counts ``.gguf`` model files inside it.  Used by the setup wizard to
        give real-time feedback when the user enters or pastes a model
        directory path.

        Args:
            data: JSON body containing the ``path`` field to validate.

        Returns:
            JSON with ``valid`` boolean, ``exists``, ``is_directory``,
            ``readable``, ``gguf_count``, and a human-readable ``message``.
            HTTP 400 when ``path`` is missing from the request body.
        """
        from vetinari.web.responses import litestar_error_response

        path_str = (data.get("path") or "").strip()
        if not path_str:
            return litestar_error_response("path is required", 400)

        p = pathlib.Path(path_str)
        exists = p.exists()
        is_dir = p.is_dir() if exists else False
        readable = False
        gguf_count = 0

        if is_dir:
            try:
                list(p.iterdir())
                readable = True
                for f in p.glob("*.gguf"):
                    if f.is_file():
                        gguf_count += 1
                for f in p.glob("*/*.gguf"):
                    if f.is_file():
                        gguf_count += 1
            except PermissionError:
                readable = False

        if not exists:
            message = "Directory not found"
        elif not is_dir:
            message = "Path exists but is not a directory"
        elif not readable:
            message = "Directory exists but is not readable"
        elif gguf_count == 0:
            message = "Directory exists but contains no .gguf model files"
        else:
            message = f"Found {gguf_count} model file{'s' if gguf_count != 1 else ''}"

        valid = exists and is_dir and readable and gguf_count > 0

        return Response(
            content={
                "valid": valid,
                "exists": exists,
                "is_directory": is_dir,
                "readable": readable,
                "gguf_count": gguf_count,
                "message": message,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/v1/browse-directory", media_type=MediaType.JSON)
    async def api_browse_directory() -> dict[str, Any]:
        """Suggest default model directory paths for the current OS.

        Returns sensible default directories where models are commonly stored,
        along with which ones actually exist on this system.

        Returns:
            JSON with ``suggestions`` list and ``default`` recommended path.
        """
        import contextlib
        import os
        import platform

        suggestions = []
        system = platform.system()

        if system == "Windows":
            candidates = [
                pathlib.Path(os.environ.get("USERPROFILE", "C:/Users")) / "models",
                pathlib.Path("C:/models"),
                pathlib.Path(os.environ.get("USERPROFILE", "C:/Users")) / ".cache" / "huggingface",
                pathlib.Path(os.environ.get("USERPROFILE", "C:/Users")) / "Documents" / "models",
            ]
        elif system == "Darwin":
            home = pathlib.Path.home()
            candidates = [
                home / "models",
                home / ".cache" / "huggingface",
                home / "Documents" / "models",
                pathlib.Path("/opt/models"),
            ]
        else:
            home = pathlib.Path.home()
            candidates = [
                home / "models",
                home / ".cache" / "huggingface",
                home / ".local" / "share" / "models",
                pathlib.Path("/opt/models"),
            ]

        for c in candidates:
            gguf_count = 0
            exists = c.exists()
            if exists and c.is_dir():
                with contextlib.suppress(PermissionError):
                    gguf_count = sum(1 for f in c.rglob("*.gguf") if f.is_file())
            suggestions.append({"path": str(c), "exists": exists, "gguf_count": gguf_count})

        default = str(candidates[0])
        for s in suggestions:
            if s["gguf_count"] > 0:
                default = s["path"]
                break
        else:
            for s in suggestions:
                if s["exists"]:
                    default = s["path"]
                    break

        return {"suggestions": suggestions, "default": default}

    @get("/api/v1/batch/queue-stats", media_type=MediaType.JSON)
    async def api_batch_queue_stats() -> dict[str, Any]:
        """Return current batch processor queue depths and flush configuration.

        Lets operators monitor how many telemetry events are queued per backend,
        whether the flush thread is alive, and whether batching is enabled.

        Returns:
            JSON object with per-provider queue depths, total queued count,
            flush interval, max batch size, and flush thread status.
        """
        from vetinari.adapters.batch_processor import get_batch_processor
        from vetinari.web.responses import litestar_error_response, success_response

        try:
            stats = get_batch_processor().get_queue_stats()
        except Exception:
            logger.warning(
                "Batch processor unavailable — cannot retrieve queue statistics",
                exc_info=True,
            )
            return litestar_error_response("Batch processor unavailable — cannot retrieve queue statistics", 503)
        return success_response(stats)

    @get("/api/v1/upgrade-check", media_type=MediaType.JSON)
    async def api_upgrade_check() -> dict[str, Any]:
        """Query the orchestrator for available upgrades.

        Returns:
            JSON object with an ``upgrades`` key containing the list of available
            upgrade descriptors.  Returns an empty list when upgrading is unavailable.
            Returns 503 when the orchestrator or upgrader subsystem is unreachable.
        """
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import get_orchestrator

        try:
            orb = get_orchestrator()
            if not hasattr(orb, "upgrader") or orb.upgrader is None:
                return {"upgrades": [], "message": "Upgrade checking not available"}
            upgrades = orb.upgrader.check_for_upgrades()
        except Exception:
            logger.warning(
                "Upgrade check unavailable — orchestrator or upgrader subsystem not responding",
                exc_info=True,
            )
            return litestar_error_response(
                "Upgrade check unavailable — orchestrator or upgrader subsystem not responding",
                503,
            )
        return {"upgrades": upgrades}

    return [
        api_status,
        api_token_stats,
        api_global_search,
        api_config_post,
        api_config_put,
        api_validate_path,
        api_browse_directory,
        api_batch_queue_stats,
        api_upgrade_check,
    ]
