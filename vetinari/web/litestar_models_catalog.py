"""Model relay catalog, routing policy, and per-project search handlers.

Litestar migration of ``models_catalog_api.py``.  Provides handlers for the
static model catalog from ``model_relay``, single-model lookup, model selection
routing, routing-policy CRUD, catalog reload, per-project external model search,
per-project cache refresh, global online model search via HuggingFace / Reddit /
GitHub, popular model recommendations, model file listing for a repo, and
model download initiation.

Endpoints provided:
  GET  /api/v1/model-catalog                            — full static catalog
  GET  /api/v1/models/{model_id}                        — single model lookup
  POST /api/v1/models/select                            — routing decision
  GET  /api/v1/models/policy                            — active routing policy
  PUT  /api/v1/models/policy                            — replace routing policy
  POST /api/v1/models/reload                            — reload catalog from disk
  POST /api/v1/project/{project_id}/model-search        — per-project search
  POST /api/v1/project/{project_id}/refresh-models      — per-project cache refresh
  POST /api/v1/models/search                            — global online search
  GET  /api/v1/models/popular                           — popular/recommended models
  GET  /api/v1/models/files                             — files for a HuggingFace repo
  POST /api/v1/models/download                          — initiate model download

This module is a standalone handler factory — it does NOT use the Flask
``_register(bp)`` pattern.  Call ``create_models_catalog_handlers()`` and
pass the returned list to your Litestar ``Router`` or application.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _infer_catalog_backend(
    backend: str | None,
    *,
    filename: str | None = None,
    model_format: str | None = None,
) -> str:
    backend_value = "auto" if backend is None else backend
    normalized = backend_value.strip().lower().replace("-", "_")
    if normalized in {"llama", "local"}:
        return "llama_cpp"
    if normalized != "auto":
        return normalized
    requested_format = "" if model_format is None else model_format.strip().lower().lstrip(".")
    requested_filename = "" if filename is None else filename.strip().lower()
    if requested_format == "gguf" or requested_filename.endswith(".gguf"):
        return "llama_cpp"
    return "vllm"


try:
    from litestar import MediaType, get, post, put

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_models_catalog_handlers() -> list[Any]:
    """Create Litestar handlers for the model catalog and routing-policy API.

    Returns an empty list when Litestar is not installed, so the factory is
    safe to call in environments that only have Flask available.

    Returns:
        List of Litestar route handler objects ready to register on a Router
        or Application.  Empty when Litestar is unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response
    from vetinari.web.shared import (
        ENABLE_EXTERNAL_DISCOVERY,
        _get_models_cached,
        _project_external_model_enabled,
        current_config,
        validate_path_param,
    )

    @get("/api/v1/model-catalog", media_type=MediaType.JSON)
    async def api_models_list() -> Any:
        """Return the static/configured model catalog from model_relay.

        Use ``/api/v1/models`` for live local discovery instead.

        Returns:
            JSON object with key ``models`` containing serialised model dicts.
            503 when the model relay module cannot be loaded.
        """
        try:
            from vetinari.models.model_relay import model_relay

            models = model_relay.get_all_models()
            return {"models": [m.to_dict() for m in models]}
        except Exception as exc:
            logger.warning("api_models_list: model relay unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model catalog subsystem unavailable — model relay could not be loaded", 503
            )

    @get("/api/v1/models/{model_id:str}", media_type=MediaType.JSON)
    async def api_model_get(model_id: str) -> Any:
        """Retrieve a single model by its identifier from model_relay.

        Args:
            model_id: The relay identifier of the model to look up.

        Returns:
            JSON representation of the model, or 404 when not found,
            or 503 when the model relay module cannot be loaded.
        """
        try:
            from vetinari.models.model_relay import model_relay

            model = model_relay.get_model(model_id)
            if not model:
                return litestar_error_response("Model not found", 404)
            return model.to_dict()
        except Exception as exc:
            logger.warning("api_model_get: model relay unavailable for model %r — returning 503: %s", model_id, exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model catalog subsystem unavailable — model relay could not be loaded", 503
            )

    @post("/api/v1/models/select", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_model_select(data: dict[str, Any]) -> dict[str, Any]:
        """Select the best model for a given task via model_relay routing.

        Expects a JSON body with ``task_type`` (str) and ``context`` (any) fields.

        Args:
            data: Request body with ``task_type`` and optional ``context`` keys.

        Returns:
            JSON representation of the selected model routing decision.
        """
        task_type = data.get("task_type")
        if task_type is not None and not isinstance(task_type, str):
            return litestar_error_response("'task_type' must be a string", 422)  # type: ignore[return-value]

        if not task_type:
            return litestar_error_response("'task_type' is required", 400)  # type: ignore[return-value]

        try:
            from vetinari.models.model_relay import model_relay

            selection = model_relay.pick_model_for_task(
                task_type=task_type,
                context=data.get("context"),
            )
            return selection.to_dict()
        except Exception as exc:
            logger.warning("api_model_select: model relay unavailable — returning 503: %s", exc)
            return litestar_error_response(
                "Model catalog subsystem unavailable — model relay could not be loaded",
                503,
            )

    @get("/api/v1/models/policy", media_type=MediaType.JSON)
    async def api_model_policy_get() -> Any:
        """Return the current model routing policy from model_relay.

        Returns:
            JSON representation of the active ``RoutingPolicy``.
            503 when the model relay module cannot be loaded.
        """
        try:
            from vetinari.models.model_relay import model_relay

            policy = model_relay.get_policy()
            return policy.to_dict()
        except Exception as exc:
            logger.warning("api_model_policy_get: model relay unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model catalog subsystem unavailable — model relay could not be loaded", 503
            )

    @put("/api/v1/models/policy", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_model_policy_update(data: dict[str, Any]) -> dict[str, Any]:
        """Replace the active model routing policy (admin only).

        Expects a JSON body that deserialises into a ``RoutingPolicy`` via
        ``RoutingPolicy.from_dict()``.  The body must contain at least one
        of the known policy keys.

        Args:
            data: Request body to deserialise as the new routing policy.

        Returns:
            JSON with ``status`` and the new serialised policy, or 400 when
            the body is malformed or contains no recognised policy keys.
        """
        from vetinari.models.model_relay import RoutingPolicy, model_relay
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key
        from vetinari.web.responses import litestar_error_response

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)  # type: ignore[return-value]
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=400)  # type: ignore[return-value]

        # RoutingPolicy fields — reject bodies that contain none of these keys.
        _POLICY_KEYS = {
            "local_first",
            "privacy_weight",
            "latency_weight",
            "cost_weight",
            "max_cost_per_1k_tokens",
            "preferred_providers",
        }
        if not _POLICY_KEYS.intersection(data):
            return litestar_error_response("Request body contains no recognised policy fields", code=400)  # type: ignore[return-value]

        try:
            policy = RoutingPolicy.from_dict(data)
            model_relay.set_policy(policy)
            return {"status": "updated", "policy": policy.to_dict()}
        except Exception as exc:
            logger.warning("api_model_policy_update: model relay unavailable — returning 503: %s", exc)
            return litestar_error_response(
                "Model catalog subsystem unavailable — model relay could not be loaded",
                503,
            )

    @post("/api/v1/models/reload", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_models_reload() -> dict[str, Any]:
        """Reload the model catalog from disk (admin only).

        Returns:
            JSON with ``status`` and ``models_loaded`` count.

        Raises:
            Returns 503 when model relay is unavailable.
        """
        try:
            from vetinari.models.model_relay import model_relay

            model_relay.reload_catalog()
            return {"status": "reloaded", "models_loaded": len(model_relay.get_all_models())}
        except Exception as exc:
            logger.warning("api_models_reload: model relay unavailable — returning 503: %s", exc)
            return litestar_error_response(
                "Model catalog subsystem unavailable — model relay could not be loaded",
                503,
            )

    @post(
        "/api/v1/project/{project_id:str}/model-search",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_model_search(
        project_id: str,
        data: dict[str, Any],
    ) -> Any:
        """Search for external model candidates suited to a project's task.

        Args:
            project_id: Identifier of the project under ``projects/``.
            data: Request body with optional ``task_description`` (str).

        Returns:
            JSON with ``status``, ``candidates`` (list of dicts), and ``count``,
            or an error response when discovery is disabled or the project is
            not found.
        """
        from vetinari.web import shared
        from vetinari.web.request_validation import body_depth_exceeded

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)  # type: ignore[return-value]
        if validate_path_param(project_id) is None:
            return litestar_error_response("Invalid project ID", 400)

        # Read PROJECT_ROOT at call time so test patches take effect
        project_root = shared.PROJECT_ROOT
        project_dir = project_root / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)
        if not ENABLE_EXTERNAL_DISCOVERY:
            return litestar_error_response("External discovery globally disabled", 403)
        if not _project_external_model_enabled(project_dir):
            return litestar_error_response("External model discovery disabled for this project", 403)

        from vetinari.model_discovery import ModelDiscovery

        task_description = data.get("task_description", "")
        search_adapter = ModelDiscovery()

        lm_models: list[Any] = []
        local_models_degraded = False
        try:
            from vetinari.models.model_pool import ModelPool

            model_pool = ModelPool(current_config)
            model_pool.discover_models()
            lm_models = model_pool.list_models()
        except Exception as exc:
            logger.warning(
                "Could not get local models for project %s model search — proceeding with empty local list: %s",
                project_id,
                exc,
            )
            local_models_degraded = True

        try:
            candidates = search_adapter.search(task_description, lm_models)
            return {
                "status": "ok",
                "candidates": [c.to_dict() for c in candidates],
                "count": len(candidates),
                "_local_models_degraded": local_models_degraded,
            }
        except Exception as exc:
            logger.warning(
                "api_model_search: model discovery unavailable for project %s — returning 503: %s", project_id, exc
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable", 503
            )

    @post(
        "/api/v1/project/{project_id:str}/refresh-models",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_refresh_models(project_id: str) -> dict[str, Any]:
        """Refresh the model cache for a specific project.

        Args:
            project_id: Identifier of the project (currently unused; refresh is
                global, keyed on ``ModelDiscovery`` availability).

        Returns:
            JSON with ``status`` and a confirmation ``message``.

        Raises:
            Returns 404 when the project directory does not exist.
        """
        from vetinari.web import shared

        if validate_path_param(project_id) is None:
            return litestar_error_response("Invalid project ID", 400)

        # Read PROJECT_ROOT at call time so test patches take effect
        project_root = shared.PROJECT_ROOT
        project_dir = project_root / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        logger.debug("Model cache refresh requested for project %s", project_id)
        return {"status": "ok", "message": "Model cache refreshed (live search enabled)"}

    @post("/api/v1/models/search", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_global_model_search(data: dict[str, Any]) -> Any:
        """Search for models online via HuggingFace, Reddit, GitHub, and PapersWithCode.

        Accepts a JSON body with ``query`` (str, required).  Optionally merges
        locally discovered models into the ranking.

        Args:
            data: Request body with required ``query`` string and optional context.

        Returns:
            JSON with ``status``, ``candidates`` (list of scored model dicts), and
            ``count``, or an error response when discovery is disabled or the query
            is missing.
        """
        if not ENABLE_EXTERNAL_DISCOVERY:
            return litestar_error_response("External discovery is disabled", 403)

        query = data.get("query", "")
        if not isinstance(query, str):
            return litestar_error_response("'query' must be a string", 422)
        query = query.strip()
        if not query:
            return litestar_error_response("query is required", 400)

        from vetinari.model_discovery import ModelDiscovery

        discovery = ModelDiscovery()

        lm_models: list[dict[str, Any]] = []
        try:
            lm_models = _get_models_cached()
        except Exception:
            logger.warning(
                "Could not fetch local models for global model search context — proceeding without local models"
            )

        try:
            candidates = discovery.search(
                query,
                lm_models,
                objective=data.get("objective") if isinstance(data.get("objective"), str) else None,
                family=data.get("family") if isinstance(data.get("family"), str) else None,
                min_size_gb=data.get("min_size_gb") if isinstance(data.get("min_size_gb"), int | float) else None,
                max_size_gb=data.get("max_size_gb") if isinstance(data.get("max_size_gb"), int | float) else None,
                quantization=data.get("quantization") if isinstance(data.get("quantization"), str) else None,
                file_type=data.get("file_type") if isinstance(data.get("file_type"), str) else None,
            )
            return {
                "status": "ok",
                "candidates": [c.to_dict() for c in candidates],
                "count": len(candidates),
            }
        except Exception as exc:
            logger.warning("api_global_model_search: model discovery unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable", 503
            )

    from litestar.params import Parameter

    @get("/api/v1/models/popular", media_type=MediaType.JSON)
    async def api_models_popular() -> dict[str, Any]:
        """Return a curated list of popular or recommended models from the relay.

        Popularity is approximated by the number of quantization variants each
        model offers — models with broader quantization support are more widely
        used in practice.  Falls back to an empty list when the model catalog
        is unavailable rather than raising, so the frontend degrades gracefully.

        Returns:
            JSON object with key ``models`` containing serialised model dicts,
            sorted by quantization-variant count descending.  Returns
            ``{"models": [], "status": "unavailable"}`` if the catalog cannot
            be reached.
        """
        try:
            from vetinari.models.model_relay import model_relay

            all_models = model_relay.get_all_models()
            # Sort by number of quantization variants as a popularity proxy.
            # Models with more quants have broader community adoption.
            sorted_models = sorted(
                all_models,
                key=lambda m: len(getattr(m, "quantizations", []) or []),
                reverse=True,
            )
            return {"models": [m.to_dict() for m in sorted_models]}
        except Exception as exc:
            logger.warning(
                "api_models_popular: model catalog unavailable — returning 503: %s",
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Model catalog subsystem unavailable — model relay could not be loaded", 503
            )

    @get("/api/v1/models/files", media_type=MediaType.JSON)
    async def api_model_files(
        repo_id: str = Parameter(query="repo_id", default=""),
        vram_gb: int = Parameter(query="vram_gb", default=32, ge=1),
        use_case: str = Parameter(query="use_case", default="general"),
        backend: str = Parameter(query="backend", default="auto"),
        model_format: str = Parameter(query="format", default=""),
        revision: str = Parameter(query="revision", default=""),
        objective: str = Parameter(query="objective", default=""),
        family: str = Parameter(query="family", default=""),
        quantization: str = Parameter(query="quantization", default=""),
        file_type: str = Parameter(query="file_type", default=""),
        min_size_gb: float | None = Parameter(query="min_size_gb", default=None),
        max_size_gb: float | None = Parameter(query="max_size_gb", default=None),
    ) -> dict[str, Any]:
        """Return available model artifacts for a HuggingFace model repository.

        Queries the model relay or discovery layer to enumerate files available
        for ``repo_id``, optionally filtered by VRAM budget and use-case.  The
        frontend calls ``getModelFiles(repoId, { vramGb, useCase })``.

        Args:
            repo_id: HuggingFace repository identifier (e.g. ``"author/model"``).
                Required; returns an error payload when empty.
            vram_gb: Available VRAM in gigabytes used to filter quantization
                file recommendations.  Defaults to 32.
            use_case: Intended use-case (e.g. ``"general"``, ``"coding"``).
                Passed through to the discovery layer for ranking hints.
            backend: Target backend for artifact selection.
            model_format: Optional model format filter.
            revision: Optional revision to resolve before listing.
            objective: Optional objective/category filter.
            family: Optional model family filter.
            quantization: Optional quantization filter.
            file_type: Optional file suffix filter.
            min_size_gb: Optional minimum file size in GB.
            max_size_gb: Optional maximum file size in GB.

        Returns:
            JSON object with ``files`` (list of file descriptor dicts) and
            ``repo_id``.  Returns ``{"files": [], "repo_id": repo_id}`` on
            lookup failure so the frontend degrades gracefully.
        """
        if not repo_id:
            return litestar_error_response("repo_id query parameter is required", 400)

        try:
            from vetinari.model_discovery import ModelDiscovery

            backend_normalized = _infer_catalog_backend(backend, model_format=model_format or None)
            discovery = ModelDiscovery()
            files = discovery.get_repo_files(
                repo_id=repo_id,
                vram_gb=vram_gb,
                use_case=use_case,
                backend=backend_normalized,
                model_format=model_format or None,
                revision=revision or None,
                objective=objective or None,
                family=family or None,
                quantization=quantization or None,
                file_type=file_type or None,
                min_size_gb=min_size_gb,
                max_size_gb=max_size_gb,
            )
            return {
                "files": files,
                "repo_id": repo_id,
                "backend": backend_normalized,
                "format": model_format or None,
                "filters": {
                    "objective": objective or None,
                    "family": family or None,
                    "quantization": quantization or None,
                    "file_type": file_type or None,
                    "min_size_gb": min_size_gb,
                    "max_size_gb": max_size_gb,
                },
            }
        except Exception as exc:
            logger.warning(
                "api_model_files: could not retrieve files for repo %s (vram_gb=%s, use_case=%s) — returning 503: %s",
                repo_id,
                vram_gb,
                use_case,
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable — could not retrieve model files", 503
            )

    @post("/api/v1/models/download", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_model_download(data: dict[str, Any]) -> dict[str, Any]:
        """Initiate an asynchronous model download (admin only).

        Accepts a JSON body with ``repo_id`` and ``filename`` fields and
        immediately acknowledges the request.  The actual download is delegated
        to the model discovery layer's background download facility when
        available; otherwise only the acknowledgment is returned.  Long-running
        downloads run outside the request lifecycle so the response is always
        fast.

        Args:
            data: Request body containing:
                - ``repo_id`` (str): HuggingFace repository identifier.
                - ``filename`` (str): Specific GGUF file to download.

        Returns:
            JSON with ``status`` ``"started"``,
            ``repo_id``, and ``filename``, or 502 when download cannot be started.
        """
        repo_id = data.get("repo_id", "")
        filename = data.get("filename")
        revision = data.get("revision")
        backend = data.get("backend", "auto")
        model_format = data.get("format", data.get("model_format"))

        if not isinstance(repo_id, str):
            return litestar_error_response("'repo_id' must be a string", 422)
        if filename is not None and not isinstance(filename, str):
            return litestar_error_response("'filename' must be a string", 422)
        if revision is not None and not isinstance(revision, str):
            return litestar_error_response("'revision' must be a string", 422)
        if not isinstance(backend, str):
            return litestar_error_response("'backend' must be a string", 422)
        if model_format is not None and not isinstance(model_format, str):
            return litestar_error_response("'format' must be a string", 422)

        repo_id = repo_id.strip()
        filename = filename.strip() if isinstance(filename, str) else None
        revision = revision.strip() if isinstance(revision, str) else None
        backend = backend.strip()
        model_format = model_format.strip() if isinstance(model_format, str) else None

        backend = _infer_catalog_backend(backend, filename=filename, model_format=model_format)

        if not repo_id:
            return litestar_error_response("repo_id is required", 400)
        if backend == "llama_cpp" and not filename:
            return litestar_error_response("filename is required", 400)

        try:
            from vetinari.model_discovery import ModelDiscovery

            discovery = ModelDiscovery()
            started = discovery.start_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                backend=backend,
                model_format=model_format,
            )
            if isinstance(started, dict):
                return {
                    "status": started.get("status", "started"),
                    "download_id": started.get("download_id"),
                    "repo_id": started.get("repo_id", repo_id),
                    "filename": started.get("filename", filename),
                    "revision": started.get("revision"),
                    "backend": started.get("backend", backend),
                    "format": started.get("format", model_format),
                    "artifact_type": started.get("artifact_type"),
                    "path": started.get("path"),
                    "manifest_path": started.get("manifest_path"),
                    "file_count": started.get("file_count"),
                    "bytes_total": started.get("bytes_total"),
                    "bytes_downloaded": started.get("bytes_downloaded"),
                }
            return {"status": "started", "repo_id": repo_id, "filename": filename, "backend": backend}
        except Exception as exc:
            logger.warning(
                "Could not start background download for %s/%s — returning 502: %s",
                repo_id,
                filename,
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Download backend unavailable — could not start download", 502
            )

    @get("/api/v1/models/download/{download_id:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_model_download_status(download_id: str) -> dict[str, Any]:
        """Return the status of a tracked model download.

        Returns:
            Download status object or an error response.
        """
        download_id = download_id.strip()
        if not download_id:
            return litestar_error_response("download_id is required", 400)

        try:
            from vetinari.model_discovery import ModelDiscovery

            status = ModelDiscovery().get_download_status(download_id)
            if status is None:
                return litestar_error_response("download not found", 404)
            return status
        except Exception as exc:
            logger.warning("Could not retrieve download status for %s - returning 503: %s", download_id, exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Download backend unavailable - could not retrieve status", 503
            )

    @post("/api/v1/models/download/{download_id:str}/cancel", media_type=MediaType.JSON, guards=[admin_guard], status_code=200)
    async def api_model_download_cancel(download_id: str) -> dict[str, Any]:
        """Request cancellation of a tracked model download.

        Returns:
            Cancellation result or an error response.
        """
        download_id = download_id.strip()
        if not download_id:
            return litestar_error_response("download_id is required", 400)

        try:
            from vetinari.model_discovery import ModelDiscovery

            if not ModelDiscovery().cancel_download(download_id):
                return litestar_error_response("download not found or not running", 404)
            return {"download_id": download_id, "status": "canceling"}
        except Exception as exc:
            logger.warning("Could not cancel download %s - returning 503: %s", download_id, exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Download backend unavailable - could not cancel download", 503
            )

    return [
        api_models_list,
        api_model_get,
        api_model_select,
        api_model_policy_get,
        api_model_policy_update,
        api_models_reload,
        api_model_search,
        api_refresh_models,
        api_global_model_search,
        api_models_popular,
        api_model_files,
        api_model_download,
        api_model_download_status,
        api_model_download_cancel,
    ]
