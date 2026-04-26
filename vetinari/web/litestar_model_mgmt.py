"""Model management API — native Litestar handlers for model lifecycle operations.

Provides REST routes for task-to-model assignment, catalog-wide model
enumeration, model registration and removal, speculative-decoding draft-pair
statistics, chat streaming via Server-Sent Events, VRAM phase and thermal
management, and cascade-router construction.

These handlers are registered by ``vetinari.web.litestar_app.create_app()``
and mirror the routes previously served by the Flask ``model_mgmt_api``
blueprint.  All endpoints are versioned under ``/api/v1/``.

This is part of the web layer: Intake -> **API** -> Orchestration -> Execution.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

# Optional Litestar imports — graceful degradation when not installed
try:
    from litestar import MediaType, Response, get, post
    from litestar.response import Stream

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_model_mgmt_handlers() -> list[Any]:
    """Create all Litestar route handlers for the model management API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.  Covers task assignment,
    model catalog CRUD, draft-pair statistics, SSE chat streaming, VRAM
    management, and cascade-router operations.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — model management API handlers not registered")
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    # -- POST /api/v1/models/assign-tasks -------------------------------------

    @post("/api/v1/models/assign-tasks", media_type=MediaType.JSON, guards=[admin_guard])
    async def assign_tasks_to_models(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Assign tasks to optimal models based on capability scoring.

        Accepts a JSON body with a ``tasks`` list.  Each task object is
        augmented with an ``assigned_model_id`` field and returned.

        Args:
            data: JSON request body containing a ``tasks`` list.

        Returns:
            JSON with the ``tasks`` list after assignment.
        """
        from vetinari.web.responses import success_response

        body = data if data is not None else {}
        tasks = body.get("tasks")
        if not isinstance(tasks, list):
            return litestar_error_response("'tasks' must be a list", code=400)  # type: ignore[return-value]

        try:
            from vetinari.models.model_pool import ModelPool
            from vetinari.web.shared import current_config

            pool = ModelPool(current_config.to_dict())
            pool.assign_tasks_to_models({"tasks": tasks})
        except Exception:
            logger.warning("Model pool unavailable — cannot assign tasks to models, returning 503")
            return litestar_error_response("Model pool subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response({"tasks": tasks})

    # -- GET /api/v1/models/all-available -------------------------------------

    @get("/api/v1/models/all-available", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_all_available_models() -> dict[str, Any]:
        """Return every model available to the system (local and cloud providers).

        Merges locally discovered GGUF files with configured cloud providers
        whose API tokens are present in the environment.

        Returns:
            JSON with ``models`` list containing model metadata dicts and a
            ``count`` field with the total number of available models.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.models.model_pool import ModelPool
            from vetinari.web.shared import current_config

            pool = ModelPool(current_config.to_dict())
            models = pool.get_all_available_models()
        except Exception:
            logger.warning("Model pool unavailable — cannot enumerate available models, returning 503")
            return litestar_error_response("Model pool subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response({"models": models, "count": len(models)})

    # -- POST /api/v1/models --------------------------------------------------

    @post("/api/v1/models", media_type=MediaType.JSON, guards=[admin_guard])
    async def add_model(data: dict[str, Any] | None = None) -> Any:
        """Register a new model in the relay catalog and persist the change.

        Accepts a JSON body with model fields (``model_id`` required).  On
        success the model is immediately available for task routing.

        Args:
            data: JSON request body with model registration fields.

        Returns:
            201 Created with the registered model dict, or 400 on missing
            ``model_id``.
        """
        from vetinari.web.responses import success_response

        body = data if data is not None else {}
        model_id_val = body.get("model_id")
        if not isinstance(model_id_val, str) or not model_id_val:
            return litestar_error_response("'model_id' must be a non-empty string", code=400)  # type: ignore[return-value]

        from vetinari.models.model_relay import ModelEntry, get_model_relay

        model = ModelEntry.from_dict(body)
        relay = get_model_relay()
        relay.add_model(model)
        logger.info("Model '%s' added to relay catalog via API", model.model_id)
        return Response(
            content=success_response(model.to_dict()),
            status_code=201,
            media_type=MediaType.JSON,
        )

    # -- DELETE /api/v1/models/{model_id} -------------------------------------

    @post("/api/v1/models/{model_id:str}/delete", media_type=MediaType.JSON, guards=[admin_guard])
    async def remove_model(model_id: str) -> dict[str, Any]:
        """Remove a model from the relay catalog and persist the change.

        Uses a POST-to-delete pattern because Litestar DELETE handlers that
        return JSON work best as POST with a ``/delete`` suffix in some
        proxy configurations.  The URL path keeps the same model_id segment.

        Args:
            model_id: URL path parameter identifying the model to remove.

        Returns:
            JSON with ``removed`` field on success, or 404 if the model
            was not found in the relay catalog.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.models.model_relay import get_model_relay

            relay = get_model_relay()
            if model_id not in relay.models:
                return litestar_error_response(f"Model '{model_id}' not found", code=404)  # type: ignore[return-value]
            relay.remove_model(model_id)
        except Exception:
            logger.warning("Model relay unavailable — cannot remove model '%s', returning 503", model_id)
            return litestar_error_response("Model relay subsystem unavailable", code=503)  # type: ignore[return-value]
        logger.info("Model '%s' removed from relay catalog via API", model_id)
        return success_response({"removed": model_id})

    # -- GET /api/v1/models/draft-pairs/{main}/{draft}/stats ------------------

    @get(
        "/api/v1/models/draft-pairs/{main_model_id:str}/{draft_model_id:str}/stats",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def get_pair_stats(main_model_id: str, draft_model_id: str) -> dict[str, Any]:
        """Return acceptance statistics for a speculative-decoding draft pair.

        Args:
            main_model_id: URL segment for the verifier (main) model identifier.
            draft_model_id: URL segment for the draft model identifier.

        Returns:
            JSON with ``acceptance_rate``, ``total``, and ``is_disabled`` fields.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.models.draft_pair_resolver import get_draft_pair_resolver

            resolver = get_draft_pair_resolver()
            stats = resolver.get_pair_stats(main_model_id, draft_model_id)
        except Exception:
            logger.warning(
                "Draft pair resolver unavailable — cannot get stats for %s/%s, returning 503",
                main_model_id,
                draft_model_id,
            )
            return litestar_error_response("Draft pair resolver subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response(stats)

    # -- GET /api/v1/models/draft-pairs/stats ---------------------------------

    @get("/api/v1/models/draft-pairs/stats", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_all_draft_pair_stats() -> dict[str, Any]:
        """Return acceptance statistics for all tracked speculative-decoding pairs.

        Returns:
            JSON with ``pairs`` mapping each ``main:draft`` key to its stats dict.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.models.draft_pair_resolver import get_draft_pair_resolver

            resolver = get_draft_pair_resolver()
            all_stats = resolver.get_all_stats()
        except Exception:
            logger.warning("Draft pair resolver unavailable — cannot get all pair stats, returning 503")
            return litestar_error_response("Draft pair resolver subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response({"pairs": all_stats})

    # -- POST /api/v1/models/chat-stream (SSE) --------------------------------

    @post("/api/v1/models/chat-stream", media_type="text/event-stream", guards=[admin_guard])
    async def chat_stream(data: dict[str, Any] | None = None) -> Any:
        """Stream a chat completion from a local GGUF model as Server-Sent Events.

        Accepts JSON body with ``model_id``, ``system_prompt`` (optional), and
        ``prompt`` fields.  Each token is delivered as an SSE ``data:`` line.
        A final ``event: done`` line signals end-of-stream.  If inference fails
        mid-stream an ``event: error`` line is yielded before the generator
        closes.

        Args:
            data: JSON request body with ``model_id``, ``prompt``, and
                optional ``system_prompt`` fields.

        Returns:
            ``text/event-stream`` response streaming tokens, or a 400 JSON
            response when required fields are missing.
        """
        body = data if data is not None else {}
        model_id: str = body.get("model_id", "")
        if not isinstance(model_id, str) or not model_id:
            return litestar_error_response("'model_id' is required and must be a string", code=400)
        system_prompt_raw = body.get("system_prompt", "")
        if not isinstance(system_prompt_raw, str):
            return litestar_error_response("'system_prompt' must be a string", code=422)
        system_prompt: str = system_prompt_raw
        prompt_raw = body.get("prompt", "")
        if not isinstance(prompt_raw, str):
            return litestar_error_response("'prompt' must be a string", code=422)
        prompt: str = prompt_raw
        if not prompt:
            return litestar_error_response("'prompt' is required", code=400)

        # Capture locals for use inside the generator closure.
        _model_id = model_id
        _system_prompt = system_prompt
        _prompt = prompt

        def _generate() -> Generator[str, None, None]:
            """Yield SSE-formatted token chunks from the local inference adapter.

            Yields:
                SSE ``data:`` lines for each token, followed by a terminal
                ``event: done`` line.  On error yields ``event: error`` with
                the exception message.
            """
            try:
                from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

                adapter = LocalInferenceAdapter()
                for token in adapter.chat_stream(_model_id, _system_prompt, _prompt):
                    yield f"data: {token}\n\n"
                yield "event: done\ndata: \n\n"
            except Exception as exc:
                logger.error(
                    "chat_stream failed for model '%s' — streaming terminated: %s",
                    _model_id,
                    exc,
                    exc_info=True,
                )
                yield "event: error\ndata: The model encountered an error during generation. Check server logs for details.\n\n"

        return Stream(
            _generate(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- GET /api/v1/vram/thermal-status --------------------------------------

    @get("/api/v1/vram/thermal-status", media_type=MediaType.JSON, guards=[admin_guard])
    async def vram_thermal_status() -> dict[str, Any]:
        """Return whether the primary GPU is currently thermal-throttling.

        Reads the GPU core temperature via pynvml and compares it against the
        85 °C throttle threshold configured in VRAMManager.  Returns 200 in
        all cases so the dashboard can show degraded-but-available status.

        Returns:
            JSON with ``throttled`` bool, ``temperature_c`` int-or-null, and
            ``gpu_index`` used for the reading.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.models.vram_manager import get_vram_manager

            mgr = get_vram_manager()
            gpu_index: int = 0
            throttled = mgr.is_thermal_throttled(gpu_index)
            temperature = mgr.get_gpu_temperature(gpu_index)
        except Exception:
            logger.warning("VRAM manager unavailable — cannot read thermal status, returning 503")
            return litestar_error_response("VRAM manager subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response({
            "throttled": throttled,
            "temperature_c": temperature,
            "gpu_index": gpu_index,
        })

    # -- GET /api/v1/vram/phase -----------------------------------------------

    @get("/api/v1/vram/phase", media_type=MediaType.JSON, guards=[admin_guard])
    async def vram_phase_recommendation() -> dict[str, Any]:
        """Return the VRAM budget recommendation for the current execution phase.

        The recommendation lists which models should be loaded and unloaded to
        optimise VRAM usage for the current pipeline phase (planning / execution
        / review).

        Returns:
            JSON with ``phase`` string, ``gpu_total_gb`` float, ``load`` list,
            ``unload`` list, and ``estimated_vram_gb`` float.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.models.vram_manager import get_vram_manager

            rec = get_vram_manager().get_phase_recommendation()
        except Exception:
            logger.warning("VRAM manager unavailable — cannot get phase recommendation, returning 503")
            return litestar_error_response("VRAM manager subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response(rec)

    # -- POST /api/v1/vram/phase ----------------------------------------------

    @post("/api/v1/vram/phase", media_type=MediaType.JSON, guards=[admin_guard])
    async def vram_set_phase(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Set the current pipeline execution phase for VRAM budgeting decisions.

        Accepts a JSON body with a ``phase`` field.  Valid values are
        ``"planning"``, ``"execution"``, and ``"review"``.

        Args:
            data: JSON request body with a ``phase`` string field.

        Returns:
            JSON with ``phase`` confirming the newly active phase, or 400 on
            missing or unrecognised input.
        """
        from vetinari.web.responses import success_response

        body = data if data is not None else {}
        phase_raw = body.get("phase", "")
        if not isinstance(phase_raw, str):
            return litestar_error_response("'phase' must be a string", code=422)  # type: ignore[return-value]
        phase: str = phase_raw
        if not phase:
            return litestar_error_response("'phase' is required", code=400)  # type: ignore[return-value]

        # Phase-string validation must run before the subsystem call so a bad
        # value always returns 400, even when the VRAM manager is unavailable.
        # Import the enum here so the 400 path doesn't depend on the manager.
        try:
            from vetinari.models.vram_manager import ExecutionPhase
        except Exception:
            logger.warning("VRAM manager unavailable — cannot import ExecutionPhase, returning 503")
            return litestar_error_response("VRAM manager subsystem unavailable", code=503)  # type: ignore[return-value]

        valid_phases = {ExecutionPhase.PLANNING, ExecutionPhase.EXECUTION, ExecutionPhase.REVIEW}
        if phase not in valid_phases:
            return litestar_error_response(  # type: ignore[return-value]
                f"Invalid phase '{phase}'. Valid values: {sorted(str(p) for p in valid_phases)}",
                code=400,
            )

        try:
            from vetinari.models.vram_manager import get_vram_manager

            get_vram_manager().set_phase(phase)
        except Exception:
            logger.warning("VRAM manager unavailable — cannot set phase to '%s', returning 503", phase)
            return litestar_error_response("VRAM manager subsystem unavailable", code=503)  # type: ignore[return-value]
        logger.info("VRAM phase set to '%s' via API", phase)
        return success_response({"phase": phase})

    # -- POST /api/v1/models/cascade-router/build -----------------------------

    @post("/api/v1/models/cascade-router/build", media_type=MediaType.JSON, guards=[admin_guard])
    async def build_cascade_from_router(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a CascadeRouter from the current DynamicModelRouter model registry.

        Reads optional ``task_type`` (string) and ``confidence_threshold`` (float)
        from the JSON request body.  Returns the tier configuration of the newly
        built cascade ordered cheapest-first.

        Args:
            data: JSON request body with optional ``task_type`` and
                ``confidence_threshold`` fields.

        Returns:
            JSON with ``tiers`` list and the resolved ``task_type`` string.
        """
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key
        from vetinari.web.responses import success_response

        if data is None:
            return litestar_error_response("Request body is required", code=422)  # type: ignore[return-value]
        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", code=400)  # type: ignore[return-value]
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=400)  # type: ignore[return-value]

        body = data
        # Reject bodies that contain only unrecognised keys — a sign of a fuzz
        # payload or a misconfigured client.
        _KNOWN_KEYS = {"task_type", "confidence_threshold"}
        if not _KNOWN_KEYS.intersection(body):
            return litestar_error_response("Request body contains unrecognised fields", code=422)  # type: ignore[return-value]
        task_type_raw = body.get("task_type", "general")
        if not isinstance(task_type_raw, str):
            return litestar_error_response("'task_type' must be a string", code=422)  # type: ignore[return-value]
        task_type_str: str = task_type_raw
        confidence_raw = body.get("confidence_threshold", 0.7)
        if not isinstance(confidence_raw, (int, float)):
            return litestar_error_response("'confidence_threshold' must be a number", code=422)  # type: ignore[return-value]
        confidence: float = float(confidence_raw)

        try:
            from vetinari.cascade_router import build_cascade_from_router as _build
            from vetinari.models.dynamic_model_router import get_model_router
            from vetinari.models.model_router_types import TaskType

            dynamic_router = get_model_router()
            # Map string task_type to enum where possible; fall back to raw string.
            try:
                typed_task_type = TaskType[task_type_str.upper()]
            except (KeyError, AttributeError):
                typed_task_type = task_type_str  # type: ignore[assignment]

            cascade = _build(dynamic_router, typed_task_type, confidence_threshold=confidence)
            tiers = [
                {
                    "model_id": tier.model_id,
                    "cost_per_1k_tokens": tier.cost_per_1k_tokens,
                    "priority": tier.priority,
                }
                for tier in cascade._tiers  # type: ignore[attr-defined]
            ]
        except Exception:
            logger.warning("Cascade router build failed for task_type '%s' — returning 503", task_type_str)
            return litestar_error_response("Cascade router subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response({"tiers": tiers, "task_type": task_type_str})

    # -- GET /api/v1/models/cascade/stats -------------------------------------

    @get("/api/v1/models/cascade/stats", media_type=MediaType.JSON, guards=[admin_guard])
    async def cascade_stats() -> dict[str, Any]:
        """Return routing statistics from the active cascade router.

        Shows per-tier hit counts, fallback counts, and latency percentiles so
        operators can see whether tiered routing is working as expected.

        Returns:
            JSON with cascade routing stats dict, or an empty ``stats`` dict
            when the adapter manager is not yet initialised.
        """
        from vetinari.web.responses import success_response

        try:
            from vetinari.adapter_manager import get_adapter_manager

            stats = get_adapter_manager().get_cascade_stats()
        except Exception:
            logger.warning("Adapter manager unavailable — cannot get cascade stats, returning 503")
            return litestar_error_response("Adapter manager subsystem unavailable", code=503)  # type: ignore[return-value]
        return success_response({"stats": stats})

    # -- POST /api/v1/models/cascade/disable ----------------------------------

    @post("/api/v1/models/cascade/disable", media_type=MediaType.JSON, guards=[admin_guard])
    async def cascade_disable(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Disable cascade routing so all requests go directly to the primary model.

        Useful in emergencies when tiered routing is causing unexpected failures.
        The change takes effect immediately without a server restart.  This
        endpoint accepts no body parameters; any body sent by the client is
        rejected with 422.

        Args:
            data: Optional JSON body — must be absent or None.

        Returns:
            JSON confirmation that cascade routing has been disabled.
        """
        from vetinari.web.responses import success_response

        # Reject any body — this endpoint takes no parameters.
        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)  # type: ignore[return-value]

        try:
            from vetinari.adapter_manager import get_adapter_manager

            get_adapter_manager().disable_cascade_routing()
        except Exception:
            logger.warning("Adapter manager unavailable — cannot disable cascade routing, returning 503")
            return litestar_error_response("Adapter manager subsystem unavailable", code=503)  # type: ignore[return-value]
        logger.info("Cascade routing disabled via admin API")
        return success_response({"disabled": True})

    return [
        assign_tasks_to_models,
        get_all_available_models,
        add_model,
        remove_model,
        get_pair_stats,
        get_all_draft_pair_stats,
        chat_stream,
        vram_thermal_status,
        vram_phase_recommendation,
        vram_set_phase,
        build_cascade_from_router,
        cascade_stats,
        cascade_disable,
    ]
