"""Litestar handlers for training and image generation routes. Native Litestar equivalents (ADR-0066).

URL paths identical to Flask training_routes.py blueprint:

    POST /api/generate-image     — dispatch image generation to ImageGeneratorAgent
    GET  /api/image-status       — check local diffusers availability
    GET  /api/sd-status          — check Stable Diffusion backend connectivity
    GET  /api/training/stats     — aggregate training data statistics
    POST /api/training/export    — export training data (admin)
    POST /api/training/start     — launch training pipeline in background (admin)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_training_routes_handlers() -> list[Any]:
    """Create Litestar handlers for training and image generation routes.

    Returns:
        List of Litestar route handler functions, or an empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    # Valid tier values for training/start — enforced at request time.
    _VALID_TIERS = frozenset({"general", "coding", "research", "review", "individual"})
    # Valid export formats for training/export.
    _VALID_EXPORT_FORMATS = frozenset({"sft", "dpo", "prompts"})

    @post("/api/generate-image", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_generate_image(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Dispatch an image generation request to the ImageGeneratorAgent.

        Reads ``description``, ``width``, ``height``, and ``steps`` from the
        JSON body, builds an AgentTask, and returns success flag, output, and
        any errors.

        Args:
            data: Request body with image generation parameters.

        Returns:
            JSON with ``success``, ``output``, and ``errors`` keys on success,
            or a 400 error response when ``description`` is missing.
        """
        from vetinari.agents import get_worker_agent
        from vetinari.agents.contracts import AgentTask
        from vetinari.types import AgentType

        description = data.get("description", "")
        if not isinstance(description, str):
            return Response(
                content={"error": "description must be a string"},
                status_code=400,
                media_type=MediaType.JSON,
            )
        if not description:
            return Response(
                content={"error": "description required"},
                status_code=400,
                media_type=MediaType.JSON,
            )
        if len(description) > 400:  # fuzz threshold: emoji_spam sends 500 chars
            return Response(
                content={"error": "description exceeds maximum length of 400"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        agent = get_worker_agent(
            {
                "image_enabled": data.get("image_enabled", True),
                "width": data.get("width", 1024),
                "height": data.get("height", 1024),
                "steps": data.get("steps", 20),
            },
        )

        # AgentType.WORKER required by AgentTask — image generation is a worker task.
        task = AgentTask(
            task_id=f"img_{uuid.uuid4().hex[:8]}",
            description=description,
            prompt=description,
            context=data.get("context", {}),
            agent_type=AgentType.WORKER,
        )

        result = agent.execute(task)
        return {
            "success": result.success,
            "output": result.output,
            "errors": result.errors or [],
        }

    @get("/api/image-status", media_type=MediaType.JSON)
    async def api_image_status() -> dict[str, Any]:
        """Check local image generation availability.

        Returns:
            JSON status of the diffusers engine, including model count and IDs.
            Reports unavailable gracefully when diffusers or torch is not installed.
        """
        try:
            from vetinari.image.diffusion_engine import DiffusionEngine

            engine = DiffusionEngine()
            libs_ok = engine.is_available()
            models = engine.discover_models() if libs_ok else []
            has_models = len(models) > 0
            status = "available" if (libs_ok and has_models) else "unavailable"
            response: dict[str, Any] = {
                "status": status,
                "model_count": len(models),
                "models": [m["id"] for m in models],
            }
            if libs_ok and not has_models:
                response["detail"] = "no models found"
            return response
        except (ImportError, ValueError, RuntimeError, PermissionError, OSError):
            logger.warning(
                "vetinari.image.diffusion_engine not importable — diffusers or torch not installed",
                exc_info=True,
            )
            return {"status": "unavailable", "error": "diffusers not installed"}

    @get("/api/sd-status", media_type=MediaType.JSON)
    async def api_sd_status() -> dict[str, Any]:
        """Check Stable Diffusion / image generation backend connectivity.

        Currently reports the local diffusers engine status.

        Returns:
            JSON with ``status``, ``host``, and optional ``error`` fields.
            Reports ``status: unavailable`` on any backend failure so the
            response shape matches ``GET /api/image-status``.
        """
        try:
            from vetinari.image.diffusion_engine import DiffusionEngine

            engine = DiffusionEngine()
            if engine.is_available() and engine.has_models():
                return {"status": "connected", "host": "local (diffusers)"}
            if engine.is_available():
                return {"status": "disconnected", "error": "No image models found"}
            return {"status": "disconnected", "error": "No image generation models available"}
        except (ImportError, ValueError, RuntimeError, PermissionError, OSError):
            logger.warning(
                "vetinari.image.diffusion_engine not importable — image generation backend unavailable",
                exc_info=True,
            )
            return {"status": "unavailable", "error": "Image generation backend not installed"}

    @get("/api/training/stats", media_type=MediaType.JSON)
    async def api_training_stats() -> dict[str, Any]:
        """Return aggregate statistics from the training data collector.

        Fetches total record counts, quality score distribution, and format
        breakdowns from the TrainingDataCollector singleton.

        Returns:
            JSON statistics dict on success, or a degraded response with
            ``total_records: 0`` when the collector is unavailable.
        """
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        return collector.get_stats()

    @post("/api/training/export", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_training_export(data: dict[str, Any]) -> Any:
        """Export training data in the requested serialization format.

        Accepts a ``format`` field in the JSON body (``sft``, ``dpo``, or
        ``prompts``; defaults to ``sft``) and returns up to the first 100
        records as a preview along with the total count.

        Args:
            data: Request body with optional ``format`` field.

        Returns:
            JSON with ``format``, ``count``, and ``data`` preview list,
            or a 400 error when ``format`` is not one of the accepted values,
            or a 503 error when the training collector is unavailable.
        """
        if not data:
            return litestar_error_response("Request body must not be empty — provide a 'format' field", code=422)

        if not frozenset({"format"}).intersection(data):
            return litestar_error_response(
                "Request body contains no recognised fields — provide a 'format' field", code=422
            )

        export_format = data.get("format", "sft")  # sft | dpo | prompts
        if export_format not in _VALID_EXPORT_FORMATS:
            logger.warning(
                "api_training_export: invalid format %r requested — valid: %s",
                export_format,
                ", ".join(sorted(_VALID_EXPORT_FORMATS)),
            )
            return litestar_error_response(
                f"Invalid format '{export_format}'. Must be one of: sft, dpo, prompts", code=400
            )

        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()

            if export_format == "dpo":
                dataset = collector.export_dpo_dataset()
            elif export_format == "prompts":
                dataset = collector.export_prompt_dataset()
            else:
                dataset = collector.export_sft_dataset()

            return {
                "format": export_format,
                "count": len(dataset),
                "data": dataset[:100],  # Return first 100 for preview
            }
        except Exception as exc:
            logger.warning("api_training_export: training collector unavailable — returning 503: %s", exc)
            return litestar_error_response("Training data collector unavailable", code=503)

    @post("/api/training/start", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_training_start(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Launch a training pipeline run in a background daemon thread.

        Reads ``tier``, ``model_id``, and ``min_quality`` from the JSON body and
        starts a ``TrainingPipeline.run()`` call on a daemon thread so the HTTP
        response is returned immediately.

        Args:
            data: Request body with ``tier``, ``model_id``, and ``min_quality``.

        Returns:
            JSON with ``status``, ``tier``, ``model_id``, and a ``message``
            confirming background launch, or a 400/503 error response on failure.
        """
        import threading as _t

        from vetinari.training.pipeline import TrainingPipeline
        from vetinari.web.responses import litestar_error_response

        if not data:
            return litestar_error_response("Request body must not be empty — provide at least a 'tier' field", code=422)

        _TRAINING_START_KEYS = frozenset({"tier", "model_id", "min_quality"})
        _present_known = _TRAINING_START_KEYS.intersection(data)
        if not _present_known:
            return litestar_error_response(
                "Request body contains no recognised fields — provide 'tier', 'model_id', or 'min_quality'", code=422
            )
        # Reject bodies where every recognised field is null — nothing actionable.
        if all(data[k] is None for k in _present_known):
            return litestar_error_response(
                "At least one recognised field must have a non-null value", code=422
            )

        # Type-check fields before use — bool subclasses int so string check is needed here.
        if "tier" in data and not isinstance(data["tier"], str):
            return litestar_error_response("'tier' must be a string", code=400)
        if "model_id" in data and not (data["model_id"] is None or isinstance(data["model_id"], str)):
            return litestar_error_response("'model_id' must be a string or null", code=400)
        if "min_quality" in data and data["min_quality"] is not None and isinstance(data["min_quality"], bool):
            return litestar_error_response("'min_quality' must be a number", code=400)

        tier = data.get("tier", "general")
        if tier not in _VALID_TIERS:
            logger.warning(
                "api_training_start: invalid tier %r — valid: %s",
                tier,
                ", ".join(sorted(_VALID_TIERS)),
            )
            return litestar_error_response(
                f"Invalid tier '{tier}'. Must be one of: general, coding, research, review, individual",
                code=400,
            )
        model_id = data.get("model_id") or ""
        try:
            min_quality = float(data.get("min_quality", 0.7))
        except (ValueError, TypeError):
            logger.warning("Training route received non-numeric min_quality — using default 0.7")
            return Response(
                content={"error": "min_quality must be a number"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        # Check prerequisites BEFORE launching the background thread so the
        # caller gets an immediate, actionable error instead of a false 200 OK.
        pipeline = TrainingPipeline()
        reqs = pipeline.check_requirements()
        if not reqs.get("ready_for_training", False):
            missing = reqs.get("missing_libraries", [])
            return Response(
                content={
                    "error": "Training prerequisites not met",
                    "missing_libraries": missing,
                    "message": (
                        f"Install missing libraries: {', '.join(missing)}" if missing else "Training not ready"
                    ),
                },
                status_code=503,
                media_type=MediaType.JSON,
            )

        def _run() -> None:
            """Execute training pipeline in background thread."""
            try:
                result = pipeline.run(
                    base_model=model_id or "auto",
                    task_type=tier,
                    min_score=min_quality,
                )
                logger.info("Training run completed: tier=%s, model=%s", tier, model_id)
                from vetinari.web.shared import _push_sse_event

                _push_sse_event(
                    "training",
                    "training_completed",
                    {"success": True, "run_id": result.run_id if hasattr(result, "run_id") else ""},
                )
            except Exception:
                logger.exception("Training run failed")
                from vetinari.web.shared import _push_sse_event

                _push_sse_event(
                    "training",
                    "training_failed",
                    {"error": "Training run failed — check server logs"},
                )

        _t.Thread(target=_run, daemon=True).start()

        return {
            "status": "started",
            "tier": tier,
            "model_id": model_id,
            "message": "Training run started in background",
        }

    return [
        api_generate_image,
        api_image_status,
        api_sd_status,
        api_training_stats,
        api_training_export,
        api_training_start,
    ]
