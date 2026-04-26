"""Replay API Litestar handler — re-execute pipeline decisions from a stored checkpoint.

Native Litestar equivalent of the route previously registered by
``replay_api._register(bp)``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Endpoints
---------
    POST /api/v1/replay — Plan a partial pipeline re-execution from a checkpoint
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, post

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

_ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset({"model_id", "temperature", "modified_prompt"})
_PIPELINE_STEPS: tuple[str, ...] = (
    "prevention_gate",
    "input_analysis",
    "plan_gen",
    "model_assignment",
    "execution",
    "review",
    "assembly",
)


def create_replay_api_handlers() -> list[Any]:
    """Create Litestar handlers for the pipeline replay API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — replay API handler not registered")
        return []

    # -- POST /api/v1/replay -----------------------------------------------------

    @post("/api/v1/replay", media_type=MediaType.JSON, guards=[admin_guard])
    async def replay_pipeline(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plan a partial pipeline re-execution from a checkpoint.

        Validates the trace and step, applies overrides, and returns a replay
        descriptor that a caller can use to re-run the pipeline from ``from_step``
        with the checkpointed state for all preceding steps already loaded.

        Args:
            data: JSON request body with the following fields:

                - ``trace_id`` (str, required) — ID of the pipeline trace to replay.
                - ``from_step`` (str, required) — Pipeline step to restart from.
                - ``overrides`` (dict, optional) — Allowed keys: ``model_id``,
                  ``temperature``, ``modified_prompt``.

        Returns:
            200 with replay plan containing ``replay_id``, ``pre_loaded_checkpoints``,
            ``restart_checkpoint``, and ``steps_to_replay``.
            400 on missing/invalid inputs or disallowed override keys.
            404 if no checkpoints exist for the given ``trace_id``.
        """
        body = data if data is not None else {}

        trace_id_raw = body.get("trace_id", "")
        if not isinstance(trace_id_raw, str):
            return litestar_error_response("'trace_id' must be a string", 422)  # type: ignore[return-value]
        trace_id: str = trace_id_raw

        from_step_raw = body.get("from_step", "")
        if not isinstance(from_step_raw, str):
            return litestar_error_response("'from_step' must be a string", 422)  # type: ignore[return-value]
        from_step: str = from_step_raw

        overrides_raw = body.get("overrides") or {}
        if not isinstance(overrides_raw, dict):
            return litestar_error_response("'overrides' must be a dict", 422)  # type: ignore[return-value]
        overrides: dict[str, Any] = overrides_raw

        if not trace_id:
            return litestar_error_response("trace_id is required", 400)  # type: ignore[return-value]
        if not from_step:
            return litestar_error_response("from_step is required", 400)  # type: ignore[return-value]
        if from_step not in _PIPELINE_STEPS:
            return litestar_error_response(  # type: ignore[return-value]
                f"from_step '{from_step}' is not a valid pipeline step",
                400,
                details={"valid_steps": list(_PIPELINE_STEPS)},
            )

        bad_keys = set(overrides.keys()) - _ALLOWED_OVERRIDE_KEYS
        if bad_keys:
            return litestar_error_response(  # type: ignore[return-value]
                f"Override keys not allowed: {sorted(bad_keys)}",
                400,
                details={"allowed_override_keys": sorted(_ALLOWED_OVERRIDE_KEYS)},
            )

        try:
            from vetinari.observability.checkpoints import get_checkpoint_store

            store = get_checkpoint_store()
            checkpoints = store.list_checkpoints(trace_id)
        except Exception:
            logger.exception("Failed to load checkpoints for trace_id=%s", trace_id)
            return litestar_error_response("Failed to load checkpoints", 500)  # type: ignore[return-value]

        if not checkpoints:
            return litestar_error_response(  # type: ignore[return-value]
                f"No checkpoints found for trace_id '{trace_id}'", 404
            )

        from_step_index = _PIPELINE_STEPS.index(from_step)

        pre_loaded = [
            {
                "step_name": cp.step_name,
                "step_index": cp.step_index,
                "status": cp.status,
                "output_snapshot": cp.output_snapshot,
                "tokens_used": cp.tokens_used,
                "latency_ms": cp.latency_ms,
                "model_id": cp.model_id,
                "quality_score": cp.quality_score,
                "created_at": cp.created_at,
            }
            for cp in checkpoints
            if cp.step_index < from_step_index
        ]

        restart_checkpoint = next(
            (
                {
                    "step_name": cp.step_name,
                    "step_index": cp.step_index,
                    "status": cp.status,
                    "input_snapshot": cp.input_snapshot,
                    "output_snapshot": cp.output_snapshot,
                }
                for cp in checkpoints
                if cp.step_name == from_step
            ),
            None,
        )

        # from_step must have a saved checkpoint to restart from; without it the
        # replay cannot be constructed — 409 signals the conflict (valid trace,
        # but the requested step has no stored checkpoint).
        if restart_checkpoint is None:
            return litestar_error_response(  # type: ignore[return-value]
                f"No checkpoint found for step '{from_step}' in trace '{trace_id}'",
                409,
                details={"from_step": from_step, "trace_id": trace_id},
            )

        replay_id = str(uuid.uuid4())

        return Response(  # type: ignore[return-value]
            content={
                "replay_id": replay_id,
                "trace_id": trace_id,
                "from_step": from_step,
                "from_step_index": from_step_index,
                "overrides": overrides,
                "pre_loaded_checkpoints": pre_loaded,
                "restart_checkpoint": restart_checkpoint,
                "steps_to_replay": list(_PIPELINE_STEPS[from_step_index:]),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [replay_pipeline]
