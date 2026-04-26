"""Cost analysis Litestar handler — per-stage token cost and quality-efficiency metrics.

Native Litestar equivalent of the route previously registered by
``cost_analysis_api._register(bp)``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

cost_per_quality_point = tokens_used / quality_score for records where
quality_score > 0.  Records with quality_score == 0 are reported separately
as zero-quality records (they consumed tokens but produced no measurable value).

Endpoints
---------
    GET /api/v1/cost-analysis — Aggregated token cost and quality-efficiency metrics
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Default cap on rows scanned to bound query cost.
_DEFAULT_LIMIT: int = 1_000
_MAX_LIMIT: int = 10_000


def _make_bucket() -> dict[str, Any]:
    """Return a fresh cost-aggregation bucket for a model or pipeline stage."""
    return {
        "total_tokens": 0,
        "total_quality": 0.0,
        "record_count": 0,
        "zero_quality_count": 0,
        "cost_per_quality_points": [],
    }


def _summarize(bucket: dict[str, Any]) -> dict[str, Any]:
    """Collapse a cost bucket into its summary statistics.

    Args:
        bucket: A cost-aggregation bucket as returned by ``_make_bucket()``.

    Returns:
        Dict with ``total_tokens``, ``total_quality``, ``record_count``,
        ``zero_quality_count``, and ``avg_cost_per_quality_point``.
    """
    cpq_list: list[float] = bucket["cost_per_quality_points"]
    avg_cpq = sum(cpq_list) / len(cpq_list) if cpq_list else None
    return {
        "total_tokens": bucket["total_tokens"],
        "total_quality": round(bucket["total_quality"], 4),
        "record_count": bucket["record_count"],
        "zero_quality_count": bucket["zero_quality_count"],
        "avg_cost_per_quality_point": round(avg_cpq, 4) if avg_cpq is not None else None,
    }


def create_cost_analysis_api_handlers() -> list[Any]:
    """Create Litestar handlers for the cost analysis API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — cost analysis API handler not registered")
        return []

    # -- GET /api/v1/cost-analysis -----------------------------------------------

    @get("/api/v1/cost-analysis", media_type=MediaType.JSON, guards=[admin_guard])
    async def cost_analysis(
        since: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Aggregate token costs and quality efficiency across pipeline stages.

        Reads ``pipeline_traces`` checkpoints and computes:
        - ``cost_per_quality_point`` per record (tokens_used / quality_score)
        - Aggregated totals and averages by ``model_id``
        - Aggregated totals and averages by ``step_name``
        - Cross-breakdown by stage x model

        Records with ``quality_score == 0`` or ``NULL`` are counted as
        zero-quality records and reported separately so callers can identify
        stages that consume tokens without producing measurable output.

        Args:
            since: ISO-8601 datetime string — only include traces after this time.
            limit: Maximum number of trace records to scan (default 1000, max 10000).

        Returns:
            JSON response with ``by_model``, ``by_stage``, ``by_stage_model``,
            ``zero_quality_records``, and ``summary`` sections.
        """
        clamped_limit = min(max(limit, 1), _MAX_LIMIT)

        try:
            from vetinari.observability.checkpoints import get_checkpoint_store

            store = get_checkpoint_store()
            traces = store.list_all_checkpoints(since=since, limit=clamped_limit)
        except Exception:
            logger.exception("Failed to load traces for cost analysis")
            return litestar_error_response(
                "Training trace subsystem unavailable — checkpoint store could not be reached", 503
            )  # type: ignore[return-value]

        by_model: dict[str, dict[str, Any]] = defaultdict(_make_bucket)
        by_stage: dict[str, dict[str, Any]] = defaultdict(_make_bucket)
        # Cross-breakdown: by_stage_model[stage][model] → bucket
        by_stage_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(_make_bucket))
        zero_quality_records: list[dict[str, Any]] = []
        total_tokens = 0
        total_records = 0

        for cp in traces:
            tokens = cp.tokens_used or 0
            quality = cp.quality_score  # None or float
            model = cp.model_id or "unknown"
            stage = cp.step_name or "unknown"

            total_tokens += tokens
            total_records += 1

            by_model[model]["total_tokens"] += tokens
            by_model[model]["record_count"] += 1
            by_stage[stage]["total_tokens"] += tokens
            by_stage[stage]["record_count"] += 1
            by_stage_model[stage][model]["total_tokens"] += tokens
            by_stage_model[stage][model]["record_count"] += 1

            if quality and quality > 0.0:
                cpq = tokens / quality
                by_model[model]["total_quality"] += quality
                by_model[model]["cost_per_quality_points"].append(cpq)
                by_stage[stage]["total_quality"] += quality
                by_stage[stage]["cost_per_quality_points"].append(cpq)
                by_stage_model[stage][model]["total_quality"] += quality
                by_stage_model[stage][model]["cost_per_quality_points"].append(cpq)
            else:
                by_model[model]["zero_quality_count"] += 1
                by_stage[stage]["zero_quality_count"] += 1
                by_stage_model[stage][model]["zero_quality_count"] += 1
                zero_quality_records.append({
                    "trace_id": cp.trace_id,
                    "step_name": cp.step_name,
                    "model_id": cp.model_id,
                    "tokens_used": tokens,
                    "quality_score": quality,
                    "created_at": cp.created_at,
                })

        return {
            "summary": {
                "total_records": total_records,
                "total_tokens": total_tokens,
                "zero_quality_records": len(zero_quality_records),
            },
            "by_model": {model: _summarize(data) for model, data in by_model.items()},
            "by_stage": {stage: _summarize(data) for stage, data in by_stage.items()},
            "by_stage_model": {
                stage: {model: _summarize(bucket) for model, bucket in model_dict.items()}
                for stage, model_dict in by_stage_model.items()
            },
            # Cap the detail list at 100 to avoid oversized responses
            "zero_quality_records": zero_quality_records[:100],
        }

    return [cost_analysis]
