"""Decision journal Litestar handler — filtered, paginated view of pipeline decisions.

Native Litestar equivalent of the route previously registered by
``decisions_api._register(bp)``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

This is step 3 of the observability layer: Tracing → Drift Detection → **Decision Journal**.

Endpoints
---------
    GET /api/v1/decisions — Filtered, paginated decision journal records
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get

    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Numeric confidence values for filter comparisons.
# Maps ConfidenceLevel.value strings to [0.0, 1.0] scale.
# "low" (0.25) and "very_low" (0.0) are intentionally distinct so callers can
# filter one without including the other.
_CONFIDENCE_VALUES: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.25,
    "very_low": 0.0,
}

# Hard cap on records returned per request.
_MAX_LIMIT: int = 200


def create_decisions_api_handlers() -> list[Any]:
    """Create Litestar handlers for the decision journal API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — decisions API handler not registered")
        return []

    # -- GET /api/v1/decisions ---------------------------------------------------

    @get("/api/v1/decisions", media_type=MediaType.JSON)
    async def api_decisions(
        type: str = "",
        confidence_min: str = "",
        since_iso: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return a filtered, paginated view of decision journal records.

        Supports filtering by decision type, minimum confidence level, and a
        since-ISO timestamp. The journal is queried newest-first; post-filters
        for confidence and timestamp are applied in Python after the SQL fetch.

        Args:
            type: Decision type string (e.g. ``"model_selection"``). Optional.
            confidence_min: Minimum confidence as a float string in [0.0, 1.0].
                1.0 → high only; 0.5 → medium and above; 0.0 → all. Optional.
            since_iso: ISO 8601 timestamp — only return decisions after this time.
                Optional.
            limit: Maximum records to return (default 50, max 200). Optional.

        Returns:
            JSON object with ``decisions`` list and ``total`` count, or a 503
            when the decision journal module is unavailable.
        """
        try:
            from vetinari.observability.decision_journal import get_decision_journal
            from vetinari.types import DecisionType
        except ImportError as exc:
            logger.warning(
                "Decision journal module not importable — decisions endpoint unavailable: %s",
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Decision journal module not available", 503
            )

        clamped_limit = min(max(limit, 1), _MAX_LIMIT)
        type_param = type.strip()
        confidence_min_param = confidence_min.strip()

        # Resolve optional decision_type filter — unknown type is a client error (400)
        decision_type_filter = None
        if type_param:
            try:
                decision_type_filter = DecisionType(type_param)
            except ValueError:
                logger.warning("Unknown decision type filter %r — returning 400", type_param)
                return litestar_error_response(  # type: ignore[return-value]
                    f"Unknown decision type {type_param!r} — valid types are defined in DecisionType enum", 400
                )

        # Resolve optional confidence_min filter with range validation
        confidence_min_val: float | None = None
        if confidence_min_param:
            try:
                parsed = float(confidence_min_param)
            except ValueError:
                logger.warning(
                    "Invalid confidence_min param %r — must be float 0.0-1.0",
                    confidence_min_param,
                )
                return litestar_error_response(  # type: ignore[return-value]
                    "confidence_min must be a float between 0.0 and 1.0", 400
                )
            if not (0.0 <= parsed <= 1.0):
                logger.warning(
                    "Out-of-range confidence_min param %r — must be in [0.0, 1.0]",
                    confidence_min_param,
                )
                return litestar_error_response(  # type: ignore[return-value]
                    "confidence_min must be a float between 0.0 and 1.0", 400
                )
            confidence_min_val = parsed

        # Validate since_iso before using it in string comparisons
        if since_iso:
            try:
                from datetime import datetime

                datetime.fromisoformat(since_iso)
            except ValueError:
                logger.warning("Invalid since_iso param %r — must be ISO 8601", since_iso)
                return litestar_error_response(  # type: ignore[return-value]
                    "since_iso must be a valid ISO 8601 datetime string", 400
                )

        try:
            journal = get_decision_journal()
            # Fetch up to 2x the requested limit so post-filters don't starve results
            fetch_limit = min(clamped_limit * 2 + 20, _MAX_LIMIT * 2)
            records = journal.get_decisions(
                decision_type=decision_type_filter,
                limit=fetch_limit,
            )

            if since_iso:
                records = [r for r in records if r.timestamp >= since_iso]

            if confidence_min_val is not None:
                records = [r for r in records if _CONFIDENCE_VALUES.get(r.confidence.value, 0.0) >= confidence_min_val]

            records = records[:clamped_limit]

            decisions = []
            for rec in records:
                conf_str = rec.confidence.value
                decisions.append({
                    "id": rec.decision_id,
                    "timestamp": rec.timestamp,
                    "type": rec.decision_type.value,
                    "chosen": rec.chosen,
                    "alternatives": rec.alternatives,
                    "confidence": conf_str,
                    "confidence_value": _CONFIDENCE_VALUES.get(conf_str, 0.0),
                    "reasoning": rec.reasoning,
                    "outcome": rec.outcome,
                    "status": rec.status,
                    "trace_id": rec.trace_id,
                })

            return {"status": "ok", "decisions": decisions, "total": len(decisions)}
        except Exception as exc:
            logger.warning("decisions API: journal access failed — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Decision journal subsystem unavailable", 503
            )

    return [api_decisions]
