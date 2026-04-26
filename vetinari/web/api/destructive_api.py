"""HTTP request parsing helpers for destructive-operation endpoints.

Provides ``parse_confirmed_intent_or_400`` which extracts a ``ConfirmedIntent``
from an HTTP request body (dict, dataclass, or None) and returns a plain-English
400 error message when the intent is missing or invalid.

Route handlers call this helper before passing the intent to a
``@protected_mutation``-decorated function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from vetinari.safety.protected_mutation import ConfirmedIntent

logger = logging.getLogger(__name__)


@dataclass
class ConfirmedIntentBody:
    """Parsed request body carrying confirmed-intent fields.

    Mirrors the JSON payload expected by destructive endpoints.

    Attributes:
        confirmed_by: Identity of the human confirming the action.
        reason: Justification for the destructive operation.
        confirmed_at_utc: Optional ISO-8601 UTC timestamp; defaults to now.
    """

    confirmed_by: str
    reason: str
    confirmed_at_utc: str = ""


def parse_confirmed_intent_or_400(
    body: dict[str, Any] | ConfirmedIntentBody | None,
) -> tuple[ConfirmedIntent | None, str]:
    """Parse a request body into a ``ConfirmedIntent`` or return a 400 error message.

    Accepts a plain dict (from JSON body parsing), a ``ConfirmedIntentBody``
    dataclass, or ``None`` (missing body).

    Args:
        body: The parsed request body.  May be a dict, a ``ConfirmedIntentBody``,
            or ``None`` when the body was absent.

    Returns:
        A tuple ``(intent, error_message)``.  When ``error_message`` is non-empty
        the caller should return HTTP 400 with that message.  When empty the
        ``intent`` is a valid ``ConfirmedIntent`` ready for use.
    """
    if body is None:
        return None, (
            "Request body is required for this destructive operation. "
            "Provide a JSON body with 'confirmed_by' and 'reason' fields."
        )

    if isinstance(body, ConfirmedIntentBody):
        confirmed_by = body.confirmed_by
        reason = body.reason
        confirmed_at_utc = body.confirmed_at_utc
    elif isinstance(body, dict):
        confirmed_by = body.get("confirmed_by", "")
        reason = body.get("reason", "")
        confirmed_at_utc = body.get("confirmed_at_utc", "")
    else:
        return None, (
            f"Unexpected body type {type(body).__name__!r}. "
            "Expected a JSON object with 'confirmed_by' and 'reason' fields."
        )

    if not confirmed_by or not str(confirmed_by).strip():
        return None, (
            "Missing required field 'confirmed_by'. Provide the identity of the person confirming this operation."
        )

    if not reason or not str(reason).strip():
        return None, (
            "Missing required field 'reason'. Provide a plain-English justification for this destructive operation."
        )

    effective_at = str(confirmed_at_utc).strip() if confirmed_at_utc else datetime.now(timezone.utc).isoformat()

    try:
        intent = ConfirmedIntent(
            confirmed_by=str(confirmed_by).strip(),
            reason=str(reason).strip(),
            confirmed_at_utc=effective_at,
        )
        return intent, ""
    except ValueError as exc:
        logger.debug("parse_confirmed_intent_or_400: invalid intent fields — %s", exc)
        return None, f"Invalid confirmed intent: {exc}"


__all__ = [
    "ConfirmedIntentBody",
    "parse_confirmed_intent_or_400",
]
