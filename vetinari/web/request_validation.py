"""Request body validation helpers for Litestar route handlers.

Provides deterministic checks (depth, size, shape) that guard handlers
accepting ``dict[str, Any]`` bodies from overly nested or malformed payloads.
These are applied at the start of handler execution, before any business logic.

This is part of the web request pipeline: Auth -> CSRF -> **Body Validation** -> Handler.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maximum nesting depth accepted for any request body dict.
# A legitimate API body is at most 2-3 levels deep.  Anything beyond 5 is
# either a client bug or a fuzzing/injection attempt.
_MAX_BODY_DEPTH = 5


def _measure_depth(value: Any, current: int = 0) -> int:
    """Recursively measure the nesting depth of a JSON-like value.

    Args:
        value: The value to measure.
        current: Depth of the current call frame (starts at 0).

    Returns:
        Maximum nesting depth found in ``value``.
    """
    if isinstance(value, dict):
        if not value:
            return current
        return max(_measure_depth(v, current + 1) for v in value.values())
    if isinstance(value, list):
        if not value:
            return current
        return max(_measure_depth(v, current + 1) for v in value)
    return current


# Maximum byte length for any single key in a request body dict.
# A legitimate API key is a short ASCII identifier.  10 000-byte keys are
# either a client bug or a fuzzing/injection attempt.
_MAX_KEY_LENGTH = 256


def body_has_oversized_key(data: dict[str, Any] | None, max_key_length: int = _MAX_KEY_LENGTH) -> bool:
    """Return True when any key in the request body exceeds ``max_key_length`` bytes.

    Recursively inspects nested dicts so deeply-nested oversized keys are also
    caught.  Handlers that accept ``data: dict[str, Any]`` should call this
    alongside ``body_depth_exceeded`` to guard against key-length bombs.

    Args:
        data: The parsed request body dict, or None for empty bodies.
        max_key_length: Maximum allowed key length in characters (default: 256).

    Returns:
        True if any dict key exceeds ``max_key_length``, else False.
    """
    if data is None:
        return False
    return _has_oversized_key(data, max_key_length)


def json_object_body(data: Any | None) -> dict[str, Any] | None:
    """Return a JSON object body or ``None`` when the payload has the wrong shape.

    Litestar route handlers often receive parsed request bodies through a
    ``data`` parameter.  Tests and malformed clients can still provide non-dict
    JSON values, so handlers should validate the shape before calling
    ``dict.get`` or membership checks.

    Args:
        data: Parsed JSON body, or ``None`` when no body was sent.

    Returns:
        ``{}`` for a missing body, the original dict for a JSON object, or
        ``None`` when the body is a non-object JSON value.
    """
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    logger.warning("Request body rejected - expected JSON object, got %s", type(data).__name__)
    return None


def _has_oversized_key(value: Any, max_key_length: int) -> bool:
    """Recursively scan ``value`` for any dict key longer than ``max_key_length``.

    Args:
        value: The value to scan.
        max_key_length: Maximum allowed key length in characters.

    Returns:
        True if an oversized key is found anywhere in the structure.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and len(k) > max_key_length:
                logger.warning(
                    "Request body rejected — key length %d exceeds maximum %d",
                    len(k),
                    max_key_length,
                )
                return True
            if _has_oversized_key(v, max_key_length):
                return True
    elif isinstance(value, list):
        for item in value:
            if _has_oversized_key(item, max_key_length):
                return True
    return False


def body_depth_exceeded(data: dict[str, Any] | None, max_depth: int = _MAX_BODY_DEPTH) -> bool:
    """Return True when the request body nesting depth exceeds ``max_depth``.

    Handlers that accept ``data: dict[str, Any]`` should call this at the top
    of their body and return 400 when it returns True.  This prevents deeply
    nested payloads (JSON bombs, fuzz inputs) from reaching business logic.

    Args:
        data: The parsed request body dict, or None for empty bodies.
        max_depth: Maximum allowed nesting depth (default: 5).

    Returns:
        True if the body is more deeply nested than ``max_depth``, else False.
    """
    if data is None:
        return False
    depth = _measure_depth(data)
    if depth > max_depth:
        logger.warning(
            "Request body rejected — nesting depth %d exceeds maximum %d",
            depth,
            max_depth,
        )
        return True
    return False
