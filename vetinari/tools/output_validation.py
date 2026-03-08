"""Shared output validation helpers for tool/skill results.

Provides a ``validate_output`` function that checks tool responses for
completeness before returning them to the caller.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def validate_output(
    result: Any,
    required_fields: Sequence[str] = (),
    *,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Validate a tool/skill output and return a normalised report.

    Parameters
    ----------
    result:
        The raw result object (dict, dataclass with ``to_dict()``, or any
        truthy/falsy value).
    required_fields:
        Field names that must be present *and* non-``None`` in the result
        dict.
    allow_empty:
        When ``False`` (the default), a ``None`` or empty-dict result is
        treated as invalid.

    Returns
    -------
    dict with keys:
        ``valid`` (bool) -- whether the output passed all checks.
        ``data``  (dict | None) -- the normalised result dict, or ``None``.
        ``errors`` (list[str]) -- human-readable validation failures.
    """
    errors: List[str] = []

    # ── Normalise to dict ────────────────────────────────────────────────
    if result is None:
        data = None
    elif isinstance(result, dict):
        data = result
    elif hasattr(result, "to_dict"):
        try:
            data = result.to_dict()
        except Exception as exc:
            logger.warning("validate_output: to_dict() failed: %s", exc)
            data = None
            errors.append(f"to_dict() raised {type(exc).__name__}: {exc}")
    else:
        # Treat as opaque truthy/falsy value
        data = {"output": result} if result else None

    # ── Emptiness check ──────────────────────────────────────────────────
    if not allow_empty and (data is None or data == {}):
        errors.append("Output is empty or None")

    # ── Required-fields check ────────────────────────────────────────────
    if data and required_fields:
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

    valid = len(errors) == 0
    if not valid:
        logger.debug("validate_output: %d error(s): %s", len(errors), errors)

    return {"valid": valid, "data": data, "errors": errors}
