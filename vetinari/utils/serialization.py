"""Serialization utilities for dataclass-to-dict conversion.

Replaces hand-written ``to_dict()`` methods on dataclasses with a single
recursive converter that handles enums, datetimes, and nested dataclasses.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import Enum
from typing import Any


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-serializable dictionary.

    Recursively processes all fields, converting enums to their ``.value``,
    datetimes to ISO-8601 strings, and nested dataclasses to dicts.  Lists
    and dicts are traversed element-wise so nested structures are handled
    correctly.

    Args:
        obj: A dataclass instance to serialize.

    Returns:
        A plain dictionary suitable for ``json.dumps()``.

    Raises:
        TypeError: If *obj* is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")

    result: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        result[f.name] = _convert_value(getattr(obj, f.name))
    return result


def _convert_value(value: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    if isinstance(value, tuple):
        # JSON has no tuple type; convert to list so json.dumps() can serialise it.
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    return value
