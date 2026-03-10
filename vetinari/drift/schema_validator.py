"""
Schema Validator — vetinari.drift.schema_validator  (Phase 7)

Validates dataclass instances and plain dicts against a set of JSON-Schema-
style rules without requiring any external ``jsonschema`` dependency.

Rules supported
---------------
    required_keys   — list of keys that must be present
    forbidden_keys  — list of keys that must NOT be present
    key_types       — dict mapping key → expected Python type name
    version_pattern — regex the ``version`` field must match
    non_empty_keys  — keys whose value must be non-None and non-empty-string

Usage
-----
    from vetinari.drift.schema_validator import get_schema_validator

    v = get_schema_validator()
    v.register_schema("Plan", {
        "required_keys":   ["plan_id", "goal", "status"],
        "key_types":       {"plan_id": "str", "risk_score": "float"},
        "version_pattern": "^v\\d+\\.\\d+\\.\\d+$",
        "non_empty_keys":  ["plan_id", "goal"],
    })

    errors = v.validate("Plan", plan_instance)
    assert errors == [], errors
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TYPE_MAP: Dict[str, type] = {
    "str":   str,
    "int":   int,
    "float": float,
    "bool":  bool,
    "list":  list,
    "dict":  dict,
}


def _to_dict(obj: Any) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"__value__": obj}


class SchemaValidator:
    """
    Thread-safe schema validator.  Singleton — use
    ``get_schema_validator()``.
    """

    _instance:   Optional["SchemaValidator"] = None
    _class_lock  = threading.Lock()

    def __new__(cls) -> "SchemaValidator":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock    = threading.RLock()
        self._schemas: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register (or replace) a named validation schema."""
        with self._lock:
            self._schemas[name] = schema
        logger.debug("Registered schema '%s'", name)

    def unregister_schema(self, name: str) -> bool:
        with self._lock:
            existed = name in self._schemas
            self._schemas.pop(name, None)
            return existed

    def list_schemas(self) -> List[str]:
        with self._lock:
            return sorted(self._schemas.keys())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, schema_name: str, obj: Any) -> List[str]:
        """
        Validate ``obj`` against the named schema.

        Returns a list of error strings (empty = valid).
        """
        with self._lock:
            schema = self._schemas.get(schema_name)
        if schema is None:
            return [f"Unknown schema '{schema_name}'"]

        data   = _to_dict(obj)
        errors: List[str] = []

        # required_keys
        for key in schema.get("required_keys", []):
            if key not in data:
                errors.append(f"Missing required key '{key}'")

        # forbidden_keys
        for key in schema.get("forbidden_keys", []):
            if key in data:
                errors.append(f"Forbidden key present: '{key}'")

        # key_types
        for key, type_name in schema.get("key_types", {}).items():
            if key in data and data[key] is not None:
                expected = _TYPE_MAP.get(type_name)
                if expected and not isinstance(data[key], expected):
                    errors.append(
                        f"Key '{key}': expected {type_name}, "
                        f"got {type(data[key]).__name__}"
                    )

        # version_pattern
        pattern = schema.get("version_pattern")
        if pattern and "version" in data:
            v = str(data["version"])
            if not re.match(pattern, v):
                errors.append(
                    f"Field 'version' value '{v}' does not match "
                    f"pattern '{pattern}'"
                )

        # non_empty_keys
        for key in schema.get("non_empty_keys", []):
            val = data.get(key)
            if val is None or val == "" or val == [] or val == {}:
                errors.append(f"Key '{key}' must not be empty")

        # allowed_status_values
        allowed_statuses = schema.get("allowed_status_values")
        if allowed_statuses and "status" in data:
            if data["status"] not in allowed_statuses:
                errors.append(
                    f"Invalid status '{data['status']}'; "
                    f"allowed: {allowed_statuses}"
                )

        return errors

    def validate_many(
        self,
        schema_name: str,
        objects: List[Any],
    ) -> Dict[int, List[str]]:
        """Validate a list of objects.  Returns index → error list (only failures)."""
        failures: Dict[int, List[str]] = {}
        for i, obj in enumerate(objects):
            errs = self.validate(schema_name, obj)
            if errs:
                failures[i] = errs
        return failures

    def is_valid(self, schema_name: str, obj: Any) -> bool:
        """Return True when validation produces no errors."""
        return len(self.validate(schema_name, obj)) == 0

    # ------------------------------------------------------------------
    # Built-in Vetinari schemas
    # ------------------------------------------------------------------

    def register_vetinari_schemas(self) -> None:
        """Register all canonical Vetinari contract schemas."""
        self.register_schema("Plan", {
            "required_keys":   ["plan_id", "goal", "status"],
            "non_empty_keys":  ["plan_id", "goal"],
            "key_types":       {"plan_id": "str", "goal": "str",
                                "risk_score": "float"},
            "version_pattern": r"^v\d+\.\d+\.\d+$",
            "allowed_status_values": [
                "draft", "approved", "executing",
                "completed", "failed", "rejected", "cancelled",
            ],
        })
        self.register_schema("Subtask", {
            "required_keys":  ["subtask_id", "plan_id", "description", "status"],
            "non_empty_keys": ["subtask_id", "description"],
            "key_types":      {"subtask_id": "str", "plan_id": "str",
                               "depth": "int"},
        })
        self.register_schema("LogRecord", {
            "required_keys":  ["message", "level", "timestamp"],
            "non_empty_keys": ["message", "level"],
            "key_types":      {"message": "str", "level": "str",
                               "timestamp": "float"},
        })
        self.register_schema("AlertThreshold", {
            "required_keys":  ["name", "metric_key", "condition",
                               "threshold_value", "severity"],
            "non_empty_keys": ["name", "metric_key"],
            "key_types":      {"name": "str", "threshold_value": "float"},
        })
        self.register_schema("CostEntry", {
            "required_keys":  ["provider", "model"],
            "non_empty_keys": ["provider", "model"],
            "key_types":      {"cost_usd": "float", "input_tokens": "int",
                               "output_tokens": "int"},
        })
        self.register_schema("ForecastResult", {
            "required_keys":  ["metric", "method", "horizon", "predictions"],
            "non_empty_keys": ["metric", "method"],
            "key_types":      {"horizon": "int", "rmse": "float"},
        })
        logger.info("Registered %d Vetinari schemas", len(self._schemas))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_schemas": len(self._schemas),
                "schema_names":       list(self._schemas.keys()),
            }

    def clear(self) -> None:
        with self._lock:
            self._schemas.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_schema_validator() -> SchemaValidator:
    return SchemaValidator()


def reset_schema_validator() -> None:
    with SchemaValidator._class_lock:
        SchemaValidator._instance = None
