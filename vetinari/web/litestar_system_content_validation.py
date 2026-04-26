"""Validation and filesystem helpers for Litestar system-content routes."""

from __future__ import annotations

import contextlib
import math
import os
import pathlib
import re
import unicodedata
from typing import Any

_UNSAFE_FILENAME_RE = re.compile(r"[^\w.\-]", re.ASCII)


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    """Write text to *path* using a same-directory temporary replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _secure_filename(filename: str) -> str:
    """Sanitise a filename for safe filesystem storage.

    Normalises unicode, strips path separators and leading dots,
    and replaces non-alphanumeric characters with underscores.
    Equivalent to werkzeug.utils.secure_filename (removed with Flask).
    """
    # Normalise unicode characters to ASCII decomposition
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    # Replace path separators with space so they become underscores
    for sep in ("/", "\\"):
        filename = filename.replace(sep, " ")

    # Keep only safe characters, collapse whitespace, strip dots/spaces
    filename = _UNSAFE_FILENAME_RE.sub("_", filename).strip("._")

    return filename or "unnamed"


_SETTINGS_VALID_KEYS: frozenset[str] = frozenset({"inference", "agent_timeouts", "log_level"})
_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_INFERENCE_INT_RANGES: dict[str, tuple[int, int]] = {
    "gpu_layers": (-1, 10_000),
    "context_length": (1, 1_048_576),
    "batch_size": (1, 262_144),
    "n_threads": (0, 1024),
    "max_agent_retries": (0, 100),
}
_INFERENCE_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "inference_timeout": (0.001, 86_400.0),
}
_INFERENCE_BOOL_KEYS: frozenset[str] = frozenset({"flash_attn"})
_INFERENCE_PATH_KEYS: frozenset[str] = frozenset({"models_dir", "local_models_dir", "default_model"})


def _validate_settings_update(data: dict[str, Any]) -> list[str]:
    """Return validation errors for operator settings updates."""
    errors: list[str] = []

    inference = data.get("inference")
    if isinstance(inference, dict):
        for key, value in inference.items():
            if key in _INFERENCE_INT_RANGES:
                low, high = _INFERENCE_INT_RANGES[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    errors.append(f"'inference.{key}' must be an integer")
                    continue
                if not low <= value <= high:
                    errors.append(f"'inference.{key}' must be between {low} and {high}")
            elif key in _INFERENCE_FLOAT_RANGES:
                low, high = _INFERENCE_FLOAT_RANGES[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    errors.append(f"'inference.{key}' must be a finite number")
                    continue
                if not low <= float(value) <= high:
                    errors.append(f"'inference.{key}' must be between {low} and {high}")
            elif key in _INFERENCE_BOOL_KEYS:
                if not isinstance(value, bool):
                    errors.append(f"'inference.{key}' must be a boolean")
            elif key in _INFERENCE_PATH_KEYS:
                if not isinstance(value, str) or not value.strip() or "\x00" in value:
                    errors.append(f"'inference.{key}' must be a non-empty path string")

    timeouts = data.get("agent_timeouts")
    if isinstance(timeouts, dict):
        for key, value in timeouts.items():
            if not isinstance(key, str) or not key.strip():
                errors.append("'agent_timeouts' keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"'agent_timeouts.{key}' must be a finite number")
            elif float(value) <= 0:
                errors.append(f"'agent_timeouts.{key}' must be greater than 0")

    log_level = data.get("log_level")
    if isinstance(log_level, str) and log_level.strip().upper() not in _LOG_LEVELS:
        errors.append(f"'log_level' must be one of {', '.join(sorted(_LOG_LEVELS))}")

    return errors
