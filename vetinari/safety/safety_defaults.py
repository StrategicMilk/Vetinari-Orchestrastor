"""Safety defaults loader — reads config/safety_defaults.yaml once per process.

Exposes ``load_safety_defaults()`` which returns a frozen ``SafetyDefaults``
dataclass consumed by ``RecycleStore`` and ``ArchiveStore`` when no explicit
arguments are passed.  The YAML is read at most once thanks to
``functools.lru_cache``; the cache can be cleared in tests via
``load_safety_defaults.cache_clear()``.

Fail-loud contract: if the YAML is absent or malformed the function raises
``ConfigurationError`` rather than silently using compiled-in fallback values.
Lifecycle stores depend on this config, and silent fallback hides
misconfigured deployments until something goes wrong at delete time.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from vetinari.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Project root is two levels above this file: vetinari/safety/safety_defaults.py
# -> vetinari/ -> <project_root>
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Canonical config location.  Tests can override _SAFETY_DEFAULTS_PATH before
# calling load_safety_defaults() then clear the cache to test alternate YAMLs.
_SAFETY_DEFAULTS_PATH: Path = _PROJECT_ROOT / "config" / "safety_defaults.yaml"


@dataclass(frozen=True)
class SafetyDefaults:
    """Frozen snapshot of safety_defaults.yaml values.

    Attributes:
        recycle_root: Root directory for recycle-bin records.
        grace_hours: How many hours a recycled entity stays restorable.
        archive_root: Root directory for archive records.
        recent_days: Records younger than this appear in the ``"recent"`` tier.
        cooling_days: Records older than ``recent_days`` but younger than this
            are in the ``"cooling"`` tier; older records are ``"cold"``.
    """

    recycle_root: Path
    grace_hours: int
    archive_root: Path
    recent_days: int
    cooling_days: int

    def __repr__(self) -> str:
        """Show the key threshold values and roots for debugging."""
        return (
            f"SafetyDefaults("
            f"grace_hours={self.grace_hours}, "
            f"recycle_root={self.recycle_root!r}, "
            f"recent_days={self.recent_days}, "
            f"cooling_days={self.cooling_days}, "
            f"archive_root={self.archive_root!r})"
        )


def _resolve_path(raw: str) -> Path:
    """Resolve a raw config path string against the project root.

    Relative paths are resolved against the project root.  Absolute paths are
    returned as-is.

    Args:
        raw: Path string from the YAML file.

    Returns:
        Resolved absolute ``Path``.
    """
    p = Path(raw)
    if not p.is_absolute():
        return (_PROJECT_ROOT / p).resolve()
    return p.resolve()


@functools.lru_cache(maxsize=1)
def load_safety_defaults() -> SafetyDefaults:
    """Load and cache safety defaults from ``config/safety_defaults.yaml``.

    The YAML is read at most once per process; subsequent calls return the
    cached result.  Call ``load_safety_defaults.cache_clear()`` in tests to
    force a re-read.

    Returns:
        Frozen ``SafetyDefaults`` dataclass with validated fields.

    Raises:
        ConfigurationError: If the YAML file does not exist, cannot be parsed,
            or is missing required fields.
    """
    # Allow tests to monkeypatch the module-level variable before cache hit.
    path = _SAFETY_DEFAULTS_PATH

    if not path.exists():
        raise ConfigurationError(
            f"Safety defaults config not found at {path} — "
            "ensure config/safety_defaults.yaml exists in the project root. "
            "This file is required for RecycleStore and ArchiveStore to initialise."
        )

    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"config/safety_defaults.yaml is malformed and could not be parsed — "
            f"fix the YAML syntax before starting the server: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"config/safety_defaults.yaml must be a YAML mapping at the top level, got {type(data).__name__!r}"
        )

    try:
        rp = data["recycle_policy"]
        ap = data["archive_policy"]
        return SafetyDefaults(
            recycle_root=_resolve_path(rp["recycle_root"]),
            grace_hours=int(rp["grace_hours"]),
            archive_root=_resolve_path(ap["archive_root"]),
            recent_days=int(ap["recent_days"]),
            cooling_days=int(ap["cooling_days"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"config/safety_defaults.yaml is missing required fields or has invalid values — "
            f"required: recycle_policy.{{grace_hours, recycle_root}}, "
            f"archive_policy.{{recent_days, cooling_days, archive_root}}. "
            f"Error: {exc}"
        ) from exc


__all__ = ["SafetyDefaults", "load_safety_defaults"]
