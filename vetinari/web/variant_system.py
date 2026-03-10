"""Variant system -- processing depth control inspired by OpenCodes.

Controls how deeply Vetinari analyses tasks: from fast/minimal context (LOW)
through balanced (MEDIUM) to full deep-analysis (HIGH).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class VariantLevel(Enum):
    """Processing depth levels."""

    LOW = "low"  # Fast, minimal context
    MEDIUM = "medium"  # Balanced (default)
    HIGH = "high"  # Deep analysis, full context


@dataclass
class VariantConfig:
    """Configuration for a variant level."""

    level: VariantLevel
    max_context_tokens: int
    max_planning_depth: int
    enable_verification: bool
    enable_self_improvement: bool
    description: str


VARIANT_CONFIGS: Dict[VariantLevel, VariantConfig] = {
    VariantLevel.LOW: VariantConfig(
        level=VariantLevel.LOW,
        max_context_tokens=4096,
        max_planning_depth=2,
        enable_verification=False,
        enable_self_improvement=False,
        description="Fast mode -- minimal context, quick responses",
    ),
    VariantLevel.MEDIUM: VariantConfig(
        level=VariantLevel.MEDIUM,
        max_context_tokens=16384,
        max_planning_depth=5,
        enable_verification=True,
        enable_self_improvement=True,
        description="Balanced -- good context, verification enabled",
    ),
    VariantLevel.HIGH: VariantConfig(
        level=VariantLevel.HIGH,
        max_context_tokens=32768,
        max_planning_depth=10,
        enable_verification=True,
        enable_self_improvement=True,
        description="Deep analysis -- full context, thorough verification",
    ),
}


class VariantManager:
    """Manages variant level selection and configuration."""

    def __init__(self, default_level: str = "medium"):
        self._current = VariantLevel(default_level)

    def get_config(self) -> VariantConfig:
        """Return the configuration for the current variant level."""
        return VARIANT_CONFIGS[self._current]

    def set_level(self, level: str) -> VariantConfig:
        """Switch to a different variant level and return its config."""
        self._current = VariantLevel(level)
        return self.get_config()

    @property
    def current_level(self) -> str:
        """Return the current level as a plain string."""
        return self._current.value

    def get_all_levels(self) -> list:
        """Return metadata for every available level."""
        return [
            {"level": v.level.value, "description": v.description}
            for v in VARIANT_CONFIGS.values()
        ]
