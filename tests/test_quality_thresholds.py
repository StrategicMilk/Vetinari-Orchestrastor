"""Tests for quality threshold loading in vetinari.learning.self_refinement."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from vetinari.learning.self_refinement import get_quality_threshold

# -- Known task types ---------------------------------------------------------


class TestKnownTaskTypes:
    """Thresholds for known task types match the config/quality_thresholds.yaml values."""

    def test_security_audit_returns_0_85(self) -> None:
        """security_audit requires 0.85 quality before refinement stops."""
        threshold = get_quality_threshold("security_audit")
        assert threshold == pytest.approx(0.85)

    def test_documentation_returns_0_65(self) -> None:
        """documentation has a lower bar of 0.65 — human-reviewed anyway."""
        threshold = get_quality_threshold("documentation")
        assert threshold == pytest.approx(0.65)


# -- Default for unknown types ------------------------------------------------


class TestDefaultThreshold:
    """Unknown task types fall back to the configured default (0.70)."""

    def test_unknown_type_returns_default_0_70(self) -> None:
        """A task type not in the YAML falls back to the 'default' key value."""
        threshold = get_quality_threshold("completely_unknown_type_xyz")
        assert threshold == pytest.approx(0.70)


# -- Missing config file ------------------------------------------------------


class TestMissingConfigFile:
    """When the YAML file is absent, the fallback constant is used."""

    def test_missing_file_returns_fallback(self) -> None:
        """get_quality_threshold() returns the module fallback when yaml is unavailable."""
        # Force the except branch by making yaml.safe_load raise
        with patch("yaml.safe_load", side_effect=Exception("mocked yaml failure")):
            threshold = get_quality_threshold("any_type")
        # Must be a valid float in 0..1 range (the module-level fallback)
        assert 0.0 <= threshold <= 1.0
