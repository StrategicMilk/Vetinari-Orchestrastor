"""Tests for YAML-based runtime configuration of agent practices."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vetinari.agents.practices import (
    AGENT_PRACTICES,
    get_practices_for_mode,
    load_practices,
    reset_practices_cache,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the practices cache before and after every test."""
    reset_practices_cache()
    yield
    reset_practices_cache()


class TestLoadPracticesFromYaml:
    def test_load_practices_from_yaml(self, tmp_path: Path) -> None:
        """load_practices() reads label+text entries from a valid YAML file."""
        yaml_content = textwrap.dedent("""\
            practices:
              plan_before_act:
                label: "Plan Before Act"
                text: "Custom plan text."
              verify_before_report:
                label: "Verify Before Report"
                text: "Custom verify text."
        """)
        config_file = tmp_path / "practices.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        result = load_practices(config_path=config_file)

        assert result["plan_before_act"] == "Custom plan text."
        assert result["verify_before_report"] == "Custom verify text."
        # Only the two keys defined in YAML are present (not the full 10)
        assert set(result.keys()) == {"plan_before_act", "verify_before_report"}

    def test_load_practices_cache_is_reused(self, tmp_path: Path) -> None:
        """Second call returns the cached dict without re-reading the file."""
        yaml_content = textwrap.dedent("""\
            practices:
              minimal_scope:
                label: "Minimal Scope"
                text: "Cached text."
        """)
        config_file = tmp_path / "practices.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        first = load_practices(config_path=config_file)
        # Delete the file — cache should still serve the result
        config_file.unlink()
        second = load_practices(config_path=config_file)

        assert first is second


class TestLoadPracticesFallback:
    def test_load_practices_fallback_when_missing(self, tmp_path: Path) -> None:
        """load_practices() returns hardcoded AGENT_PRACTICES when YAML is absent."""
        missing = tmp_path / "nonexistent.yaml"

        result = load_practices(config_path=missing)

        assert result == AGENT_PRACTICES

    def test_load_practices_fallback_on_invalid_yaml(self, tmp_path: Path) -> None:
        """load_practices() falls back to defaults when YAML cannot be parsed."""
        config_file = tmp_path / "practices.yaml"
        config_file.write_text(":: this is not valid yaml ::", encoding="utf-8")

        result = load_practices(config_path=config_file)

        assert result == AGENT_PRACTICES

    def test_load_practices_fallback_on_wrong_structure(self, tmp_path: Path) -> None:
        """load_practices() falls back when YAML is valid but has wrong top-level key."""
        config_file = tmp_path / "practices.yaml"
        config_file.write_text("not_practices:\n  key: value\n", encoding="utf-8")

        result = load_practices(config_path=config_file)

        assert result == AGENT_PRACTICES

    def test_load_practices_fallback_on_empty_practices(self, tmp_path: Path) -> None:
        """load_practices() falls back when all entries are malformed."""
        yaml_content = textwrap.dedent("""\
            practices:
              bad_entry:
                label: "No text field here"
        """)
        config_file = tmp_path / "practices.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        result = load_practices(config_path=config_file)

        assert result == AGENT_PRACTICES


class TestGetPracticesForModeUsesLoadedConfig:
    def test_get_practices_for_mode_uses_loaded_config(self, tmp_path: Path) -> None:
        """get_practices_for_mode() uses custom YAML text when practices are overridden."""
        # Patch the default path via load_practices first, then call get_practices_for_mode.
        yaml_content = textwrap.dedent("""\
            practices:
              plan_before_act:
                label: "Plan Before Act"
                text: "Overridden plan text."
              verify_before_report:
                label: "Verify Before Report"
                text: "Overridden verify text."
              evidence_over_assumption:
                label: "Evidence Over Assumption"
                text: "Overridden evidence text."
              explore_before_modify:
                label: "Explore Before Modify"
                text: "Overridden explore text."
              minimal_scope:
                label: "Minimal Scope"
                text: "Overridden minimal text."
              context_discipline:
                label: "Context Discipline"
                text: "Overridden context text."
              escalate_uncertainty:
                label: "Escalate Uncertainty"
                text: "Overridden escalate text."
              delegation_depth:
                label: "Delegation Depth"
                text: "Overridden delegation text."
              checkpoint_frequently:
                label: "Checkpoint Frequently"
                text: "Overridden checkpoint text."
              fail_informatively:
                label: "Fail Informatively"
                text: "Overridden fail text."
        """)
        config_file = tmp_path / "practices.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        # Pre-warm the cache with the custom YAML
        load_practices(config_path=config_file)

        # "build" mode uses plan_before_act, verify_before_report,
        # evidence_over_assumption, minimal_scope
        result = get_practices_for_mode("build")

        assert "Overridden plan text." in result
        assert "Overridden verify text." in result
        assert "Overridden evidence text." in result
        assert "Overridden minimal text." in result

    def test_get_practices_for_mode_unknown_mode_returns_defaults(self) -> None:
        """get_practices_for_mode() returns minimal default for unknown modes."""
        result = get_practices_for_mode("totally_unknown_mode_xyz")

        assert result  # non-empty
        assert "## Agent Best Practices" in result

    def test_get_practices_for_mode_returns_subset(self) -> None:
        """get_practices_for_mode() returns only the relevant subset, not all 10."""
        # "clarify" mode maps to: evidence_over_assumption, escalate_uncertainty
        result = get_practices_for_mode("clarify")

        # Should NOT include plan_before_act (not in clarify mapping)
        assert "PLAN BEFORE ACT" not in result
        # Should include at least one of the mapped practices
        assert result
