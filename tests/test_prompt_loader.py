"""Tests for vetinari.agents.prompt_loader and vetinari.agents.practices."""

from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from vetinari.agents.practices import (
    AGENT_CODE_STANDARDS,
    AGENT_PRACTICES,
    MODE_VERIFICATION_REQUIREMENTS,
    format_practices,
    get_code_standards,
    get_verification_requirements,
)
from vetinari.agents.prompt_loader import (
    _agent_type_to_filename,
    _extract_mode_section,
    _extract_sections,
    clear_prompt_cache,
    get_cached_agent_names,
    load_agent_prompt,
)
from vetinari.types import AgentType
from vetinari.utils.frontmatter import FrontmatterError
from vetinari.utils.frontmatter import parse_frontmatter as _parse_frontmatter

# ---------------------------------------------------------------------------
# prompt_loader tests
# ---------------------------------------------------------------------------


class TestAgentTypeToFilename:
    """Test mapping from AgentType values to filenames."""

    def test_simple_type(self) -> None:
        assert _agent_type_to_filename(AgentType.WORKER.value) == "worker"

    def test_foreman(self) -> None:
        assert _agent_type_to_filename(AgentType.FOREMAN.value) == "foreman"

    def test_inspector(self) -> None:
        assert _agent_type_to_filename(AgentType.INSPECTOR.value) == "inspector"


class TestParseFrontmatter:
    """Test YAML frontmatter parsing."""

    def test_with_frontmatter(self) -> None:
        content = "---\nname: builder\nruntime: true\n---\n# Hello\nBody text"
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "builder"
        assert meta["runtime"] is True
        assert "# Hello" in body

    def test_without_frontmatter(self) -> None:
        content = "# No frontmatter\nJust body text"
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_invalid_yaml(self) -> None:
        content = "---\n{invalid: [yaml\n---\nBody"
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert "Body" in body

    def test_invalid_yaml_strict_raises(self) -> None:
        content = "---\n{invalid: [yaml\n---\nBody"
        with pytest.raises(FrontmatterError):
            _parse_frontmatter(content, strict=True)


class TestExtractSections:
    """Test markdown section extraction."""

    def test_multiple_sections(self) -> None:
        body = textwrap.dedent("""\
            # Title

            ## Identity

            You are the Builder.

            ## Modes

            ### build
            Build stuff.

            ## Project Standards

            Standard rules.

            ## Development Notes

            Dev notes here.
        """)
        sections = _extract_sections(body)
        assert "Identity" in sections
        assert "Modes" in sections
        assert "Project Standards" in sections
        assert "Development Notes" in sections
        assert "You are the Builder." in sections["Identity"]

    def test_empty_body(self) -> None:
        sections = _extract_sections("")
        assert sections == {}


class TestExtractModeSection:
    """Test mode subsection extraction."""

    def test_extract_backtick_mode(self) -> None:
        modes_text = textwrap.dedent("""\
            ### `build`
            Build features.

            ### `image_generation`
            Generate images.
        """)
        result = _extract_mode_section(modes_text, "build")
        assert "Build features" in result
        assert "Generate images" not in result

    def test_extract_plain_mode(self) -> None:
        modes_text = textwrap.dedent("""\
            ### code_review
            Review code.

            ### security_audit
            Audit security.
        """)
        result = _extract_mode_section(modes_text, "code_review")
        assert "Review code" in result

    def test_missing_mode(self) -> None:
        modes_text = "### build\nBuild stuff."
        result = _extract_mode_section(modes_text, "nonexistent")
        assert result == ""


class TestLoadAgentPrompt:
    """Test the main load_agent_prompt function."""

    def test_load_worker_prompt(self) -> None:
        """Load worker prompt from per-mode directory or monolithic file."""
        clear_prompt_cache()
        result = load_agent_prompt(AgentType.WORKER.value)
        # Per-mode path loads identity.md directly (no "## Identity" header)
        # Monolithic path wraps in "## Identity" header
        assert "Worker" in result
        assert len(result) > 50

    def test_load_with_mode(self) -> None:
        """Load worker prompt with a specific mode included."""
        clear_prompt_cache()
        result = load_agent_prompt(AgentType.WORKER.value, mode="build")
        # Per-mode path loads mode-build.md content directly
        # Monolithic path wraps in "## Current Mode:" header
        assert "build" in result.lower()
        assert len(result) > 50

    def test_load_foreman(self) -> None:
        """FOREMAN maps to foreman.md."""
        clear_prompt_cache()
        result = load_agent_prompt(AgentType.FOREMAN.value)
        assert len(result) > 50

    def test_load_inspector(self) -> None:
        """INSPECTOR maps to inspector.md."""
        clear_prompt_cache()
        result = load_agent_prompt(AgentType.INSPECTOR.value)
        assert len(result) > 50

    def test_missing_agent_file(self) -> None:
        """Missing file returns empty string gracefully."""
        clear_prompt_cache()
        result = load_agent_prompt("NONEXISTENT_AGENT_TYPE")
        assert result == ""

    def test_caching(self) -> None:
        """Second call uses cache (same mtime)."""
        clear_prompt_cache()
        result1 = load_agent_prompt(AgentType.WORKER.value)
        assert "worker" in get_cached_agent_names()
        result2 = load_agent_prompt(AgentType.WORKER.value)
        assert result1 == result2

    def test_accepts_enum(self) -> None:
        """Accepts AgentType enum values."""
        from vetinari.types import AgentType

        clear_prompt_cache()
        result = load_agent_prompt(AgentType.WORKER)
        assert len(result) > 50

    def test_runtime_false_skipped(self) -> None:
        """Files with runtime: false in frontmatter are skipped."""
        clear_prompt_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_file = Path(tmpdir) / "test_agent.md"
            agent_file.write_text(
                "---\nname: test\nruntime: false\n---\n## Identity\nSkipped.",
                encoding="utf-8",
            )
            with patch(
                "vetinari.agents.prompt_loader._AGENTS_DIR",
                Path(tmpdir),
            ):
                result = load_agent_prompt("TEST_AGENT")
                assert result == ""

    def test_malformed_frontmatter_fails_closed(self) -> None:
        clear_prompt_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_file = Path(tmpdir) / "test_agent.md"
            agent_file.write_text(
                "---\nname: test\nruntime: [broken\n---\n## Identity\nShould not load.",
                encoding="utf-8",
            )
            with patch(
                "vetinari.agents.prompt_loader._AGENTS_DIR",
                Path(tmpdir),
            ):
                result = load_agent_prompt("TEST_AGENT")
                assert result == ""

    def test_invalid_mode_name_rejected_before_file_resolution(self) -> None:
        clear_prompt_cache()
        with pytest.raises(ValueError, match="Invalid prompt mode name"):
            load_agent_prompt(AgentType.WORKER.value, mode="../worker")

    def test_clear_cache(self) -> None:
        """clear_prompt_cache empties the cache."""
        load_agent_prompt(AgentType.WORKER.value)
        assert len(get_cached_agent_names()) > 0
        clear_prompt_cache()
        assert len(get_cached_agent_names()) == 0


# ---------------------------------------------------------------------------
# practices tests
# ---------------------------------------------------------------------------


class TestAgentPractices:
    """Test AGENT_PRACTICES constant and format_practices."""

    def test_practices_has_10_entries(self) -> None:
        assert len(AGENT_PRACTICES) == 10

    def test_practices_keys(self) -> None:
        expected_keys = {
            "plan_before_act",
            "explore_before_modify",
            "verify_before_report",
            "context_discipline",
            "evidence_over_assumption",
            "escalate_uncertainty",
            "minimal_scope",
            "delegation_depth",
            "checkpoint_frequently",
            "fail_informatively",
        }
        assert set(AGENT_PRACTICES.keys()) == expected_keys

    def test_practices_no_claude_specific_language(self) -> None:
        """Practices must be model-agnostic."""
        for key, value in AGENT_PRACTICES.items():
            lower = value.lower()
            assert "claude" not in lower, f"Practice {key} mentions 'claude'"
            assert "anthropic" not in lower, f"Practice {key} mentions 'anthropic'"
            assert "lm studio" not in lower, f"Practice {key} mentions 'lm studio'"

    def test_format_practices_output(self) -> None:
        result = format_practices()
        assert "## Agent Best Practices" in result
        assert "PLAN BEFORE ACT" in result
        assert "FAIL INFORMATIVELY" in result
        assert len(result.split("\n")) >= 11  # header + 10 practices


class TestCodeStandards:
    """Test AGENT_CODE_STANDARDS and get_code_standards."""

    def test_code_standards_not_empty(self) -> None:
        assert len(AGENT_CODE_STANDARDS) > 100

    def test_get_code_standards_build_mode(self) -> None:
        result = get_code_standards("build")
        assert "Code Generation Standards" in result

    def test_get_code_standards_test_generation_mode(self) -> None:
        result = get_code_standards("test_generation")
        assert "Code Generation Standards" in result

    def test_get_code_standards_non_code_mode(self) -> None:
        result = get_code_standards("code_review")
        assert result == ""

    def test_get_code_standards_none_mode(self) -> None:
        result = get_code_standards(None)
        assert result == ""


class TestVerificationRequirements:
    """Test MODE_VERIFICATION_REQUIREMENTS and get_verification_requirements."""

    def test_worker_build_requirements(self) -> None:
        reqs = get_verification_requirements(AgentType.WORKER.value, "build")
        assert "pytest_passes" in reqs
        assert "type_hints_present" in reqs

    def test_inspector_code_review_requirements(self) -> None:
        reqs = get_verification_requirements(AgentType.INSPECTOR.value, "code_review")
        assert "findings_have_file_line_refs" in reqs
        assert "gate_decision_binary" in reqs

    def test_worker_architecture_requirements(self) -> None:
        reqs = get_verification_requirements(AgentType.WORKER.value, "architecture")
        assert "min_3_alternatives_evaluated" in reqs
        assert "adr_created" in reqs

    def test_unknown_mode_returns_empty(self) -> None:
        reqs = get_verification_requirements(AgentType.WORKER.value, "nonexistent_mode")
        assert reqs == []

    def test_returns_copy_not_reference(self) -> None:
        """Ensure returned list is a copy, not a reference to the original."""
        reqs = get_verification_requirements(AgentType.WORKER.value, "build")
        reqs.append("extra_item")
        original = MODE_VERIFICATION_REQUIREMENTS["WORKER:build"]
        assert "extra_item" not in original


class TestBaseAgentPracticesIntegration:
    """Test that practices are injected into BaseAgent's prompt framework."""

    def test_build_system_prompt_includes_practices(self) -> None:
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        prompt = agent._build_system_prompt(mode="build")
        # Should include Tier 1 core principles
        assert "Correctness above all" in prompt
        # Should include mode-relevant practices
        assert "Agent Best Practices" in prompt
        assert "PLAN BEFORE ACT" in prompt

    def test_build_system_prompt_includes_mode_practices(self) -> None:
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        prompt = agent._build_system_prompt(mode="build")
        # Should include Tier 1 core principles
        assert "Correctness above all" in prompt
        # Should include mode-relevant practices (not all 10)
        assert "Agent Best Practices" in prompt


class TestMultiModeAgentPromptLoading:
    """Test that MultiModeAgent uses prompt_loader."""

    def test_builder_loads_from_markdown(self) -> None:
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        prompt = agent.get_system_prompt()
        # Should load from vetinari/config/agents/worker.md — contains "Identity"
        assert "Identity" in prompt or "Worker" in prompt or "Builder" in prompt


class TestPlannerValidateAgentOutput:
    """Test PlannerAgent.validate_agent_output."""

    def test_passing_validation(self) -> None:
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        output = {
            "verification": {
                "pytest_passes": True,
                "type_hints_present": True,
                "no_print_statements": True,
                "imports_canonical": True,
            }
        }
        passed, unmet = planner.validate_agent_output(AgentType.WORKER.value, "build", output)
        assert passed is True
        assert unmet == []

    def test_failing_validation(self) -> None:
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        output = {
            "verification": {
                "pytest_passes": True,
                "type_hints_present": False,
                "no_print_statements": True,
            }
        }
        passed, unmet = planner.validate_agent_output(AgentType.WORKER.value, "build", output)
        assert passed is False
        assert "type_hints_present" in unmet
        assert "imports_canonical" in unmet

    def test_none_output_fails(self) -> None:
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        passed, unmet = planner.validate_agent_output(AgentType.WORKER.value, "build", None)
        assert passed is False
        assert len(unmet) > 0

    def test_unknown_mode_passes(self) -> None:
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        passed, unmet = planner.validate_agent_output(AgentType.WORKER.value, "nonexistent", {})
        assert passed is True
        assert unmet == []


# ---------------------------------------------------------------------------
# Cross-model validation tests
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Test BaseAgent._cross_validate."""

    def test_skips_non_critical_modes(self) -> None:
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        result = agent._cross_validate("output", "prompt", mode="build")
        assert result["validated"] is False
        assert result["skipped"] is True
        assert result["agreement"] == 1.0
        assert "not required" in result["notes"]

    def test_critical_mode_triggers_validation(self) -> None:
        """architecture mode triggers cross-validation — fails closed without adapter."""
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        # Without an adapter_manager, cross-validation must fail closed
        # (validated=False) — never silently approve un-validated output
        result = agent._cross_validate("output", "prompt", mode="architecture")
        assert result["validated"] is False
        assert result["agreement"] == 0.0
        assert "model_used" in result

    def test_explicit_cross_validate_flag(self) -> None:
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        agent._context = {"cross_validate": True}
        result = agent._cross_validate("output", "prompt", mode="build")
        # Should attempt validation (graceful fallback without adapter)
        assert "validated" in result


# ---------------------------------------------------------------------------
# Prompt versioning tests
# ---------------------------------------------------------------------------


class TestPromptVersionManager:
    """Test PromptVersionManager."""

    def test_save_and_retrieve(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        v = mgr.save_version(AgentType.WORKER.value, "build", "test prompt text")
        assert v.version == "1.0.0"
        assert v.prompt_text == "test prompt text"

        history = mgr.get_history(AgentType.WORKER.value, "build")
        assert len(history) == 1
        assert history[0].version == "1.0.0"

    def test_increments_version(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        mgr.save_version(AgentType.WORKER.value, "build", "v1 text")
        v2 = mgr.save_version(AgentType.WORKER.value, "build", "v2 text")
        assert v2.version == "1.0.1"

    def test_skips_duplicate(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        mgr.save_version(AgentType.WORKER.value, "build", "same text")
        mgr.save_version(AgentType.WORKER.value, "build", "same text")  # duplicate
        history = mgr.get_history(AgentType.WORKER.value, "build")
        assert len(history) == 1

    def test_rollback(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        mgr.save_version(AgentType.WORKER.value, "build", "original")
        mgr.save_version(AgentType.WORKER.value, "build", "updated")

        result = mgr.rollback(AgentType.WORKER.value, "build", "1.0.0")
        assert result is not None
        assert result.prompt_text == "original"
        assert result.version == "1.0.2"  # new version after rollback

    def test_rollback_nonexistent_version(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        mgr.save_version(AgentType.WORKER.value, "build", "text")
        result = mgr.rollback(AgentType.WORKER.value, "build", "99.0.0")
        assert result is None

    def test_get_latest(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        assert mgr.get_latest(AgentType.WORKER.value, "build") is None
        mgr.save_version(AgentType.WORKER.value, "build", "v1")
        mgr.save_version(AgentType.WORKER.value, "build", "v2")
        latest = mgr.get_latest(AgentType.WORKER.value, "build")
        assert latest is not None
        assert latest.prompt_text == "v2"

    def test_auto_rollback_on_regression(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        mgr.save_version(AgentType.INSPECTOR.value, "code_review", "v1 prompt", quality_score=0.9)
        mgr.save_version(AgentType.INSPECTOR.value, "code_review", "v2 prompt", quality_score=0.85)

        # No rollback — regression within threshold
        result = mgr.auto_rollback_on_regression(AgentType.INSPECTOR.value, "code_review", 0.80, threshold=0.1)
        assert result is None

        # Rollback — regression exceeds threshold
        result = mgr.auto_rollback_on_regression(AgentType.INSPECTOR.value, "code_review", 0.70, threshold=0.1)
        assert result is not None
        assert result.prompt_text == "v1 prompt"

    def test_quality_score_tracking(self, tmp_path: Path) -> None:
        from vetinari.prompts import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        v = mgr.save_version(AgentType.WORKER.value, "build", "text", quality_score=0.95)
        assert v.quality_score == 0.95
