"""Tests for vetinari.config.standards_loader.

Covers StandardsLoader: file loading, YAML parsing, section parsing,
selective context injection, constraints, verification checklists,
quality criteria, defect warnings, caching, and singleton management.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vetinari.config.standards_loader import (
    CONTEXT_RELEVANCE,
    StandardsLoader,
    get_standards_loader,
    reset_standards_loader,
)
from vetinari.types import AgentType

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def standards_dir(tmp_path: Path) -> Path:
    """Create a minimal standards directory for testing."""
    d = tmp_path / "standards"
    d.mkdir()

    # universal.md with two sections
    (d / "universal.md").write_text(
        textwrap.dedent("""\
        # Universal Standards

        ## Core Principles

        Do things correctly. No shortcuts.

        ### Correctness Above All

        Always be correct.

        ## Code Generation Rules

        Use type hints. Use logging.

        ## Import Rules

        Canonical imports only.

        ## Documentation Rules

        Google-style docstrings required.
        """),
        encoding="utf-8",
    )

    # style_guide.md
    (d / "style_guide.md").write_text(
        "# Style Guide\n\nPEP 8 compliance mandatory.\n",
        encoding="utf-8",
    )

    # escalation.md
    (d / "escalation.md").write_text(
        "# Escalation Rules\n\nStop when ambiguous.\n",
        encoding="utf-8",
    )

    # constraints.yaml
    (d / "constraints.yaml").write_text(
        textwrap.dedent("""\
        agents:
          foreman:
            max_tokens: 16384
            timeout_seconds: 180
            max_retries: 3
          worker:
            max_tokens: 32768
            timeout_seconds: 120
            max_retries: 2
        global:
          default_max_tokens: 4096
          default_timeout_seconds: 120
          default_max_retries: 3
        """),
        encoding="utf-8",
    )

    # verification.yaml
    (d / "verification.yaml").write_text(
        textwrap.dedent("""\
        modes:
          build:
            checks:
              - "All functions have type annotations"
              - "No TODO/FIXME/placeholder code"
            auto_fail:
              - "Hallucinated import"
          code_review:
            checks:
              - "Security patterns checked"
            auto_fail:
              - "Critical vulnerability unaddressed"
        """),
        encoding="utf-8",
    )

    # quality_criteria.md
    (d / "quality_criteria.md").write_text(
        textwrap.dedent("""\
        # Quality Criteria

        ## Worker — build mode

        ### Good work looks like:
        - Type hints on all functions

        ### Bad work looks like:
        - Missing type hints
        """),
        encoding="utf-8",
    )

    # defect_catalog.md
    (d / "defect_catalog.md").write_text(
        textwrap.dedent("""\
        # Defect Catalog

        ## Worker — Common Defects

        ### 1. Unwired Features (40%)
        New code never called.

        ### 2. Missing Tests (25%)
        No test coverage.

        ## Inspector — Common Defects

        ### 1. Rubber-Stamp Approval (35%)
        No substantive review.
        """),
        encoding="utf-8",
    )

    return d


@pytest.fixture
def loader(standards_dir: Path) -> StandardsLoader:
    """Create a StandardsLoader pointed at the test standards directory."""
    return StandardsLoader(standards_dir)


# ── File Loading Tests ─────────────────────────────────────────────────


class TestFileLoading:
    """Test raw file loading with caching."""

    def test_read_universal_rules(self, loader: StandardsLoader) -> None:
        content = loader.get_universal_rules()
        assert "Core Principles" in content
        assert "Do things correctly" in content

    def test_read_style_guide(self, loader: StandardsLoader) -> None:
        content = loader.get_style_guide()
        assert "PEP 8" in content

    def test_read_escalation_rules(self, loader: StandardsLoader) -> None:
        content = loader.get_escalation_rules()
        assert "Stop when ambiguous" in content

    def test_read_nonexistent_file_raises(self, loader: StandardsLoader) -> None:
        with pytest.raises(FileNotFoundError, match="Standards file not found"):
            loader._read_file("nonexistent.md")

    def test_mtime_cache_invalidation(self, loader: StandardsLoader, standards_dir: Path) -> None:
        """Verify that modifying a file invalidates the cache."""
        content1 = loader.get_style_guide()
        assert "PEP 8" in content1

        # Modify the file
        (standards_dir / "style_guide.md").write_text(
            "# Updated Style Guide\n\nNew content.\n",
            encoding="utf-8",
        )

        content2 = loader.get_style_guide()
        assert "New content" in content2
        assert content2 != content1


# ── YAML Loading Tests ─────────────────────────────────────────────────


class TestYAMLLoading:
    """Test YAML parsing for constraints and verification."""

    def test_get_constraints_known_agent(self, loader: StandardsLoader) -> None:
        constraints = loader.get_constraints(AgentType.FOREMAN.value)
        assert constraints["max_tokens"] == 16384
        assert constraints["timeout_seconds"] == 180
        assert constraints["max_retries"] == 3

    def test_get_constraints_worker(self, loader: StandardsLoader) -> None:
        constraints = loader.get_constraints(AgentType.WORKER.value)
        assert constraints["max_tokens"] == 32768

    def test_get_constraints_unknown_prefix(self, loader: StandardsLoader) -> None:
        """Unknown agent types fall back to global defaults."""
        constraints = loader.get_constraints("UNKNOWN_TYPE")
        assert constraints["max_tokens"] == 4096  # global default

    def test_get_constraints_unknown_agent(self, loader: StandardsLoader) -> None:
        constraints = loader.get_constraints("UNKNOWN_AGENT")
        assert constraints["max_tokens"] == 4096  # global default

    def test_get_verification_checklist(self, loader: StandardsLoader) -> None:
        checks = loader.get_verification_checklist("build")
        assert len(checks) == 2
        assert "All functions have type annotations" in checks

    def test_get_verification_checklist_unknown_mode(self, loader: StandardsLoader) -> None:
        checks = loader.get_verification_checklist("nonexistent_mode")
        assert checks == []

    def test_get_auto_fail_criteria(self, loader: StandardsLoader) -> None:
        criteria = loader.get_auto_fail_criteria("build")
        assert "Hallucinated import" in criteria

    def test_get_auto_fail_criteria_unknown(self, loader: StandardsLoader) -> None:
        criteria = loader.get_auto_fail_criteria("nonexistent")
        assert criteria == []

    def test_yaml_nonexistent_returns_empty(self, loader: StandardsLoader) -> None:
        """Missing YAML files must return {} (crash-safe) so callers can use defaults."""
        result = loader._read_yaml("nonexistent.yaml")
        assert result == {}, "absent YAML file must produce empty dict, not raise"


# ── Section Parsing Tests ──────────────────────────────────────────────


class TestSectionParsing:
    """Test markdown section extraction."""

    def test_parse_sections_universal(self, loader: StandardsLoader) -> None:
        sections = loader._parse_sections("universal.md")
        assert "core_principles" in sections
        assert "code_generation_rules" in sections
        assert "import_rules" in sections
        assert "documentation_rules" in sections

    def test_section_content(self, loader: StandardsLoader) -> None:
        sections = loader._parse_sections("universal.md")
        assert "Do things correctly" in sections["core_principles"]

    def test_parse_sections_nonexistent(self, loader: StandardsLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader._parse_sections("nonexistent.md")


# ── Quality Criteria Tests ─────────────────────────────────────────────


class TestQualityCriteria:
    """Test quality criteria lookup."""

    def test_get_quality_criteria_worker_build(self, loader: StandardsLoader) -> None:
        criteria = loader.get_quality_criteria(AgentType.WORKER.value, "build")
        assert "Type hints" in criteria

    def test_get_quality_criteria_unknown(self, loader: StandardsLoader) -> None:
        criteria = loader.get_quality_criteria("UNKNOWN", "unknown")
        assert criteria == ""


# ── Defect Warnings Tests ─────────────────────────────────────────────


class TestDefectWarnings:
    """Test defect catalog lookup."""

    def test_get_defect_warnings_worker(self, loader: StandardsLoader) -> None:
        warnings = loader.get_defect_warnings(AgentType.WORKER.value)
        assert len(warnings) == 2
        assert "1. Unwired Features (40%)" in warnings
        assert "2. Missing Tests (25%)" in warnings

    def test_get_defect_warnings_inspector(self, loader: StandardsLoader) -> None:
        warnings = loader.get_defect_warnings(AgentType.INSPECTOR.value)
        assert len(warnings) == 1
        assert "Rubber-Stamp" in warnings[0]

    def test_get_defect_warnings_unknown(self, loader: StandardsLoader) -> None:
        warnings = loader.get_defect_warnings("UNKNOWN_AGENT")
        assert warnings == []


# ── Selective Context Injection Tests ──────────────────────────────────


class TestSelectiveContext:
    """Test mode-aware context injection."""

    def test_worker_build_includes_code_rules(self, loader: StandardsLoader) -> None:
        context = loader.get_context_for_mode(AgentType.WORKER.value, "build")
        assert "Code Generation Rules" in context
        assert "Import Rules" in context

    def test_foreman_plan_excludes_code_rules(self, loader: StandardsLoader) -> None:
        context = loader.get_context_for_mode(AgentType.FOREMAN.value, "plan")
        assert "Code Generation Rules" not in context
        # Core principles should always be present
        assert "Core Principles" in context

    def test_unknown_mode_gets_core_only(self, loader: StandardsLoader) -> None:
        context = loader.get_context_for_mode(AgentType.WORKER.value, "unknown_mode")
        assert "Core Principles" in context
        # Should NOT include code generation rules for unknown mode
        assert "Code Generation Rules" not in context

    def test_worker_architecture_has_core(self, loader: StandardsLoader) -> None:
        """WORKER architecture mode should include core principles."""
        context = loader.get_context_for_mode(AgentType.WORKER.value, "architecture")
        assert "Core Principles" in context

    def test_context_hash_deterministic(self, loader: StandardsLoader) -> None:
        hash1 = loader.get_context_hash(AgentType.WORKER.value, "build")
        hash2 = loader.get_context_hash(AgentType.WORKER.value, "build")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_context_hash_varies_by_mode(self, loader: StandardsLoader) -> None:
        hash_build = loader.get_context_hash(AgentType.WORKER.value, "build")
        hash_plan = loader.get_context_hash(AgentType.FOREMAN.value, "plan")
        assert hash_build != hash_plan


# ── Context Relevance Map Tests ────────────────────────────────────────


class TestContextRelevanceMap:
    """Test the CONTEXT_RELEVANCE map coverage."""

    def test_all_worker_modes_have_entries(self) -> None:
        worker_modes = [k for k in CONTEXT_RELEVANCE if k[0] == AgentType.WORKER.value]
        assert len(worker_modes) >= 2  # build + image_generation

    def test_all_entries_include_core_principles(self) -> None:
        for key, sections in CONTEXT_RELEVANCE.items():
            assert "core_principles" in sections, f"{key} missing core_principles"

    def test_code_generating_modes_include_code_rules(self) -> None:
        code_modes = [(AgentType.WORKER.value, "build"), (AgentType.INSPECTOR.value, "test_generation")]
        for key in code_modes:
            sections = CONTEXT_RELEVANCE.get(key, [])
            assert "code_generation_rules" in sections, f"{key} should include code_generation_rules"


# ── Singleton Tests ────────────────────────────────────────────────────


class TestSingleton:
    """Test singleton management."""

    def test_get_standards_loader_returns_same_instance(self) -> None:
        reset_standards_loader()
        loader1 = get_standards_loader()
        loader2 = get_standards_loader()
        assert loader1 is loader2
        reset_standards_loader()

    def test_reset_clears_singleton(self) -> None:
        reset_standards_loader()
        loader1 = get_standards_loader()
        reset_standards_loader()
        loader2 = get_standards_loader()
        assert loader1 is not loader2
        reset_standards_loader()
