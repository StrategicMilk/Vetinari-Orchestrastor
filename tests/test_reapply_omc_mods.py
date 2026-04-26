"""Tests for scripts/reapply_omc_mods.py.

Loads the script as a module via importlib so it can be exercised without
installing it as a package. The source lives in scripts/ not vetinari/,
so VET124 (no matching vetinari source module) does not apply here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_module():
    """Load reapply_omc_mods.py as an importable module."""
    module_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "reapply_omc_mods.py"
    )
    spec = importlib.util.spec_from_file_location("reapply_omc_mods", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
MODEL_ASSIGNMENTS = _mod.MODEL_ASSIGNMENTS
apply_model_assignments = _mod.apply_model_assignments
find_agents_dir = _mod.find_agents_dir


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_agent(
    tmp_path: Path,
    name: str,
    model: str,
    description: str = "An agent",
) -> Path:
    """Write a minimal agent markdown file with YAML frontmatter.

    Args:
        tmp_path: Directory in which to write the file.
        name: Filename stem (file will be ``{name}.md``).
        model: Model ID to embed in the ``model:`` frontmatter field.
        description: Agent description string.

    Returns:
        Path to the created file.
    """
    content = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"model: {model}\n"
        "level: 3\n"
        "---\n"
        "Body text here.\n"
    )
    agent_file = tmp_path / f"{name}.md"
    agent_file.write_text(content, encoding="utf-8")
    return agent_file


# ---------------------------------------------------------------------------
# TestModelAssignments
# ---------------------------------------------------------------------------

class TestModelAssignments:
    """Verify MODEL_ASSIGNMENTS contains the expected tier assignments."""

    def test_opus_4_7_agents_present(self) -> None:
        """architect and tracer must be assigned claude-opus-4-7."""
        assert MODEL_ASSIGNMENTS["architect"] == "claude-opus-4-7"
        assert MODEL_ASSIGNMENTS["tracer"] == "claude-opus-4-7"

    def test_opus_4_6_frozen_agents_present(self) -> None:
        """analyst, critic, planner must be assigned claude-opus-4-6."""
        assert MODEL_ASSIGNMENTS["analyst"] == "claude-opus-4-6"
        assert MODEL_ASSIGNMENTS["critic"] == "claude-opus-4-6"
        assert MODEL_ASSIGNMENTS["planner"] == "claude-opus-4-6"

    def test_sonnet_agents_present(self) -> None:
        """code-simplifier, code-reviewer, security-reviewer must be claude-sonnet-4-6."""
        assert MODEL_ASSIGNMENTS["code-simplifier"] == "claude-sonnet-4-6"
        assert MODEL_ASSIGNMENTS["code-reviewer"] == "claude-sonnet-4-6"
        assert MODEL_ASSIGNMENTS["security-reviewer"] == "claude-sonnet-4-6"

    def test_total_count(self) -> None:
        """MODEL_ASSIGNMENTS must contain exactly 8 entries."""
        assert len(MODEL_ASSIGNMENTS) == 8


# ---------------------------------------------------------------------------
# TestApplyModelAssignments
# ---------------------------------------------------------------------------

class TestApplyModelAssignments:
    """Behavioural tests for apply_model_assignments."""

    def test_wrong_model_is_corrected(self, tmp_path: Path) -> None:
        """architect.md with the wrong model is rewritten and the change is reported."""
        _make_agent(tmp_path, "architect", "claude-opus-4-6")

        changes = apply_model_assignments(tmp_path)

        assert changes == [("architect", "claude-opus-4-6", "claude-opus-4-7")]
        updated = (tmp_path / "architect.md").read_text(encoding="utf-8")
        assert "model: claude-opus-4-7" in updated

    def test_already_correct_not_reported(self, tmp_path: Path) -> None:
        """architect.md already at claude-opus-4-7 produces an empty changes list."""
        _make_agent(tmp_path, "architect", "claude-opus-4-7")

        changes = apply_model_assignments(tmp_path)

        assert changes == []

    def test_multiple_agents_updated(self, tmp_path: Path) -> None:
        """All wrong agents are updated and reported in one call."""
        _make_agent(tmp_path, "architect", "claude-haiku-3-5")
        _make_agent(tmp_path, "analyst", "claude-haiku-3-5")
        _make_agent(tmp_path, "code-simplifier", "claude-haiku-3-5")

        changes = apply_model_assignments(tmp_path)

        changed_names = {c[0] for c in changes}
        assert changed_names == {"architect", "analyst", "code-simplifier"}

        for _name, old, _new in changes:
            assert old == "claude-haiku-3-5"

        assert (tmp_path / "architect.md").read_text(encoding="utf-8").count(
            "model: claude-opus-4-7"
        ) == 1
        assert (tmp_path / "analyst.md").read_text(encoding="utf-8").count(
            "model: claude-opus-4-6"
        ) == 1
        assert (tmp_path / "code-simplifier.md").read_text(encoding="utf-8").count(
            "model: claude-sonnet-4-6"
        ) == 1

    def test_missing_agent_file_skipped(self, tmp_path: Path) -> None:
        """apply_model_assignments does not raise when agent files are absent."""
        # Only put architect.md in the directory; all others are absent.
        _make_agent(tmp_path, "architect", "claude-haiku-3-5")

        # Must not raise even though 7 out of 8 expected files are missing.
        changes = apply_model_assignments(tmp_path)

        assert len(changes) == 1
        assert changes[0][0] == "architect"

    def test_file_content_preserved_outside_frontmatter(self, tmp_path: Path) -> None:
        """Body text and description are unchanged after model line replacement."""
        _make_agent(tmp_path, "architect", "claude-opus-4-6", description="Design agent")

        apply_model_assignments(tmp_path)

        content = (tmp_path / "architect.md").read_text(encoding="utf-8")
        assert "Body text here." in content
        assert "description: Design agent" in content
        # Frontmatter delimiters must still be present
        assert content.startswith("---\n")
        assert "\n---\n" in content

    def test_dry_run_makes_no_changes(self, tmp_path: Path) -> None:
        """dry_run=True reports changes but leaves files on disk unmodified."""
        _make_agent(tmp_path, "architect", "claude-opus-4-6")
        original = (tmp_path / "architect.md").read_text(encoding="utf-8")

        changes = apply_model_assignments(tmp_path, dry_run=True)

        # Change is still reported
        assert changes == [("architect", "claude-opus-4-6", "claude-opus-4-7")]
        # But the file is unchanged on disk
        assert (tmp_path / "architect.md").read_text(encoding="utf-8") == original

    def test_return_type_is_list_of_tuples(self, tmp_path: Path) -> None:
        """Each element of the return value is a 3-tuple of strings."""
        _make_agent(tmp_path, "architect", "claude-opus-4-6")

        changes = apply_model_assignments(tmp_path)

        assert isinstance(changes, list)
        assert len(changes) == 1
        item = changes[0]
        assert isinstance(item, tuple)
        assert len(item) == 3
        assert all(isinstance(s, str) for s in item)


# ---------------------------------------------------------------------------
# TestFindAgentsDir
# ---------------------------------------------------------------------------

class TestFindAgentsDir:
    """Tests for find_agents_dir path resolution."""

    def test_base_path_returned_directly(self, tmp_path: Path) -> None:
        """When base_path is provided it is returned without any filesystem glob."""
        result = find_agents_dir(base_path=tmp_path)
        assert result == tmp_path

    def test_returns_none_when_no_omc_cache(self, tmp_path: Path, monkeypatch) -> None:
        """When the OMC cache directory does not exist, None is returned."""
        # Point home to an empty temp dir so the glob finds nothing.
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        result = find_agents_dir()
        assert result is None
