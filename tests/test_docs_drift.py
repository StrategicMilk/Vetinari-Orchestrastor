"""Direct behavior tests for scripts/docs_drift.py.

Verifies that docs_drift correctly scans the public docs/ tree, extracts
file/module references, validates them against the project root, and rejects
unknown CLI flags.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv312" / "Scripts" / "python.exe"


@pytest.fixture
def drift_mod():
    """Import docs_drift as a module for direct function testing.

    Inserts scripts/ on sys.path, reloads to get a clean state after each
    test, then removes it on teardown to avoid polluting the module namespace.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        mod = importlib.import_module("docs_drift")
        importlib.reload(mod)
        yield mod
    finally:
        sys.path.pop(0)
        sys.modules.pop("docs_drift", None)


class TestFindMarkdownFiles:
    def test_find_markdown_files_includes_public_docs(self, drift_mod, tmp_path):
        """Files under docs/ must appear in the scan results."""
        # Create a minimal project structure in tmp_path
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        audit_dir = docs_dir / "audit"
        audit_dir.mkdir()
        (audit_dir / "MASTER-INDEX.md").write_text("# Master Index\n", encoding="utf-8")
        plans_dir = docs_dir / "plans"
        plans_dir.mkdir()
        (plans_dir / "MASTER-PLAN.md").write_text("# Plan\n", encoding="utf-8")

        with patch.object(drift_mod, "PROJECT_ROOT", tmp_path):
            results = drift_mod.find_markdown_files()

        result_names = {f.name for f in results}
        assert "MASTER-INDEX.md" in result_names, "docs/audit/MASTER-INDEX.md not found"
        assert "MASTER-PLAN.md" in result_names, "docs/plans/MASTER-PLAN.md not found"

    def test_find_markdown_files_includes_rules(self, drift_mod, tmp_path):
        """Files under .claude/rules/ must appear in the scan results."""
        rules_dir = tmp_path / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "style.md").write_text("# Style\n", encoding="utf-8")
        (rules_dir / "testing.md").write_text("# Testing\n", encoding="utf-8")

        with patch.object(drift_mod, "PROJECT_ROOT", tmp_path):
            results = drift_mod.find_markdown_files()

        result_names = {f.name for f in results}
        assert "style.md" in result_names, ".claude/rules/style.md not found"
        assert "testing.md" in result_names, ".claude/rules/testing.md not found"

    def test_find_markdown_files_excludes_nonexistent_dirs(self, drift_mod, tmp_path):
        """Missing optional directories must not cause errors — return empty for that dir."""
        # tmp_path has no docs/ or .claude/ directories at all
        with patch.object(drift_mod, "PROJECT_ROOT", tmp_path):
            results = drift_mod.find_markdown_files()

        # Should not raise; results may be empty or only contain existing root-level files
        assert isinstance(results, list)


class TestExtractFileRefs:
    def test_extract_file_refs_finds_backtick_paths(self, drift_mod):
        """Backtick-quoted paths with file extensions must be extracted."""
        content = "See `scripts/check_vetinari_rules.py` and `config/settings.yaml` for details."
        source_dir = Path("/project")

        refs = drift_mod.extract_file_refs(content, source_dir)

        assert "scripts/check_vetinari_rules.py" in refs
        assert "config/settings.yaml" in refs

    def test_extract_file_refs_skips_skip_patterns(self, drift_mod):
        """Known non-file patterns like threading.Lock must not appear in results."""
        content = "Use `threading.Lock` for thread safety and `pathlib.Path` for paths."
        source_dir = Path("/project")

        refs = drift_mod.extract_file_refs(content, source_dir)

        assert "threading.Lock" not in refs
        assert "pathlib.Path" not in refs

    def test_extract_file_refs_skips_http_urls(self, drift_mod):
        """HTTP URLs must not be treated as file references."""
        content = "See `https://example.com/docs.md` for the external docs."
        source_dir = Path("/project")

        refs = drift_mod.extract_file_refs(content, source_dir)

        assert not any("http" in r for r in refs)


class TestExtractModuleRefs:
    def test_extract_module_refs_finds_vetinari_modules(self, drift_mod):
        """vetinari.X.Y references in backticks must be extracted."""
        content = "Use `vetinari.agents.contracts` for the spec and `vetinari.types` for enums."

        refs = drift_mod.extract_module_refs(content)

        assert "vetinari.agents.contracts" in refs
        assert "vetinari.types" in refs

    def test_extract_module_refs_ignores_bare_vetinari(self, drift_mod):
        """Bare `vetinari` with no submodule must not be extracted (regex requires at least one dot)."""
        content = "The `vetinari` package is imported at startup."

        refs = drift_mod.extract_module_refs(content)

        # Pattern requires vetinari.something — bare vetinari is not a module ref
        assert "vetinari" not in refs

    def test_extract_module_refs_ignores_non_vetinari(self, drift_mod):
        """References to other packages must not be extracted."""
        content = "Use `pathlib.Path` and `os.path.join` in your code."

        refs = drift_mod.extract_module_refs(content)

        assert refs == []


class TestCheckFileRef:
    def test_check_file_ref_existing(self, drift_mod, tmp_path):
        """check_file_ref returns True when the file exists under PROJECT_ROOT."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "my_script.py").write_text("# script\n", encoding="utf-8")

        with patch.object(drift_mod, "PROJECT_ROOT", tmp_path):
            result = drift_mod.check_file_ref("scripts/my_script.py")

        assert result is True

    def test_check_file_ref_missing(self, drift_mod, tmp_path):
        """check_file_ref returns False when the file does not exist."""
        with patch.object(drift_mod, "PROJECT_ROOT", tmp_path):
            result = drift_mod.check_file_ref("scripts/nonexistent_file.py")

        assert result is False

    def test_check_file_ref_dotslash_prefix(self, drift_mod, tmp_path):
        """check_file_ref strips leading ./ when resolving, so ./foo.py finds foo.py."""
        (tmp_path / "foo.py").write_text("# content\n", encoding="utf-8")

        with patch.object(drift_mod, "PROJECT_ROOT", tmp_path):
            result = drift_mod.check_file_ref("./foo.py")

        assert result is True


class TestFixFlagRejected:
    def test_fix_flag_not_accepted(self):
        """Passing --fix must exit with a nonzero code (argparse error), not be silently ignored."""
        python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        script = str(SCRIPTS_DIR / "docs_drift.py")

        proc = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [python, script, "--fix"],
            capture_output=True,
            text=True,
        )

        assert proc.returncode != 0, (
            f"Expected nonzero exit from --fix flag, got 0.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        # argparse writes error messages to stderr
        assert "error" in proc.stderr.lower(), f"Expected argparse error in stderr, got: {proc.stderr!r}"
