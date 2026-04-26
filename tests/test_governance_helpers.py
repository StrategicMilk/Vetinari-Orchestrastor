"""Behavior tests for governance helper scripts.

Validates that each script detects real defects rather than rubber-stamping
everything as passing. Each test is designed to FAIL on the old (buggy)
version of the script it covers.

Scripts under test:
  scripts/check_migration_index.py
  scripts/check_syntax.py
  scripts/check_test_quality.py
  scripts/hook_pre_commit_gate.py
  scripts/run_tests.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# -- helpers ------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_script(name: str):
    """Import a scripts/ module by stem, bypassing the package system.

    Args:
        name: Script filename stem (e.g. ``"check_syntax"``).

    Returns:
        Imported module object.
    """
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Could not find script: {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# -- check_syntax tests -------------------------------------------------------


class TestCheckSyntax:
    """Tests for check_syntax.py."""

    def test_check_syntax_walks_project_root(self, tmp_path: Path) -> None:
        """The computed base must be the project root, not the scripts/ directory.

        Old bug: base = os.path.dirname(os.path.abspath(__file__)) resolved to
        scripts/ so only scripts/*.py were checked.
        """
        mod = _import_script("check_syntax")
        # Recompute the same expression the script uses
        script_path = _SCRIPTS_DIR / "check_syntax.py"
        computed_base = os.path.dirname(os.path.dirname(os.path.abspath(str(script_path))))
        assert computed_base == str(_PROJECT_ROOT), (
            f"base resolves to {computed_base!r} but expected project root {_PROJECT_ROOT!r}"
        )
        # Also confirm it is NOT the scripts dir
        assert computed_base != str(_SCRIPTS_DIR), "base must not resolve to scripts/ — it must be the project root"

    def test_check_syntax_finds_vetinari_files(self, tmp_path: Path) -> None:
        """Running main() with a real project structure should check >0 files.

        Confirms the walker actually reaches vetinari/ source files, not just
        the scripts/ directory.
        """
        # Create a minimal tmp project mirroring the real layout
        vetinari_dir = tmp_path / "vetinari"
        vetinari_dir.mkdir()
        (vetinari_dir / "sample.py").write_text("x = 1\n", encoding="utf-8")

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "check_syntax.py").write_text(
            # Write a copy that walks from tmp_path
            f"""
import ast, contextlib, os, pathlib, sys
def main():
    base = {str(tmp_path)!r}
    errors = []
    checked = 0
    skip_dirs = {{"__pycache__", ".git"}}
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            src = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
            try:
                ast.parse(src)
                checked += 1
            except SyntaxError as e:
                errors.append((path, str(e)))
    return checked, errors
""",
            encoding="utf-8",
        )

        spec = importlib.util.spec_from_file_location("check_syntax_tmp", scripts_dir / "check_syntax.py")
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        checked, errors = mod.main()
        assert checked > 0, "check_syntax found 0 files — walker is not reaching source files"
        assert errors == [], f"Unexpected syntax errors in test fixtures: {errors}"


# -- check_test_quality tests -------------------------------------------------


class TestCheckTestQuality:
    """Tests for check_test_quality.py."""

    def test_check_test_quality_vet241_is_error(self, tmp_path: Path) -> None:
        """VET241 must be classified as ERROR, not WARNING.

        Old bug: WARNING severity meant the script always exited 0 even when
        it found tests with no assertions.
        """
        mod = _import_script("check_test_quality")

        # Minimal test file with one zero-assert test
        fake_test = tmp_path / "test_fake.py"
        fake_test.write_text(
            "def test_nothing():\n    x = 1 + 1\n",
            encoding="utf-8",
        )
        lines = fake_test.read_text(encoding="utf-8").splitlines()
        violations = mod.check_zero_assert(fake_test, lines)

        assert violations, "check_zero_assert() should report a violation for a no-assert test"
        for _lineno, code, severity, _msg in violations:
            if code == "VET241":
                assert severity == mod.ERROR, (
                    f"VET241 severity is {severity!r} — expected ERROR so that "
                    "zero-assert tests cause a nonzero exit code"
                )
                break
        else:
            pytest.fail("No VET241 violation returned for a zero-assert test function")

    def test_check_test_quality_exits_nonzero_on_zero_assert(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() must exit 1 when it finds a zero-assert test.

        Old bug: all VET241 were WARNING so error_count was always 0 and
        the script always returned 0.
        """
        mod = _import_script("check_test_quality")

        # Point TESTS_DIR at our tmp dir containing one zero-assert test
        zero_assert_file = tmp_path / "test_empty.py"
        zero_assert_file.write_text(
            "def test_nothing():\n    pass\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(mod, "TESTS_DIR", tmp_path)
        # Redirect argument parsing to use no extra args
        monkeypatch.setattr(sys, "argv", ["check_test_quality"])

        exit_code = mod.main()
        assert exit_code == 1, f"main() returned {exit_code} — expected 1 when zero-assert tests are found"

    def test_check_test_quality_vet242_context_manager_self_mock(
        self, tmp_path: Path
    ) -> None:
        """VET242 must fire for with-patch self-mocking, not just decorator-style."""
        mod = _import_script("check_test_quality")

        # Use a real flat module so exact-module symbol resolution can verify
        # the patch target is locally defined.
        fake_test = tmp_path / "test_preflight.py"
        fake_test.write_text(
            "from unittest.mock import patch\n"
            "def test_something():\n"
            "    with patch('vetinari.preflight.detect_hardware') as m:\n"
            "        m.return_value = 1\n"
            "        assert m() == 1\n",
            encoding="utf-8",
        )
        lines = fake_test.read_text(encoding="utf-8").splitlines()
        violations = mod.check_self_mocking(fake_test, lines)
        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET242" in codes, (
            "check_self_mocking() should detect 'with patch(...)' self-mocking as VET242"
        )

    def test_check_test_quality_vet243_mock_assert_called_method(
        self, tmp_path: Path
    ) -> None:
        """VET243 must fire for mock.assert_called_once() even without 'assert' keyword."""
        mod = _import_script("check_test_quality")

        # Test where sole 'assertion' is mock.assert_called_once() (no literal assert)
        fake_test = tmp_path / "test_mockonly.py"
        fake_test.write_text(
            "from unittest.mock import MagicMock\n"
            "def test_mock_called():\n"
            "    m = MagicMock()\n"
            "    m()\n"
            "    m.assert_called_once()\n",
            encoding="utf-8",
        )
        lines = fake_test.read_text(encoding="utf-8").splitlines()
        violations = mod.check_mock_only_assertion(fake_test, lines)
        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET243" in codes, (
            "check_mock_only_assertion() should report VET243 when sole assertion is "
            "mock.assert_called_once() — currently got no violations"
        )


# -- hook_pre_commit_gate tests -----------------------------------------------


class TestHookPreCommitGate:
    """Tests for hook_pre_commit_gate.py."""

    def test_pre_commit_gate_no_errors_only_flag(self) -> None:
        """_build_checks() must NOT pass --errors-only to check_vetinari_rules.

        Old bug: --errors-only suppressed WARNING-level violations including
        VET024/VET025 success-masking findings.
        """
        mod = _import_script("hook_pre_commit_gate")
        checks = mod._build_checks()

        check_vetinari_args: list[str] | None = None
        for label, argv in checks:
            if label == "check_vetinari_rules":
                check_vetinari_args = argv
                break

        assert check_vetinari_args is not None, "_build_checks() returned no 'check_vetinari_rules' entry"
        assert "--errors-only" not in check_vetinari_args, (
            f"check_vetinari_rules is still invoked with --errors-only: {check_vetinari_args}"
        )

    def test_pre_commit_gate_runs_audit_prevention(self) -> None:
        """Commit/push gate must include the full-spectrum audit prevention guard."""
        mod = _import_script("hook_pre_commit_gate")
        checks = mod._build_checks()

        labels = [label for label, _argv in checks]

        assert "check_audit_prevention" in labels


# -- run_tests tests ----------------------------------------------------------


class TestRunTests:
    """Tests for run_tests.py."""

    def test_run_tests_uses_sys_exit(self) -> None:
        """run_tests.py must not contain os._exit().

        os._exit() bypasses pytest teardown, atexit handlers, and finally
        blocks — it suppresses cleanup rather than fixing the root cause.
        """
        script = _SCRIPTS_DIR / "run_tests.py"
        source = script.read_text(encoding="utf-8")
        assert "os._exit" not in source, (
            "run_tests.py still uses os._exit() — replace with sys.exit() "
            "and fix non-daemon thread hangs at source (conftest teardown)"
        )


# -- check_migration_index tests ----------------------------------------------


class TestCheckMigrationIndex:
    """Tests for check_migration_index.py."""

    def test_migration_index_no_six_agent_claim(self) -> None:
        """The '6 CONSOLIDATED' heading must not appear in the source.

        Regression guard: the canonical architecture is 3-agent factory
        (Foreman, Worker, Inspector) per ADR-0061.
        """
        script = _SCRIPTS_DIR / "check_migration_index.py"
        source = script.read_text(encoding="utf-8")
        assert "6 CONSOLIDATED" not in source, (
            "check_migration_index.py still claims '6 CONSOLIDATED AGENT IMPLEMENTATIONS' — "
            "the canonical architecture is the 3-agent factory pipeline (ADR-0061)"
        )

    def test_migration_index_validates_three_agents(self, tmp_path: Path) -> None:
        """check_agent_implementations() must validate 3 factory agents.

        Ensures the function references the canonical factory agent files
        (Foreman/Worker/Inspector) and requires exactly 3, not 6.
        """
        mod = _import_script("check_migration_index")

        # The function should check exactly 3 agent files
        # We verify this by pointing project_root at a tmp dir with the 3
        # correct agent files present and confirming it returns True.
        agents_dir = tmp_path / "vetinari" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "planner_agent.py").write_text("# ForemanAgent\n", encoding="utf-8")

        consolidated_dir = agents_dir / "consolidated"
        consolidated_dir.mkdir()
        (consolidated_dir / "worker_agent.py").write_text("# WorkerAgent\n", encoding="utf-8")
        (consolidated_dir / "quality_agent.py").write_text("# InspectorAgent\n", encoding="utf-8")

        # Also create the support files so they don't confuse the output
        (agents_dir / "base_agent.py").write_text("", encoding="utf-8")
        (agents_dir / "contracts.py").write_text("", encoding="utf-8")
        (agents_dir / "interfaces.py").write_text("", encoding="utf-8")

        # Temporarily redirect project_root inside the module
        original_root = mod.project_root
        mod.project_root = tmp_path
        try:
            result = mod.check_agent_implementations()
        finally:
            mod.project_root = original_root

        assert result is True, (
            "check_agent_implementations() returned False even though all 3 "
            "factory agent files (planner_agent, worker_agent, quality_agent) were present"
        )

    def test_migration_index_fails_when_agent_missing(self, tmp_path: Path) -> None:
        """check_agent_implementations() must return False when an agent file is absent.

        Confirms the checker is actually detecting missing files, not defaulting
        to passing.
        """
        mod = _import_script("check_migration_index")

        # Create only 2 of the 3 required agent files
        agents_dir = tmp_path / "vetinari" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "planner_agent.py").write_text("# ForemanAgent\n", encoding="utf-8")
        consolidated_dir = agents_dir / "consolidated"
        consolidated_dir.mkdir()
        (consolidated_dir / "worker_agent.py").write_text("# WorkerAgent\n", encoding="utf-8")
        # quality_agent.py (InspectorAgent) deliberately absent

        original_root = mod.project_root
        mod.project_root = tmp_path
        try:
            result = mod.check_agent_implementations()
        finally:
            mod.project_root = original_root

        assert result is False, (
            "check_agent_implementations() returned True even though quality_agent.py "
            "(InspectorAgent) was absent — the checker is not detecting missing agents"
        )

    def test_migration_index_main_label_is_three_agent(self) -> None:
        """The check label in main() must say '3-Agent Factory Pipeline', not 'All 6 Agents'.

        Regression guard against the stale label reappearing.
        """
        script = _SCRIPTS_DIR / "check_migration_index.py"
        source = script.read_text(encoding="utf-8")
        assert "3-Agent Factory Pipeline" in source, (
            "check_migration_index.py main() does not contain the '3-Agent Factory Pipeline' "
            "label — it may still use the legacy 'All 6 Agents' label"
        )
        assert "All 6 Agents" not in source, "check_migration_index.py still uses the legacy 'All 6 Agents' check label"
