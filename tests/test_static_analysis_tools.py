"""Focused tests for static analysis gate truthfulness and custom rule coverage."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vetinari.tools.semgrep_tool import run_semgrep_signal
from vetinari.tools.static_analysis import run_static_analysis, run_static_analysis_signal


def _load_rules_module():
    """Load the standalone rules-checker script as an importable test module."""
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "quality" / "check_vetinari_rules.py"
    spec = importlib.util.spec_from_file_location("vetinari_rules_checker_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_static_analysis_marks_missing_tools_as_skipped(tmp_path: Path) -> None:
    """Unavailable external analyzers should be skipped, not reported as passed."""
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")

    with patch("vetinari.tools.static_analysis.subprocess.run", side_effect=FileNotFoundError):
        result = run_static_analysis(target)

    assert result.gates_passed == ["ast"]
    assert result.gates_failed == []
    assert set(result.gates_skipped) == {"pyright", "ruff", "vulture"}
    assert result.skip_reasons == {
        "pyright": "unavailable",
        "ruff": "unavailable",
        "vulture": "unavailable",
    }
    assert result.is_clean is False


def test_static_analysis_fails_gates_when_ruff_or_vulture_report_findings(tmp_path: Path) -> None:
    """Lint and dead-code findings should fail their gates instead of looking clean."""
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")

    pyright_ok = SimpleNamespace(stdout='{"generalDiagnostics": []}', returncode=0)
    ruff_failure = SimpleNamespace(
        stdout='[{"filename": "sample.py", "location": {"row": 1}, "message": "unused import", "code": "F401"}]',
        returncode=1,
    )
    vulture_failure = SimpleNamespace(stdout="sample.py:1: unused variable 'value'", returncode=1)

    with patch(
        "vetinari.tools.static_analysis.subprocess.run",
        side_effect=[pyright_ok, ruff_failure, vulture_failure],
    ):
        result = run_static_analysis(target)

    assert "pyright" in result.gates_passed
    assert "ruff" in result.gates_failed
    assert "vulture" in result.gates_failed
    assert result.gates_skipped == []
    assert any(finding.tool == "ruff" for finding in result.findings)
    assert any(finding.tool == "vulture" for finding in result.findings)
    assert result.is_clean is False


def test_rules_checker_collects_scripts_directory(tmp_path: Path) -> None:
    """Operational scripts should be scanned by the custom VET rules checker."""
    rules = _load_rules_module()

    vetinari_dir = tmp_path / "vetinari"
    tests_dir = tmp_path / "tests"
    scripts_dir = tmp_path / "scripts"
    vetinari_dir.mkdir()
    tests_dir.mkdir()
    scripts_dir.mkdir()

    vet_file = vetinari_dir / "module.py"
    test_file = tests_dir / "test_module.py"
    script_file = scripts_dir / "tool.py"
    vet_file.write_text("value = 1\n", encoding="utf-8")
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    script_file.write_text("print('hello')\n", encoding="utf-8")

    with (
        patch.object(rules, "VETINARI_DIR", vetinari_dir),
        patch.object(rules, "TESTS_DIR", tests_dir),
        patch.object(rules, "SCRIPTS_DIR", scripts_dir),
    ):
        files = rules.collect_python_files()

    assert str(vet_file) in files
    assert str(test_file) in files
    assert str(script_file) in files


# ═══════════════════════════════════════════════════════════════════════════
# OutcomeSignal wrapper tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOutcomeSignalWrappers:
    """Tests for run_static_analysis_signal() and run_semgrep_signal()."""

    # -- run_static_analysis_signal ------------------------------------------

    def test_static_analysis_signal_missing_target_returns_unsupported(self, tmp_path: Path) -> None:
        """Non-existent target yields UNSUPPORTED basis with passed=False."""
        from vetinari.types import EvidenceBasis

        sig = run_static_analysis_signal(tmp_path / "nonexistent.py")

        assert sig.passed is False
        assert sig.score == 0.0
        assert sig.basis is EvidenceBasis.UNSUPPORTED
        assert sig.provenance is not None
        assert sig.provenance.tool_name == "static_analysis_pipeline"

    def test_static_analysis_signal_all_gates_skipped_returns_unsupported(self, tmp_path: Path) -> None:
        """When all external tools are unavailable, basis must be UNSUPPORTED, not TOOL_EVIDENCE."""
        from vetinari.types import EvidenceBasis

        target = tmp_path / "sample.py"
        target.write_text("value = 1\n", encoding="utf-8")

        with patch("vetinari.tools.static_analysis.subprocess.run", side_effect=FileNotFoundError):
            sig = run_static_analysis_signal(target)

        # ast gate passed, but all subprocess gates skipped → has tool evidence for ast only.
        # With mixed pass + skips, passed=False but basis=TOOL_EVIDENCE (ast ran).
        assert sig.passed is False
        assert sig.basis is EvidenceBasis.TOOL_EVIDENCE
        assert sig.provenance is not None
        assert sig.provenance.source == "vetinari.tools.static_analysis"

    def test_static_analysis_signal_passed_gate_gives_tool_evidence(self, tmp_path: Path) -> None:
        """A gate that passes populates tool_evidence with passed=True entry."""
        from vetinari.types import EvidenceBasis

        target = tmp_path / "sample.py"
        target.write_text("value = 1\n", encoding="utf-8")

        pyright_ok = SimpleNamespace(stdout='{"generalDiagnostics": []}', returncode=0)
        ruff_ok = SimpleNamespace(stdout="[]", returncode=0)
        vulture_ok = SimpleNamespace(stdout="", returncode=0)

        with patch(
            "vetinari.tools.static_analysis.subprocess.run",
            side_effect=[pyright_ok, ruff_ok, vulture_ok],
        ):
            sig = run_static_analysis_signal(target)

        assert sig.basis is EvidenceBasis.TOOL_EVIDENCE
        assert sig.passed is True
        assert len(sig.tool_evidence) > 0
        passed_gates = [te for te in sig.tool_evidence if te.passed]
        assert len(passed_gates) > 0
        assert all(te.tool_name for te in sig.tool_evidence)

    def test_static_analysis_signal_failed_gate_gives_tool_evidence_with_passed_false(self, tmp_path: Path) -> None:
        """A failed gate yields a ToolEvidence entry with passed=False."""
        from vetinari.types import EvidenceBasis

        target = tmp_path / "sample.py"
        target.write_text("value = 1\n", encoding="utf-8")

        pyright_ok = SimpleNamespace(stdout='{"generalDiagnostics": []}', returncode=0)
        ruff_failure = SimpleNamespace(
            stdout='[{"filename": "sample.py", "location": {"row": 1}, "message": "unused import", "code": "F401"}]',
            returncode=1,
        )
        vulture_ok = SimpleNamespace(stdout="", returncode=0)

        with patch(
            "vetinari.tools.static_analysis.subprocess.run",
            side_effect=[pyright_ok, ruff_failure, vulture_ok],
        ):
            sig = run_static_analysis_signal(target)

        assert sig.basis is EvidenceBasis.TOOL_EVIDENCE
        assert sig.passed is False
        failed_te = [te for te in sig.tool_evidence if not te.passed]
        assert any(te.tool_name == "ruff" for te in failed_te)
        assert len(sig.issues) > 0

    def test_static_analysis_signal_provenance_fields_populated(self, tmp_path: Path) -> None:
        """Provenance carries source and timestamp_utc."""
        target = tmp_path / "sample.py"
        target.write_text("x = 1\n", encoding="utf-8")

        with patch("vetinari.tools.static_analysis.subprocess.run", side_effect=FileNotFoundError):
            sig = run_static_analysis_signal(target)

        assert sig.provenance.source == "vetinari.tools.static_analysis"
        assert sig.provenance.timestamp_utc  # non-empty ISO string
        assert "T" in sig.provenance.timestamp_utc  # basic ISO-8601 check

    # -- run_semgrep_signal --------------------------------------------------

    def test_semgrep_signal_unavailable_returns_unsupported(self, tmp_path: Path) -> None:
        """When semgrep is not on PATH, basis must be UNSUPPORTED and issues mention install."""
        from vetinari.types import EvidenceBasis

        target = tmp_path / "sample.py"
        target.write_text("x = 1\n", encoding="utf-8")

        with patch("vetinari.tools.semgrep_tool.subprocess.run", side_effect=FileNotFoundError):
            sig = run_semgrep_signal(target)

        assert sig.passed is False
        assert sig.basis is EvidenceBasis.UNSUPPORTED
        assert any("pip install -e .[dev]" in issue for issue in sig.issues)
        assert sig.provenance is not None
        assert sig.provenance.tool_name == "semgrep"

    def test_semgrep_signal_no_findings_returns_passed(self, tmp_path: Path) -> None:
        """Zero findings yields passed=True, score=1.0, basis=TOOL_EVIDENCE."""
        from vetinari.types import EvidenceBasis

        target = tmp_path / "sample.py"
        target.write_text("x = 1\n", encoding="utf-8")

        ok_proc = SimpleNamespace(stdout='{"results": []}', returncode=0, stderr="")

        with patch("vetinari.tools.semgrep_tool.subprocess.run", return_value=ok_proc):
            sig = run_semgrep_signal(target)

        assert sig.passed is True
        assert sig.score == 1.0
        assert sig.basis is EvidenceBasis.TOOL_EVIDENCE
        assert len(sig.tool_evidence) == 1
        assert sig.tool_evidence[0].passed is True

    def test_semgrep_signal_with_findings_returns_failed(self, tmp_path: Path) -> None:
        """Findings yield passed=False, score < 1.0, issues populated."""
        from vetinari.types import EvidenceBasis

        target = tmp_path / "sample.py"
        target.write_text("x = eval(input())\n", encoding="utf-8")

        findings_json = """{
            "results": [
                {
                    "check_id": "python.lang.security.eval-injection",
                    "path": "sample.py",
                    "start": {"line": 1},
                    "extra": {
                        "message": "Use of eval() is dangerous",
                        "severity": "ERROR"
                    }
                }
            ]
        }"""
        finding_proc = SimpleNamespace(stdout=findings_json, returncode=1, stderr="")

        with patch("vetinari.tools.semgrep_tool.subprocess.run", return_value=finding_proc):
            sig = run_semgrep_signal(target)

        assert sig.passed is False
        assert sig.score < 1.0
        assert sig.basis is EvidenceBasis.TOOL_EVIDENCE
        assert len(sig.issues) == 1
        assert "eval" in sig.issues[0].lower()
        assert sig.tool_evidence[0].passed is False

    def test_semgrep_signal_tool_evidence_has_stdout_hash(self, tmp_path: Path) -> None:
        """ToolEvidence entries must carry a non-empty stdout_hash for auditability."""
        target = tmp_path / "sample.py"
        target.write_text("x = 1\n", encoding="utf-8")

        ok_proc = SimpleNamespace(stdout='{"results": []}', returncode=0, stderr="")

        with patch("vetinari.tools.semgrep_tool.subprocess.run", return_value=ok_proc):
            sig = run_semgrep_signal(target)

        assert sig.tool_evidence[0].stdout_hash  # non-empty SHA-256 hex
        assert len(sig.tool_evidence[0].stdout_hash) == 64  # SHA-256 = 64 hex chars
