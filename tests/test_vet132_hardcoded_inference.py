"""Tests for VET129 — hardcoded inference parameter detection (originally filed as VET132).

VET129 flags literal numeric keyword arguments for inference sampling
parameters (temperature, max_tokens, top_p, top_k) in vetinari/ production
code.  Only files under vetinari/ are in scope; config/, tests/, and scripts/
are exempt because is_in_vetinari() returns False for them.

Uses checker.check_file() with is_in_vetinari monkeypatched True so scope-
guarded rules fire on tmp_path fixtures.  Exemption tests omit the patch so
the real is_in_vetinari() returns False and the rule is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable when tests run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import check_vetinari_rules as checker

_RULE = "VET129"


def _findings_in_scope(path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Run check_file with is_in_vetinari forced True (simulates vetinari/ file).

    Args:
        path: Absolute path to the fixture file to check.
        monkeypatch: pytest monkeypatch fixture for automatic teardown.

    Returns:
        List of (filepath, line, code, message) tuples from check_file().
    """
    monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
    return checker.check_file(path)


def _findings_out_of_scope(path: Path) -> list[tuple]:
    """Run check_file without patching — real is_in_vetinari() applies.

    Files in tmp_path return False from is_in_vetinari(), so VET129 is skipped.
    Used to verify exemptions for non-vetinari/ paths (config/, tests/, scripts/).

    Args:
        path: Absolute path to the fixture file to check.

    Returns:
        List of (filepath, line, code, message) tuples from check_file().
    """
    return checker.check_file(path)


class TestCheckHardcodedInferenceParams:
    """Tests for check_hardcoded_inference_params (VET129)."""

    def test_vet129_flags_hardcoded_temperature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A literal temperature= keyword argument must produce a VET129 finding."""
        bad = tmp_path / "agent.py"
        bad.write_text("response = client.create(temperature=0.3)\n", encoding="utf-8")
        findings = _findings_in_scope(bad, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"Expected {_RULE} for hardcoded temperature, got: {codes!r}"

    def test_vet129_passes_config_driven(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A variable reference for temperature must not be flagged."""
        good = tmp_path / "agent.py"
        good.write_text("response = client.create(temperature=config.temperature)\n", encoding="utf-8")
        findings = _findings_in_scope(good, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"Config-driven temperature must not trigger {_RULE}, got: {codes!r}"

    def test_vet129_flags_hardcoded_max_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A literal max_tokens= keyword argument must be flagged."""
        bad = tmp_path / "inference.py"
        bad.write_text("params = dict(max_tokens=1024)\n", encoding="utf-8")
        findings = _findings_in_scope(bad, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"Hardcoded max_tokens must trigger {_RULE}, got: {codes!r}"

    def test_vet129_flags_hardcoded_top_p(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A literal top_p= keyword argument must be flagged."""
        bad = tmp_path / "worker.py"
        bad.write_text("result = call(top_p=0.95)\n", encoding="utf-8")
        findings = _findings_in_scope(bad, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"Hardcoded top_p must trigger {_RULE}, got: {codes!r}"

    def test_vet129_flags_hardcoded_top_k(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A literal top_k= keyword argument must be flagged."""
        bad = tmp_path / "foreman.py"
        bad.write_text("result = infer(top_k=40)\n", encoding="utf-8")
        findings = _findings_in_scope(bad, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"Hardcoded top_k must trigger {_RULE}, got: {codes!r}"

    def test_vet129_exempt_outside_vetinari(self, tmp_path: Path) -> None:
        """Files outside vetinari/ (config/, tests/, scripts/) are exempt from VET129.

        is_in_vetinari() returns False for tmp_path, so the rule is skipped.
        This mirrors the real exemption: only vetinari/ files are in scope.
        """
        config_file = tmp_path / "defaults.py"
        config_file.write_text("response = call(temperature=0.7)\n", encoding="utf-8")
        findings = _findings_out_of_scope(config_file)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"Out-of-scope files must be exempt from {_RULE}, got: {codes!r}"

    def test_vet129_violation_message_names_param_and_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The violation message must include the parameter name and its literal value."""
        bad = tmp_path / "runner.py"
        bad.write_text("out = model.generate(temperature=0.1)\n", encoding="utf-8")
        findings = _findings_in_scope(bad, monkeypatch)
        matching = [f for f in findings if f[2] == _RULE]
        assert matching, f"Expected a {_RULE} finding"
        msg = matching[0][3]
        assert "temperature" in msg, f"Message must name the parameter, got: {msg!r}"
        assert "0.1" in msg, f"Message must include the literal value, got: {msg!r}"

    def test_vet129_skips_non_python_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-.py files must produce no VET129 findings (scan_file reads only .py)."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("temperature=0.3\n", encoding="utf-8")
        # check_file() calls scan_file() which parses with ast — .txt files will
        # have no AST tree, so no findings are produced.
        findings = _findings_in_scope(txt_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"Non-.py files must produce no {_RULE} findings, got: {codes!r}"

    def test_vet129_reports_correct_line_number(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The violation must reference the line where the literal value appears."""
        bad = tmp_path / "agent.py"
        bad.write_text("# comment\nx = 1\nresult = infer(max_tokens=512)\n", encoding="utf-8")
        findings = _findings_in_scope(bad, monkeypatch)
        matching = [f for f in findings if f[2] == _RULE]
        assert matching, f"Expected a {_RULE} finding"
        assert matching[0][1] == 3, f"Expected line 3, got line {matching[0][1]}"
