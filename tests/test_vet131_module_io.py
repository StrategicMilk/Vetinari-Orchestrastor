"""Tests for VET128 — module-level I/O detection (originally filed as VET131).

VET128 flags I/O calls (open, yaml.safe_load, json.load, Path.read_text,
Path.read_bytes) that appear at module scope — outside any function or class
body.  Calls inside functions or class methods are intentional lazy-init and
must not be flagged.

Uses checker.check_file() with is_in_vetinari monkeypatched True so the
scope guard fires on tmp_path fixtures.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Ensure the quality scripts are importable when tests run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "quality"))

import check_vetinari_rules as checker

_RULE = "VET128"


def _findings(path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Run check_file with is_in_vetinari forced True and return all findings.

    VET128 is guarded by is_in_vetinari(), so fixtures in tmp_path would be
    silently skipped without this patch.

    Args:
        path: Absolute path to the fixture file to check.
        monkeypatch: pytest monkeypatch fixture for automatic teardown.

    Returns:
        List of (filepath, line, code, message) tuples from check_file().
    """
    monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
    return checker.check_file(path)


class TestCheckModuleLevelIo:
    """Tests for check_module_level_io (VET128)."""

    def test_vet128_flags_yaml_safe_load_at_module_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """yaml.safe_load at module scope must produce a VET128 violation."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(
            textwrap.dedent("""\
                import yaml
                config = yaml.safe_load(open("config.yaml"))
            """),
            encoding="utf-8",
        )
        findings = _findings(bad_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"Expected {_RULE} for module-level yaml.safe_load, got: {codes!r}"

    def test_vet128_passes_lazy_init_in_function(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """I/O calls inside a function body must not be flagged."""
        good_file = tmp_path / "good.py"
        good_file.write_text(
            textwrap.dedent("""\
                import yaml
                def get_config():
                    return yaml.safe_load(open("config.yaml"))
            """),
            encoding="utf-8",
        )
        findings = _findings(good_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"{_RULE} must not fire for I/O inside a function, got: {codes!r}"

    def test_vet128_flags_open_at_module_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare open() call at module scope must be flagged."""
        bad_file = tmp_path / "bad_open.py"
        bad_file.write_text(
            textwrap.dedent("""\
                data = open("data.txt", encoding="utf-8").read()
            """),
            encoding="utf-8",
        )
        findings = _findings(bad_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"open() at module level must trigger {_RULE}, got: {codes!r}"

    def test_vet128_passes_open_inside_class_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """open() inside a class method must not be flagged."""
        good_file = tmp_path / "good_class.py"
        good_file.write_text(
            textwrap.dedent("""\
                class Config:
                    def load(self):
                        with open("cfg.yaml", encoding="utf-8") as f:
                            return f.read()
            """),
            encoding="utf-8",
        )
        findings = _findings(good_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"open() inside a class method must not trigger {_RULE}, got: {codes!r}"

    def test_vet128_flags_json_load_at_module_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """json.load() at module scope must be flagged."""
        bad_file = tmp_path / "bad_json.py"
        bad_file.write_text(
            textwrap.dedent("""\
                import json
                data = json.load(open("data.json"))
            """),
            encoding="utf-8",
        )
        findings = _findings(bad_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"json.load() at module level must trigger {_RULE}, got: {codes!r}"

    def test_vet128_violation_message_names_the_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The violation message must identify the offending I/O call."""
        bad_file = tmp_path / "bad_msg.py"
        bad_file.write_text("cfg = open('x.txt')\n", encoding="utf-8")
        findings = _findings(bad_file, monkeypatch)
        matching = [f for f in findings if f[2] == _RULE]
        assert matching, f"Expected a {_RULE} finding"
        assert "open()" in matching[0][3], f"Message must name the I/O call, got: {matching[0][3]!r}"

    def test_vet128_violation_has_positive_line_number(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each violation must include a positive 1-based line number."""
        bad_file = tmp_path / "bad_line.py"
        bad_file.write_text(
            textwrap.dedent("""\
                import json
                import os
                data = open("x.json")
            """),
            encoding="utf-8",
        )
        findings = _findings(bad_file, monkeypatch)
        matching = [f for f in findings if f[2] == _RULE]
        assert matching, f"Expected a {_RULE} finding"
        assert all(f[1] > 0 for f in matching), "All violations must have positive line numbers"

    def test_vet128_handles_syntax_error_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files that cannot be parsed must produce zero VET128 violations (no crash)."""
        broken_file = tmp_path / "broken.py"
        broken_file.write_text("def (broken syntax:\n", encoding="utf-8")
        findings = _findings(broken_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"Unparseable files must produce no {_RULE} violations"
