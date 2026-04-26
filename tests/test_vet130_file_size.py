"""Tests for VET127 — file size limit enforcement (originally filed as VET130).

VET127 flags vetinari/ Python files that exceed 550 lines of code (non-blank,
non-comment lines).  Files at or under the limit must not be flagged.
Exempt files: __init__.py and conftest.py.

Uses checker.check_file() — the project-standard testable API — with
monkeypatching of is_in_vetinari to make scope-guarded rules fire on
tmp_path fixtures that live outside vetinari/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the quality scripts are importable when tests run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "quality"))

import check_vetinari_rules as checker

_RULE = "VET127"


def _findings(path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Run check_file with is_in_vetinari forced True and return all findings.

    VET127 is guarded by is_in_vetinari(), so fixtures in tmp_path would be
    silently skipped without this patch.

    Args:
        path: Absolute path to the fixture file to check.
        monkeypatch: pytest monkeypatch fixture for automatic teardown.

    Returns:
        List of (filepath, line, code, message) tuples from check_file().
    """
    monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
    return checker.check_file(path)


class TestCheckFileSize:
    """Tests for check_file_size (VET127)."""

    def test_vet127_flags_oversized_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file with more than 550 lines of code must produce a VET127 warning."""
        big_file = tmp_path / "huge.py"
        big_file.write_text("x = 1\n" * 600, encoding="utf-8")
        findings = _findings(big_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"Expected {_RULE} for 600-loc file but got: {codes!r}"

    def test_vet127_passes_normal_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file within the 550-line limit must not produce a VET127 finding."""
        normal_file = tmp_path / "normal.py"
        normal_file.write_text("x = 1\n" * 200, encoding="utf-8")
        findings = _findings(normal_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"{_RULE} must not fire for 200-loc file, got: {codes!r}"

    def test_vet127_passes_exactly_at_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file with exactly 550 lines of code must not be flagged."""
        at_limit = tmp_path / "at_limit.py"
        at_limit.write_text("x = 1\n" * 550, encoding="utf-8")
        findings = _findings(at_limit, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"550 loc is within limit; {_RULE} must not fire"

    def test_vet127_flags_at_551_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file with 551 lines of code is one over the limit and must be flagged."""
        over_limit = tmp_path / "over_limit.py"
        over_limit.write_text("x = 1\n" * 551, encoding="utf-8")
        findings = _findings(over_limit, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE in codes, f"551 loc must trigger {_RULE}"

    def test_vet127_skips_init_py(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """__init__.py files are exempt regardless of size."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("x = 1\n" * 700, encoding="utf-8")
        findings = _findings(init_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"__init__.py must be exempt from {_RULE}"

    def test_vet127_violation_message_references_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The violation message must mention the 550-line limit."""
        big_file = tmp_path / "huge.py"
        big_file.write_text("x = 1\n" * 700, encoding="utf-8")
        findings = _findings(big_file, monkeypatch)
        matching = [f for f in findings if f[2] == _RULE]
        assert matching, f"Expected a {_RULE} finding"
        assert "550" in matching[0][3], "Message must reference the 550-line limit"

    def test_vet127_violation_points_to_line_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """File-level violations are anchored to line 1."""
        big_file = tmp_path / "huge.py"
        big_file.write_text("x = 1\n" * 600, encoding="utf-8")
        findings = _findings(big_file, monkeypatch)
        matching = [f for f in findings if f[2] == _RULE]
        assert matching, f"Expected a {_RULE} finding"
        assert matching[0][1] == 1, f"File-level violation must point to line 1, got line {matching[0][1]}"

    def test_vet127_blank_and_comment_lines_excluded_from_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank lines and comment-only lines must not count toward the 550 limit."""
        # 400 code lines + 300 blank/comment lines = still 400 loc, well under 550
        content = ("x = 1\n" * 400) + ("\n" * 200) + ("# comment\n" * 100)
        mixed_file = tmp_path / "mixed.py"
        mixed_file.write_text(content, encoding="utf-8")
        findings = _findings(mixed_file, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"Blank/comment lines must not count toward the 550 limit; got {codes!r}"

    def test_vet127_docstrings_excluded_from_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Module and function docstrings must not count as executable code."""
        docstring_heavy = tmp_path / "docstring_heavy.py"
        docstring_heavy.write_text(
            '"""\\n' + ("Documentation line.\\n" * 300) + '"""\\n'
            "\n"
            "def explain():\n"
            '    """\\n'
            + ("More documentation.\\n" * 300)
            + '    """\\n'
            "    return 1\n",
            encoding="utf-8",
        )
        findings = _findings(docstring_heavy, monkeypatch)
        codes = [f[2] for f in findings]
        assert _RULE not in codes, f"Docstrings must not count toward the 550 limit; got {codes!r}"
