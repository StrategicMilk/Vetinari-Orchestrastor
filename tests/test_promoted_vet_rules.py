"""Regression coverage for promoted VET300+ external rules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
_CONFIG = _ROOT / "config" / "vet_rules.yaml"
_FIXTURE_DIR = _ROOT / "tests" / "fixtures" / "vet_rules" / "promoted"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_vetinari_rules as checker


def _load_promoted_rules() -> dict[str, dict[str, Any]]:
    rules = checker.load_external_rules(_CONFIG)
    return {str(rule["id"]): rule for rule in rules}


def _isolated_external_findings(files: list[Path], rules: list[dict[str, Any]]) -> list[tuple]:
    saved_errors = checker.errors[:]
    saved_warnings = checker.warnings[:]
    checker.errors.clear()
    checker.warnings.clear()
    try:
        checker.apply_external_rules([str(path) for path in files], rules)
        return [*checker.errors, *checker.warnings]
    finally:
        checker.errors.clear()
        checker.errors.extend(saved_errors)
        checker.warnings.clear()
        checker.warnings.extend(saved_warnings)


def _rule_for_fixture(rule: dict[str, Any], fixture_path: Path) -> dict[str, Any]:
    patched = dict(rule)
    patched["include_globs"] = [fixture_path.relative_to(_ROOT).as_posix()]
    patched["exclude_globs"] = []
    return patched


PROMOTED_RULE_IDS = [f"VET{number}" for number in range(300, 321)]


@pytest.mark.parametrize("rule_id", PROMOTED_RULE_IDS)
def test_promoted_rule_bad_fixture_triggers(rule_id: str) -> None:
    """Every active promoted rule must have a known-bad fixture that fires."""
    rules = _load_promoted_rules()
    bad_path = next(_FIXTURE_DIR.glob(f"bad_{rule_id.lower()}.*"))
    findings = _isolated_external_findings([bad_path], [_rule_for_fixture(rules[rule_id], bad_path)])
    codes = [finding[2] for finding in findings]

    assert rule_id in codes, f"{rule_id} did not fire for {bad_path.name}; findings={findings!r}"


@pytest.mark.parametrize("rule_id", PROMOTED_RULE_IDS)
def test_promoted_rule_good_fixture_passes(rule_id: str) -> None:
    """Every active promoted rule must have a good fixture that avoids the rule."""
    rules = _load_promoted_rules()
    good_path = next(_FIXTURE_DIR.glob(f"good_{rule_id.lower()}.*"))
    findings = _isolated_external_findings([good_path], [_rule_for_fixture(rules[rule_id], good_path)])
    codes = [finding[2] for finding in findings]

    assert rule_id not in codes, f"{rule_id} fired for good fixture {good_path.name}; findings={findings!r}"


def test_include_globs_scan_non_python_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """include_globs must expand non-Python scopes rather than documenting planned enforcement."""
    static_dir = tmp_path / "web"
    static_dir.mkdir()
    css_path = static_dir / "app.css"
    css_path.write_text(".button { transition: all 300ms; }\n", encoding="utf-8")
    monkeypatch.setattr(checker, "project_root", tmp_path)

    rule = {
        "id": "VET999",
        "description": "No unguarded motion",
        "pattern_type": "regex",
        "pattern": r"transition:\s*[^;\n]+",
        "include_globs": ["web/**/*.css"],
        "severity": "warning",
    }

    files = checker.collect_external_rule_files([], [rule])
    findings = _isolated_external_findings([Path(path) for path in files], [rule])

    assert str(css_path) in files
    assert [(finding[1], finding[2]) for finding in findings] == [(1, "VET999")]


def test_exclude_globs_remove_matching_files() -> None:
    """exclude_globs must prevent historical fixtures from becoming active violations."""
    bad_path = _FIXTURE_DIR / "bad_vet300.ps1"
    rule = {
        "id": "VET999",
        "description": "Excluded uvx fixture",
        "pattern_type": "regex",
        "pattern": r"uvx\s+[^\n]*@latest",
        "include_globs": [bad_path.relative_to(_ROOT).as_posix()],
        "exclude_globs": [bad_path.relative_to(_ROOT).as_posix()],
        "severity": "error",
    }

    findings = _isolated_external_findings([bad_path], [rule])

    assert findings == []


def test_pyyaml_absence_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing external-rules file may not be silently skipped when PyYAML is missing."""
    rules_path = tmp_path / "vet_rules.yaml"
    rules_path.write_text("version: 1\nrules: []\n", encoding="utf-8")
    saved_errors = checker.errors[:]
    monkeypatch.setattr(checker, "_yaml", None)
    checker.errors.clear()
    try:
        rules = checker.load_external_rules(rules_path)
        findings = checker.errors[:]
    finally:
        checker.errors.clear()
        checker.errors.extend(saved_errors)

    assert rules == []
    assert any(finding[2] == "VETCFG" for finding in findings)


def test_invalid_gating_regex_fails_closed() -> None:
    """Invalid regexes for release-gating rules must be errors, not skipped warnings."""
    rule = {
        "id": "VET999",
        "description": "Invalid regex should block",
        "pattern_type": "regex",
        "pattern": "(",
        "severity": "warning",
    }

    findings = _isolated_external_findings([], [rule])

    assert findings
    assert findings[0][2] == "VET999"


def test_gating_warning_match_is_reported_as_error() -> None:
    """Release-gating warning rules must remain visible to --errors-only callers."""
    bad_path = _FIXTURE_DIR / "bad_vet301.md"
    rule = {
        "id": "VET999",
        "description": "Raw install command",
        "pattern_type": "regex",
        "pattern": r"pip\s+install\s+\w+",
        "include_globs": [bad_path.relative_to(_ROOT).as_posix()],
        "severity": "warning",
    }

    findings = _isolated_external_findings([bad_path], [rule])

    assert findings[0][2] == "VET999"
    assert findings[0] in checker.errors or "release-gating warning" in findings[0][3]


def test_noqa_suppression_is_line_and_rule_specific(tmp_path: Path) -> None:
    """Inline noqa suppresses only the named rule on the same physical line."""
    fixture = tmp_path / "sample.py"
    fixture.write_text(
        "blocked_call()  # noqa: VET998\n"
        "blocked_call()  # noqa: VET997\n"
        "# noqa: VET998\n"
        "blocked_call()\n",
        encoding="utf-8",
    )
    rule = {
        "id": "VET998",
        "description": "Blocked call",
        "pattern_type": "regex",
        "pattern": r"blocked_call\(",
        "severity": "error",
    }

    findings = _isolated_external_findings([fixture], [rule])

    assert [(finding[1], finding[2]) for finding in findings] == [(2, "VET998"), (4, "VET998")]
