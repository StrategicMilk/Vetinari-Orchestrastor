"""Parametrized fixture tests for the VET rules checker.

Each VET rule has a paired bad/good fixture file in tests/fixtures/vet_rules/.
These tests prove that:
  1. The bad fixture triggers the expected rule code (checker catches violations).
  2. The good fixture does NOT trigger that rule code (checker accepts valid code).

This satisfies the project rule that checks need real failing and passing inputs:
The checker must be exercised with real failing and passing inputs.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

# -- Locate the scripts directory and import the checker ----------------------
# The test file lives at tests/test_vet_rules_fixtures.py.
# The rules checker lives under scripts/quality/ relative to this file.
_TESTS_DIR = Path(__file__).parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts" / "quality"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_vetinari_rules as checker

# -- Fixture directory -------------------------------------------------------
FIXTURES_DIR = _TESTS_DIR / "fixtures" / "vet_rules"

# -- Parametrize tables -------------------------------------------------------

# (rule_id, bad_fixture_filename, good_fixture_filename, scope)
#
# scope values:
#   "any"                  — call checker.check_file() directly, no scope patching
#   "vetinari_scoped"      — monkeypatch is_in_vetinari to return True so that
#                            rules guarded by is_in_vetinari() fire
#   "vetinari_analytics"   — monkeypatch is_in_vetinari=True + expand _ANALYTICS_DIRS
#                            to include the fixture's parent dir so VET220 fires
#   "not_in_tests"         — monkeypatch is_in_tests to return False so that rules
#                            that skip test files (VET030, VET040, VET041) fire on
#                            fixtures that live under tests/fixtures/
#   "vetinari_not_in_tests" — monkeypatch both is_in_vetinari=True and
#                            is_in_tests=False (VET230 requires both)
#   "hot_path"             — monkeypatch _HOT_PATH_FILES to include the fixture
#                            so that VET130 fires
#   "init_py"              — copy the fixture content to a temp __init__.py so
#                            that filename-gated rules (VET110) fire; also patches
#                            is_in_vetinari=True for the temp file path
#   "vet142_scoped"        — monkeypatch is_in_vetinari=True + expand _VET142_SCOPE_DIRS
#                            to include the fixture's parent dir so VET142 fires on
#                            files that live under tests/fixtures/ (not web/ or safety/)
SINGLE_FILE_RULES: list[tuple[str, str, str, str]] = [
    ("VET001", "bad_vet001.py", "good_vet001.py", "any"),
    ("VET002", "bad_vet002.py", "good_vet002.py", "any"),
    ("VET003", "bad_vet003.py", "good_vet003.py", "any"),
    ("VET004", "bad_vet004.py", "good_vet004.py", "any"),
    ("VET005", "bad_vet005.py", "good_vet005.py", "any"),
    ("VET006", "bad_vet006.py", "good_vet006.py", "any"),
    ("VET010", "bad_vet010.py", "good_vet010.py", "vetinari_scoped"),
    ("VET020", "bad_vet020.py", "good_vet020.py", "any"),
    ("VET022", "bad_vet022.py", "good_vet022.py", "any"),
    ("VET023", "bad_vet023.py", "good_vet023.py", "vetinari_scoped"),
    ("VET024", "bad_vet024.py", "good_vet024.py", "vetinari_scoped"),
    ("VET025", "bad_vet025.py", "good_vet025.py", "vetinari_scoped"),
    ("VET030", "bad_vet030.py", "good_vet030.py", "not_in_tests"),
    ("VET031", "bad_vet031.py", "good_vet031.py", "any"),
    ("VET032", "bad_vet032.py", "good_vet032.py", "any"),
    ("VET033", "bad_vet033.py", "good_vet033.py", "any"),
    ("VET034", "bad_vet034.py", "good_vet034.py", "not_in_tests"),
    ("VET035", "bad_vet035.py", "good_vet035.py", "vetinari_scoped"),
    ("VET036", "bad_vet036.py", "good_vet036.py", "not_in_tests"),
    ("VET040", "bad_vet040.py", "good_vet040.py", "not_in_tests"),
    ("VET041", "bad_vet041.py", "good_vet041.py", "not_in_tests"),
    ("VET050", "bad_vet050.py", "good_vet050.py", "vetinari_scoped"),
    ("VET051", "bad_vet051.py", "good_vet051.py", "vetinari_scoped"),
    ("VET060", "bad_vet060.py", "good_vet060.py", "vetinari_scoped"),
    ("VET061", "bad_vet061.py", "good_vet061.py", "vetinari_scoped"),
    ("VET062", "bad_vet062.py", "good_vet062.py", "vetinari_scoped"),
    ("VET063", "bad_vet063.py", "good_vet063.py", "vetinari_scoped"),
    ("VET070", "bad_vet070.py", "good_vet070.py", "vetinari_scoped"),
    ("VET082", "bad_VET082.py", "good_vet082.py", "any"),
    ("VET090", "bad_vet090.py", "good_vet090.py", "vetinari_scoped"),
    ("VET091", "bad_vet091.py", "good_vet091.py", "vetinari_scoped"),
    ("VET092", "bad_vet092.py", "good_vet092.py", "vetinari_scoped"),
    ("VET093", "bad_vet093.py", "good_vet093.py", "vetinari_scoped"),
    ("VET094", "bad_vet094.py", "good_vet094.py", "vetinari_scoped"),
    ("VET095", "bad_vet095.py", "good_vet095.py", "vetinari_scoped"),
    ("VET096", "bad_vet096.py", "good_vet096.py", "vetinari_scoped"),
    ("VET103", "bad_vet103.py", "good_vet103.py", "vetinari_scoped"),
    ("VET104", "bad_vet104.py", "good_vet104.py", "vetinari_scoped"),
    ("VET105", "bad_vet105.py", "good_vet105.py", "vetinari_scoped"),
    ("VET106", "bad_vet106.py", "good_vet106.py", "vetinari_scoped"),
    ("VET107", "bad_vet107.py", "good_vet107.py", "vetinari_scoped"),
    ("VET108", "bad_vet108.py", "good_vet108.py", "vetinari_scoped"),
    ("VET110", "bad_vet110.py", "good_vet110.py", "init_py"),
    ("VET111", "bad_vet111.py", "good_vet111.py", "vetinari_scoped"),
    ("VET112", "bad_vet112.py", "good_vet112.py", "vetinari_scoped"),
    ("VET113", "bad_vet113.py", "good_vet113.py", "vetinari_scoped"),
    ("VET114", "bad_vet114.py", "good_vet114.py", "vetinari_scoped"),
    ("VET115", "bad_vet115.py", "good_vet115.py", "vetinari_scoped"),
    ("VET116", "bad_vet116.py", "good_vet116.py", "vetinari_scoped"),
    ("VET130", "bad_vet130.py", "good_vet130.py", "hot_path"),
    ("VET141", "bad_vet141.py", "good_vet141.py", "vetinari_scoped"),
    ("VET142", "vet142_bad.py", "vet142_good.py", "vet142_scoped"),
    ("VET210", "bad_vet210.py", "good_vet210.py", "vetinari_scoped"),
    ("VET220", "bad_vet220.py", "good_vet220.py", "vetinari_analytics"),
    ("VET230", "bad_vet230.py", "good_vet230.py", "vetinari_not_in_tests"),
]

MARKDOWN_RULES: list[tuple[str, str, str]] = [
    ("VET100", "bad_vet100.md", "good_vet100.md"),
    ("VET101", "bad_vet101.md", "good_vet101.md"),
    ("VET102", "bad_vet102.md", "good_vet102.md"),
]


# -- Helpers ------------------------------------------------------------------


def _apply_scope_patch(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    fixture_path: Path,
    tmp_path: Path | None = None,
) -> Path:
    """Apply the appropriate monkeypatch for the given rule scope.

    Scoped rules guard their checks behind helper predicates that inspect the
    file path.  Without patching, a fixture in tests/fixtures/vet_rules/ would
    fail those path checks and the rule would never fire.

    Args:
        monkeypatch: pytest monkeypatch fixture for automatic teardown.
        scope: One of "any", "vetinari_scoped", "vetinari_analytics",
            "vet142_scoped", "not_in_tests", "vetinari_not_in_tests", "hot_path",
            or "init_py".
        fixture_path: Absolute path to the fixture file being tested.
        tmp_path: pytest tmp_path fixture for temporary file creation; required
            when scope is "init_py".

    Returns:
        The effective path to pass to check_file().  Normally this is the
        original fixture_path; for "init_py" it is a temp __init__.py copy
        that satisfies the filename-based guard inside the rule.
    """
    if scope == "vetinari_scoped":
        # Make is_in_vetinari() always return True so vetinari-scoped rules fire
        # on the fixture path even though it lives under tests/fixtures/.
        monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)

    elif scope == "vetinari_analytics":
        # VET220 requires is_in_vetinari=True AND the file to be in an analytics
        # directory (or have "metrics"/"quality" in its basename).  Patch both
        # is_in_vetinari and _ANALYTICS_DIRS to include the fixture's parent dir
        # so the rule fires on files under tests/fixtures/vet_rules/.
        monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
        patched_dirs = checker._ANALYTICS_DIRS | {fixture_path.parent.name}
        monkeypatch.setattr(checker, "_ANALYTICS_DIRS", patched_dirs)

    elif scope == "vet142_scoped":
        # VET142 requires is_in_vetinari=True AND the file path to start with
        # vetinari/web/ or vetinari/safety/.  Patch both is_in_vetinari and
        # _VET142_SCOPE_DIRS to include a prefix that the fixture path satisfies,
        # so the rule fires on files under tests/fixtures/vet_rules/.
        monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
        rel_prefix = os.path.relpath(str(fixture_path.parent), str(checker.project_root)).replace("\\", "/")
        patched_scope_dirs = checker._VET142_SCOPE_DIRS | {rel_prefix}
        monkeypatch.setattr(checker, "_VET142_SCOPE_DIRS", patched_scope_dirs)

    elif scope == "vetinari_not_in_tests":
        # VET230 requires is_in_vetinari=True AND is_in_tests=False.
        # Patch both so the rule fires on fixtures that live under tests/.
        monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
        monkeypatch.setattr(checker, "is_in_tests", lambda _: False)

    elif scope == "not_in_tests":
        # VET030, VET040, VET041 skip files under tests/ via is_in_tests().
        # Patch to return False so those rules fire on fixtures in tests/fixtures/.
        monkeypatch.setattr(checker, "is_in_tests", lambda _: False)

    elif scope == "hot_path":
        # VET130 only fires for files whose relative path is in _HOT_PATH_FILES.
        # Add the fixture's relative path (forward-slash normalised) to the set.
        rel_path = os.path.relpath(str(fixture_path), str(checker.project_root)).replace("\\", "/")
        patched_set = checker._HOT_PATH_FILES | {rel_path}
        monkeypatch.setattr(checker, "_HOT_PATH_FILES", patched_set)

    elif scope == "init_py":
        # VET110 checks Path(filepath).name == "__init__.py" before firing.
        # Copy the fixture content to a temporary __init__.py so the filename
        # guard passes, then patch is_in_vetinari so the outer vetinari check
        # also passes.  tmp_path is a pytest-managed temp dir cleaned up
        # automatically after each test.
        assert tmp_path is not None, "init_py scope requires tmp_path fixture"
        tmp_init = tmp_path / "__init__.py"
        shutil.copy2(str(fixture_path), str(tmp_init))
        monkeypatch.setattr(checker, "is_in_vetinari", lambda _: True)
        return tmp_init

    # scope == "any": no patching needed
    return fixture_path


def _findings_for(
    fixture_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    tmp_path: Path | None = None,
) -> list[tuple]:
    """Run the checker against a fixture file and return findings.

    Applies any required scope patch before calling check_file() so that
    path-guarded rules are triggered appropriately.  For "init_py" scope,
    _apply_scope_patch returns a temp __init__.py path to check instead of
    the original fixture path.

    Args:
        fixture_path: Absolute path to the .py fixture file.
        monkeypatch: pytest monkeypatch fixture for automatic teardown.
        scope: Rule scope ("any", "vetinari_scoped", "vet142_scoped",
            "not_in_tests", "vetinari_not_in_tests", "hot_path", or "init_py").
        tmp_path: pytest tmp_path fixture; required when scope is "init_py".

    Returns:
        List of (filepath, line, code, message) findings from check_file().
    """
    effective_path = _apply_scope_patch(monkeypatch, scope, fixture_path, tmp_path)
    return checker.check_file(effective_path)


# -- Strategy A: bad fixtures must trigger the rule --------------------------

_bad_params = [
    pytest.param(rule_id, FIXTURES_DIR / bad_file, scope, id=f"{rule_id}-bad")
    for rule_id, bad_file, _good, scope in SINGLE_FILE_RULES
]


@pytest.mark.parametrize("rule_id,bad_path,scope", _bad_params)
def test_bad_fixture_triggers_rule(
    rule_id: str,
    bad_path: Path,
    scope: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that the bad fixture for each rule is caught by the checker.

    A bad fixture file is written to contain an intentional violation of the
    rule under test.  This test asserts the checker returns at least one
    finding with the expected rule code, proving the rule is live and capable
    of catching real violations.

    Args:
        rule_id: The VET rule code expected in the findings (e.g. "VET001").
        bad_path: Absolute path to the bad fixture file.
        scope: Scope tag controlling which monkeypatches are applied.
        monkeypatch: pytest monkeypatch fixture for automatic teardown.
        tmp_path: pytest-managed temporary directory for "init_py" scope.
    """
    if not bad_path.exists():
        pytest.skip(f"Fixture not yet created: {bad_path.name}")

    findings = _findings_for(bad_path, monkeypatch, scope, tmp_path)
    codes = [f[2] for f in findings]

    assert rule_id in codes, (
        f"Expected rule {rule_id} to fire on {bad_path.name} but got: {codes!r}\nAll findings: {findings!r}"
    )


# -- Strategy B: good fixtures must NOT trigger the rule ---------------------

_good_params = [
    pytest.param(rule_id, FIXTURES_DIR / good_file, scope, id=f"{rule_id}-good")
    for rule_id, _bad, good_file, scope in SINGLE_FILE_RULES
]


@pytest.mark.parametrize("rule_id,good_path,scope", _good_params)
def test_good_fixture_passes_rule(
    rule_id: str,
    good_path: Path,
    scope: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that the good fixture for each rule is NOT flagged by the checker.

    A good fixture contains code that deliberately avoids the violation the
    rule detects.  This test asserts the checker does not report the rule code
    for that file, proving the rule has a correct acceptance boundary.

    Args:
        rule_id: The VET rule code that must NOT appear in findings.
        good_path: Absolute path to the good fixture file.
        scope: Scope tag controlling which monkeypatches are applied.
        monkeypatch: pytest monkeypatch fixture for automatic teardown.
        tmp_path: pytest-managed temporary directory for "init_py" scope.
    """
    if not good_path.exists():
        pytest.skip(f"Fixture not yet created: {good_path.name}")

    findings = _findings_for(good_path, monkeypatch, scope, tmp_path)
    codes = [f[2] for f in findings]

    assert rule_id not in codes, (
        f"Rule {rule_id} fired on good fixture {good_path.name} — the rule is too broad.\nAll findings: {findings!r}"
    )


# -- Strategy C: markdown rules via _check_single_markdown --------------------

_md_bad_params = [
    pytest.param(rule_id, FIXTURES_DIR / bad_file, id=f"{rule_id}-md-bad")
    for rule_id, bad_file, _good in MARKDOWN_RULES
]

_md_good_params = [
    pytest.param(rule_id, FIXTURES_DIR / good_file, id=f"{rule_id}-md-good")
    for rule_id, _bad, good_file in MARKDOWN_RULES
]


def _md_findings_for(fixture_path: Path) -> list[tuple]:
    """Run the markdown checker against a single .md fixture file.

    check_file() only processes .py files (it calls scan_file which uses
    ast.parse).  Markdown files must go through _check_single_markdown()
    directly.  This helper saves and restores the module-level result
    collectors so the call is isolated.

    Args:
        fixture_path: Absolute path to the .md fixture file.

    Returns:
        List of (filepath, line, code, message) findings — errors and
        warnings combined — produced while checking this file.
    """
    saved_errors = checker.errors[:]
    saved_warnings = checker.warnings[:]

    checker.errors.clear()
    checker.warnings.clear()

    try:
        checker._check_single_markdown(fixture_path)
        findings: list[tuple] = [*checker.errors, *checker.warnings]
    finally:
        checker.errors.clear()
        checker.errors.extend(saved_errors)
        checker.warnings.clear()
        checker.warnings.extend(saved_warnings)

    return findings


@pytest.mark.parametrize("rule_id,bad_path", _md_bad_params)
def test_markdown_bad_fixture_triggers_rule(rule_id: str, bad_path: Path) -> None:
    """Verify that the bad markdown fixture triggers the expected VET rule.

    Uses _check_single_markdown() directly to exercise the markdown checker
    on a single file without scanning entire docs/ directories.

    Args:
        rule_id: The VET rule code expected in the findings (e.g. "VET100").
        bad_path: Absolute path to the bad .md fixture file.
    """
    if not bad_path.exists():
        pytest.skip(f"Fixture not yet created: {bad_path.name}")

    findings = _md_findings_for(bad_path)
    codes = [f[2] for f in findings]

    assert rule_id in codes, (
        f"Expected markdown rule {rule_id} to fire on {bad_path.name} but got: {codes!r}\nAll findings: {findings!r}"
    )


@pytest.mark.parametrize("rule_id,good_path", _md_good_params)
def test_markdown_good_fixture_passes_rule(rule_id: str, good_path: Path) -> None:
    """Verify that the good markdown fixture is NOT flagged by the checker.

    Uses _check_single_markdown() directly to exercise the markdown checker
    on a single file without scanning entire docs/ directories.

    Args:
        rule_id: The VET rule code that must NOT appear in findings.
        good_path: Absolute path to the good .md fixture file.
    """
    if not good_path.exists():
        pytest.skip(f"Fixture not yet created: {good_path.name}")

    findings = _md_findings_for(good_path)
    codes = [f[2] for f in findings]

    assert rule_id not in codes, (
        f"Markdown rule {rule_id} fired on good fixture {good_path.name} — the rule is too broad.\n"
        f"All findings: {findings!r}"
    )
