"""Tests for the program-master plan_type handler in scripts/check_agent_routing.py.

Covers:

- Good fixture passes validation.
- Bad fixture (missing required section) fails with the expected error text.
- Existing plan packs are not regressed by the new handler.
- Direct unit tests of `extract_owned_scopes` and `audit_program_scope_disjointness`
  helpers added for the program tier.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_agent_routing.py"

# Make the helpers importable for unit tests without depending on the script being a package.
sys.path.insert(0, str(ROOT / "scripts"))
import check_agent_routing as car


def _run_validator(plan_relpath: str) -> tuple[int, str]:
    """Run check_agent_routing.py against a single plan and return (exit_code, combined_output).

    Inputs are fully controlled: sys.executable + the repo's own validator script
    path + a plan_relpath chosen from the in-repo fixture set.
    """
    proc = subprocess.run(  # noqa: S603 -- controlled inputs: own script + fixture paths
        [sys.executable, str(SCRIPT), "--plan", plan_relpath],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@pytest.mark.parametrize(
    "plan_relpath",
    [
        "tests/fixtures/program_master/good/PROGRAM.md",
    ],
)
def test_good_program_master_passes(plan_relpath: str) -> None:
    code, out = _run_validator(plan_relpath)
    assert code == 0, f"good program-master fixture should pass; output:\n{out}"


@pytest.mark.parametrize(
    ("plan_relpath", "expected_error_substring"),
    [
        (
            "tests/fixtures/program_master/bad_missing_section/PROGRAM.md",
            "Cross-Pack Dependencies",
        ),
    ],
)
def test_bad_program_master_missing_section_fails(plan_relpath: str, expected_error_substring: str) -> None:
    code, out = _run_validator(plan_relpath)
    assert code != 0, f"bad fixture should fail; output:\n{out}"
    assert expected_error_substring in out, f"expected '{expected_error_substring}' in failure output; got:\n{out}"


@pytest.mark.parametrize(
    "plan_relpath",
    [
        ".ai-codex/plans/2026-04-25-program-tier/MASTER-INDEX.md",
    ],
)
def test_existing_master_plans_unaffected(plan_relpath: str) -> None:
    """Targeted runs on existing in-tree plans must still pass after adding program-master."""
    code, out = _run_validator(plan_relpath)
    assert code == 0, f"{plan_relpath} regressed; output:\n{out}"


def test_program_master_in_hierarchical_set() -> None:
    assert "program-master" in car.HIERARCHICAL_PLAN_TYPES


def test_program_master_required_sections_complete() -> None:
    sections = car.REQUIRED_PLAN_SECTIONS["program-master"]
    assert "Load Rule" in sections
    assert "Program Waves" in sections
    assert "Pack Registry" in sections
    assert "Cross-Pack Dependencies" in sections
    assert "Routing Map" in sections


def test_is_child_plan_link_recognizes_master_index() -> None:
    assert car.is_child_plan_link("program-master", "CHILD/MASTER-INDEX.md") is True
    assert car.is_child_plan_link("program-master", "child/master-index.md") is True
    assert car.is_child_plan_link("program-master", "SESSION-01/SESSION.md") is False
    assert car.is_child_plan_link("program-master", "SHARD-01.md") is False


def test_extract_owned_scopes_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    fake = tmp_path / "missing" / "MASTER-INDEX.md"
    assert car.extract_owned_scopes(fake) == set()


def test_extract_owned_scopes_pulls_paths_from_session_file(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    master = pack / "MASTER-INDEX.md"
    master.write_text("# stub\n", encoding="utf-8")
    session_dir = pack / "SESSION-01-test"
    session_dir.mkdir()
    session = session_dir / "SESSION.md"
    session.write_text(
        "## Owned Write Scope\n\n- `vetinari/foo.py`\n- `tests/test_foo.py`\n\n## Next\n",
        encoding="utf-8",
    )
    scopes = car.extract_owned_scopes(master)
    assert "vetinari/foo.py" in scopes
    assert "tests/test_foo.py" in scopes


def test_audit_program_scope_disjointness_flags_overlap(tmp_path: Path) -> None:
    pack_a = tmp_path / "pack_a"
    pack_a.mkdir()
    (pack_a / "MASTER-INDEX.md").write_text("# stub\n", encoding="utf-8")
    sd_a = pack_a / "SESSION-01"
    sd_a.mkdir()
    (sd_a / "SESSION.md").write_text("## Owned Write Scope\n\n- `shared/file.py`\n\n## Next\n", encoding="utf-8")

    pack_b = tmp_path / "pack_b"
    pack_b.mkdir()
    (pack_b / "MASTER-INDEX.md").write_text("# stub\n", encoding="utf-8")
    sd_b = pack_b / "SESSION-01"
    sd_b.mkdir()
    (sd_b / "SESSION.md").write_text("## Owned Write Scope\n\n- `shared/file.py`\n\n## Next\n", encoding="utf-8")

    program = tmp_path / "PROGRAM.md"
    program.write_text("# stub\n", encoding="utf-8")
    failures = car.audit_program_scope_disjointness(program, [pack_a / "MASTER-INDEX.md", pack_b / "MASTER-INDEX.md"])
    assert failures, "expected at least one disjointness failure"
    assert any("scope overlap" in f.message for f in failures)


def test_audit_program_scope_disjointness_clean_when_disjoint(tmp_path: Path) -> None:
    pack_a = tmp_path / "pack_a"
    pack_a.mkdir()
    (pack_a / "MASTER-INDEX.md").write_text("# stub\n", encoding="utf-8")
    sd_a = pack_a / "SESSION-01"
    sd_a.mkdir()
    (sd_a / "SESSION.md").write_text("## Owned Write Scope\n\n- `unique/a.py`\n\n## Next\n", encoding="utf-8")

    pack_b = tmp_path / "pack_b"
    pack_b.mkdir()
    (pack_b / "MASTER-INDEX.md").write_text("# stub\n", encoding="utf-8")
    sd_b = pack_b / "SESSION-01"
    sd_b.mkdir()
    (sd_b / "SESSION.md").write_text("## Owned Write Scope\n\n- `unique/b.py`\n\n## Next\n", encoding="utf-8")

    program = tmp_path / "PROGRAM.md"
    program.write_text("# stub\n", encoding="utf-8")
    failures = car.audit_program_scope_disjointness(program, [pack_a / "MASTER-INDEX.md", pack_b / "MASTER-INDEX.md"])
    assert failures == []
