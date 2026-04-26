"""Tests for the noqa suppression inventory gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "quality"
MODULE_PATH = SCRIPTS_DIR / "check_noqa_suppressions.py"
SPEC = importlib.util.spec_from_file_location("check_noqa_suppressions", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
check_noqa_suppressions = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_noqa_suppressions
SPEC.loader.exec_module(check_noqa_suppressions)


def test_collect_noqa_directives_ignores_string_literals(tmp_path: Path) -> None:
    """Only real comments are treated as suppressions."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        'TEXT = "value  # noqa: F401"\n'
        "import os  # noqa: F401 - imported for monkeypatch target\n",
        encoding="utf-8",
    )

    directives = check_noqa_suppressions.collect_noqa_directives([sample])

    assert len(directives) == 1
    assert directives[0].line == 2
    assert directives[0].codes == ("F401",)
    assert directives[0].has_rationale is True


def test_evaluate_directives_blocks_blanket_and_unlisted_file_level() -> None:
    """Blanket and file-level suppressions require stronger review."""
    blanket = check_noqa_suppressions.NoqaDirective(
        path="sample.py",
        line=10,
        comment="# noqa",
        codes=(),
    )
    file_level = check_noqa_suppressions.NoqaDirective(
        path="sample.py",
        line=1,
        comment="# noqa: F401 - generated compatibility shim",
        codes=("F401",),
    )

    findings = check_noqa_suppressions.evaluate_directives([blanket, file_level])

    assert any("blanket noqa is forbidden" in finding for finding in findings)
    assert any("file-level noqa requires" in finding for finding in findings)


def test_evaluate_directives_requires_rationale_only_on_new_lines() -> None:
    """Existing debt can remain, but newly added suppressions need rationale text."""
    old = check_noqa_suppressions.NoqaDirective(
        path="sample.py",
        line=2,
        comment="# noqa: F401",
        codes=("F401",),
    )
    new = check_noqa_suppressions.NoqaDirective(
        path="sample.py",
        line=3,
        comment="# noqa: F401",
        codes=("F401",),
    )

    findings = check_noqa_suppressions.evaluate_directives(
        [old, new],
        changed_lines={"sample.py": {3}},
    )

    assert findings == ["sample.py:3: noqa requires a rationale, e.g. '# noqa: F401 - reason'"]
