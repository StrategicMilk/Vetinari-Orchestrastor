#!/usr/bin/env python3
"""Audit noqa suppressions and block new unreviewable ones."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = ("scripts", "tests", "vetinari", "conftest.py", "setup.py")
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv312",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "vetinari.egg-info",
}

# Existing debt that must remain visible and rare. New file-level suppressions
# should be fixed structurally or added here only with a concrete rationale.
FILE_LEVEL_ALLOWLIST: dict[tuple[str, int, tuple[str, ...]], str] = {}


@dataclass(frozen=True)
class NoqaDirective:
    """A single active noqa directive found in a Python comment."""

    path: str
    line: int
    comment: str
    codes: tuple[str, ...]

    @property
    def signature(self) -> tuple[str, int, tuple[str, ...]]:
        """Stable allowlist key for this directive."""
        return (self.path, self.line, self.codes)

    @property
    def is_file_level(self) -> bool:
        """Return True for line-1 or ruff file-level directives."""
        return self.line == 1 or self.comment.startswith("# ruff: noqa")

    @property
    def has_rationale(self) -> bool:
        """Return True when text after the directive explains why it exists."""
        marker = "# ruff: noqa" if self.comment.startswith("# ruff: noqa") else "# noqa"
        after = self.comment.split(marker, 1)[1]
        if ":" in after:
            after = after.split(":", 1)[1]
            for code in self.codes:
                after = after.replace(code, "")
            after = after.replace(",", " ")
        return any(char.isalpha() for char in after)


def _iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        full_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not full_path.exists():
            continue
        if full_path.is_file() and full_path.suffix == ".py":
            files.append(full_path)
            continue
        for candidate in full_path.rglob("*.py"):
            if any(part in SKIP_DIRS for part in candidate.relative_to(PROJECT_ROOT).parts):
                continue
            files.append(candidate)
    return sorted(files)


def _extract_codes(comment: str) -> tuple[str, ...]:
    directive = "# ruff: noqa" if comment.startswith("# ruff: noqa") else "# noqa"
    after = comment.split(directive, 1)[1]
    if ":" not in after:
        return ()
    raw_codes = after.split(":", 1)[1].replace(",", " ").split()
    return tuple(code for code in raw_codes if code and code[0].isalpha() and any(char.isdigit() for char in code))


def collect_noqa_directives(paths: list[Path]) -> list[NoqaDirective]:
    """Collect active noqa comments from Python files."""
    directives: list[NoqaDirective] = []
    for path in _iter_python_files(paths):
        try:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        try:
            with tokenize.open(path) as handle:
                tokens = tokenize.generate_tokens(handle.readline)
                for token in tokens:
                    comment = token.string.strip()
                    if token.type != tokenize.COMMENT:
                        continue
                    if "# noqa" not in comment and "# ruff: noqa" not in comment:
                        continue
                    directives.append(
                        NoqaDirective(
                            path=rel,
                            line=token.start[0],
                            comment=comment,
                            codes=_extract_codes(comment),
                        )
                    )
        except (OSError, tokenize.TokenError, SyntaxError):
            continue
    return directives


def staged_added_lines() -> dict[str, set[int]]:
    """Return Python line numbers added in the staged diff."""
    git_exe = shutil.which("git") or "git"
    proc = subprocess.run(  # noqa: S603 - git executable is resolved from PATH with fixed read-only arguments.
        [git_exe, "diff", "--cached", "--unified=0", "--", "*.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        return {}

    changed: dict[str, set[int]] = {}
    current_path: str | None = None
    new_line: int | None = None
    for raw_line in proc.stdout.splitlines():
        if raw_line.startswith("+++ b/"):
            current_path = raw_line.removeprefix("+++ b/").replace("\\", "/")
            changed.setdefault(current_path, set())
            continue
        if raw_line.startswith("@@"):
            marker = raw_line.split("+", 1)[1].split(" ", 1)[0]
            start = marker.split(",", 1)[0]
            new_line = int(start)
            continue
        if current_path is None or new_line is None:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            changed.setdefault(current_path, set()).add(new_line)
            new_line += 1
        elif not raw_line.startswith("-"):
            new_line += 1
    return changed


def evaluate_directives(
    directives: list[NoqaDirective],
    changed_lines: dict[str, set[int]] | None = None,
    *,
    enforce_all_rationales: bool = False,
) -> list[str]:
    """Return policy violations for collected directives."""
    changed_lines = changed_lines or {}
    findings: list[str] = []
    for directive in directives:
        location = f"{directive.path}:{directive.line}"
        if not directive.codes:
            findings.append(f"{location}: blanket noqa is forbidden; specify rule codes and rationale")
            continue
        if directive.is_file_level and directive.signature not in FILE_LEVEL_ALLOWLIST:
            findings.append(f"{location}: file-level noqa requires an explicit allowlist entry")
        rationale_required = enforce_all_rationales or directive.line in changed_lines.get(directive.path, set())
        if rationale_required and not directive.has_rationale:
            findings.append(f"{location}: noqa requires a rationale, e.g. '# noqa: {directive.codes[0]} - reason'")
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the suppression inventory gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Optional paths to scan")
    parser.add_argument(
        "--all-rationales",
        action="store_true",
        help="Require rationale text on every existing noqa, not just newly staged lines.",
    )
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.paths] if args.paths else [Path(root) for root in DEFAULT_ROOTS]
    directives = collect_noqa_directives(paths)
    findings = evaluate_directives(
        directives,
        staged_added_lines(),
        enforce_all_rationales=args.all_rationales,
    )

    if findings:
        print("Noqa suppression policy failed:")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print(f"Noqa suppression policy passed ({len(directives)} active directives scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
