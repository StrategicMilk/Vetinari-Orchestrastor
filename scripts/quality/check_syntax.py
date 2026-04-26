#!/usr/bin/env python3
"""Syntax check all Python files in the Vetinari project.

Walks from the project root so that vetinari/, tests/, scripts/, and other
top-level packages are all covered.
"""

from __future__ import annotations

import ast
import contextlib
import os
import pathlib


def main() -> int:
    """Check all Python files for syntax errors.

    Returns:
        Exit code: 0 if no errors, 1 if syntax errors found.
    """
    # Category scripts live under scripts/<category>/, two levels below the
    # project root.
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    errors: list[tuple[str, str]] = []
    checked = 0
    # Skip generated output directories and test artifacts
    skip_dirs = {
        "venv",
        ".git",
        "__pycache__",
        "vetinari.egg-info",
        "build",
        "dist",
        ".pytest_cache",
        "outputs",
        "projects",  # Skip generated artifacts
    }

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, base)

            # Read file — skip on I/O errors (permissions, encoding failures)
            src = None
            with contextlib.suppress(OSError):
                src = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
            if src is None:
                continue

            try:
                ast.parse(src)
                checked += 1
            except SyntaxError as e:
                errors.append((rel, str(e).encode("ascii", "replace").decode()))

    print(f"Checked {checked} Python source files (excl. outputs/)")
    if errors:
        print(f"\nFOUND {len(errors)} SYNTAX ERROR(S):")
        for rel, e in errors:
            print(f"  {rel}: {e}")
        return 1
    print("All source files OK -- no syntax errors found!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
