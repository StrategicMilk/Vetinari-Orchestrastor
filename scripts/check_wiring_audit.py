#!/usr/bin/env python3
"""Wiring Audit Checker for Vetinari.

Detects wiring and structural issues in the Vetinari source:

  VET200  Singleton get_X() factory missing double-checked locking
  VET201  __init__.py re-exports without __all__ declaration
  VET202  Event published via event_bus.publish() with no subscriber
  VET203  Event subscribed via event_bus.subscribe() with no publisher

Exit codes:
  0  — no errors found (warnings may be present)
  1  — one or more errors found

Usage:
  python scripts/check_wiring_audit.py [--errors-only] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VETINARI_DIR = PROJECT_ROOT / "vetinari"

SKIP_DIRS = {"__pycache__", "venv", ".git", "vetinari.egg-info", "build", "dist"}

# Severity levels for output formatting
ERROR = "ERROR"
WARNING = "WARNING"

# Pattern: def get_<name>( possibly followed by args/type hints/return type
_GET_FUNC_DEF = re.compile(r"^def (get_\w+)\s*\(")

# Pattern for global _ statement inside a function body
_GLOBAL_UNDERSCORE = re.compile(r"\bglobal\s+_\w+")

# Pattern for with-lock usage (double-checked locking)
_WITH_LOCK = re.compile(r"\bwith\s+\w*[Ll]ock\b")

# Patterns for event bus publish / subscribe calls
# Matches get_event_bus().publish("event_name", ...) or bus.publish("event_name", ...)
_PUBLISH_CALL = re.compile(r"""\.publish\(\s*["']([^"']+)["']""")
_SUBSCRIBE_CALL = re.compile(r"""\.subscribe\(\s*["']([^"']+)["']""")

# Matches event_type= keyword form: .publish(event_type="foo", ...)
_PUBLISH_KW = re.compile(r"""\.publish\(.*?event_type\s*=\s*["']([^"']+)["']""")
_SUBSCRIBE_KW = re.compile(r"""\.subscribe\(.*?event_type\s*=\s*["']([^"']+)["']""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_python_files(directory: Path) -> list[Path]:
    """Return sorted list of .py files under directory, skipping cache dirs.

    Args:
        directory: Root directory to walk.

    Returns:
        Sorted list of Path objects for every .py file found.
    """
    results = []
    for f in sorted(directory.rglob("*.py")):
        if any(skip in f.parts for skip in SKIP_DIRS):
            continue
        results.append(f)
    return results


def _read_lines(filepath: Path) -> list[str] | None:
    """Read a file and return its lines, or None on error.

    Args:
        filepath: Path to the file.

    Returns:
        List of lines (with newlines stripped) or None if unreadable.
    """
    try:
        return filepath.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# VET200 — Singleton missing double-checked locking
# ---------------------------------------------------------------------------


def check_singleton_locking(filepath: Path, lines: list[str]) -> list[tuple[int, str, str, str]]:
    """Find get_X() singletons that use 'global _x' without double-checked locking.

    A proper singleton factory must use a threading.Lock to avoid race
    conditions when multiple threads call the factory simultaneously.  The
    required pattern is::

        _instance = None
        _lock = threading.Lock()
        def get_X():
            global _instance
            if _instance is None:
                with _lock:
                    if _instance is None:
                        _instance = X()
            return _instance

    This check flags any ``def get_X()`` whose body contains ``global _``
    but lacks any ``with <lock>:`` usage.

    Args:
        filepath: Path to the file being checked.
        lines: Lines of the file content.

    Returns:
        List of (line_number, code, severity, message) tuples.
    """
    violations: list[tuple[int, str, str, str]] = []

    # Parse function boundaries and bodies
    func_start: int | None = None
    func_name: str = ""
    func_body: list[str] = []

    # D7: nil-check pattern for double-checked locking (outer + inner both required)
    _NIL_CHECK = re.compile(r"\bif\s+_\w+\s+is\s+None\s*:")

    def _check_body(name: str, start: int, body: list[str]) -> None:
        body_text = "\n".join(body)
        if not _GLOBAL_UNDERSCORE.search(body_text):
            return
        # D7: require both with-lock AND two separate if _instance is None: checks
        has_lock = bool(_WITH_LOCK.search(body_text))
        nil_checks = len(_NIL_CHECK.findall(body_text))
        if not has_lock or nil_checks < 2:
            violations.append((
                start,
                "VET200",
                WARNING,
                f"get_{name.split('get_', 1)[-1]}() uses 'global _' without "
                "double-checked locking (requires outer + inner 'if _x is None:' "
                "and 'with <lock>:')",
            ))

    for i, line in enumerate(lines, 1):
        # Detect top-level function definitions (no indentation)
        m = _GET_FUNC_DEF.match(line)
        if m:
            # Flush previous function
            if func_start is not None and func_body:
                _check_body(func_name, func_start, func_body)
            func_name = m.group(1)
            func_start = i
            func_body = []
            continue

        if func_start is not None:
            # Function body ends when we see a non-indented non-blank line
            stripped = line.lstrip()
            if stripped and not line.startswith((" ", "\t")):
                # End of current function
                _check_body(func_name, func_start, func_body)
                func_start = None
                func_name = ""
                func_body = []
                # Check if this new line is itself a get_ function
                m2 = _GET_FUNC_DEF.match(line)
                if m2:
                    func_name = m2.group(1)
                    func_start = i
                    func_body = []
            else:
                func_body.append(line)

    # Flush last function
    if func_start is not None and func_body:
        _check_body(func_name, func_start, func_body)

    return violations


# ---------------------------------------------------------------------------
# VET201 — __init__.py re-exports without __all__
# ---------------------------------------------------------------------------


def check_init_all(filepath: Path, lines: list[str]) -> list[tuple[int, str, str, str]]:
    """Find __init__.py files that re-export symbols without declaring __all__.

    Every package ``__init__.py`` that contains ``from X import Y`` statements
    should declare ``__all__`` so that callers using ``from package import *``
    get a predictable public API.

    Args:
        filepath: Path to the __init__.py file.
        lines: Lines of the file content.

    Returns:
        List of (line_number, code, severity, message) tuples (at most one).
    """
    if filepath.name != "__init__.py":
        return []

    has_reexport = any(re.match(r"\s*from\s+\S+\s+import\s+", line) for line in lines)
    has_all = any("__all__" in line for line in lines)

    if has_reexport and not has_all:
        return [
            (
                1,
                "VET201",
                WARNING,
                f"{filepath.name} re-exports symbols with 'from X import Y' but has no __all__ declaration",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# VET202 / VET203 — EventBus publish/subscribe pairing
# ---------------------------------------------------------------------------


def _strip_comments_and_docstrings(line: str) -> str:
    """Remove comments and docstring content from a single line of code.

    Strips:
    - Inline comments (text after # outside of strings)
    - Content of triple-quoted strings (docstrings)

    Regular single/double-quoted strings are preserved because they may
    contain event names that are live references.

    Args:
        line: A single line of source code.

    Returns:
        Line with comment and docstring content removed.
    """
    in_triple_double = False
    in_triple_single = False
    in_single = False
    in_double = False
    out: list[str] = []
    i = 0

    while i < len(line):
        # Check for triple quotes first
        if i + 2 < len(line):
            triple = line[i:i + 3]
            if triple == '"""' and not in_single and not in_triple_single:
                in_triple_double = not in_triple_double
                out.append('"""')
                i += 3
                continue
            elif triple == "'''" and not in_double and not in_triple_double:
                in_triple_single = not in_triple_single
                out.append("'''")
                i += 3
                continue

        # If in triple-quoted string, skip character
        if in_triple_double or in_triple_single:
            out.append(" ")
            i += 1
            continue

        ch = line[i]
        if ch == "'" and not in_double and not in_triple_double:
            in_single = not in_single
            out.append(ch)
        elif ch == '"' and not in_single and not in_triple_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "#" and not in_single and not in_double and not in_triple_double and not in_triple_single:
            # Start of comment — stop processing
            break
        else:
            out.append(ch)
        i += 1

    return "".join(out)


def _collect_event_names(lines: list[str], pattern: re.Pattern[str], kw_pattern: re.Pattern[str]) -> set[str]:
    """Extract event name string literals from matching call sites.

    Args:
        lines: Source lines to scan.
        pattern: Regex matching positional event name argument.
        kw_pattern: Regex matching keyword event_type= argument.

    Returns:
        Set of event name strings found.
    """
    names: set[str] = set()
    for line in lines:
        names.update(m.group(1) for m in pattern.finditer(line))
        names.update(m.group(1) for m in kw_pattern.finditer(line))
    return names


def check_event_bus_pairing(
    all_files: list[Path],
) -> list[tuple[str, int, str, str, str]]:
    """Detect unpaired event_bus publish/subscribe calls across all files.

    Scans the full codebase and reports:
    - VET202: event names published but never subscribed
    - VET203: event names subscribed but never published

    Args:
        all_files: List of all Python files in the scanned directory.

    Returns:
        List of (filepath_str, line_number, code, severity, message) tuples.
    """
    # Collect all publish sites: {event_name: [(filepath, line_num), ...]}
    publish_sites: dict[str, list[tuple[str, int]]] = {}
    # Collect all subscribe sites: {event_name: [(filepath, line_num), ...]}
    subscribe_sites: dict[str, list[tuple[str, int]]] = {}

    for filepath in all_files:
        raw_lines = _read_lines(filepath)
        if raw_lines is None:
            continue
        rel = str(filepath)
        # D8+D9: join continuation lines and skip comments before pattern matching
        logical_lines: list[tuple[int, str]] = []
        pending: list[str] = []
        start_lineno: int = 0
        open_parens: int = 0
        for i, raw in enumerate(raw_lines, 1):
            stripped = raw.lstrip()
            # Skip pure comment lines
            if stripped.startswith("#"):
                continue
            # Remove inline comments and docstrings while preserving string literals
            code_part = _strip_comments_and_docstrings(raw)
            if not pending:
                start_lineno = i
            pending.append(code_part)
            open_parens += code_part.count("(") - code_part.count(")")
            has_continuation = code_part.rstrip().endswith("\\")
            if has_continuation or open_parens > 0:
                continue
            # Logical line complete
            logical = " ".join(part.rstrip("\\").strip() for part in pending)
            logical_lines.append((start_lineno, logical))
            pending = []
            open_parens = 0
        # Flush any trailing unclosed logical line
        if pending:
            logical = " ".join(part.rstrip("\\").strip() for part in pending)
            logical_lines.append((start_lineno, logical))

        for lineno, line in logical_lines:
            for m in _PUBLISH_CALL.finditer(line):
                publish_sites.setdefault(m.group(1), []).append((rel, lineno))
            for m in _PUBLISH_KW.finditer(line):
                publish_sites.setdefault(m.group(1), []).append((rel, lineno))
            for m in _SUBSCRIBE_CALL.finditer(line):
                subscribe_sites.setdefault(m.group(1), []).append((rel, lineno))
            for m in _SUBSCRIBE_KW.finditer(line):
                subscribe_sites.setdefault(m.group(1), []).append((rel, lineno))

    violations: list[tuple[str, int, str, str, str]] = []

    # VET202: published but not subscribed
    for event_name, sites in sorted(publish_sites.items()):
        if event_name not in subscribe_sites:
            for fp, ln in sites:
                violations.append((
                    fp,
                    ln,
                    "VET202",
                    WARNING,
                    f'Event "{event_name}" is published but has no subscriber in the codebase',
                ))

    # VET203: subscribed but not published
    for event_name, sites in sorted(subscribe_sites.items()):
        if event_name not in publish_sites:
            for fp, ln in sites:
                violations.append((
                    fp,
                    ln,
                    "VET203",
                    WARNING,
                    f'Event "{event_name}" is subscribed but never published in the codebase',
                ))

    return violations


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Detect wiring issues in Vetinari source (VET200-VET203).",
        epilog="Exit 0 if no errors, exit 1 if any ERROR-level violations found.",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Suppress warnings; only report ERROR-level violations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file as it is scanned.",
    )
    return parser


def main() -> int:
    """Scan vetinari/ for wiring violations and report results.

    Returns:
        0 if no errors found, 1 if errors found.
    """
    args = _build_parser().parse_args()

    if not VETINARI_DIR.exists():
        print(f"error: directory not found: {VETINARI_DIR}", file=sys.stderr)
        return 1

    all_files = _iter_python_files(VETINARI_DIR)

    # (filepath_str, line_num, code, severity, message)
    all_violations: list[tuple[str, int, str, str, str]] = []

    files_checked = 0
    for filepath in all_files:
        lines = _read_lines(filepath)
        if lines is None:
            continue
        files_checked += 1

        if args.verbose:
            try:
                rel = filepath.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = filepath
            print(f"  scanning {rel}")

        rel_str = str(filepath)

        # VET200: singleton locking
        for ln, code, sev, msg in check_singleton_locking(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        # VET201: __init__.py missing __all__
        for ln, code, sev, msg in check_init_all(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

    # VET202/VET203: event bus pairing (cross-file, run once)
    for fp, ln, code, sev, msg in check_event_bus_pairing(all_files):
        all_violations.append((fp, ln, code, sev, msg))

    # Filter by severity if --errors-only
    if args.errors_only:
        display = [(fp, ln, code, sev, msg) for fp, ln, code, sev, msg in all_violations if sev == ERROR]
    else:
        display = all_violations

    # Print sorted by file then line
    for fp, ln, code, sev, msg in sorted(display, key=lambda v: (v[0], v[1])):
        print(f"{fp}:{ln}: {code} [{sev}] {msg}")

    error_count = sum(1 for _, _, _, sev, _ in all_violations if sev == ERROR)
    warning_count = sum(1 for _, _, _, sev, _ in all_violations if sev == WARNING)

    if not args.errors_only:
        status_parts = []
        if error_count:
            status_parts.append(f"{error_count} error{'s' if error_count != 1 else ''}")
        if warning_count:
            status_parts.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")
        status = ", ".join(status_parts) if status_parts else "clean"
        print(f"\n{files_checked} file{'s' if files_checked != 1 else ''} checked — {status}")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
