#!/usr/bin/env python3
"""Monkey-Chain Anti-Pattern Detector for Vetinari.

Detects three anti-patterns from the 140-agent audit (2026-03-28) that
caused 500+ bugs:

  MC001  String literals used in place of enum values (monkey chains)
  MC002  Raw task dict access without normalization
  MC003  Write-only context keys (written but never read in same file)

Exit codes:
  0  — no violations found
  1  — one or more violations found

Usage:
  python scripts/check_monkey_chains.py [directory] [--errors-only] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# -- Configuration ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VETINARI_DIR = PROJECT_ROOT / "vetinari"

# Agent type strings that should use AgentType enum
AGENT_TYPE_LITERALS = {"FOREMAN", "WORKER", "INSPECTOR"}

# Status strings that should use StatusEnum enum
STATUS_LITERALS = {
    "pending",
    "blocked",
    "ready",
    "assigned",
    "in_progress",
    "running",
    "completed",
    "failed",
    "cancelled",
    "waiting",
    "skipped",
}

# Files exempt from checking (enum definitions, config, docs)
EXEMPT_FILES = {
    VETINARI_DIR / "types.py",  # enum definitions themselves
}

# Task dict keys that must be normalized before access (from anti-patterns audit)
TASK_DICT_KEYS = {"inputs", "outputs", "dependencies", "id", "agent_type", "type"}

# Context variable names that trigger write-only detection
CONTEXT_VAR_NAMES = {"context", "ctx", "shared_context", "task_context"}

# Context keys consumed cross-file (context dicts are passed between modules).
# These are intentionally written in one module and read in another.
CROSS_FILE_CONTEXT_KEYS = {
    "mode",  # consumed by multi_mode_agent.py
    "prior_memories",  # consumed by inference.py via _current_task_memories
    "cross_validate",  # consumed by collaboration pipeline consumers
    "prior_adrs",  # consumed by worker_agent handler dispatch
    "dependency_results",  # consumed by downstream task handlers
    "incorporated_results",  # consumed by rework decision logic
    "info_response",  # consumed by agent info aggregation
    "original_output",  # consumed by rework comparison
    "verification_issues",  # consumed by quality gate evaluation
    "intake_pattern_key",  # consumed by pattern-matched routing
    "_exec_id",  # consumed by durable execution correlation
    "_rules_prefix",  # consumed by rules-based routing
    "routing_decision",  # consumed by stage selection logic
    "complexity",  # consumed by model selection routing
    "add_stages",  # consumed by pipeline extension logic
}

# Patterns in lines that are exempt (docstrings, comments, prompt templates)
EXEMPT_LINE_PATTERNS = [
    r"^\s*#",  # comment lines
    r'^\s*"""',  # docstring boundary
    r"^\s*'''",  # docstring boundary
    r"^\s*\.\.\.",  # Ellipsis (type stubs)
    r"e\.g\.",  # example references in docstrings
    r"Args:",  # docstring sections
    r"Returns:",
    r"Raises:",
    r"Example:",
    r"noqa:\s*MC",  # inline suppression
]


def _is_exempt_line(line: str) -> bool:
    """Check if a line is exempt from monkey chain detection."""
    stripped = line.strip()
    return any(re.match(pattern, stripped) for pattern in EXEMPT_LINE_PATTERNS)


def _check_agent_type_literals(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str]]:
    """Find agent type string literals that should use AgentType enum.

    Args:
        filepath: Path to the file being checked.
        lines: Lines of the file.

    Returns:
        List of (line_number, literal_found, line_text) violations.
    """
    violations = []
    in_docstring = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track docstring boundaries
        triple_count = stripped.count('"""') + stripped.count("'''")
        if triple_count % 2 == 1:
            in_docstring = not in_docstring

        if in_docstring or _is_exempt_line(line):
            continue

        for agent in AGENT_TYPE_LITERALS:
            # Match "FOREMAN" or 'FOREMAN' but not AgentType.FOREMAN
            pattern = rf"""(?<!\w)["']{agent}["']"""
            if re.search(pattern, line):
                # Skip if it's already using the enum
                if f"AgentType.{agent}" in line:
                    continue
                # Skip long f-string prompt template lines (natural language, not code)
                if line.strip().startswith(("f'", 'f"', "f'''", 'f"""')) and len(line.strip()) > 80:
                    continue
                violations.append((i, agent, line.rstrip()))

    return violations


def _check_status_literals(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str]]:
    """Find status string literals that should use StatusEnum enum.

    Args:
        filepath: Path to the file being checked.
        lines: Lines of the file.

    Returns:
        List of (line_number, literal_found, line_text) violations.
    """
    violations = []
    in_docstring = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track docstring boundaries
        triple_count = stripped.count('"""') + stripped.count("'''")
        if triple_count % 2 == 1:
            in_docstring = not in_docstring

        if in_docstring or _is_exempt_line(line):
            continue

        for status in STATUS_LITERALS:
            # Match "completed" or 'completed' but not StatusEnum.COMPLETED
            pattern = rf"""(?<!\w)["']{status}["']"""
            if re.search(pattern, line):
                # Skip if it's already using the enum
                if f"StatusEnum.{status.upper()}" in line:
                    continue
                # Skip dict literals in test data, config, or serialization
                if "to_dict" in line or "from_dict" in line:
                    continue
                # Skip SSE event type registry mappings (protocol-level identifiers)
                if "Event" in line and ":" in line:
                    continue
                # Skip event_type field defaults in dataclasses (SSE protocol names)
                if "event_type" in line:
                    continue
                # Skip JSON/YAML config patterns
                if "yaml" in line.lower() or "json" in line.lower():
                    continue
                # Skip lines where this specific status is already using the enum
                # (narrowed: only skip if StatusEnum.STATUS.value is on the line, not any .value)
                if f"StatusEnum.{status.upper()}.value" in line:
                    continue
                # Skip lines with inline noqa suppression
                if "noqa: MC001" in line or "noqa:MC001" in line:
                    continue
                # Skip enum definitions (e.g., FAILED = "failed" in ImprovementStatus)
                if re.match(rf'^\s+{status.upper()}\s*=\s*["\']', line):
                    continue
                violations.append((i, status, line.rstrip()))

    return violations


def _check_raw_task_dict_access(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str]]:
    """Find raw task dict key accesses that bypass normalization guards.

    Detects patterns like ``task["inputs"]`` or ``task['outputs']`` where the
    caller accesses a task dict key without first normalising the dict.  These
    patterns cause KeyError when a task dict is missing expected keys.

    Args:
        filepath: Path to the file being checked.
        lines: Lines of the file.

    Returns:
        List of (line_number, key_found, line_text) violations.
    """
    violations = []
    in_docstring = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        triple_count = stripped.count('"""') + stripped.count("'''")
        if triple_count % 2 == 1:
            in_docstring = not in_docstring

        if in_docstring or _is_exempt_line(line):
            continue

        for key in TASK_DICT_KEYS:
            # Match task["inputs"] or task['inputs'] but not task.get("inputs")
            pattern = rf"""\btask\[["']{key}["']\]"""
            if re.search(pattern, line):
                # Skip if it's a .get() call on the same line (safe access)
                if f'task.get("{key}"' in line or f"task.get('{key}'" in line:
                    continue
                # Skip normalization / definition sites themselves
                if "normalize" in line.lower() or key + '"' + ":" in line or f"'{key}':" in line:
                    continue
                violations.append((i, key, line.rstrip()))

    return violations


def _check_write_only_context_keys(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str]]:
    """Find context dict writes that have no corresponding read in the same file.

    Detects ``context[key] = value`` assignments where the same key is never
    read back via ``context[key]`` or ``context.get(key)`` anywhere in the
    file.  These are write-only paths where computed data is silently discarded.

    Args:
        filepath: Path to the file being checked.
        lines: Lines of the file.

    Returns:
        List of (line_number, key_written, line_text) violations.
    """
    violations = []

    # Collect all write sites: context["key"] = ... or ctx["key"] = ...
    write_pattern = re.compile(r"""\b(?:""" + "|".join(CONTEXT_VAR_NAMES) + r""")\[["'](\w+)["']\]\s*=""")
    # Collect all read sites: context["key"], context.get("key"), or context.get('key')
    read_pattern = re.compile(r"""\b(?:""" + "|".join(CONTEXT_VAR_NAMES) + r""")\[["'](\w+)["']\](?!\s*=)""")
    get_pattern = re.compile(r"""\b(?:""" + "|".join(CONTEXT_VAR_NAMES) + r""")\.get\(["'](\w+)["']""")

    full_text = "\n".join(lines)

    # Find all writes
    writes: dict[str, list[tuple[int, str]]] = {}
    for i, line in enumerate(lines, 1):
        for m in write_pattern.finditer(line):
            key = m.group(1)
            writes.setdefault(key, []).append((i, line.rstrip()))

    if not writes:
        return violations

    # Find all reads (direct access + .get())
    read_keys: set[str] = set()
    read_keys.update(m.group(1) for m in read_pattern.finditer(full_text))
    read_keys.update(m.group(1) for m in get_pattern.finditer(full_text))

    # Report keys that are written but never read
    for key, write_sites in writes.items():
        if key not in read_keys and key not in CROSS_FILE_CONTEXT_KEYS:
            for line_num, line_text in write_sites:
                # Check for inline noqa suppression
                if "noqa: MC003" in line_text or "noqa:MC003" in line_text:
                    continue
                violations.append((line_num, key, line_text))

    return violations


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Detect monkey-chain anti-patterns in Vetinari source.",
        epilog="All three checks (MC001, MC002, MC003) run by default.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=VETINARI_DIR,
        help="Directory to scan (default: vetinari/)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Print only violations, suppress the summary line.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file as it is scanned.",
    )
    return parser


def main() -> int:
    """Scan source directory for monkey chain anti-pattern violations.

    Returns:
        0 if no violations found, 1 if violations found.
    """
    args = _build_parser().parse_args()
    directory: Path = Path(args.directory).resolve()

    if not directory.exists():
        print(f"error: directory not found: {directory}", file=sys.stderr)
        return 1

    all_violations: list[tuple[str, int, str, str]] = []
    files_checked = 0

    py_files = sorted(directory.rglob("*.py"))

    for filepath in py_files:
        if filepath.resolve() in EXEMPT_FILES:
            continue
        if "__pycache__" in str(filepath):
            continue

        if args.verbose:
            try:
                rel = filepath.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = filepath
            print(f"  scanning {rel}")

        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        files_checked += 1
        rel_str = str(filepath)

        # MC001: agent type string literals (always on)
        for line_num, literal, _line_text in _check_agent_type_literals(filepath, lines):
            all_violations.append((
                rel_str,
                line_num,
                "MC001",
                f'String literal "{literal}" should use AgentType.{literal}',
            ))

        # MC001b: status string literals in comparisons
        for line_num, literal, _line_text in _check_status_literals(filepath, lines):
            all_violations.append((
                rel_str,
                line_num,
                "MC001",
                f'String literal "{literal}" should use StatusEnum.{literal.upper()} (or .value comparison)',
            ))

        # MC002: raw task dict access
        for line_num, key, _line_text in _check_raw_task_dict_access(filepath, lines):
            all_violations.append((
                rel_str,
                line_num,
                "MC002",
                f'Raw task["{key}"] access — normalize task dict first or use task.get("{key}", ...)',
            ))

        # MC003: write-only context keys
        for line_num, key, _line_text in _check_write_only_context_keys(filepath, lines):
            all_violations.append((
                rel_str,
                line_num,
                "MC003",
                f'context["{key}"] is written but never read in this file — wire the reader or remove the write',
            ))

    # Print violations sorted by file then line
    for filepath_str, line_num, code, message in sorted(all_violations, key=lambda v: (v[0], v[1])):
        print(f"{filepath_str}:{line_num}: {code} {message}")

    if not args.errors_only:
        n = len(all_violations)
        status = "clean" if n == 0 else f"{n} violation{'s' if n != 1 else ''} found"
        print(f"\n{files_checked} file{'s' if files_checked != 1 else ''} checked — {status}")

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
