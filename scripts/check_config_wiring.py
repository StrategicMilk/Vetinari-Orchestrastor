#!/usr/bin/env python3
"""Config Wiring Checker for Vetinari.

Parses all YAML/YML config files under config/ and checks whether each config key
is actually referenced in vetinari/ Python source code.  Flags keys that no
Python code consumes (write-only config).

  VET250  Config key defined in YAML but not referenced in any Python file

Exit codes:
  0  — no errors found (warnings may be present)
  1  — one or more errors found

Usage:
  python scripts/check_config_wiring.py [--errors-only] [--verbose]

Notes:
  - Only top-level section keys and their direct leaf children are checked.
    Deep nesting (3+ levels) is skipped to avoid false positives from
    generic key names (``enabled``, ``threshold``, etc.).
  - Structural / schema meta-keys are always exempted (see EXEMPT_KEYS).
  - A key is considered "referenced" only in live Python expressions after
    comments and docstrings are stripped.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    print(
        "error: PyYAML is not installed. Run: pip install pyyaml",  # noqa: VET301 — user guidance string
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VETINARI_DIR = PROJECT_ROOT / "vetinari"
CONFIG_DIR = PROJECT_ROOT / "config"

SKIP_DIRS = {"__pycache__", "venv", ".git", "vetinari.egg-info", "build", "dist"}

ERROR = "ERROR"
WARNING = "WARNING"

# Keys that are purely structural / schema metadata — never flag these.
# These appear in many YAML files but are not expected in Python source.
EXEMPT_KEYS: frozenset[str] = frozenset({
    # Schema / JSON-Schema meta
    "$schema",
    "$ref",
    "$defs",
    "$id",
    # Generic structural keys common across all config formats
    "description",
    "type",
    "default",
    "example",
    "examples",
    "required",
    "optional",
    "nullable",
    "format",
    "enum",
    "properties",
    "items",
    "additionalProperties",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    # YAML-document metadata
    "version",
    "comment",
    "notes",
    # Deeply generic names that appear in dozens of config files and
    # are always resolved indirectly (e.g. by iterating dict.items()).
    "enabled",
    "threshold",
    "model",
    "models",
    "topics",
    "rules",
    "style",
    "tone",
    "person",
})

# Top-level config keys that correspond to entire config-file sections.
# These are always loaded by name (e.g. config["thompson_sampling"]), so
# a section key is "referenced" if ANY key inside that section is used.
# We check section keys only at depth=1; deeper structural nesting is
# checked at depth=2 (direct children of sections).
MAX_DEPTH = 2  # check top-level keys (depth 1) and their direct children (depth 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_python_files(directory: Path) -> list[Path]:
    """Return sorted list of .py files under directory, skipping cache dirs.

    Args:
        directory: Root directory to scan.

    Returns:
        Sorted list of Path objects for every .py file found.
    """
    results = []
    for f in sorted(directory.rglob("*.py")):
        if any(skip in f.parts for skip in SKIP_DIRS):
            continue
        results.append(f)
    return results


def _collect_keys(
    data: dict[str, Any],
    current_depth: int = 1,
    prefix: str = "",
) -> list[tuple[str, str, int]]:
    """Recursively extract config keys up to MAX_DEPTH.

    Args:
        data: Parsed YAML dict to traverse.
        current_depth: Current nesting depth (1 = top-level).
        prefix: Dot-separated key path accumulated so far.

    Returns:
        List of (full_dotted_path, leaf_key_name, depth) tuples.
    """
    results: list[tuple[str, str, int]] = []
    for key, value in data.items():
        full_path = f"{prefix}.{key}" if prefix else key
        results.append((full_path, key, current_depth))
        if isinstance(value, dict) and current_depth < MAX_DEPTH:
            results.extend(_collect_keys(value, current_depth + 1, full_path))
    return results


def _strip_comments(source: str) -> str:
    """Remove comment lines and inline comments from Python source.

    Strips lines that are pure comments (``# ...``) and the comment portion of
    lines that have inline comments.  Does not attempt full tokenisation — this
    is a best-effort filter sufficient to prevent false-positive key matches on
    commented-out references.

    Args:
        source: Raw Python source text.

    Returns:
        Source text with comment content removed.
    """
    result: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # Entire line is a comment — drop it
            result.append("")
        else:
            # Remove inline comment, but only outside string literals (rough heuristic:
            # find first unquoted '#' by scanning left-to-right with a simple state machine)
            in_single = False
            in_double = False
            out: list[str] = []
            i = 0
            while i < len(line):
                ch = line[i]
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == "#" and not in_single and not in_double:
                    break
                out.append(ch)
                i += 1
            result.append("".join(out))
    return "\n".join(result)


def _strip_comments_and_strings(source: str) -> str:
    """Remove comments and docstring content from Python source.

    Removes:
    - Pure comment lines (lines starting with #)
    - Inline comments (text after # outside of strings)
    - Content of triple-quoted docstrings only

    Regular single/double-quoted strings are preserved because they may
    contain config keys that are live references (e.g., config["key"]).

    This ensures that:
    - Keys mentioned only in comments are not matched as live references
    - Keys mentioned only in docstrings are not matched as live references
    - Keys in actual function arguments like config["key"] are still matched

    Args:
        source: Raw Python source text.

    Returns:
        Source text with comment and docstring content removed.
    """
    result: list[str] = []
    in_triple_double = False
    in_triple_single = False

    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # Entire line is a comment
            result.append("")
            continue

        out: list[str] = []
        in_single = False
        in_double = False
        i = 0

        while i < len(line):
            # Check for triple quotes first (before single characters)
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

            # If currently inside a triple-quoted string, skip the character (replace with space)
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
                # Start of comment — stop processing this line
                break
            else:
                out.append(ch)
            i += 1

        result.append("".join(out))

    return "\n".join(result)


def _build_python_corpus(files: list[Path]) -> str:
    """Concatenate all Python source into a single searchable string.

    Comment lines and inline comments are stripped so that keys mentioned
    only in comments are not treated as live references (D5).

    Args:
        files: List of Python source files to read.

    Returns:
        Concatenated source text (comments removed) of all files.
    """
    parts: list[str] = []
    for f in files:
        with contextlib.suppress(OSError):
            raw = f.read_text(encoding="utf-8")
            parts.append(_strip_comments_and_strings(raw))
    return "\n".join(parts)


def _key_is_referenced(key: str, corpus: str) -> bool:
    """Return True if the key appears in a real code expression in the corpus.

    Requires the key to appear in one of the following expression contexts (D6):
    - Dict/attribute access: ``config["key"]``, ``cfg['key']``, ``config.key``
    - ``.get()`` call: ``.get("key")`` or ``.get('key')``
    - Assignment target: ``key =`` or ``key=``
    - Function argument: ``(key,``, ``(key)``

    A bare word-boundary match is intentionally NOT sufficient — it would fire
    on string literals and comments that merely mention the key text.

    Args:
        key: The config key string to search for.
        corpus: Concatenated Python source (comments already stripped).

    Returns:
        True if any live-code reference is found.
    """
    ek = re.escape(key)
    patterns = [
        # dict access: ["key"] or ['key']
        re.compile(r"""\[["']""" + ek + r"""["']\]"""),
        # .get("key") or .get('key')
        re.compile(r"""\.get\(\s*["']""" + ek + r"""["']"""),
        # attribute access: .key (preceded by dot, followed by non-word or end)
        re.compile(r"""\.""" + ek + r"""\b"""),
        # assignment: key = ... (config key being set or unpacked)
        re.compile(r"""\b""" + ek + r"""\s*="""),
        # function argument or tuple: (key, or (key) or , key,
        re.compile(r"""[\(,]\s*""" + ek + r"""\s*[\),]"""),
    ]
    return any(bool(p.search(corpus)) for p in patterns)


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


def check_config_file(
    config_path: Path,
    python_corpus: str,
    verbose: bool = False,
) -> list[tuple[str, int, str, str, str]]:
    """Parse one YAML config file and report unreferenced keys.

    Args:
        config_path: Path to the YAML config file.
        python_corpus: Concatenated Python source to search within.
        verbose: If True, print each key as it is checked.

    Returns:
        List of (filepath_str, line_number, code, severity, message) tuples.
        Line number is always 1 (YAML doesn't track line numbers easily).
    """
    violations: list[tuple[str, int, str, str, str]] = []

    try:
        raw = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        violations.append((
            str(config_path),
            1,
            "VET250",
            ERROR,
            f"Failed to parse YAML: {exc}",
        ))
        return violations
    except OSError as exc:
        violations.append((
            str(config_path),
            1,
            "VET250",
            ERROR,
            f"Cannot read file: {exc}",
        ))
        return violations

    if not isinstance(data, dict):
        # Not a mapping — nothing to check
        return violations

    keys = _collect_keys(data)

    for full_path, leaf_key, _depth in keys:
        if leaf_key in EXEMPT_KEYS:
            if verbose:
                print(f"    exempt: {full_path}")
            continue

        # Skip very short keys (1-2 chars) — too generic, high false-positive rate
        if len(leaf_key) <= 2:
            if verbose:
                print(f"    skip (too short): {full_path}")
            continue

        if verbose:
            print(f"    checking: {full_path}")

        if not _key_is_referenced(leaf_key, python_corpus):
            violations.append((
                str(config_path),
                1,
                "VET250",
                WARNING,
                f'Config key "{full_path}" (leaf: "{leaf_key}") in '
                f"{config_path.name} is not referenced in any vetinari/ Python file",
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
        description="Detect config keys in config/*.yaml not referenced in vetinari/ (VET250).",
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
        help="Print each config key as it is checked.",
    )
    return parser


def main() -> int:
    """Parse config/ YAML files and cross-reference against vetinari/ Python source.

    Returns:
        0 if no errors found, 1 if errors found.
    """
    args = _build_parser().parse_args()

    if not CONFIG_DIR.exists():
        print(f"error: config directory not found: {CONFIG_DIR}", file=sys.stderr)
        return 1
    if not VETINARI_DIR.exists():
        print(f"error: vetinari directory not found: {VETINARI_DIR}", file=sys.stderr)
        return 1

    # Build corpus once (expensive) before iterating config files
    python_files = _iter_python_files(VETINARI_DIR)
    if args.verbose:
        print(f"Loading {len(python_files)} Python source files...")
    python_corpus = _build_python_corpus(python_files)

    config_files = sorted({*CONFIG_DIR.rglob("*.yaml"), *CONFIG_DIR.rglob("*.yml")})
    all_violations: list[tuple[str, int, str, str, str]] = []

    for config_path in config_files:
        if args.verbose:
            try:
                rel = config_path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = config_path
            print(f"  checking {rel}")

        violations = check_config_file(config_path, python_corpus, verbose=args.verbose)
        all_violations.extend(violations)

    # Filter by severity
    if args.errors_only:
        display = [(fp, ln, code, sev, msg) for fp, ln, code, sev, msg in all_violations if sev == ERROR]
    else:
        display = all_violations

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
        n_configs = len(config_files)
        print(
            f"\n{n_configs} config file{'s' if n_configs != 1 else ''} checked against {len(python_files)} Python files — {status}"
        )

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
