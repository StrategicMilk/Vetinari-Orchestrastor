#!/usr/bin/env python3
"""Vetinari Project Rules Checker — Custom Linter.

Enforces project-specific rules that ruff and other standard linters cannot.
Scans vetinari/ and tests/ directories for violations.

Usage:
    python scripts/check_vetinari_rules.py [--errors-only] [--verbose] [--fix] [--yaml PATH]

Exit codes:
    0   No errors (warnings may be present).
    1   One or more errors found.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

project_root = Path(__file__).parent.parent
VETINARI_DIR = project_root / "vetinari"
TESTS_DIR = project_root / "tests"
SCRIPTS_DIR = project_root / "scripts"

SKIP_DIRS = {
    "venv",
    ".git",
    "__pycache__",
    "vetinari.egg-info",
    "build",
    "dist",
    ".pytest_cache",
    "outputs",
    "projects",
    ".claude",
    "node_modules",
    "vet_rules",  # Intentional bad-fixture files — tested via check_file() directly
}

# ── Canonical import sources ─────────────────────────────────────────────────
CANONICAL_ENUMS = {"AgentType", "TaskStatus", "ExecutionMode", "PlanStatus"}
CANONICAL_SOURCE = "vetinari.types"

# ── Placeholder patterns ─────────────────────────────────────────────────────
PLACEHOLDER_STRINGS = re.compile(r"\b(placeholder|lorem\s*ipsum|foo|bar|baz)\b", re.IGNORECASE)

# ── Credential patterns ──────────────────────────────────────────────────────
CREDENTIAL_PATTERN = re.compile(
    r"""(?:password|secret|api_key|apikey|token|auth_token|secret_key)\s*=\s*["'][^"']{3,}["']""",
    re.IGNORECASE,
)

# ── Localhost URL pattern ─────────────────────────────────────────────────────
LOCALHOST_PATTERN = re.compile(
    r"""["'](https?://)?(localhost|127\.0\.0\.1)(:\d+)""",
)

# ── TODO/FIXME pattern ───────────────────────────────────────────────────────
TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|TEMP)\b", re.IGNORECASE)
TODO_WITH_ISSUE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|TEMP)\s*\(#\d+\)", re.IGNORECASE)

# ── Root logger pattern ──────────────────────────────────────────────────────
ROOT_LOGGER_PATTERN = re.compile(r"\blogging\.(debug|info|warning|error|critical|exception)\(")

# ── f-string in logger pattern ────────────────────────────────────────────────
FSTRING_LOGGER_PATTERN = re.compile(r"\blogger\.(debug|info|warning|error|critical|exception)\(f[\"']")

# ── open() without encoding ──────────────────────────────────────────────────
OPEN_CALL_PATTERN = re.compile(r"\bopen\s*\(")

# ── os.path.join pattern ─────────────────────────────────────────────────────
OS_PATH_JOIN_PATTERN = re.compile(r"\bos\.path\.join\s*\(")

# ── sleep pattern ─────────────────────────────────────────────────────────────
SLEEP_PATTERN = re.compile(r"\btime\.sleep\s*\(\s*(\d+(?:\.\d+)?)\s*\)")

# ── Debug code patterns ──────────────────────────────────────────────────────
DEBUG_PATTERNS = [
    re.compile(r"\bbreakpoint\s*\("),
    re.compile(r"\bimport\s+pdb\b"),
    re.compile(r"\bpdb\.set_trace\s*\("),
]

# ── AI Anti-Pattern patterns (VET103-116) ────────────────────────────────────
NAIVE_DATETIME_NOW = re.compile(r"\bdatetime\.now\s*\(\s*\)")
DEPRECATED_UTCNOW = re.compile(r"\bdatetime\.utcnow\s*\(")
ENTRY_EXIT_LOG = re.compile(
    r"""\blogger\.\w+\s*\(\s*["'](?:entering|exiting|starting function|leaving function|entering function)""",
    re.IGNORECASE,
)
FORMULAIC_DOCSTRING = re.compile(
    r"""^\s*["]{3}\s*(?:Initialize the \w|Get the \w|Return the \w|Create a new \w|The (?:result|jsonify|f) (?:result|string)|Tuple of results|Api \w+ \w+)\b""",
)
STR_E_RETURN = re.compile(r"""(?:jsonify|return)\s*\(.*\bstr\s*\(\s*e\s*\)""")

# ── Result collectors ─────────────────────────────────────────────────────────
errors = []
warnings = []
checked_files = 0


def add_error(filepath: str, line: int, code: str, message: str) -> None:
    """Record an error (blocks commit/CI)."""
    errors.append((filepath, line, code, message))


def add_warning(filepath: str, line: int, code: str, message: str) -> None:
    """Record a warning (informational, does not block)."""
    warnings.append((filepath, line, code, message))


def has_noqa(line_text: str, code: str) -> bool:
    """Check if a line has a noqa suppression for the given code.

    Supports both single-code (``# noqa: VET006``) and comma-separated
    multi-code (``# noqa: F403, VET006``) noqa annotations.
    """
    noqa_match = re.search(r"#\s*noqa\b", line_text)
    if not noqa_match:
        return False
    noqa_section = line_text[noqa_match.start() :]
    # Blanket noqa with no specific codes
    if ":" not in noqa_section:
        return True
    codes_part = noqa_section.split(":", 1)[1]
    codes = [c.strip() for c in re.split(r"[,\s]+", codes_part) if c.strip()]
    return code in codes


def is_in_vetinari(filepath: str) -> bool:
    """Check if file is under vetinari/ source directory."""
    try:
        Path(filepath).relative_to(VETINARI_DIR)
        return True
    except ValueError:
        return False


def is_in_tests(filepath: str) -> bool:
    """Check if file is under tests/ directory."""
    try:
        Path(filepath).relative_to(TESTS_DIR)
        return True
    except ValueError:
        return False


def is_in_scripts(filepath: str) -> bool:
    """Check if file is under scripts/ directory."""
    try:
        Path(filepath).relative_to(SCRIPTS_DIR)
        return True
    except ValueError:
        return False


def is_cli_module(filepath: str) -> bool:
    """Check if file is a CLI entry point (print allowed)."""
    name = Path(filepath).name
    # preflight.py is CLI-facing: prints hardware/dependency status to stdout at startup
    return name in ("__main__.py", "cli.py", "preflight.py") or name.startswith("cli_")


def is_init_version_only(filepath: str, source: str) -> bool:
    """Check if __init__.py only contains version and simple imports."""
    if Path(filepath).name != "__init__.py":
        return False
    stripped = [line.strip() for line in source.splitlines() if line.strip() and not line.strip().startswith("#")]
    return all(line.startswith(("__version__", "from", "import")) for line in stripped) and len(stripped) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Rule implementations
# ═══════════════════════════════════════════════════════════════════════════════


def check_import_rules(filepath: str, source: str, lines: list[str]) -> None:
    """VET001-006: Import canonicalization rules."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # VET001-004: Enum imported from wrong source
        for enum_name in CANONICAL_ENUMS:
            pattern = rf"from\s+(?!{re.escape(CANONICAL_SOURCE)})\S+\s+import\s+.*\b{enum_name}\b"
            if re.match(pattern, stripped):
                # Allow re-exports in contracts.py itself
                if "contracts.py" in str(filepath):
                    continue
                code_map = {
                    "AgentType": "VET001",
                    "TaskStatus": "VET002",
                    "ExecutionMode": "VET003",
                    "PlanStatus": "VET004",
                }
                code = code_map.get(enum_name, "VET001")
                if not has_noqa(line, code):
                    add_error(filepath, i, code, f"Import {enum_name} from {CANONICAL_SOURCE}, not from this module")

        # VET005: Duplicate enum definition
        if (
            re.match(r"class\s+(AgentType|TaskStatus|ExecutionMode|PlanStatus)\s*\(", stripped)
            and "types.py" not in str(filepath)
            and not has_noqa(line, "VET005")
        ):
            add_error(filepath, i, "VET005", f"Duplicate enum definition — this enum is defined in {CANONICAL_SOURCE}")

        # VET006: Wildcard import from vetinari
        if re.match(r"from\s+vetinari\.\S+\s+import\s+\*", stripped) and not has_noqa(line, "VET006"):
            add_error(filepath, i, "VET006", "Wildcard import from vetinari.* is forbidden")


def check_future_annotations(filepath: str, source: str, lines: list[str]) -> None:
    """VET010: Missing from __future__ import annotations."""
    if not is_in_vetinari(filepath):
        return
    if is_in_scripts(filepath):
        return
    if is_init_version_only(filepath, source):
        return

    has_code = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped != '"""' and stripped != "'''":
            has_code = True
            break

    if not has_code:
        return

    has_future = "from __future__ import annotations" in source
    if not has_future:
        add_warning(filepath, 1, "VET010", "Missing 'from __future__ import annotations' as first import")


def _except_body_has_logging(body: list[ast.stmt]) -> bool:
    """Return True if an except block body contains a logger call."""
    for stmt in body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.func.attr in ("debug", "info", "warning", "error", "critical", "exception")
            ):
                return True
    return False


def _except_body_returns_success(body: list[ast.stmt]) -> bool:
    """Return True if an except block returns a truthy/success indicator."""
    for stmt in body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            val = stmt.value
            # return True
            if isinstance(val, ast.Constant) and val.value is True:
                return True
            # return {"success": True, ...}
            if isinstance(val, ast.Dict):
                for key, value in zip(val.keys, val.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in ("success", "passed", "ok")
                        and isinstance(value, ast.Constant)
                        and value.value is True
                    ):
                        return True
    return False


def check_error_handling(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET020-025: Error handling rules."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue

        lineno = node.lineno

        # VET020: Bare except
        if node.type is None:
            line_text = lines[lineno - 1] if lineno <= len(lines) else ""
            if not has_noqa(line_text, "VET020"):
                add_error(filepath, lineno, "VET020", "Bare 'except:' without exception type")

        # VET022: Empty except block (only pass or ...)
        body = node.body
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...
            ):
                line_text = lines[lineno - 1] if lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET022"):
                    add_error(filepath, lineno, "VET022", "Empty except block (swallowed exception without logging)")

        # VET023: Debug-level exception logging for non-optional failures
        # Catches except blocks that use logger.debug() for real errors
        # (the exception is not something expected/optional)
        if node.type is not None and is_in_vetinari(filepath):
            has_debug_log = False
            has_higher_log = False
            for stmt in body:
                for child in ast.walk(stmt):
                    if not isinstance(child, ast.Call):
                        continue
                    if not isinstance(child.func, ast.Attribute):
                        continue
                    if not isinstance(child.func.value, ast.Name):
                        continue
                    if child.func.value.id != "logger":
                        continue
                    if child.func.attr == "debug":
                        has_debug_log = True
                    elif child.func.attr in ("info", "warning", "error", "critical", "exception"):
                        has_higher_log = True
            if has_debug_log and not has_higher_log:
                # Only flag if the exception type suggests a real failure
                # Skip expected/optional exceptions (NotFound, FileNotFoundError, etc.)
                optional_exceptions = {
                    "FileNotFoundError",
                    "KeyError",
                    "AttributeError",
                    "ImportError",
                    "ModuleNotFoundError",
                    "StopIteration",
                    "TypeError",
                    "ValueError",
                }
                exc_name = ""
                all_optional = False
                if isinstance(node.type, ast.Name):
                    exc_name = node.type.id
                    all_optional = exc_name in optional_exceptions
                elif isinstance(node.type, ast.Attribute):
                    exc_name = node.type.attr
                    all_optional = exc_name in optional_exceptions
                elif isinstance(node.type, ast.Tuple):
                    # Tuple: except (A, B, C) — safe only if ALL are optional
                    exc_names: set[str] = set()
                    for elt in node.type.elts:
                        if isinstance(elt, ast.Name):
                            exc_names.add(elt.id)
                        elif isinstance(elt, ast.Attribute):
                            exc_names.add(elt.attr)
                    exc_name = ", ".join(sorted(exc_names)) if exc_names else "Exception"
                    all_optional = bool(exc_names) and exc_names.issubset(optional_exceptions)
                if not all_optional:
                    line_text = lines[lineno - 1] if lineno <= len(lines) else ""
                    if not has_noqa(line_text, "VET023"):
                        add_warning(
                            filepath,
                            lineno,
                            "VET023",
                            f"Exception ({exc_name or 'Exception'}) logged at DEBUG level "
                            "— use logger.warning/error/exception for real failures",
                        )

        # VET024: Success-masking except block (returns True/success in except)
        if is_in_vetinari(filepath) and _except_body_returns_success(body):
            line_text = lines[lineno - 1] if lineno <= len(lines) else ""
            if not has_noqa(line_text, "VET024"):
                add_warning(
                    filepath,
                    lineno,
                    "VET024",
                    "Except block returns success/True — failures should not be masked as success",
                )

        # VET025: Broad except without logging (has return/continue but no logger call)
        if is_in_vetinari(filepath) and node.type is not None:
            has_early_exit = False
            for stmt in body:
                if isinstance(stmt, (ast.Return, ast.Continue)):
                    has_early_exit = True
                    break
            if has_early_exit and not _except_body_has_logging(body):
                line_text = lines[lineno - 1] if lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET025"):
                    add_warning(
                        filepath,
                        lineno,
                        "VET025",
                        "Except block exits early (return/continue) without logging — "
                        "add logger.warning/exception to aid debugging",
                    )


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function/method has @abstractmethod decorator."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
            return True
    return False


def _is_exception_init(node: ast.FunctionDef, parent: ast.AST) -> bool:
    """Check if this is an __init__ of an Exception subclass."""
    if not isinstance(parent, ast.ClassDef):
        return False
    if node.name != "__init__":
        return False
    for base in parent.bases:
        base_name = ""
        if isinstance(base, ast.Name):
            base_name = base.id
        elif isinstance(base, ast.Attribute):
            base_name = base.attr
        if "Error" in base_name or "Exception" in base_name:
            return True
    return False


def _get_effective_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Strip leading docstring from a function body, return remaining statements."""
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body


def _check_stub_bodies(filepath: str, lines: list[str], tree: ast.Module) -> None:
    """VET031-033: Detect stub function bodies and NotImplementedError."""
    parent_map = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        effective_body = _get_effective_body(node.body)
        if not effective_body:
            continue
        parent = parent_map.get(node)

        if len(effective_body) == 1:
            stmt = effective_body[0]
            line_text = lines[stmt.lineno - 1] if stmt.lineno <= len(lines) else ""

            # VET031: pass as sole body
            if (
                isinstance(stmt, ast.Pass)
                and not _is_abstract_method(node)
                and not _is_exception_init(node, parent)
                and not has_noqa(line_text, "VET031")
            ):
                add_error(
                    filepath, stmt.lineno, "VET031", f"'pass' as sole body of '{node.name}' — implement or remove"
                )

            # VET032: Ellipsis as sole body
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is ...
                and not filepath.endswith(".pyi")
                and not has_noqa(line_text, "VET032")
            ):
                add_error(filepath, stmt.lineno, "VET032", f"'...' as sole body of '{node.name}' — implement or remove")

        # VET033: raise NotImplementedError outside @abstractmethod
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Raise) or stmt.exc is None:
                continue
            exc = stmt.exc
            is_not_impl = (
                isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError"
            ) or (isinstance(exc, ast.Name) and exc.id == "NotImplementedError")
            if is_not_impl and not _is_abstract_method(node):
                line_text = lines[stmt.lineno - 1] if stmt.lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET033"):
                    add_error(filepath, stmt.lineno, "VET033", "raise NotImplementedError outside @abstractmethod")


def _check_line_patterns(filepath: str, lines: list[str], source: str = "") -> None:
    """VET034-036: Line-based completeness checks (placeholders, print, dead code)."""
    in_tests = is_in_tests(filepath)

    # VET034: Placeholder strings (skip tests)
    # Only flag when the placeholder word is used AS a value (bare assignment or
    # short string literal), not when it appears as a parameter name, in longer
    # sentences (agent prompts, docstrings), or in HTML attributes.
    if not in_tests:
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if stripped_line.startswith("#"):
                continue
            match = PLACEHOLDER_STRINGS.search(line)
            if not match:
                continue
            word = match.group(0).lower()
            # Skip: parameter names (placeholder=, placeholder:)
            after = line[match.end() : match.end() + 2].lstrip()
            if after.startswith(("=", ":")):
                continue
            # Skip: HTML data attributes (data-placeholder)
            before_start = max(0, match.start() - 5)
            if "data-" in line[before_start : match.start()]:
                continue
            # Skip: the word appears inside a longer string (sentence context, not a bare value)
            # Heuristic: if there are 3+ other alphabetic words on the same line
            # besides the match, it's likely a sentence/prompt, not a bare placeholder value.
            other_words = re.findall(r"\b[a-zA-Z]{2,}\b", line[: match.start()] + line[match.end() :])
            if len(other_words) >= 3:
                continue
            if not has_noqa(line, "VET034"):
                add_warning(filepath, i, "VET034", f"Possible placeholder string detected: '{word}'")

    # VET035: print() in production code
    # Use AST to find actual print() calls (function-level statements), which
    # naturally skips string literals, docstrings, and multi-line template strings
    # that *mention* print().
    if is_in_vetinari(filepath) and not is_cli_module(filepath):
        try:
            tree_local = ast.parse(source)
        except SyntaxError:
            tree_local = None
        if tree_local is not None:
            for node in ast.walk(tree_local):
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "print"
                ):
                    line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if not has_noqa(line_text, "VET035"):
                        add_error(filepath, node.lineno, "VET035", "print() in production code — use logging instead")

    # VET036: Commented-out code blocks (3+ consecutive lines)
    if not in_tests:
        consecutive = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#!") and len(stripped) > 2:
                consecutive.append((i, stripped[1:].strip()))
            else:
                _check_commented_block(filepath, lines, consecutive)
                consecutive = []
        _check_commented_block(filepath, lines, consecutive)


def _check_commented_block(filepath: str, lines: list[str], consecutive: int) -> None:
    """Helper: check if a consecutive comment block is commented-out code."""
    if len(consecutive) < 3:
        return
    code_text = "\n".join(text for _, text in consecutive)
    try:
        ast.parse(code_text)
        first_line = consecutive[0][0]
        line_text = lines[first_line - 1] if first_line <= len(lines) else ""
        if not has_noqa(line_text, "VET036"):
            add_warning(
                filepath,
                first_line,
                "VET036",
                f"Commented-out code block ({len(consecutive)} lines) — delete dead code",
            )
    except SyntaxError:
        return


def check_completeness(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET030-036: Completeness rules."""
    # VET030: Untracked TODO/FIXME (skip tests)
    if not is_in_tests(filepath):
        for i, line in enumerate(lines, 1):
            if TODO_PATTERN.search(line) and not TODO_WITH_ISSUE.search(line) and not has_noqa(line, "VET030"):
                add_error(filepath, i, "VET030", "TODO/FIXME/HACK/XXX/TEMP without issue reference (use TODO(#123))")

    _check_stub_bodies(filepath, lines, tree)
    _check_line_patterns(filepath, lines, source)


def check_security(filepath: str, source: str, lines: list[str]) -> None:
    """VET040-041: Security rules."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # VET040: Hardcoded credentials (skip tests — test credentials are not real)
        if CREDENTIAL_PATTERN.search(line) and not is_in_tests(filepath) and not has_noqa(line, "VET040"):
            add_error(filepath, i, "VET040", "Hardcoded credential pattern detected")

        # VET041: Hardcoded localhost URLs
        # Skip: test files, config files, CLI modules, env-var-overridable defaults,
        # CORS origin lists, and localhost identity checks.
        if (
            LOCALHOST_PATTERN.search(line)
            and not is_in_tests(filepath)
            and "config" not in str(filepath).lower()
            and not is_cli_module(filepath)
            and "os.environ.get(" not in line
            and not re.search(r"(?i)cors|allowed_origins", line)
            and "remote" not in stripped.lower()
            and not has_noqa(line, "VET041")
        ):
            add_warning(filepath, i, "VET041", "Hardcoded localhost URL — use config instead")


def check_logging(filepath: str, source: str, lines: list[str]) -> None:
    """VET050-051: Logging rules."""
    if not is_in_vetinari(filepath):
        return

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # VET050: Root logger usage
        if ROOT_LOGGER_PATTERN.search(line) and not has_noqa(line, "VET050"):
            add_warning(
                filepath, i, "VET050", "Using root logger (logging.info()) — use module logger (logger.info()) instead"
            )

        # VET051: f-string in logger call
        if FSTRING_LOGGER_PATTERN.search(line) and not has_noqa(line, "VET051"):
            add_warning(filepath, i, "VET051", "f-string in logger call — use %-style: logger.info('msg %s', val)")


def check_robustness(filepath: str, source: str, lines: list[str]) -> None:
    """VET060-063: Robustness rules."""
    if not is_in_vetinari(filepath):
        return

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # VET060: open() without encoding
        if (
            OPEN_CALL_PATTERN.search(line)
            and "encoding=" not in line
            and "encoding =" not in line
            # Skip binary mode opens
            and not re.search(r"""['"](r?b|['"]*wb|['"]*rb)['""]""", line)
            and not has_noqa(line, "VET060")
        ):
            add_warning(filepath, i, "VET060", "open() without encoding= parameter — use encoding='utf-8'")

        # VET061: Debug code
        for pattern in DEBUG_PATTERNS:
            if pattern.search(line) and not has_noqa(line, "VET061"):
                add_error(filepath, i, "VET061", "Debug code (breakpoint/pdb) — remove before committing")

        # VET062: Long sleep
        match = SLEEP_PATTERN.search(line)
        if match:
            sleep_val = float(match.group(1))
            if sleep_val > 5 and not has_noqa(line, "VET062"):
                add_warning(filepath, i, "VET062", f"time.sleep({sleep_val}) > 5 seconds — use configurable timeout")

        # VET063: os.path.join
        if OS_PATH_JOIN_PATTERN.search(line) and not has_noqa(line, "VET063"):
            add_warning(filepath, i, "VET063", "os.path.join() — prefer pathlib.Path for cross-platform paths")


# Known Python standard library top-level module names (3.10+)
_STDLIB_MODULES = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "atexit",
    "base64",
    "bisect",
    "builtins",
    "calendar",
    "cgi",
    "cmd",
    "codecs",
    "collections",
    "colorsys",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "csv",
    "ctypes",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "dis",
    "email",
    "enum",
    "errno",
    "faulthandler",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getpass",
    "gettext",
    "glob",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "imaplib",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "math",
    "mimetypes",
    "mmap",
    "multiprocessing",
    "netrc",
    "numbers",
    "operator",
    "os",
    "pathlib",
    "pdb",
    "pkgutil",
    "platform",
    "pprint",
    "profile",
    "pstats",
    "queue",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtplib",
    "socket",
    "socketserver",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "struct",
    "subprocess",
    "sys",
    "sysconfig",
    "tarfile",
    "tempfile",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "token",
    "tokenize",
    "tomllib",
    "trace",
    "traceback",
    "tracemalloc",
    "turtle",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "_thread",
    "__future__",
    "typing_extensions",
    # Easter-egg / seldom-used stdlib modules
    "this",
    "antigravity",
    "idlelib",
}


def check_integration(filepath: str, source: str, lines: list[str]) -> None:
    """VET070-072: Integration rules."""
    if not is_in_vetinari(filepath):
        return

    # VET070: Hallucinated imports (non-stdlib, non-project, not in pyproject.toml)
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    else:
        pyproject_text = ""

    # Common package name aliases (import name -> pyproject.toml name)
    pkg_aliases = {
        "yaml": "pyyaml",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
        "bs4": "beautifulsoup4",
        "attr": "attrs",
        "dateutil": "python-dateutil",
        "google": "google-generativeai",
        "gi": "pygobject",
        "ddgs": "ddgs",
        "duckduckgo_search": "ddgs",  # Legacy import path, current package is ddgs.
        "pynvml": "nvidia-ml-py",  # nvidia-ml-py provides the pynvml namespace
    }

    in_multiline_string = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track triple-quoted string state to skip docstring content
        triple_count = stripped.count('"""') + stripped.count("'''")
        if in_multiline_string:
            if triple_count % 2 == 1:
                in_multiline_string = False
            continue
        if triple_count % 2 == 1:
            in_multiline_string = True
            continue

        m = re.match(r"^(?:from\s+(\S+)|import\s+(\S+))", stripped)
        if not m:
            continue
        module = (m.group(1) or m.group(2)).split(".")[0]

        # Skip stdlib, vetinari, relative imports, and template placeholders
        if module in _STDLIB_MODULES or module == "vetinari" or stripped.startswith("from ."):
            continue
        if "{" in module or "}" in module:  # template placeholder like from {target} import X
            continue

        # Check if module appears in pyproject.toml
        if module not in pyproject_text and module.replace("_", "-") not in pyproject_text:
            pkg_name = pkg_aliases.get(module, module)
            if (
                pkg_name not in pyproject_text
                and pkg_name.replace("_", "-") not in pyproject_text
                and not has_noqa(line, "VET070")
            ):
                add_error(filepath, i, "VET070", f"Import '{module}' not found in pyproject.toml dependencies")


def check_organization(filepath: str, source: str, lines: list[str]) -> None:
    """VET080-082: Organization rules."""
    fpath = Path(filepath)

    # VET081: Python directory without __init__.py
    if fpath.suffix == ".py" and is_in_vetinari(filepath):
        parent = fpath.parent
        init_file = parent / "__init__.py"
        if not init_file.exists() and parent != VETINARI_DIR.parent:
            add_error(filepath, 1, "VET081", f"Directory {parent.name}/ has no __init__.py")

    # VET082: Non-snake_case Python filename
    if fpath.suffix == ".py":
        stem = fpath.stem
        if stem.startswith("__") and stem.endswith("__"):
            pass  # Dunder files are ok
        elif stem.startswith("test_") or stem == "conftest":
            pass  # Test files are ok
        elif not re.match(r"^_?[a-z][a-z0-9_]*$", stem):
            # Allow suppression: add a VET082 noqa directive on line 1 of the file
            try:
                first_line = fpath.read_text(encoding="utf-8").splitlines()[0]
            except Exception:
                first_line = ""
            if not has_noqa(first_line, "VET082"):
                add_warning(
                    filepath, 1, "VET082", f"File '{fpath.name}' is not snake_case — rename to '{stem.lower()}.py'"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# J. Documentation Quality Rules (VET090-096)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_docstring(node: ast.AST) -> str | None:
    """Extract docstring from a function, class, or module node."""
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.value.value
    return None


def _get_function_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Get parameter names from a function def, excluding 'self' and 'cls'."""
    return [arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")]


def _has_return_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function has a return statement with a value (not bare return/None)."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            if isinstance(child.value, ast.Constant) and child.value.value is None:
                continue
            return True
    return False


def _has_raise_statement(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function body directly raises exceptions (not in nested funcs)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Raise):
            return True
        # Walk into if/for/with/try but NOT into nested function defs
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for grandchild in ast.walk(child):
                if isinstance(grandchild, ast.Raise):
                    return True
    return False


def _is_public(name: str) -> bool:
    """Check if a name is public (not prefixed with underscore)."""
    return not name.startswith("_")


def _is_property_or_simple(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function is a property, setter, or very simple (single return/pass)."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in ("property", "staticmethod"):
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "getter", "deleter"):
            return True
    # Single-line functions (just return something) don't need full docstrings
    effective = _get_effective_body(node.body)
    return bool(len(effective) == 1 and isinstance(effective[0], ast.Return))


def _is_dunder(name: str) -> bool:
    """Check if name is a dunder method."""
    return name.startswith("__") and name.endswith("__")


def check_documentation(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET090-096: Documentation quality rules."""
    if not is_in_vetinari(filepath):
        return

    # VET095: Module missing module-level docstring
    module_doc = _get_docstring(tree)
    if (
        module_doc is None
        and not is_init_version_only(filepath, source)
        and Path(filepath).name != "__init__.py"
        and not any(has_noqa(lines[0] if lines else "", f"VET09{x}") for x in range(10))
    ):
        add_warning(filepath, 1, "VET095", "Module missing module-level docstring")

    # Walk classes and functions
    parent_map = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _check_class_docstring(filepath, lines, node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parent_map.get(node)
            _check_function_docstring(filepath, lines, node, parent)


def _check_class_docstring(filepath: str, lines: list[str], node: ast.ClassDef) -> None:
    """Check class-level docstring quality."""
    if not _is_public(node.name):
        return

    docstring = _get_docstring(node)
    lineno = node.lineno
    line_text = lines[lineno - 1] if lineno <= len(lines) else ""

    # VET090: Public class missing docstring
    if docstring is None:
        if not has_noqa(line_text, "VET090"):
            add_warning(filepath, lineno, "VET090", f"Public class '{node.name}' missing docstring")
        return

    # VET091: Docstring too short
    stripped_doc = docstring.strip()
    if len(stripped_doc) < 10 and not has_noqa(line_text, "VET091"):
        add_warning(
            filepath, lineno, "VET091", f"Docstring for class '{node.name}' is too short ({len(stripped_doc)} chars)"
        )

    # VET096: Docstring is just the class name repeated
    if stripped_doc.rstrip(".").lower() == node.name.lower() and not has_noqa(line_text, "VET096"):
        add_warning(
            filepath,
            lineno,
            "VET096",
            f"Docstring for class '{node.name}' just restates the name — add meaningful description",
        )


def _check_function_docstring(
    filepath: str, lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef, parent: ast.AST
) -> None:
    """Check function/method docstring quality."""
    if not _is_public(node.name):
        return
    if _is_dunder(node.name) and node.name != "__init__":
        return
    if _is_property_or_simple(node):
        return

    docstring = _get_docstring(node)
    lineno = node.lineno
    line_text = lines[lineno - 1] if lineno <= len(lines) else ""

    # VET090: Public function/method missing docstring
    if docstring is None:
        if not has_noqa(line_text, "VET090"):
            add_warning(filepath, lineno, "VET090", f"Public function '{node.name}' missing docstring")
        return

    # VET091: Docstring too short
    stripped_doc = docstring.strip()
    if len(stripped_doc) < 10 and not has_noqa(line_text, "VET091"):
        add_warning(filepath, lineno, "VET091", f"Docstring for '{node.name}' is too short ({len(stripped_doc)} chars)")

    # VET092: Missing Args section when function has 2+ params
    params = _get_function_params(node)
    if len(params) >= 2 and "Args:" not in docstring and "args:" not in docstring and not has_noqa(line_text, "VET092"):
        add_warning(
            filepath, lineno, "VET092", f"Docstring for '{node.name}' missing Args section ({len(params)} parameters)"
        )

    # VET093: Missing Returns section when function returns non-None
    if (
        _has_return_value(node)
        and "Returns:" not in docstring
        and "returns:" not in docstring
        and "Return:" not in docstring
        and not has_noqa(line_text, "VET093")
    ):
        add_warning(filepath, lineno, "VET093", f"Docstring for '{node.name}' missing Returns section")

    # VET094: Missing Raises section when function raises exceptions
    if (
        _has_raise_statement(node)
        and "Raises:" not in docstring
        and "raises:" not in docstring
        and not has_noqa(line_text, "VET094")
    ):
        add_warning(filepath, lineno, "VET094", f"Docstring for '{node.name}' missing Raises section")


# ═══════════════════════════════════════════════════════════════════════════════
# K. Markdown Documentation Quality Rules (VET100-102)
# ═══════════════════════════════════════════════════════════════════════════════


def check_markdown_files() -> None:
    """VET100-102: Markdown documentation quality rules."""
    docs_dirs = [
        project_root / "docs",
        project_root / ".claude" / "docs",
    ]
    md_files = [md_file for docs_dir in docs_dirs if docs_dir.exists() for md_file in docs_dir.rglob("*.md")]
    # Also check root-level markdown files
    md_files.extend(project_root.glob("*.md"))

    for md_file in md_files:
        _check_single_markdown(md_file)


def _check_single_markdown(filepath: str) -> None:
    """Check a single markdown file for quality issues."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    lines = content.splitlines()
    rel = os.path.relpath(filepath, project_root)

    # Build a set of line indices that are inside fenced code blocks
    in_code_block = set()
    inside = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            if inside:
                in_code_block.add(i)
                inside = False
            else:
                inside = True
                in_code_block.add(i)
        elif inside:
            in_code_block.add(i)

    # VET100: Missing top-level heading
    has_h1 = False
    for i, line in enumerate(lines):
        if i in in_code_block:
            continue
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            has_h1 = True
            break
    if not has_h1 and lines:
        add_warning(str(filepath), 1, "VET100", f"Markdown file '{rel}' missing top-level heading (# Title)")

    # VET101: Empty sections (heading followed by same-or-higher-level heading with no content)
    for i, line in enumerate(lines):
        if i in in_code_block:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            current_level = len(stripped) - len(stripped.lstrip("#"))
            # Look ahead for next non-blank, non-separator, non-code-block line
            for j in range(i + 1, min(i + 5, len(lines))):
                if j in in_code_block:
                    break  # Code block counts as content
                next_stripped = lines[j].strip()
                if next_stripped == "" or next_stripped == "---":
                    continue
                if next_stripped.startswith("#"):
                    next_level = len(next_stripped) - len(next_stripped.lstrip("#"))
                    if next_level <= current_level:
                        add_warning(
                            str(filepath),
                            i + 1,
                            "VET101",
                            f"Empty section in '{rel}' — heading with no content before next heading",
                        )
                break

    # VET102: Very short content (less than 50 non-whitespace chars, excluding headings and code blocks)
    content_chars = 0
    for i, line in enumerate(lines):
        if i in in_code_block:
            continue
        stripped = line.strip()
        if not stripped.startswith("#") and stripped:
            content_chars += len(stripped)
    if content_chars < 50 and lines:
        add_warning(
            str(filepath), 1, "VET102", f"Markdown file '{rel}' has very little content ({content_chars} chars)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AI Anti-Pattern Detection (VET103-115)
# ═══════════════════════════════════════════════════════════════════════════════


def _to_dict_has_custom_logic(func_node: ast.FunctionDef, class_node: ast.ClassDef) -> bool:
    """Return True if a to_dict() method has custom logic beyond field copying.

    Detects: function calls (round, .value, .to_dict), field renaming,
    conditional expressions, list comprehensions, computed properties,
    slicing, and any non-trivial transformation.
    """
    # Collect dataclass field names for comparison
    field_names = {
        item.target.id
        for item in class_node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }

    # Walk the function body looking for any non-trivial operation
    for child in ast.walk(func_node):
        # Function calls (round(), .value, .to_dict(), str(), etc.)
        if isinstance(child, ast.Call):
            # Allow dataclass_to_dict delegation (handled separately)
            if isinstance(child.func, ast.Name) and child.func.id == "dataclass_to_dict":
                continue
            return True
        # List/dict comprehensions
        if isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            return True
        # Conditional expressions (ternary)
        if isinstance(child, ast.IfExp):
            return True
        # If/for/while statements
        if isinstance(child, (ast.If, ast.For, ast.While)):
            return True
        # Slicing / subscript
        if isinstance(child, ast.Subscript):
            return True
        # BinOp (arithmetic, string concat)
        if isinstance(child, ast.BinOp):
            return True
        # BoolOp (or, and) — defensive defaults
        if isinstance(child, ast.BoolOp):
            return True

    # Check for field renaming: dict keys don't match self.field_name
    for child in ast.walk(func_node):
        if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
            for key in child.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value not in field_names:
                    return True  # Key doesn't match any field — renaming

    return False


def _annotation_allows_none(annotation: ast.AST | None) -> bool:
    """Return True when an annotation explicitly permits None."""
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant):
        value = annotation.value
        if value is None:
            return True
        if isinstance(value, str):
            return "None" in value or "Optional" in value
    if isinstance(annotation, ast.Name):
        return annotation.id in {"None", "Optional"}
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Optional"
    if isinstance(annotation, ast.Subscript):
        return _annotation_allows_none(annotation.value) or _annotation_allows_none(annotation.slice)
    if isinstance(annotation, ast.Tuple):
        return any(_annotation_allows_none(elt) for elt in annotation.elts)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_allows_none(annotation.left) or _annotation_allows_none(annotation.right)
    return False


def _non_optional_local_names(tree: ast.Module) -> set[str]:
    """Collect simple local names whose annotations prove they are non-optional."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None and not _annotation_allows_none(node.annotation):
            names.add(node.arg)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not _annotation_allows_none(node.annotation)
        ):
            names.add(node.target.id)
    return names


def check_ai_antipatterns(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """Check for common AI code generation anti-patterns (VET103-115)."""
    if not is_in_vetinari(filepath):
        return
    non_optional_names = _non_optional_local_names(tree)

    for i, line in enumerate(lines, 1):
        # VET103: Timezone-naive datetime.now()
        if NAIVE_DATETIME_NOW.search(line) and not has_noqa(line, "VET103"):
            add_warning(filepath, i, "VET103", "datetime.now() without timezone — use datetime.now(timezone.utc)")

        # VET103b: Deprecated datetime.utcnow()
        if DEPRECATED_UTCNOW.search(line) and not has_noqa(line, "VET103"):
            add_warning(
                filepath, i, "VET103", "datetime.utcnow() is deprecated (Python 3.12) — use datetime.now(timezone.utc)"
            )

        # VET107: Entry/exit logging noise
        if ENTRY_EXIT_LOG.search(line) and not has_noqa(line, "VET107"):
            add_warning(
                filepath, i, "VET107", "Entry/exit logging adds noise — log meaningful state transitions instead"
            )

        # VET116: str(e) leaked to client in web route error handler
        if STR_E_RETURN.search(line) and not has_noqa(line, "VET116"):
            add_warning(filepath, i, "VET116", "str(e) leaked to client — log the error, return generic message")

    if tree is None:
        return

    # VET104: Missing exception chaining (raise X without 'from' in except block)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name:
            for child in ast.walk(node):
                if isinstance(child, ast.Raise) and child.exc is not None and child.cause is None:
                    line_no = child.lineno
                    line_text = lines[line_no - 1] if line_no <= len(lines) else ""
                    if not has_noqa(line_text, "VET104"):
                        add_warning(
                            filepath, line_no, "VET104", f"raise without 'from {node.name}' — exception chain lost"
                        )

    # VET105: Mechanical to_dict() on @dataclass
    # Only flag truly mechanical copies (every dict key matches a self.field
    # with no transformation).  Skip methods that have custom logic: field
    # renaming, rounding, computed properties, conditional fields, nested
    # serialization, slicing, enum .value access, etc.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "dataclass")
                for d in node.decorator_list
            )
            if is_dataclass:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "to_dict":
                        # Skip to_dict() that delegates to dataclass_to_dict()
                        delegates = False
                        for child in ast.walk(item):
                            if (
                                isinstance(child, ast.Call)
                                and isinstance(child.func, ast.Name)
                                and child.func.id == "dataclass_to_dict"
                            ):
                                delegates = True
                                break
                        if delegates:
                            continue
                        # Check if the body has custom logic — if so, skip
                        has_custom_logic = _to_dict_has_custom_logic(item, node)
                        if has_custom_logic:
                            continue
                        line_text = lines[item.lineno - 1] if item.lineno <= len(lines) else ""
                        if not has_noqa(line_text, "VET105"):
                            add_warning(
                                filepath,
                                item.lineno,
                                "VET105",
                                f"Hand-written to_dict() on @dataclass '{node.name}' — use dataclasses.asdict()",
                            )

    # VET106: Zero-logic property (just returns self._x)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            is_property = any(
                (isinstance(d, ast.Name) and d.id == "property")
                or (isinstance(d, ast.Attribute) and d.attr == "getter")
                for d in node.decorator_list
            )
            if is_property and len(node.body) == 1:
                body = node.body[0]
                if isinstance(body, ast.Return) and body.value is not None:
                    val = body.value
                    if (
                        isinstance(val, ast.Attribute)
                        and isinstance(val.value, ast.Name)
                        and val.value.id == "self"
                        and val.attr.startswith("_")
                    ):
                        line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        if not has_noqa(line_text, "VET106"):
                            add_warning(
                                filepath,
                                node.lineno,
                                "VET106",
                                f"Zero-logic @property '{node.name}' just returns self.{val.attr} — use public attribute",
                            )

    # VET108: Redundant boolean conversion (return True if X else False)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.IfExp):
            ifexp = node.value
            if (
                isinstance(ifexp.body, ast.Constant)
                and ifexp.body.value is True
                and isinstance(ifexp.orelse, ast.Constant)
                and ifexp.orelse.value is False
            ):
                line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET108"):
                    add_warning(
                        filepath, node.lineno, "VET108", "Redundant 'return True if X else False' — use 'return X'"
                    )

    # VET110: Empty __init__.py (stub without re-exports)
    if Path(filepath).name == "__init__.py":
        non_trivial_lines = [
            line.strip()
            for line in lines
            if line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("'''")
            and line.strip() != "from __future__ import annotations"
        ]
        # If file has a docstring and future import but nothing else
        if len(non_trivial_lines) <= 1:
            has_all = any("__all__" in line for line in lines)
            has_exports = any(
                line.strip().startswith("from ") and " import " in line
                for line in lines
                if "from __future__" not in line
            )
            if not has_all and not has_exports:
                line_text = lines[0] if lines else ""
                if not has_noqa(line_text, "VET110"):
                    add_warning(filepath, 1, "VET110", "Empty __init__.py — should re-export package's public API")

    # VET111: Redundant intermediate variable (x = func(); return x)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            for i_node in range(len(body) - 1):
                stmt = body[i_node]
                next_stmt = body[i_node + 1]
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(next_stmt, ast.Return)
                    and isinstance(next_stmt.value, ast.Name)
                    and next_stmt.value.id == stmt.targets[0].id
                ):
                    line_text = lines[stmt.lineno - 1] if stmt.lineno <= len(lines) else ""
                    if not has_noqa(line_text, "VET111"):
                        add_warning(
                            filepath,
                            stmt.lineno,
                            "VET111",
                            f"Redundant intermediate variable '{stmt.targets[0].id}' — return directly",
                        )

    # VET113: Missing __repr__ on dataclass with >3 fields
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "dataclass")
                for d in node.decorator_list
            )
            if is_dataclass:
                # Count annotated fields (instance variables)
                field_count = sum(1 for item in node.body if isinstance(item, ast.AnnAssign))
                has_repr = any(isinstance(item, ast.FunctionDef) and item.name == "__repr__" for item in node.body)
                if field_count > 3 and not has_repr:
                    line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if not has_noqa(line_text, "VET113"):
                        add_warning(
                            filepath,
                            node.lineno,
                            "VET113",
                            f"@dataclass '{node.name}' has {field_count} fields but no __repr__ — add meaningful repr showing key fields",
                        )

    # VET114: Mutable value-type dataclass (Event/Spec/Config/Entry without frozen=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            name = node.name
            if any(name.endswith(suffix) for suffix in ("Event", "Spec", "Config", "Entry", "Record")):
                is_dataclass_call = False
                is_frozen = False
                for d in node.decorator_list:
                    if isinstance(d, ast.Call):
                        func = d.func
                        if (isinstance(func, ast.Name) and func.id == "dataclass") or (
                            isinstance(func, ast.Attribute) and func.attr == "dataclass"
                        ):
                            is_dataclass_call = True
                            for kw in d.keywords:
                                if kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    is_frozen = True
                    elif isinstance(d, ast.Name) and d.id == "dataclass":
                        is_dataclass_call = True  # bare @dataclass, no frozen
                if is_dataclass_call and not is_frozen:
                    line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if not has_noqa(line_text, "VET114"):
                        add_warning(
                            filepath, node.lineno, "VET114", f"Value-type @dataclass '{name}' should use frozen=True"
                        )

    # VET112: Defensive 'or ""' / 'or []' on possibly non-optional value
    # Only flag when the left operand is a bare local variable name.
    # Skip attribute access (obj.field — often Optional), function/method calls
    # (return type often Optional), dict.get(), and getattr() — all commonly
    # return Optional values where the defensive default is correct.
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        last_val = node.values[-1]
        is_empty_fallback = (
            (isinstance(last_val, ast.Constant) and last_val.value == "")
            or (isinstance(last_val, ast.Constant) and last_val.value == 0)
            or (isinstance(last_val, ast.List) and not last_val.elts)
            or (isinstance(last_val, ast.Dict) and not last_val.keys)
        )
        if not is_empty_fallback:
            continue
        first_val = node.values[0]
        # Skip attribute access (self.x, obj.field) — commonly Optional
        if isinstance(first_val, ast.Attribute):
            continue
        # Skip function/method calls — return types are commonly Optional
        if isinstance(first_val, ast.Call):
            continue
        # Skip subscripts (dict[key], list[idx]) — may return None
        if isinstance(first_val, ast.Subscript):
            continue
        if not isinstance(first_val, ast.Name) or first_val.id not in non_optional_names:
            continue
        line_no = node.lineno
        line_text = lines[line_no - 1] if line_no <= len(lines) else ""
        if not has_noqa(line_text, "VET112"):
            add_warning(
                filepath,
                line_no,
                "VET112",
                "Defensive 'or \"\"' / 'or []' — only use when value is genuinely Optional (str | None)",
            )

    # VET115: Config file read inside @bp.route / @app.route request handler
    # Only flag reads of hardcoded/static config paths.  Skip per-request data
    # reads where the file path is dynamically constructed from request params
    # (e.g., project_dir / "project.yaml" where project_dir varies per request).
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_route_handler = any(
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "route")
            or (isinstance(d, ast.Attribute) and d.attr == "route")
            for d in node.decorator_list
        )
        if not is_route_handler:
            continue
        # Scan the function body for config file reads
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            config_read = isinstance(func, ast.Attribute) and (
                (func.attr == "safe_load" and isinstance(func.value, ast.Name) and func.value.id == "yaml")
                or (func.attr == "load" and isinstance(func.value, ast.Name) and func.value.id == "json")
                or func.attr == "read_text"
            )
            if config_read:
                # Check if the file argument is a hardcoded string (static config)
                # vs a variable/expression (per-request dynamic path — skip)
                is_static_path = False
                if func.attr in ("safe_load", "load"):
                    # yaml.safe_load(f) / json.load(f) — the file handle comes
                    # from open(); check the open() call's argument.  Heuristic:
                    # if the call's first positional arg is a string constant,
                    # it's static.  If it's a variable, it's dynamic.
                    is_static_path = (
                        child.args and isinstance(child.args[0], ast.Constant) and isinstance(child.args[0].value, str)
                    )
                elif func.attr == "read_text":
                    # path.read_text() — check if path is a module-level constant
                    is_static_path = isinstance(func.value, ast.Name) and func.value.id.isupper()

                if not is_static_path:
                    continue  # Dynamic path — per-request data, not global config

                line_no = child.lineno
                line_text = lines[line_no - 1] if line_no <= len(lines) else ""
                if not has_noqa(line_text, "VET115"):
                    add_warning(
                        filepath,
                        line_no,
                        "VET115",
                        "Config file read inside route handler — cache at module level or use TTL cache",
                    )
                break  # One warning per handler is sufficient


# ═══════════════════════════════════════════════════════════════════════════════
# M. Structural Reinforcement Rules (VET210-230)
# ═══════════════════════════════════════════════════════════════════════════════


# Regex patterns for structural rules
_RAW_SINGLETON_PATTERN = re.compile(
    r"""^\s*(?:global\s+_\w+|_\w+\s*=\s*None)\s*$""",
)
_SINGLETON_INSTANCE_CHECK = re.compile(
    r"""if\s+_\w+\s+is\s+None\s*:""",
)
_UNBOUNDED_METRICS_ANNOTATION = re.compile(
    r""":\s*list\s*\[\s*(?:float|dict)\s*\]""",
)
_RELATIVE_DATA_PATH = re.compile(
    r"""(?:["'](?:\.\/|\.\.\/)?(?:data|db|database|training_data|models|projects|outputs|logs)\/[^"']+["'])""",
)

# Directory names whose files are subject to VET220 unbounded-metrics checks.
# Extracted as a module-level constant so tests can monkeypatch this set to
# make VET220 fire on fixture files that live outside the real analytics dirs.
_ANALYTICS_DIRS: frozenset[str] = frozenset({"analytics", "drift", "learning", "workflow"})


# ── Hot-path lazy import detection (VET130) ─────────────────────────────────

# Files where function-body imports are forbidden (hot path).
# Relative paths from project root, forward-slash normalized.
_HOT_PATH_FILES = {
    "vetinari/agents/base_agent.py",
    "vetinari/agents/inference.py",
    "vetinari/agents/consolidated/worker_agent.py",
    "vetinari/agents/consolidated/quality_agent.py",
    "vetinari/adapters/llama_cpp_adapter.py",
    "vetinari/orchestration/agent_graph.py",
    "vetinari/orchestration/graph_executor.py",
    "vetinari/orchestration/two_layer.py",
}

# Functions where lazy imports are acceptable (cold path — run once).
_COLD_PATH_FUNCTIONS = {
    "__init__",  # Constructors — run once per instance
    "__init_subclass__",
    "initialize",  # One-time initialization (e.g., AgentGraph.initialize)
    "_get_agent_constraints",  # Helper called once
}


def check_hot_path_imports(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET130: Flag import statements inside function bodies in hot-path files.

    Lazy imports in per-request/per-inference functions cause repeated
    sys.modules lookups. Move them to module level or use a cached lazy
    getter pattern.
    """
    rel_path = os.path.relpath(filepath, project_root).replace("\\", "/")
    if rel_path not in _HOT_PATH_FILES:
        return

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Skip cold-path functions and module-level lazy getters (def _get_X)
        func_name = node.name
        if func_name in _COLD_PATH_FUNCTIONS:
            continue
        if func_name.startswith(("_get_", "_lazy_get_")):
            continue

        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                line_no = child.lineno
                # Check noqa on any line spanned by this import (multi-line imports)
                end_line = getattr(child, "end_lineno", line_no) or line_no
                _suppressed = False
                for _ln in range(line_no, end_line + 1):
                    _lt = lines[_ln - 1] if _ln <= len(lines) else ""
                    if has_noqa(_lt, "VET130"):
                        _suppressed = True
                        break
                if _suppressed:
                    continue
                # Build a readable module name for the message
                if isinstance(child, ast.ImportFrom) and child.module:
                    mod = child.module
                elif isinstance(child, ast.Import) and child.names:
                    mod = child.names[0].name
                else:
                    mod = "unknown"
                add_error(
                    filepath,
                    line_no,
                    "VET130",
                    f"import '{mod}' inside hot-path function {func_name}() — move to module level or use a cached lazy getter",
                )


_SANDBOX_MODULE = "vetinari.security.sandbox"
_SANDBOX_GUARD_CALL = "enforce_blocked_paths"
_WRITE_METHODS = frozenset({"write_text", "write_bytes"})


def _imports_sandbox(tree: ast.Module) -> bool:
    """Return True if the module imports vetinari.security.sandbox.

    Catches both ``import vetinari.security.sandbox`` and
    ``from vetinari.security.sandbox import ...``. Does NOT match deeper
    paths like ``vetinari.security.sandbox.subsidiary`` — the sandbox
    module is a leaf module, so an exact name compare is enough.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == _SANDBOX_MODULE for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == _SANDBOX_MODULE:
            return True
    return False


def _is_write_call(node: ast.Call) -> bool:
    """Return True when *node* is a filesystem write call VET141 should guard.

    Matches three patterns:
      * ``open(path, "w"...)`` / ``open(path, "wb"...)`` / ``open(path, "w+")`` etc.
      * ``<any>.write_text(...)``
      * ``<any>.write_bytes(...)``

    The detection is intentionally syntactic — it does not try to prove the
    target path is attacker-influenced, because the sandbox policy applies
    to every write in a sandbox-importing module.
    """
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _WRITE_METHODS:
        return True
    if isinstance(func, ast.Name) and func.id == "open" and len(node.args) >= 2:
        mode_arg = node.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str) and "w" in mode_arg.value:
            return True
    return False


def _is_guard_call(node: ast.Call) -> bool:
    """Return True when *node* calls enforce_blocked_paths (bare or attribute)."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == _SANDBOX_GUARD_CALL:
        return True
    return isinstance(func, ast.Attribute) and func.attr == _SANDBOX_GUARD_CALL


def check_unguarded_sandbox_writes(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET141: Flag filesystem writes that precede no enforce_blocked_paths guard.

    Scope: only modules that import vetinari.security.sandbox. The rule makes
    the sandbox import a load-bearing signal — once a module imports the
    sandbox, every write inside it MUST be preceded by enforce_blocked_paths
    within the same scope (function body, or module top-level for module
    writes).

    This closes the Rule 2 governance gap from SESSION-03 SHARD-02 Task 2.3:
    a security helper that is defined but never called certifies nothing.
    VET141 forces the helper to stay wired — a refactor that removes the
    guard without removing the sandbox import is flagged at commit time.

    Allow-listed context:
      * Calls inside `vetinari/security/sandbox.py` itself — the guard
        implementation and its helpers write to disk during YAML parsing,
        and guarding them would be circular.
    """
    if not is_in_vetinari(filepath):
        return
    rel = os.path.relpath(filepath, project_root).replace("\\", "/")
    if rel == "vetinari/security/sandbox.py":
        return
    if not _imports_sandbox(tree):
        return

    # Scope = (module-level top, each FunctionDef, each AsyncFunctionDef).
    # Each scope is analysed in isolation: walk the scope's own statements but
    # stop at any nested FunctionDef/AsyncFunctionDef boundary so writes inside
    # nested functions are only counted in the function's own scope (avoids
    # double-counting one write at module + function level).
    def _walk_scope(scope_body: list[ast.stmt]) -> tuple[list[int], list[tuple[int, ast.Call]]]:
        """Return (guard_lines, writes) for a single lexical scope.

        Walks *scope_body* but does not descend into nested function or class
        bodies — those define their own scopes and are visited separately.
        """
        guards: list[int] = []
        writes: list[tuple[int, ast.Call]] = []
        stack: list[ast.AST] = list(scope_body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # Different scope, handled by the outer loop
            if isinstance(node, ast.Call):
                if _is_guard_call(node):
                    guards.append(node.lineno)
                elif _is_write_call(node):
                    writes.append((node.lineno, node))
            stack.extend(ast.iter_child_nodes(node))
        return guards, writes

    def _flag_scope(scope_body: list[ast.stmt]) -> None:
        """Flag writes in *scope_body* that are not preceded by a guard call."""
        guard_lines, writes = _walk_scope(scope_body)
        for write_line, _node in writes:
            has_preceding_guard = any(g_line <= write_line for g_line in guard_lines)
            if has_preceding_guard:
                continue
            line_text = lines[write_line - 1] if write_line <= len(lines) else ""
            if has_noqa(line_text, "VET141"):
                continue
            add_error(
                filepath,
                write_line,
                "VET141",
                "Unguarded write in a sandbox-importing module — call enforce_blocked_paths(target) "
                "before the write, or drop the sandbox import if this module does not enforce the policy.",
            )

    # Module-level writes (top-level only, function bodies excluded by _walk_scope)
    _flag_scope(tree.body)
    # Function / method writes (each function body is its own scope)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _flag_scope(list(node.body))


# R. Destructive-Operation Guard (VET142) — per-file, scoped to web/ and safety/

_VET142_SCOPE_DIRS = frozenset({"vetinari/web", "vetinari/safety", "vetinari/lifecycle"})
# Protected implementation files that legitimately call shutil.rmtree / os.remove
# and must NOT trigger the rule.
_VET142_EXCLUDED_FILES = frozenset({"protected_mutation.py"})
# Method name in vetinari/safety/recycle.py that is the one legitimate hard-delete path.
_VET142_EXCLUDED_METHOD = "purge_expired"
_VET142_EXCLUDED_RECYCLE_MODULE = "vetinari/safety/recycle.py"

# AST attribute / name patterns that identify destructive FS calls.
# Only shutil.rmtree and os.remove are flagged — Path.unlink is used for
# transient temp-file cleanup which is a different concern.
_DESTRUCTIVE_CALL_NAMES = frozenset({"rmtree"})  # shutil.rmtree
_DESTRUCTIVE_OS_NAMES = frozenset({"remove"})  # os.remove


def _is_destructive_fs_call(node: ast.Call) -> bool:
    """Return True when *node* is shutil.rmtree(...) or os.remove(...)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        # shutil.rmtree(path) → Attribute(value=Name(id='shutil'), attr='rmtree')
        # os.remove(path)     → Attribute(value=Name(id='os'), attr='remove')
        if func.attr in _DESTRUCTIVE_CALL_NAMES:
            return True
        if func.attr in _DESTRUCTIVE_OS_NAMES:
            return True
    return False


def _func_has_protected_mutation_decorator(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when the function has @protected_mutation(...) in its decorator list."""
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "protected_mutation":
            return True
        if isinstance(dec, ast.Name) and dec.id == "protected_mutation":
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "protected_mutation":
            return True
    return False


def _func_has_vet142_exclusion_comment(func_node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> bool:
    """Return True when the function body contains a # VET142-excluded: comment."""
    start = func_node.lineno
    end = getattr(func_node, "end_lineno", None) or len(lines)
    for ln in range(start, min(end + 1, len(lines) + 1)):
        line_text = lines[ln - 1] if ln <= len(lines) else ""
        if "# VET142-excluded:" in line_text:
            return True
    return False


def check_unguarded_destructive_ops(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET142: Flag shutil.rmtree / os.remove without @protected_mutation guard.

    Scope: files under vetinari/web/ or vetinari/safety/.  Excludes the
    protected_mutation.py implementation itself and purge_expired in recycle.py,
    both of which are the designated lifecycle-primitive endpoints where hard
    deletion is intentional.

    Two valid paths exist for a function to pass:
    1. The function carries ``@protected_mutation(...)`` in its decorator stack.
    2. The function calls RecycleStore.retire (lifecycle-fenced) and is annotated
       with a ``# VET142-excluded: lifecycle-fenced ...`` comment.

    Functions that do neither are flagged as errors so that new routes cannot
    accidentally re-introduce bare directory-tree deletion.

    Allow-listed:
      * ``vetinari/safety/protected_mutation.py`` — the guard implementation itself.
      * ``vetinari/safety/recycle.py::purge_expired`` — the single sanctioned
        hard-delete path; removing the exclusion without updating this rule is
        detectable by the pre-commit check.
    """
    if not is_in_vetinari(filepath):
        return

    rel = os.path.relpath(filepath, project_root).replace("\\", "/")

    # Must be under one of the scoped directories.
    if not any(rel.startswith(scope_dir + "/") for scope_dir in _VET142_SCOPE_DIRS):
        return

    # Exclude protected_mutation.py itself.
    if Path(filepath).name in _VET142_EXCLUDED_FILES:
        return

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name

        # Exclude purge_expired in recycle.py — it IS the sanctioned hard-delete.
        if func_name == _VET142_EXCLUDED_METHOD and rel == _VET142_EXCLUDED_RECYCLE_MODULE:
            continue

        # Walk only direct child nodes of this function body (not nested functions).
        destructive_calls: list[int] = []
        stack: list[ast.AST] = list(node.body)
        while stack:
            child = stack.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # Different lexical scope — analysed separately.
            if isinstance(child, ast.Call) and _is_destructive_fs_call(child):
                destructive_calls.append(child.lineno)
            stack.extend(ast.iter_child_nodes(child))

        if not destructive_calls:
            continue

        # Guard check 1: @protected_mutation in decorator stack.
        if _func_has_protected_mutation_decorator(node):
            continue

        # Guard check 2: lifecycle-fenced exclusion comment in the function body.
        if _func_has_vet142_exclusion_comment(node, lines):
            continue

        # Neither guard found — flag each destructive call site.
        for call_line in destructive_calls:
            line_text = lines[call_line - 1] if call_line <= len(lines) else ""
            if has_noqa(line_text, "VET142"):
                continue
            add_error(
                filepath,
                call_line,
                "VET142",
                f"{func_name} performs a destructive action without @protected_mutation"
                " -- wrap it or move the deletion into RecycleStore.purge_expired.",
            )


def check_structural_rules(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET210-230: Structural reinforcement rules."""
    if not is_in_vetinari(filepath):
        return

    # VET210: Singleton uses raw _instance instead of SingletonMeta or thread_safe_singleton
    # Detect the pattern:  global _instance / _instance = None ... if _instance is None:
    # Skip: vetinari/utils/__init__.py (defines SingletonMeta itself), singleton.py
    basename = Path(filepath).name
    if basename not in ("__init__.py", "singleton.py") or "utils" not in str(filepath):
        has_raw_instance = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Look for the raw singleton pattern: _something_instance = None at module level
            if re.match(r"^_\w*instance\w*\s*[:=]\s*None", stripped) and not line[0].isspace():
                has_raw_instance = True
                instance_line = i
        if has_raw_instance:
            # Check if there's a corresponding "if _instance is None:" without a lock
            has_lock_guard = "with _lock:" in source or "with cls._lock:" in source
            if not has_lock_guard:
                line_text = lines[instance_line - 1] if instance_line <= len(lines) else ""
                if not has_noqa(line_text, "VET210"):
                    add_warning(
                        filepath,
                        instance_line,
                        "VET210",
                        "Raw singleton _instance pattern without lock — "
                        "use SingletonMeta or @thread_safe_singleton from vetinari.utils",
                    )

    # VET220: Unbounded list[float] or list[dict] in metrics/analytics classes
    # Flag class-level annotations that use list[float] instead of deque or BoundedMetrics
    file_dir = Path(filepath).parent.name
    if file_dir in _ANALYTICS_DIRS or "metrics" in basename or "quality" in basename:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                # Check the annotation string in source
                ann_line = lines[item.lineno - 1] if item.lineno <= len(lines) else ""
                if _UNBOUNDED_METRICS_ANNOTATION.search(ann_line):
                    field_name = item.target.id
                    # Skip private fields that are managed internally with bounds
                    if field_name.startswith("_"):
                        continue
                    if not has_noqa(ann_line, "VET220"):
                        add_warning(
                            filepath,
                            item.lineno,
                            "VET220",
                            f"Unbounded list[float]/list[dict] field '{field_name}' "
                            "in analytics class — use collections.deque or BoundedMetrics",
                        )

    # VET230: Relative path to data/DB file (not using constants)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if _RELATIVE_DATA_PATH.search(line):
            # Skip: lines that reference constants (already using central paths)
            if any(
                const in line
                for const in (
                    "PROJECTS_DIR",
                    "OUTPUTS_DIR",
                    "LOGS_DIR",
                    "CHECKPOINT_DIR",
                    "VETINARI_STATE_DIR",
                    "MODEL_CACHE_DIR",
                    "TRAINING_DATA_FILE",
                    "_PROJECT_ROOT",
                    "constants.",
                )
            ):
                continue
            # Skip: test files, config files, comments, docstrings
            if is_in_tests(filepath) or is_in_scripts(filepath):
                continue
            if not has_noqa(line, "VET230"):
                add_warning(
                    filepath,
                    i,
                    "VET230",
                    "Relative path to data/DB file — use constants from vetinari.constants",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# L. ABC Single Implementation Check (VET109) — cross-file
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_abc_info(
    filepath: str,
    tree: ast.Module,
    abc_defs: dict[str, tuple[str, int]],
    subclass_counts: dict[str, int],
) -> None:
    """Collect ABC class definitions and subclass relationships for VET109.

    Args:
        filepath: Path to the Python source file being scanned.
        tree: Parsed AST module for the file.
        abc_defs: Output dict mapping class names to (filepath, lineno) for ABCs.
        subclass_counts: Output dict mapping base class names to concrete subclass counts.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        # Is this class itself an ABC?
        if any(b in ("ABC", "ABCMeta") for b in base_names):
            if node.name not in abc_defs:
                abc_defs[node.name] = (filepath, node.lineno)
        else:
            # Count it as a concrete subclass of each base
            for base_name in base_names:
                if base_name:
                    subclass_counts[base_name] = subclass_counts.get(base_name, 0) + 1


def check_abc_implementations() -> None:
    """VET109: Flag ABC classes with fewer than 2 concrete implementations.

    ABCs with 0 or 1 concrete subclasses add abstraction overhead without
    enabling polymorphism. They should be collapsed into a concrete class.
    """
    abc_defs: dict[str, tuple[str, int]] = {}
    subclass_counts: dict[str, int] = {}

    for scan_dir in [VETINARI_DIR]:
        if not scan_dir.exists():
            continue
        for root, dirs, filenames in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    src = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(src)
                except (OSError, SyntaxError):
                    continue
                _collect_abc_info(fpath, tree, abc_defs, subclass_counts)

    for class_name, (filepath, lineno) in sorted(abc_defs.items()):
        count = subclass_counts.get(class_name, 0)
        if count <= 1:
            try:
                file_lines = Path(filepath).read_text(encoding="utf-8").splitlines()
                line_text = file_lines[lineno - 1] if lineno <= len(file_lines) else ""
            except OSError:
                line_text = ""
            if not has_noqa(line_text, "VET109"):
                add_warning(
                    filepath,
                    lineno,
                    "VET109",
                    f"ABC '{class_name}' has {count} concrete implementation(s) "
                    "— collapse into a single class or add a second implementation",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# N. Unwired Code Detection (VET120-124) — cross-file
# ═══════════════════════════════════════════════════════════════════════════════

# Decorators that register functions with frameworks (not called by name)
_FRAMEWORK_DECORATORS = frozenset({
    # ABC / typing
    "abstractmethod",
    "overload",
    # Descriptors
    "property",
    "staticmethod",
    "classmethod",
    "cached_property",
    # Testing
    "fixture",
    "pytest.fixture",
    # Litestar / web framework route verbs
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "route",
    "head",
    "options",
    "websocket",
    "websocket_listener",
    # Litestar lifecycle
    "listener",
    "on_app_init",
})

# Decorator suffixes — if a dotted decorator ends with one of these, it's framework-registered
_FRAMEWORK_DECORATOR_SUFFIXES = frozenset({
    "route",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "options",
    "websocket",
    "before_request",
    "after_request",
    "errorhandler",
    "listener",
    "on_app_init",
    "on_shutdown",
    "on_startup",
    "teardown_appcontext",
    "context_processor",
})

# Base classes whose subclasses are used as types, not instantiated directly
_TYPE_ONLY_BASES = frozenset({
    "BaseModel",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "AttributeError",
    "OSError",
    "IOError",
    "NotImplementedError",
    "ABC",
    "ABCMeta",
    "Protocol",
    "TypedDict",
    "Enum",
    "IntEnum",
    "StrEnum",
    "Flag",
    "IntFlag",
    "Controller",  # Litestar controllers are registered, not instantiated by user code
})


def _extract_decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract human-readable decorator names from AST decorator_list.

    Handles plain names (``@abstractmethod``), attribute access
    (``@bp.route``), and calls (``@bp.route("/path")``).

    Args:
        node: AST node with a decorator_list attribute.

    Returns:
        List of dotted decorator name strings.
    """
    names: list[str] = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        parts: list[str] = []
        current = target
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        if parts:
            names.append(".".join(reversed(parts)))
    return names


def _extract_base_class_names(node: ast.ClassDef) -> list[str]:
    """Extract base class names from a ClassDef AST node.

    Args:
        node: ClassDef AST node.

    Returns:
        List of base class name strings (rightmost component for dotted names).
    """
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _is_framework_registered(decorators: list[str]) -> bool:
    """Check if any decorator indicates framework registration.

    Args:
        decorators: List of dotted decorator name strings.

    Returns:
        True if the function is registered by a framework and should not be
        flagged as unwired.
    """
    for dec in decorators:
        if dec in _FRAMEWORK_DECORATORS:
            return True
        # Dotted decorators: check suffix (e.g. "self.router.get" -> "get")
        parts = dec.split(".")
        if len(parts) >= 2 and parts[-1] in _FRAMEWORK_DECORATOR_SUFFIXES:
            return True
    return False


def _extract_all_exports(tree: ast.Module) -> set[str]:
    """Extract literal string names from a module-level __all__ assignment."""
    exports: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                exports.add(element.value)
    return exports


def _build_wiring_index() -> dict:
    """Build a cross-file index of all definitions and references.

    Walks ``vetinari/`` with AST for definitions, then scans ``vetinari/`` +
    ``tests/`` + ``scripts/`` with text tokenization for references.  Parses
    ``pyproject.toml`` for CLI entry points.

    Returns:
        Dictionary with keys: ``func_defs``, ``class_defs``, ``refs``,
        ``module_importers``, ``init_reexports``, ``source_stems``,
        ``test_stems``, ``entry_funcs``, ``class_methods``.
    """
    # ── Definitions (vetinari/ only) ─────────────────────────────────
    # name -> list of (filepath, lineno, kind, decorators, parent_class, bases)
    func_defs: dict[str, list[tuple[str, int, str, list[str], str | None]]] = {}
    class_defs: dict[str, list[tuple[str, int, list[str]]]] = {}
    # class_name -> set of method names (for override detection)
    class_methods: dict[str, set[str]] = {}
    # module dotted path -> set of filepaths that import it
    module_importers: dict[str, set[str]] = {}
    # name -> list of (filepath, lineno) for __init__.py re-exports
    init_reexports: dict[str, list[tuple[str, int]]] = {}
    # filepath -> names explicitly listed in __all__
    init_all_exports: dict[str, set[str]] = {}
    # stems of source modules, package directory names, and test files
    source_stems: set[str] = set()
    package_names: set[str] = set()  # directory names under vetinari/
    test_stems: set[str] = set()

    # Phase 1: AST walk of vetinari/ for definitions
    if VETINARI_DIR.exists():
        for root, dirs, filenames in os.walk(VETINARI_DIR):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            # Record this directory name as a package name
            dir_name = Path(root).name
            if dir_name != "vetinari" and (Path(root) / "__init__.py").exists():
                package_names.add(dir_name)
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                stem = Path(fname).stem
                is_init = fname == "__init__.py"
                if not is_init:
                    source_stems.add(stem)

                try:
                    src = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(src)
                except (OSError, SyntaxError):
                    continue

                # Walk top-level nodes only (not ast.walk — we need parent context)
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        decorators = _extract_decorator_names(node)
                        bases = _extract_base_class_names(node)
                        class_defs.setdefault(node.name, []).append((fpath, node.lineno, bases))
                        # Collect methods
                        methods: set[str] = set()
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                methods.add(item.name)
                                m_decorators = _extract_decorator_names(item)
                                func_defs.setdefault(item.name, []).append((
                                    fpath,
                                    item.lineno,
                                    "method",
                                    m_decorators,
                                    node.name,
                                ))
                        class_methods.setdefault(node.name, set()).update(methods)
                        # Also merge base class methods for multi-level override detection
                        for base_name in bases:
                            if base_name in class_methods:
                                class_methods[node.name].update(class_methods[base_name])

                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        decorators = _extract_decorator_names(node)
                        func_defs.setdefault(node.name, []).append((fpath, node.lineno, "function", decorators, None))

                # __init__.py re-exports: collect names from "from .X import Y"
                # Only track relative imports (from .X) and project imports (from vetinari.X).
                # Stdlib and third-party imports are implementation details, not re-exports.
                if is_init:
                    init_all_exports[fpath] = _extract_all_exports(tree)
                    for node in tree.body:
                        if isinstance(node, ast.ImportFrom):
                            if node.module == "__future__":
                                continue  # Skip from __future__ import annotations — not a re-export
                            is_relative = node.level > 0  # from .X import Y
                            is_project = node.module is not None and node.module.startswith("vetinari")
                            if not (is_relative or is_project):
                                continue  # Skip stdlib/third-party imports — not re-exports
                            for alias in node.names:
                                export_name = alias.asname or alias.name
                                init_reexports.setdefault(export_name, []).append((fpath, node.lineno))

    # Phase 2: Text scan for references (vetinari/ + tests/ + scripts/)
    all_names = set(func_defs.keys()) | set(class_defs.keys())
    # name -> {filepath: occurrence_count}
    refs: dict[str, dict[str, int]] = {}

    scan_dirs = [d for d in [VETINARI_DIR, TESTS_DIR, SCRIPTS_DIR] if d.exists()]
    for scan_dir in scan_dirs:
        for root, dirs, filenames in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                stem = Path(fname).stem
                if stem.startswith("test_") and scan_dir == TESTS_DIR:
                    test_stems.add(stem)

                try:
                    src = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                # Collect import relationships for VET122
                for raw_line in src.splitlines():
                    stripped = raw_line.strip()
                    m = re.match(r"from\s+(vetinari\S*)\s+import", stripped)
                    if m:
                        module_importers.setdefault(m.group(1), set()).add(fpath)
                    m2 = re.match(r"import\s+(vetinari\S*)", stripped)
                    if m2:
                        module_importers.setdefault(m2.group(1), set()).add(fpath)
                    m3 = re.match(r"from\s+(\.+)([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?\s+import", stripped)
                    if m3 and Path(fpath).resolve().is_relative_to(VETINARI_DIR.resolve()):
                        rel_module = str(Path(fpath).relative_to(project_root)).replace(os.sep, ".").removesuffix(".py")
                        package_parts = rel_module.split(".")[:-1]
                        level = len(m3.group(1))
                        base_parts = package_parts[: max(0, len(package_parts) - level + 1)]
                        imported_tail = m3.group(2)
                        if imported_tail:
                            base_parts.extend(imported_tail.split("."))
                        if base_parts:
                            module_importers.setdefault(".".join(base_parts), set()).add(fpath)

                # Tokenize: extract all identifiers in one pass
                word_counts = Counter(re.findall(r"\b[A-Za-z_]\w*\b", src))
                for name in set(word_counts.keys()) & all_names:
                    refs.setdefault(name, {})[fpath] = word_counts[name]

    # Phase 3: Parse pyproject.toml for entry point function names
    entry_funcs: set[str] = set()
    toml_path = project_root / "pyproject.toml"
    if toml_path.exists():
        try:
            content = toml_path.read_text(encoding="utf-8")
            in_scripts = False
            for raw_line in content.splitlines():
                stripped = raw_line.strip()
                if stripped == "[project.scripts]":
                    in_scripts = True
                    continue
                if stripped.startswith("[") and in_scripts:
                    break
                if in_scripts and "=" in stripped:
                    m = re.search(r'"([^"]+)"', stripped)
                    if m and ":" in m.group(1):
                        entry_funcs.add(m.group(1).rsplit(":", 1)[1])
        except OSError:
            entry_funcs = set()

    return {
        "func_defs": func_defs,
        "class_defs": class_defs,
        "refs": refs,
        "module_importers": module_importers,
        "init_reexports": init_reexports,
        "init_all_exports": init_all_exports,
        "source_stems": source_stems,
        "package_names": package_names,
        "test_stems": test_stems,
        "entry_funcs": entry_funcs,
        "class_methods": class_methods,
    }


def _get_line_text(filepath: str, lineno: int) -> str:
    """Read a single line from a file for noqa checking.

    Args:
        filepath: Absolute path to the file.
        lineno: 1-based line number.

    Returns:
        The line text, or empty string on error.
    """
    try:
        file_lines = Path(filepath).read_text(encoding="utf-8").splitlines()
        return file_lines[lineno - 1] if lineno <= len(file_lines) else ""
    except OSError:
        return ""


def _check_unwired_functions(index: dict) -> None:
    """VET120: Flag public functions/methods defined but never referenced.

    A function is unwired if its name appears in no file other than its
    definition file, and appears only once (the ``def`` itself) within that
    file.  Framework-registered, private, dunder, and overridden methods are
    excluded.
    """
    func_defs = index["func_defs"]
    refs = index["refs"]
    entry_funcs = index["entry_funcs"]
    class_methods = index["class_methods"]

    for name, defs in func_defs.items():
        for filepath, lineno, kind, decorators, parent_class in defs:
            # Skip private and dunder
            if name.startswith("_"):
                continue

            # Skip __init__.py definitions (handled by VET123)
            if Path(filepath).name == "__init__.py":
                continue

            # Skip framework-registered functions
            if _is_framework_registered(decorators):
                continue

            # Skip CLI entry points
            if name in entry_funcs:
                continue

            # Skip method overrides: if parent class inherits from a base
            # that defines the same method name
            if kind == "method" and parent_class:
                is_override = False
                for cname, cdefs in index["class_defs"].items():
                    if cname != parent_class:
                        continue
                    for _cfpath, _clineno, bases in cdefs:
                        for base_name in bases:
                            base_meths = class_methods.get(base_name, set())
                            if name in base_meths:
                                is_override = True
                                break
                        if is_override:
                            break
                    if is_override:
                        break
                if is_override:
                    continue

            # Check references
            ref_files = refs.get(name, {})
            other_file_refs = {f: c for f, c in ref_files.items() if f != filepath}

            if other_file_refs:
                continue  # Referenced from another file — wired

            # Same-file check: the name appears in this file.  Count of 1
            # means only the ``def`` line.  Count >= 2 means it is also called.
            same_file_count = ref_files.get(filepath, 0)
            if same_file_count >= 2:
                continue  # Called within the same file — wired

            line_text = _get_line_text(filepath, lineno)
            if not has_noqa(line_text, "VET120"):
                ctx = f" in class '{parent_class}'" if parent_class else ""
                add_error(
                    filepath,
                    lineno,
                    "VET120",
                    f"Public {kind} '{name}'{ctx} is defined but never referenced — wire a call site; only remove if the code is superseded or deprecated",
                )


def _check_unwired_classes(index: dict) -> None:
    """VET121: Flag public classes defined but never referenced.

    Excludes exception subclasses, enums, ABCs, Protocols, TypedDicts,
    Pydantic models, and Litestar controllers — these are used as types or
    registered by frameworks, not instantiated by user code.
    """
    class_defs = index["class_defs"]
    refs = index["refs"]

    for name, defs in class_defs.items():
        for filepath, lineno, bases in defs:
            # Skip private and dunder
            if name.startswith("_"):
                continue

            # Skip __init__.py
            if Path(filepath).name == "__init__.py":
                continue

            # Skip type-only base classes (exceptions, enums, ABCs, etc.)
            if any(b in _TYPE_ONLY_BASES for b in bases):
                continue

            # Skip dataclasses (often used as types in annotations, not instantiated)
            # We detect @dataclass decorator by reading the file
            line_text = _get_line_text(filepath, lineno)
            # Look up to 3 lines above the class for decorators
            try:
                file_lines = Path(filepath).read_text(encoding="utf-8").splitlines()
                start = max(0, lineno - 4)
                preceding = "\n".join(file_lines[start : lineno - 1])
                if "@dataclass" in preceding:
                    continue
            except OSError:
                file_lines = []
                preceding = ""

            # Check references
            ref_files = refs.get(name, {})
            other_file_refs = {f: c for f, c in ref_files.items() if f != filepath}

            if other_file_refs:
                continue

            same_file_count = ref_files.get(filepath, 0)
            if same_file_count >= 2:
                continue

            if not has_noqa(line_text, "VET121"):
                add_error(
                    filepath,
                    lineno,
                    "VET121",
                    f"Public class '{name}' is defined but never referenced — wire a usage; only remove if the code is superseded or deprecated",
                )


def _check_unwired_modules(index: dict) -> None:
    """VET122: Flag Python modules in vetinari/ never imported by any file.

    Skips ``__init__.py``, ``__main__.py``, and ``conftest.py``.
    """
    module_importers = index["module_importers"]

    if not VETINARI_DIR.exists():
        return

    for root, dirs, filenames in os.walk(VETINARI_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname in ("__init__.py", "__main__.py", "conftest.py"):
                continue

            fpath = os.path.join(root, fname)
            try:
                rel = Path(fpath).relative_to(project_root)
            except ValueError:
                continue

            # Convert path to dotted module: vetinari/foo/bar.py -> vetinari.foo.bar
            module_path = str(rel).replace(os.sep, ".").replace("/", ".").removesuffix(".py")

            # Check if this module is imported (exact or as parent of deeper import)
            is_imported = False
            for imported_module in module_importers:
                if imported_module == module_path or imported_module.startswith(module_path + "."):
                    is_imported = True
                    break
                # Also check: "from vetinari.foo import bar" when file is vetinari/foo/bar.py
                # In this case imported_module is "vetinari.foo" and we need to check
                # if any name from bar.py is imported via that parent
                parent_module = module_path.rsplit(".", 1)[0] if "." in module_path else ""
                if parent_module and imported_module == parent_module:
                    is_imported = True
                    break
            if not is_imported:
                line_text = _get_line_text(fpath, 1)
                if not has_noqa(line_text, "VET122"):
                    add_error(
                        fpath,
                        1,
                        "VET122",
                        f"Module '{module_path}' is never imported — wire an import; only remove if the module is superseded or deprecated",
                    )


def _check_dead_init_reexports(index: dict) -> None:
    """VET123: Flag __init__.py re-exports that nothing outside the package imports."""
    init_reexports = index["init_reexports"]
    init_all_exports = index["init_all_exports"]
    refs = index["refs"]

    for name, locations in init_reexports.items():
        for filepath, lineno in locations:
            if name in init_all_exports.get(filepath, set()):
                continue
            pkg_dir = Path(filepath).parent

            ref_files = refs.get(name, {})
            # Only count references from outside this package directory
            outside_refs = {f for f in ref_files if not Path(f).resolve().is_relative_to(pkg_dir.resolve())}

            if outside_refs:
                continue

            line_text = _get_line_text(filepath, lineno)
            if not has_noqa(line_text, "VET123"):
                add_warning(
                    filepath,
                    lineno,
                    "VET123",
                    f"Re-export '{name}' in __init__.py is never imported from outside the package",
                )


_VET124_EXEMPT_TEST_DIRS = {
    "benchmarks",
    "chaos",
    "integration",
    "replay",
    "runtime",
    "security",
}

_VET124_EXEMPT_TEST_STEMS = {
    "test_check_scripts_c",
    "test_fixer_scripts_e",
    "test_governance_helpers",
    "test_hook_scripts_a",
    "test_hook_scripts_b",
    "test_misc_scripts_d",
    "test_promoted_vet_rules",
    "test_reapply_omc_mods",
    "test_release_certifier",
    "test_startup_contract",
    "test_temp_harness",
    "test_update_scaffold",
    "test_vet130_file_size",
    "test_vet131_module_io",
    "test_vet_rules_fixtures",
}


def _is_vet124_exempt_test_file(path: Path, source: str) -> bool:
    """Return True for legitimate non-module-mapped test categories."""
    try:
        relative_parts = path.relative_to(TESTS_DIR).parts
    except ValueError:
        relative_parts = path.parts
    if len(relative_parts) > 1 and relative_parts[0] in _VET124_EXEMPT_TEST_DIRS:
        return True
    if path.stem in _VET124_EXEMPT_TEST_STEMS:
        return True
    return bool(
        "scripts/" in source
        or "scripts\\" in source
        or re.search(r"^\s*(?:from|import)\s+scripts(?:\.|\b)", source, re.MULTILINE)
        or "SCRIPTS_DIR" in source
        or re.search(r"sys\.path\.(?:insert|append)\([^)]*scripts", source)
    )


def _check_orphaned_tests(index: dict) -> None:
    """VET124: Flag test files whose corresponding source module does not exist.

    Maps ``test_foo.py`` to ``foo.py`` and also tries splitting compound
    names (``test_foo_bar`` checks for ``foo_bar``, ``foo``, and ``bar``
    individually) to handle the project convention of flattening nested
    module paths into underscored test names.
    """
    test_stems = index["test_stems"]
    source_stems = index["source_stems"]
    package_names = index["package_names"]
    all_known_names = source_stems | package_names

    for test_stem in sorted(test_stems):
        if not test_stem.startswith("test_"):
            continue
        source_stem = test_stem[5:]  # Remove "test_" prefix
        if not source_stem:
            continue

        # Direct match: test_foo.py -> foo.py or vetinari/foo/ package
        if source_stem in all_known_names:
            continue

        # Compound name match: test_adapters_base -> check if "adapters" or "base"
        # is a known source stem or package (covers vetinari/adapters/base.py)
        parts = source_stem.split("_")
        has_partial_match = any(part in all_known_names for part in parts if len(part) > 2)
        if has_partial_match:
            continue

        # Two-part compound: test_foo_bar -> check "foo_bar" but also "foo" as a
        # package name containing "bar.py"
        if len(parts) >= 2:
            first_part = parts[0]
            rest = "_".join(parts[1:])
            if first_part in all_known_names or rest in all_known_names:
                continue

        # Find the actual test file path for reporting
        if TESTS_DIR.exists():
            for root, dirs, filenames in os.walk(TESTS_DIR):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fname in filenames:
                    if Path(fname).stem == test_stem:
                        fpath = os.path.join(root, fname)
                        try:
                            test_source = Path(fpath).read_text(encoding="utf-8")
                        except OSError:
                            test_source = ""
                        if _is_vet124_exempt_test_file(Path(fpath), test_source):
                            break
                        if re.search(r"^\s*(?:from|import)\s+vetinari(?:\.|\b)", test_source, re.MULTILINE):
                            break
                        line_text = _get_line_text(fpath, 1)
                        if not has_noqa(line_text, "VET124"):
                            add_error(
                                fpath,
                                1,
                                "VET124",
                                f"Test file '{test_stem}.py' has no matching source "
                                f"module '{source_stem}.py' in vetinari/",
                            )
                        break


def check_unwired_code() -> None:
    """VET120-124: Cross-file unwired code detection.

    Builds a codebase-wide index of definitions and references in a single
    pass, then runs five analysis checks to detect dead functions, classes,
    modules, init re-exports, and orphaned tests.
    """
    index = _build_wiring_index()

    _check_unwired_functions(index)
    _check_unwired_classes(index)
    _check_unwired_modules(index)
    _check_dead_init_reexports(index)
    _check_orphaned_tests(index)


# ═══════════════════════════════════════════════════════════════════════════════
# O. Same-Name Class Detection (VET125) — cross-file
# WARNING not ERROR: 43 pre-existing violations on introduction — promote to ERROR once backlog is resolved
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns for VET126 deprecated code detection
_DEPRECATED_DECORATOR_PATTERN = re.compile(r"^\s*@(?:\w+\.)*deprecated\b")
_DEPRECATION_WARNING_PATTERN = re.compile(r"raise\s+DeprecationWarning\s*\(|warnings\.warn\s*\([^)]*DeprecationWarning")


def check_same_name_classes() -> None:
    """VET125: Flag classes with identical names defined in multiple vetinari/ modules.

    Two classes with the same name in different modules are forbidden because
    they cause silent behavior differences depending on which module is imported.
    Reports the second (and later) definition, pointing to the first in the message.

    Emits WARNING (not ERROR) until the 43 pre-existing violations are resolved.
    """
    # Map from class name -> (filepath, lineno) of the first definition seen
    first_seen: dict[str, tuple[str, int]] = {}

    if not VETINARI_DIR.exists():
        return

    for root, dirs, filenames in os.walk(VETINARI_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            # __init__.py re-exports don't count as definitions
            if fname == "__init__.py":
                continue
            fpath = os.path.join(root, fname)
            try:
                src = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(src)
            except (OSError, SyntaxError):
                continue

            file_lines = src.splitlines()

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                name = node.name
                # Skip private classes
                if name.startswith("_"):
                    continue
                # Skip pure exception subclasses — they duplicate names by convention
                # (e.g., ValueError in each module), but exception classes share a
                # common base so the anti-pattern rule explicitly calls these out.
                # Per spec: exception subclasses are NOT excluded; same rule applies.
                line_text = file_lines[node.lineno - 1] if node.lineno <= len(file_lines) else ""
                if has_noqa(line_text, "VET125"):
                    continue

                if name in first_seen:
                    first_filepath, _ = first_seen[name]
                    first_rel = os.path.relpath(first_filepath, project_root)
                    add_warning(
                        fpath,
                        node.lineno,
                        "VET125",
                        f"Class '{name}' is already defined in {first_rel} "
                        "— same-name classes in different modules are forbidden; "
                        "rename one at the source",
                    )
                else:
                    first_seen[name] = (fpath, node.lineno)


# ═══════════════════════════════════════════════════════════════════════════════
# P. Deprecated Code Detection (VET126) — per-file
# ═══════════════════════════════════════════════════════════════════════════════


def check_deprecated_code(filepath: str, source: str, lines: list[str]) -> None:
    """VET126: Flag deprecated code patterns in vetinari/ production files.

    Catches three patterns:
    - @deprecated decorator (any form)
    - raise DeprecationWarning(...) or warnings.warn(..., DeprecationWarning)
    - Files whose stem ends with _legacy or _old

    Args:
        filepath: Path to the source file being checked.
        source: Full source text of the file.
        lines: Source lines (1-indexed via enumerate).
    """
    if not is_in_vetinari(filepath):
        return
    # Exclude __init__.py and tests (tests/ excluded by is_in_vetinari guard above)
    if Path(filepath).name == "__init__.py":
        return

    # Item 3: filename pattern — checked once per file, reported on line 1
    stem = Path(filepath).stem
    if stem.endswith("_legacy") or stem.endswith("_old"):
        line1 = lines[0] if lines else ""
        if not has_noqa(line1, "VET126"):
            add_error(
                filepath,
                1,
                "VET126",
                "File name suggests deprecated code ('_legacy'/'_old' suffix) "
                "— rename or delete; deprecated modules must not remain in the codebase",
            )

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if has_noqa(line, "VET126"):
            continue

        # Item 1: @deprecated decorator
        if _DEPRECATED_DECORATOR_PATTERN.match(line):
            add_error(
                filepath,
                i,
                "VET126",
                "@deprecated decorator found — delete the deprecated code and wire callers to the replacement",
            )
            continue

        # Item 2: DeprecationWarning raised or warned
        if _DEPRECATION_WARNING_PATTERN.search(line):
            add_error(
                filepath,
                i,
                "VET126",
                "DeprecationWarning raised in production code — delete the deprecated "
                "code; warnings belong in migration notes, not source",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Q. File Size Limit (VET127), Module-Level I/O (VET128), Hardcoded Inference Params (VET129)
# ═══════════════════════════════════════════════════════════════════════════════

# Names that indicate module-level I/O (VET128)
_MODULE_IO_CALLS: frozenset[str] = frozenset({
    "open",
    "safe_load",  # yaml.safe_load
    "read_text",  # Path.read_text
    "read_bytes",  # Path.read_bytes
})

# json.load is caught by checking the attribute name "load" combined with the
# object being "json" — handled separately in the AST walk.

# Inference parameter names subject to VET129
_INFERENCE_PARAMS: frozenset[str] = frozenset({
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
})


def _docstring_line_numbers(source: str) -> set[int]:
    """Return source lines occupied by Python docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        ):
            continue
        end_lineno = getattr(first, "end_lineno", first.lineno)
        docstring_lines.update(range(first.lineno, end_lineno + 1))
    return docstring_lines


def check_file_size(filepath: str, source: str, lines: list[str]) -> None:
    """VET127: Warn when a file exceeds 550 lines of code.

    Counts non-blank, non-comment, non-docstring lines as "lines of code".
    Reports at line 1 so the warning appears at the top of the file in CI
    output. Exempt files: ``__init__.py``, ``conftest.py``, and generated or
    vendor files.

    Args:
        filepath: Absolute path to the file being checked.
        source: Full source text of the file.
        lines: Source lines (1-indexed via enumerate).
    """
    if not is_in_vetinari(filepath):
        return

    name = Path(filepath).name
    if name in ("__init__.py", "conftest.py"):
        return

    docstring_lines = _docstring_line_numbers(source)
    loc = sum(
        1
        for idx, line in enumerate(lines, start=1)
        if line.strip() and not line.strip().startswith("#") and idx not in docstring_lines
    )

    if loc > 550:
        line1 = lines[0] if lines else ""
        if not has_noqa(line1, "VET127"):
            add_warning(
                filepath,
                1,
                "VET127",
                f"File has {loc} lines of code (limit: 550). Split by dependency boundary.",
            )


def _is_module_level_io_call(node: ast.expr) -> str | None:
    """Return a human-readable call description if node is a module-level I/O call.

    Detects the following patterns:
    - ``open(...)``
    - ``yaml.safe_load(...)``
    - ``json.load(...)``
    - ``Path(...).read_text(...)``
    - ``Path(...).read_bytes(...)``
    - any ``<expr>.read_text(...)`` or ``<expr>.read_bytes(...)``

    Args:
        node: An AST expression node to inspect.

    Returns:
        A short string description of the call (e.g. ``"open()"``), or None
        if the node is not a recognised module-level I/O call.
    """
    if not isinstance(node, ast.Call):
        return None

    func = node.func

    # bare open()
    if isinstance(func, ast.Name) and func.id == "open":
        return "open()"

    if isinstance(func, ast.Attribute):
        method = func.attr

        # yaml.safe_load()
        if method == "safe_load":
            obj = func.value
            if isinstance(obj, ast.Name) and obj.id == "yaml":
                return "yaml.safe_load()"

        # json.load()
        if method == "load":
            obj = func.value
            if isinstance(obj, ast.Name) and obj.id == "json":
                return "json.load()"

        # Path(...).read_text() or Path(...).read_bytes()
        if method in ("read_text", "read_bytes"):
            return f"Path.{method}()"

    return None


def _stmt_contains_module_io(stmt: ast.stmt) -> list[str]:
    """Collect module-level I/O call descriptions from a statement node.

    Walks all sub-expressions of the statement, returning the description of
    every I/O call found. Assignment nodes (``x = open(...)``) and bare
    expression statements (``open(...)``) are both detected.

    Args:
        stmt: A top-level AST statement from ``ast.Module.body``.

    Returns:
        List of human-readable call descriptions found in this statement.
    """
    found: list[str] = []
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            desc = _is_module_level_io_call(node)
            if desc:
                found.append(desc)
    return found


def check_module_level_io(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET128: Flag file/config I/O calls at module body level (not inside functions or classes).

    I/O at import time causes side effects on every import, slows startup,
    and makes testing harder. Use lazy init (load inside a function, cache
    the result) instead.

    Args:
        filepath: Absolute path to the file being checked.
        source: Full source text of the file.
        lines: Source lines (1-indexed via enumerate).
        tree: Parsed AST module for the file.
    """
    if not is_in_vetinari(filepath):
        return

    # Only inspect the top-level module body — not function/class bodies
    for stmt in tree.body:
        # Skip import statements, function/class defs — these are fine at module level
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # Skip pure string expressions (docstrings, comments-in-code)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue

        # Check if this module-level statement contains any I/O call
        io_calls = _stmt_contains_module_io(stmt)
        for call_desc in io_calls:
            line_no = stmt.lineno
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            if not has_noqa(line_text, "VET128"):
                add_error(
                    filepath,
                    line_no,
                    "VET128",
                    f"Module-level I/O detected: {call_desc} — use lazy init instead",
                )
            break  # One error per statement is enough


def check_hardcoded_inference_params(filepath: str, source: str, lines: list[str], tree: ast.Module) -> None:
    """VET129: Flag hardcoded inference parameter keyword arguments.

    Parameters like ``temperature=0.3`` or ``max_tokens=2048`` hardcoded at
    call sites override the project's ``InferenceConfigManager`` and
    ``task_inference_profiles.json`` system, causing silent per-call overrides
    that conflict with tuned profiles. All sampling parameters must flow through
    the profile system.

    Exempt: ``config/``, ``tests/``, ``scripts/``.

    Args:
        filepath: Absolute path to the file being checked.
        source: Full source text of the file.
        lines: Source lines (1-indexed via enumerate).
        tree: Parsed AST module for the file.
    """
    if not is_in_vetinari(filepath):
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in _INFERENCE_PARAMS:
                continue
            # Only flag constant literal values — variable references are OK
            if not isinstance(kw.value, ast.Constant):
                continue
            line_no = kw.value.lineno
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            if has_noqa(line_text, "VET129"):
                continue
            # WARNING not ERROR: 12 pre-existing violations on introduction — promote to ERROR once backlog is resolved
            add_warning(
                filepath,
                line_no,
                "VET129",
                f"Hardcoded inference param: {kw.arg}={kw.value.value!r} — use InferenceConfigManager instead",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main scanner
# ═══════════════════════════════════════════════════════════════════════════════


def scan_file(filepath: str) -> None:
    """Run all rules against a single Python file."""
    global checked_files
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    lines = source.splitlines()
    checked_files += 1

    # Try to parse AST
    tree = None
    with contextlib.suppress(SyntaxError):
        tree = ast.parse(source)

    # Run all rule checkers
    check_import_rules(filepath, source, lines)
    check_future_annotations(filepath, source, lines)
    check_security(filepath, source, lines)
    check_logging(filepath, source, lines)
    check_robustness(filepath, source, lines)
    check_organization(filepath, source, lines)
    check_deprecated_code(filepath, source, lines)
    check_file_size(filepath, source, lines)  # VET127

    if tree is not None:
        check_error_handling(filepath, source, lines, tree)
        check_completeness(filepath, source, lines, tree)
        check_documentation(filepath, source, lines, tree)
        check_ai_antipatterns(filepath, source, lines, tree)
        check_hot_path_imports(filepath, source, lines, tree)
        check_unguarded_sandbox_writes(filepath, source, lines, tree)  # VET141
        check_unguarded_destructive_ops(filepath, source, lines, tree)  # VET142
        check_structural_rules(filepath, source, lines, tree)
        check_module_level_io(filepath, source, lines, tree)  # VET128
        check_hardcoded_inference_params(filepath, source, lines, tree)  # VET129

    # Integration checks (heavier, do last)
    check_integration(filepath, source, lines)


def collect_python_files() -> list[str]:
    """Collect all Python files to check."""
    files = []
    for scan_dir in [VETINARI_DIR, TESTS_DIR, SCRIPTS_DIR]:
        if not scan_dir.exists():
            continue
        for root, dirs, filenames in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            files.extend(os.path.join(root, f) for f in filenames if f.endswith(".py"))
    return files


def _rule_globs(rule: dict, key: str) -> list[str]:
    """Return normalized glob patterns from an external-rule field."""
    raw = rule.get(key, [])
    if isinstance(raw, str):
        return [raw.replace("\\", "/")]
    if not isinstance(raw, list):
        return []
    return [str(item).replace("\\", "/") for item in raw if str(item).strip()]


def _relative_posix(filepath: str | Path) -> str:
    """Return a project-relative POSIX path when possible."""
    path = Path(filepath)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _matches_any_glob(filepath: str | Path, patterns: list[str]) -> bool:
    """Return True when filepath matches any project-relative glob pattern."""
    rel = _relative_posix(filepath)
    full = Path(filepath).as_posix()
    for pattern in patterns:
        variants = {pattern}
        if "**/" in pattern:
            variants.add(pattern.replace("**/", ""))
        if any(fnmatch.fnmatchcase(rel, item) or fnmatch.fnmatchcase(full, item) for item in variants):
            return True
    return False


def _rule_is_gating(rule: dict) -> bool:
    """Return whether a promoted rule should block release gates when matched."""
    if "gating" in rule:
        return bool(rule["gating"])
    if "release_gating" in rule:
        return bool(rule["release_gating"])
    return True


def collect_external_rule_files(default_files: list[str], rules: list[dict]) -> list[str]:
    """Collect files that external rules can scan.

    Rules without ``include_globs`` keep the historical Python-file default.
    Rules with ``include_globs`` expand those globs relative to the project root,
    which makes promoted rules for Markdown/YAML/TOML/static files enforceable
    instead of planned-only metadata.
    """
    files = {str(Path(filepath)) for filepath in default_files}

    for rule in rules:
        for pattern in _rule_globs(rule, "include_globs"):
            for matched in project_root.glob(pattern):
                if matched.is_file():
                    files.add(str(matched))

    return sorted(files)


def load_external_rules(yaml_path: str | Path) -> list[dict]:
    """Load external VET rules from a YAML file.

    Args:
        yaml_path: Path to vet_rules.yaml containing promoted audit findings.

    Returns:
        List of rule dicts, each with keys: id, category, description,
        pattern_type, pattern, severity. A missing file is non-fatal, but an
        existing rules file that cannot be loaded records a fail-closed error.
    """
    path = Path(yaml_path)
    if not path.exists():
        print(f"INFO: External rules file not found at {path} — no external rules loaded")
        return []
    if _yaml is None:
        add_error(
            str(path),
            1,
            "VETCFG",
            "PyYAML is required to load external promoted rules; install pyyaml or remove the rules file",
        )
        return []
    try:
        data = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        add_error(str(path), 1, "VETCFG", f"External promoted rules failed to load: {exc}")
        return []
    rules = data.get("rules") or []
    print(f"INFO: Loaded {len(rules)} external rule(s) from {path.name}")
    return rules


def apply_external_rules(files: list[str], rules: list[dict]) -> None:
    """Apply external regex rules from vet_rules.yaml to collected files.

    Each rule is applied to every file. Matches are recorded via the module-level
    ``errors`` / ``warnings`` lists using the same convention as the hardcoded rules.
    Supports ``include_globs`` and ``exclude_globs`` using project-relative paths.

    Args:
        files: List of absolute file paths to scan.
        rules: External rule dicts loaded by ``load_external_rules``.
    """
    if not rules:
        return

    import re as _re

    for rule in rules:
        rule_id = rule.get("id", "VET???")
        description = rule.get("description", rule_id)
        pattern_raw = rule.get("pattern", "")
        pattern_type = rule.get("pattern_type", "regex")
        severity = str(rule.get("severity", "warning")).lower()
        include_globs = _rule_globs(rule, "include_globs")
        exclude_globs = _rule_globs(rule, "exclude_globs")
        gating = _rule_is_gating(rule)

        if not pattern_raw:
            continue
        if pattern_type != "regex":
            message = f"External rule {rule_id} uses unsupported pattern_type={pattern_type!r}"
            if gating:
                errors.append((str(project_root / "config" / "vet_rules.yaml"), 1, rule_id, message))
            else:
                warnings.append((str(project_root / "config" / "vet_rules.yaml"), 1, rule_id, message))
            continue

        try:
            compiled = _re.compile(pattern_raw, _re.MULTILINE)
        except _re.error as exc:
            message = f"External rule {rule_id} has invalid regex and cannot enforce its finding: {exc}"
            if gating:
                errors.append((str(project_root / "config" / "vet_rules.yaml"), 1, rule_id, message))
            else:
                warnings.append((str(project_root / "config" / "vet_rules.yaml"), 1, rule_id, message))
            continue

        for filepath in files:
            if include_globs and not _matches_any_glob(filepath, include_globs):
                continue
            if exclude_globs and _matches_any_glob(filepath, exclude_globs):
                continue

            try:
                text = Path(filepath).read_text(encoding="utf-8")
            except OSError:
                continue
            for m in compiled.finditer(text):
                line_num = text[: m.start()].count("\n") + 1
                # Extract the line and check for noqa suppression
                lines = text.split("\n")
                if 0 <= line_num - 1 < len(lines):
                    line_text = lines[line_num - 1]
                    if has_noqa(line_text, rule_id):
                        continue  # Skip lines with noqa suppression

                if severity == "error" or gating:
                    message = description
                    if severity != "error":
                        message = f"{description} (release-gating warning; set gating: false only for advisory rules)"
                    errors.append((filepath, line_num, rule_id, message))
                else:
                    warnings.append((filepath, line_num, rule_id, description))


def check_file(filepath: str | Path) -> list[tuple[str, int, str, str]]:
    """Run all per-file rules against one file and return findings.

    Temporarily resets the module-level result collectors, runs scan_file(),
    captures findings, then restores previous state so callers are unaffected.
    This is the testable API that fixture tests use to assert bad files are caught
    and good files pass.

    Cross-file rules (VET120-124: unwired code detection) are NOT run here because
    they require a full-codebase index built by check_unwired_code(). Test those
    via the integration test path.

    Args:
        filepath: Path to the Python file to check (absolute or project-relative).

    Returns:
        List of (filepath, line, code, message) tuples — errors and warnings combined.
        Empty list means the file passes all per-file rules.
    """
    global errors, warnings, checked_files

    saved_errors = errors[:]
    saved_warnings = warnings[:]
    saved_checked = checked_files

    errors.clear()
    warnings.clear()
    checked_files = 0

    try:
        scan_file(str(filepath))
        findings: list[tuple[str, int, str, str]] = [*errors, *warnings]
    finally:
        errors.clear()
        errors.extend(saved_errors)
        warnings.clear()
        warnings.extend(saved_warnings)
        checked_files = saved_checked

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Vetinari Project Rules Checker")
    parser.add_argument("--errors-only", action="store_true", help="Only show errors, suppress warnings")
    parser.add_argument("--verbose", action="store_true", help="Show all checked files")
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix some violations (removes TODOs without issue refs)"
    )
    parser.add_argument(
        "--yaml",
        default=str(project_root / "config" / "vet_rules.yaml"),
        help="Path to external rules YAML (default: config/vet_rules.yaml)",
    )
    args = parser.parse_args()

    files = collect_python_files()

    for filepath in sorted(files):
        if args.verbose:
            rel = os.path.relpath(filepath, project_root)
            print(f"  checking {rel}")
        scan_file(filepath)

    # Check markdown documentation quality
    check_markdown_files()

    # VET109: Cross-file ABC single-implementation check
    check_abc_implementations()

    # VET120-124: Cross-file unwired code detection
    check_unwired_code()

    # Apply any rules promoted from audit findings via promote_findings_to_rules.py
    external_rules = load_external_rules(args.yaml)
    external_files = collect_external_rule_files(files, external_rules)
    apply_external_rules(external_files, external_rules)

    # VET125: Same-name classes across modules (cross-file)
    check_same_name_classes()

    # Print results
    print(f"\nVetinari Rules Checker — scanned {checked_files} files")
    print("=" * 60)

    all_issues = []
    for filepath, line, code, message in errors:
        rel = os.path.relpath(filepath, project_root)
        all_issues.append((rel, line, code, message, "ERROR"))

    if not args.errors_only:
        for filepath, line, code, message in warnings:
            rel = os.path.relpath(filepath, project_root)
            all_issues.append((rel, line, code, message, "WARN"))

    all_issues.sort(key=lambda x: (x[0], x[1]))

    for rel, line, code, message, severity in all_issues:
        print(f"  {rel}:{line}: {code} [{severity}] {message}")

    error_count = len(errors)
    warn_count = len(warnings)

    print(f"\n  {error_count} error(s), {warn_count} warning(s)")

    if error_count > 0:
        print("\n  FAIL: Fix errors before committing.")
        return 1

    if warn_count > 0 and not args.errors_only:
        print("\n  PASS (with warnings)")
    else:
        print("\n  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
