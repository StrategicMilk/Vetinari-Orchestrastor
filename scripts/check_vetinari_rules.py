#!/usr/bin/env python3
"""Vetinari Project Rules Checker — Custom Linter.

Enforces project-specific rules that ruff and other standard linters cannot.
Scans vetinari/ and tests/ directories for violations.

Usage:
    python scripts/check_vetinari_rules.py [--errors-only] [--verbose] [--fix]

Exit codes:
    0   No errors (warnings may be present).
    1   One or more errors found.
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
VETINARI_DIR = project_root / "vetinari"
TESTS_DIR = project_root / "tests"
SCRIPTS_DIR = project_root / "scripts"

SKIP_DIRS = {
    "venv", ".git", "__pycache__", "vetinari.egg-info", "build", "dist",
    ".pytest_cache", "outputs", "projects", ".claude", "node_modules",
}

# ── Canonical import sources ─────────────────────────────────────────────────
CANONICAL_ENUMS = {"AgentType", "TaskStatus", "ExecutionMode", "PlanStatus"}
CANONICAL_SOURCE = "vetinari.types"

# ── Placeholder patterns ─────────────────────────────────────────────────────
PLACEHOLDER_STRINGS = re.compile(
    r"\b(placeholder|lorem\s*ipsum|foo|bar|baz)\b", re.IGNORECASE
)

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

# ── Result collectors ─────────────────────────────────────────────────────────
errors = []
warnings = []
checked_files = 0


def add_error(filepath, line, code, message):
    """Record an error (blocks commit/CI)."""
    errors.append((filepath, line, code, message))


def add_warning(filepath, line, code, message):
    """Record a warning (informational, does not block)."""
    warnings.append((filepath, line, code, message))


def has_noqa(line_text, code):
    """Check if a line has a noqa suppression for the given code.

    Supports both single-code (``# noqa: VET006``) and comma-separated
    multi-code (``# noqa: F403, VET006``) noqa annotations.
    """
    noqa_match = re.search(r"#\s*noqa\b", line_text)
    if not noqa_match:
        return False
    noqa_section = line_text[noqa_match.start():]
    # Blanket noqa with no specific codes
    if ":" not in noqa_section:
        return True
    codes_part = noqa_section.split(":", 1)[1]
    codes = [c.strip() for c in re.split(r"[,\s]+", codes_part) if c.strip()]
    return code in codes


def is_in_vetinari(filepath):
    """Check if file is under vetinari/ source directory."""
    try:
        Path(filepath).relative_to(VETINARI_DIR)
        return True
    except ValueError:
        return False


def is_in_tests(filepath):
    """Check if file is under tests/ directory."""
    try:
        Path(filepath).relative_to(TESTS_DIR)
        return True
    except ValueError:
        return False


def is_in_scripts(filepath):
    """Check if file is under scripts/ directory."""
    try:
        Path(filepath).relative_to(SCRIPTS_DIR)
        return True
    except ValueError:
        return False


def is_cli_module(filepath):
    """Check if file is a CLI entry point (print allowed)."""
    name = Path(filepath).name
    return name in ("__main__.py", "cli.py")


def is_init_version_only(filepath, source):
    """Check if __init__.py only contains version and simple imports."""
    if Path(filepath).name != "__init__.py":
        return False
    stripped = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return all(
        line.startswith("__version__") or line.startswith("from") or line.startswith("import")
        for line in stripped
    ) and len(stripped) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Rule implementations
# ═══════════════════════════════════════════════════════════════════════════════


def check_import_rules(filepath, source, lines):
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
                    "AgentType": "VET001", "TaskStatus": "VET002",
                    "ExecutionMode": "VET003", "PlanStatus": "VET004",
                }
                code = code_map.get(enum_name, "VET001")
                if not has_noqa(line, code):
                    add_error(filepath, i, code, f"Import {enum_name} from {CANONICAL_SOURCE}, not from this module")

        # VET005: Duplicate enum definition
        if re.match(r"class\s+(AgentType|TaskStatus|ExecutionMode|PlanStatus)\s*\(", stripped):
            if "types.py" not in str(filepath):
                if not has_noqa(line, "VET005"):
                    add_error(filepath, i, "VET005", f"Duplicate enum definition — this enum is defined in {CANONICAL_SOURCE}")

        # VET006: Wildcard import from vetinari
        if re.match(r"from\s+vetinari\.\S+\s+import\s+\*", stripped):
            if not has_noqa(line, "VET006"):
                add_error(filepath, i, "VET006", "Wildcard import from vetinari.* is forbidden")


def check_future_annotations(filepath, source, lines):
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


def check_error_handling(filepath, source, lines, tree):
    """VET020-022: Error handling rules."""
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
            if isinstance(stmt, ast.Pass):
                line_text = lines[lineno - 1] if lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET022"):
                    add_error(filepath, lineno, "VET022", "Empty except block (swallowed exception without logging)")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                line_text = lines[lineno - 1] if lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET022"):
                    add_error(filepath, lineno, "VET022", "Empty except block (swallowed exception without logging)")


def _is_abstract_method(node):
    """Check if a function/method has @abstractmethod decorator."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
            return True
    return False


def _is_exception_init(node, parent):
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


def _get_effective_body(body):
    """Strip leading docstring from a function body, return remaining statements."""
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body


def _check_stub_bodies(filepath, lines, tree):
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
            if isinstance(stmt, ast.Pass):
                if not _is_abstract_method(node) and not _is_exception_init(node, parent):
                    if not has_noqa(line_text, "VET031"):
                        add_error(filepath, stmt.lineno, "VET031", f"'pass' as sole body of '{node.name}' — implement or remove")

            # VET032: Ellipsis as sole body
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                if not filepath.endswith(".pyi"):
                    if not has_noqa(line_text, "VET032"):
                        add_error(filepath, stmt.lineno, "VET032", f"'...' as sole body of '{node.name}' — implement or remove")

        # VET033: raise NotImplementedError outside @abstractmethod
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Raise) or stmt.exc is None:
                continue
            exc = stmt.exc
            is_not_impl = (
                (isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError")
                or (isinstance(exc, ast.Name) and exc.id == "NotImplementedError")
            )
            if is_not_impl and not _is_abstract_method(node):
                line_text = lines[stmt.lineno - 1] if stmt.lineno <= len(lines) else ""
                if not has_noqa(line_text, "VET033"):
                    add_error(filepath, stmt.lineno, "VET033", "raise NotImplementedError outside @abstractmethod")


def _check_line_patterns(filepath, lines):
    """VET034-036: Line-based completeness checks (placeholders, print, dead code)."""
    in_tests = is_in_tests(filepath)

    # VET034: Placeholder strings (skip tests)
    if not in_tests:
        for i, line in enumerate(lines, 1):
            if not line.strip().startswith("#") and PLACEHOLDER_STRINGS.search(line):
                if not has_noqa(line, "VET034"):
                    add_warning(filepath, i, "VET034", "Possible placeholder string detected")

    # VET035: print() in production code
    if is_in_vetinari(filepath) and not is_cli_module(filepath):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped.startswith("#") and re.search(r"\bprint\s*\(", stripped):
                if not has_noqa(line, "VET035"):
                    add_error(filepath, i, "VET035", "print() in production code — use logging instead")

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


def _check_commented_block(filepath, lines, consecutive):
    """Helper: check if a consecutive comment block is commented-out code."""
    if len(consecutive) < 3:
        return
    code_text = "\n".join(text for _, text in consecutive)
    try:
        ast.parse(code_text)
        first_line = consecutive[0][0]
        line_text = lines[first_line - 1] if first_line <= len(lines) else ""
        if not has_noqa(line_text, "VET036"):
            add_warning(filepath, first_line, "VET036",
                        f"Commented-out code block ({len(consecutive)} lines) — delete dead code")
    except SyntaxError:
        pass


def check_completeness(filepath, source, lines, tree):
    """VET030-036: Completeness rules."""
    # VET030: Untracked TODO/FIXME (skip tests)
    if not is_in_tests(filepath):
        for i, line in enumerate(lines, 1):
            if TODO_PATTERN.search(line) and not TODO_WITH_ISSUE.search(line):
                if not has_noqa(line, "VET030"):
                    add_error(filepath, i, "VET030", "TODO/FIXME/HACK/XXX/TEMP without issue reference (use TODO(#123))")

    _check_stub_bodies(filepath, lines, tree)
    _check_line_patterns(filepath, lines)


def check_security(filepath, source, lines):
    """VET040-041: Security rules."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # VET040: Hardcoded credentials
        if CREDENTIAL_PATTERN.search(line):
            if not has_noqa(line, "VET040"):
                add_error(filepath, i, "VET040", "Hardcoded credential pattern detected")

        # VET041: Hardcoded localhost URLs
        if LOCALHOST_PATTERN.search(line):
            if not is_in_tests(filepath) and "config" not in str(filepath).lower():
                if not has_noqa(line, "VET041"):
                    add_warning(filepath, i, "VET041", "Hardcoded localhost URL — use config instead")


def check_logging(filepath, source, lines):
    """VET050-051: Logging rules."""
    if not is_in_vetinari(filepath):
        return

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # VET050: Root logger usage
        if ROOT_LOGGER_PATTERN.search(line):
            if not has_noqa(line, "VET050"):
                add_warning(filepath, i, "VET050", "Using root logger (logging.info()) — use module logger (logger.info()) instead")

        # VET051: f-string in logger call
        if FSTRING_LOGGER_PATTERN.search(line):
            if not has_noqa(line, "VET051"):
                add_warning(filepath, i, "VET051", "f-string in logger call — use %-style: logger.info('msg %s', val)")


def check_robustness(filepath, source, lines):
    """VET060-063: Robustness rules."""
    if not is_in_vetinari(filepath):
        return

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # VET060: open() without encoding
        if OPEN_CALL_PATTERN.search(line):
            if "encoding=" not in line and "encoding =" not in line:
                # Skip binary mode opens
                if not re.search(r"""['"](r?b|['"]*wb|['"]*rb)['""]""", line):
                    if not has_noqa(line, "VET060"):
                        add_warning(filepath, i, "VET060", "open() without encoding= parameter — use encoding='utf-8'")

        # VET061: Debug code
        for pattern in DEBUG_PATTERNS:
            if pattern.search(line):
                if not has_noqa(line, "VET061"):
                    add_error(filepath, i, "VET061", "Debug code (breakpoint/pdb) — remove before committing")

        # VET062: Long sleep
        match = SLEEP_PATTERN.search(line)
        if match:
            sleep_val = float(match.group(1))
            if sleep_val > 5:
                if not has_noqa(line, "VET062"):
                    add_warning(filepath, i, "VET062", f"time.sleep({sleep_val}) > 5 seconds — use configurable timeout")

        # VET063: os.path.join
        if OS_PATH_JOIN_PATTERN.search(line):
            if not has_noqa(line, "VET063"):
                add_warning(filepath, i, "VET063", "os.path.join() — prefer pathlib.Path for cross-platform paths")


# Known Python standard library top-level module names (3.10+)
_STDLIB_MODULES = {
    "abc", "argparse", "ast", "asyncio", "atexit", "base64", "bisect",
    "builtins", "calendar", "cgi", "cmd", "codecs", "collections",
    "colorsys", "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "csv", "ctypes", "dataclasses", "datetime", "decimal",
    "difflib", "dis", "email", "enum", "errno", "faulthandler",
    "filecmp", "fileinput", "fnmatch", "fractions", "ftplib", "functools",
    "gc", "getpass", "gettext", "glob", "gzip", "hashlib", "heapq",
    "hmac", "html", "http", "imaplib", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "linecache", "locale",
    "logging", "lzma", "math", "mimetypes", "mmap", "multiprocessing",
    "netrc", "numbers", "operator", "os", "pathlib", "pdb",
    "pkgutil", "platform", "pprint", "profile", "pstats",
    "queue", "random", "re", "readline", "reprlib", "resource",
    "runpy", "sched", "secrets", "select", "selectors", "shelve",
    "shlex", "shutil", "signal", "site", "smtplib", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics", "string",
    "struct", "subprocess", "sys", "sysconfig", "tarfile", "tempfile",
    "test", "textwrap", "threading", "time", "timeit", "token",
    "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
    "turtle", "types", "typing", "unicodedata", "unittest", "urllib",
    "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xml", "xmlrpc", "zipapp",
    "zipfile", "zipimport", "zlib", "_thread", "__future__",
    "typing_extensions",
    # Easter-egg / seldom-used stdlib modules
    "this", "antigravity", "turtle", "idlelib",
}


def check_integration(filepath, source, lines):
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
        "yaml": "pyyaml", "cv2": "opencv-python", "PIL": "pillow",
        "sklearn": "scikit-learn", "bs4": "beautifulsoup4",
        "attr": "attrs", "dateutil": "python-dateutil",
        "google": "google-generativeai", "gi": "pygobject",
        "ddgs": "duckduckgo-search",  # ddgs is the newer module name for duckduckgo-search
    }

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
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
            if pkg_name not in pyproject_text and pkg_name.replace("_", "-") not in pyproject_text:
                if not has_noqa(line, "VET070"):
                    add_error(filepath, i, "VET070", f"Import '{module}' not found in pyproject.toml dependencies")


def check_organization(filepath, source, lines):
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
        elif not re.match(r"^[a-z][a-z0-9_]*$", stem):
            add_warning(filepath, 1, "VET082", f"File '{fpath.name}' is not snake_case — rename to '{stem.lower()}.py'")


# ═══════════════════════════════════════════════════════════════════════════════
# J. Documentation Quality Rules (VET090-096)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_docstring(node):
    """Extract docstring from a function, class, or module node."""
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.value.value
    return None


def _get_function_params(node):
    """Get parameter names from a function def, excluding 'self' and 'cls'."""
    params = []
    for arg in node.args.args:
        if arg.arg not in ("self", "cls"):
            params.append(arg.arg)
    return params


def _has_return_value(node):
    """Check if function has a return statement with a value (not bare return/None)."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            if isinstance(child.value, ast.Constant) and child.value.value is None:
                continue
            return True
    return False


def _has_raise_statement(node):
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


def _is_public(name):
    """Check if a name is public (not prefixed with underscore)."""
    return not name.startswith("_")


def _is_property_or_simple(node):
    """Check if function is a property, setter, or very simple (single return/pass)."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in ("property", "staticmethod"):
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "getter", "deleter"):
            return True
    # Single-line functions (just return something) don't need full docstrings
    effective = _get_effective_body(node.body)
    if len(effective) == 1 and isinstance(effective[0], ast.Return):
        return True
    return False


def _is_dunder(name):
    """Check if name is a dunder method."""
    return name.startswith("__") and name.endswith("__")


def check_documentation(filepath, source, lines, tree):
    """VET090-096: Documentation quality rules."""
    if not is_in_vetinari(filepath):
        return

    # VET095: Module missing module-level docstring
    module_doc = _get_docstring(tree)
    if module_doc is None:
        # Skip __init__.py files that are mostly empty
        if not is_init_version_only(filepath, source):
            name = Path(filepath).name
            if name != "__init__.py":
                if not any(has_noqa(lines[0] if lines else "", f"VET09{x}") for x in range(10)):
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


def _check_class_docstring(filepath, lines, node):
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
    if len(stripped_doc) < 10:
        if not has_noqa(line_text, "VET091"):
            add_warning(filepath, lineno, "VET091", f"Docstring for class '{node.name}' is too short ({len(stripped_doc)} chars)")

    # VET096: Docstring is just the class name repeated
    if stripped_doc.rstrip(".").lower() == node.name.lower():
        if not has_noqa(line_text, "VET096"):
            add_warning(filepath, lineno, "VET096", f"Docstring for class '{node.name}' just restates the name — add meaningful description")


def _check_function_docstring(filepath, lines, node, parent):
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
    if len(stripped_doc) < 10:
        if not has_noqa(line_text, "VET091"):
            add_warning(filepath, lineno, "VET091", f"Docstring for '{node.name}' is too short ({len(stripped_doc)} chars)")

    # VET092: Missing Args section when function has 2+ params
    params = _get_function_params(node)
    if len(params) >= 2 and "Args:" not in docstring and "args:" not in docstring:
        if not has_noqa(line_text, "VET092"):
            add_warning(filepath, lineno, "VET092", f"Docstring for '{node.name}' missing Args section ({len(params)} parameters)")

    # VET093: Missing Returns section when function returns non-None
    if _has_return_value(node):
        if "Returns:" not in docstring and "returns:" not in docstring and "Return:" not in docstring:
            if not has_noqa(line_text, "VET093"):
                add_warning(filepath, lineno, "VET093", f"Docstring for '{node.name}' missing Returns section")

    # VET094: Missing Raises section when function raises exceptions
    if _has_raise_statement(node):
        if "Raises:" not in docstring and "raises:" not in docstring:
            if not has_noqa(line_text, "VET094"):
                add_warning(filepath, lineno, "VET094", f"Docstring for '{node.name}' missing Raises section")


# ═══════════════════════════════════════════════════════════════════════════════
# K. Markdown Documentation Quality Rules (VET100-102)
# ═══════════════════════════════════════════════════════════════════════════════


def check_markdown_files():
    """VET100-102: Markdown documentation quality rules."""
    docs_dirs = [
        project_root / "docs",
        project_root / ".claude" / "docs",
    ]
    md_files = []
    for docs_dir in docs_dirs:
        if docs_dir.exists():
            for md_file in docs_dir.rglob("*.md"):
                md_files.append(md_file)

    # Also check root-level markdown files
    for md_file in project_root.glob("*.md"):
        md_files.append(md_file)

    for md_file in md_files:
        _check_single_markdown(md_file)


def _check_single_markdown(filepath):
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
                        add_warning(str(filepath), i + 1, "VET101", f"Empty section in '{rel}' — heading with no content before next heading")
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
        add_warning(str(filepath), 1, "VET102", f"Markdown file '{rel}' has very little content ({content_chars} chars)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main scanner
# ═══════════════════════════════════════════════════════════════════════════════


def scan_file(filepath):
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
    try:
        tree = ast.parse(source)
    except SyntaxError:
        pass

    # Run all rule checkers
    check_import_rules(filepath, source, lines)
    check_future_annotations(filepath, source, lines)
    check_security(filepath, source, lines)
    check_logging(filepath, source, lines)
    check_robustness(filepath, source, lines)
    check_organization(filepath, source, lines)

    if tree is not None:
        check_error_handling(filepath, source, lines, tree)
        check_completeness(filepath, source, lines, tree)
        check_documentation(filepath, source, lines, tree)

    # Integration checks (heavier, do last)
    check_integration(filepath, source, lines)


def collect_python_files():
    """Collect all Python files to check."""
    files = []
    for scan_dir in [VETINARI_DIR, TESTS_DIR]:
        if not scan_dir.exists():
            continue
        for root, dirs, filenames in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                if f.endswith(".py"):
                    files.append(os.path.join(root, f))
    return files


def main():
    parser = argparse.ArgumentParser(description="Vetinari Project Rules Checker")
    parser.add_argument("--errors-only", action="store_true",
                        help="Only show errors, suppress warnings")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all checked files")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix some violations (removes TODOs without issue refs)")
    args = parser.parse_args()

    files = collect_python_files()

    for filepath in sorted(files):
        if args.verbose:
            rel = os.path.relpath(filepath, project_root)
            print(f"  checking {rel}")
        scan_file(filepath)

    # Check markdown documentation quality
    check_markdown_files()

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
