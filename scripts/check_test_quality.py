#!/usr/bin/env python3
"""Test Quality Checker for Vetinari.

Detects common test quality problems in the tests/ directory:

  VET240  Ambiguous status assertion — truthy check instead of value check
  VET241  Zero-assert test — test function contains no assert statements
  VET242  Self-mocking — test patches the very module it is testing
  VET243  Mock-only assertion — sole assertion is mock.called / mock.assert_called
  VET244  Weak status assertion — checks only "not 500"
  VET245  Broad success assertion — accepts multiple success statuses
  VET246  Response-shape-only assertion — checks keys without semantics
  VET247  Module-level importorskip — silently skips an entire suite

Exit codes:
  0  — no errors found (warnings may be present)
  1  — one or more errors found

Usage:
  python scripts/check_test_quality.py [--errors-only] [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from functools import cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"

SKIP_DIRS = {"__pycache__", "venv", ".git"}

ERROR = "ERROR"
WARNING = "WARNING"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches test function definitions (regular or async)
_TEST_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+(test_\w+)\s*\(")

# VET240: assert result["status"] or assert response.status — bare truthiness check
# Fires when we see  assert <expr>["status"]  or  assert <expr>.status  with
# no comparison operator (==, !=, is, in, >=, <=, >, <) immediately after.
_AMBIGUOUS_STATUS = re.compile(
    r"""\bassert\s+\w+(?:\[["']status["']\]|\.status(?:_code)?)\s*(?:$|[,#\\])""",
    re.IGNORECASE,
)

# VET241: any assert statement (broad; detects assert keyword at start of statement)
_ASSERT_STMT = re.compile(r"\bassert\s")

# VET242: @patch("vetinari.X.Y") style decorator — capture module path
_PATCH_DECORATOR = re.compile(r"""@(?:patch|mock\.patch|unittest\.mock\.patch)\s*\(\s*["'](vetinari\.[^"']+)["']""")

# VET242: with patch("vetinari.X.Y") context manager — capture module path
_PATCH_CTX_MANAGER = re.compile(r"""\bwith\s+(?:patch|mock\.patch|unittest\.mock\.patch)\s*\(\s*["'](vetinari\.[^"']+)["']""")

# VET243: sole assertion is a weak mock call-count check.
_MOCK_ONLY_ASSERT = re.compile(
    r"\bassert\s+\w+\.called\b"
    r"|\b\w+\.assert_called(?:_once)?\s*\(",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_test_files(directory: Path) -> list[Path]:
    """Return sorted list of test .py files, skipping cache directories.

    Args:
        directory: Root directory to walk.

    Returns:
        Sorted list of Path objects for every test .py file.
    """
    results = []
    for f in sorted(directory.rglob("test_*.py")):
        if any(skip in f.parts for skip in SKIP_DIRS):
            continue
        results.append(f)
    return results


def _read_lines(filepath: Path) -> list[str] | None:
    """Read file lines or return None on I/O error.

    Args:
        filepath: Path to the file.

    Returns:
        List of stripped lines or None if unreadable.
    """
    try:
        return filepath.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None


def _derive_module_stem(test_filepath: Path) -> str:
    """Derive the expected vetinari module path from a test file name.

    Maps ``tests/test_foo_bar.py`` to ``vetinari.foo_bar``.

    Args:
        test_filepath: Path to the test file.

    Returns:
        Expected module prefix string (e.g. ``vetinari.foo_bar``).
    """
    stem = test_filepath.stem  # e.g. "test_foo_bar"
    if stem.startswith("test_"):
        module_name = stem[len("test_") :]
        return f"vetinari.{module_name}"
    return f"vetinari.{stem}"


def _call_name(node: ast.AST) -> str:
    """Return a dotted best-effort name for an AST call target."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _decorator_name(node: ast.AST) -> str:
    """Return a dotted best-effort name for a decorator target."""
    target = node.func if isinstance(node, ast.Call) else node
    return _call_name(target)


def _is_pytest_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when a function is a pytest fixture, not a test case."""
    return any(_decorator_name(decorator) in {"pytest.fixture", "fixture"} for decorator in node.decorator_list)


def _is_pytest_assertion_helper_name(name: str, helpers: set[str]) -> bool:
    """Return True for pytest assertion-helper aliases such as _pytest.raises."""
    if name in helpers:
        return True
    parts = name.split(".")
    if len(parts) < 2 or parts[-1] not in helpers:
        return False
    return parts[-2] in {"pytest", "_pytest"}


def _is_status_code_expr(node: ast.AST) -> bool:
    """Return True for response.status_code style expressions."""
    return isinstance(node, ast.Attribute) and node.attr == "status_code"


def _is_bare_status_expr(node: ast.AST) -> bool:
    """Return True for a status/status_code expression used as truthiness."""
    if isinstance(node, ast.Attribute):
        return node.attr in {"status", "status_code"}
    if isinstance(node, ast.Subscript):
        index = node.slice
        return isinstance(index, ast.Constant) and index.value == "status"
    return False


def _constant_ints(node: ast.AST) -> set[int]:
    """Return integer constants contained in a set/list/tuple AST node."""
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return set()
    values: set[int] = set()
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, int):
            values.add(item.value)
    return values


def _is_dict_key_access(node: ast.AST, var_name: str, key: str) -> bool:
    """Return True for data["key"] or data.get("key") access."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == var_name:
        index = node.slice
        if isinstance(index, ast.Constant) and index.value == key:
            return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == var_name
    ):
        return bool(node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == key)
    return False


def _assert_has_semantic_key_check(test_node: ast.AST, shape_assert: ast.Assert, var_name: str, key: str) -> bool:
    """Return True if another assertion checks a JSON field value or type."""
    for node in ast.walk(test_node):
        if not isinstance(node, ast.Assert) or node is shape_assert:
            continue
        for child in ast.walk(node.test):
            if _is_dict_key_access(child, var_name, key):
                return True
    return False


def _node_contains_name(node: ast.AST, name: str) -> bool:
    """Return True if ``node`` references a specific local variable name."""
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _assert_has_key_flow_assertion(test_node: ast.AST, shape_assert: ast.Assert, var_name: str, key: str) -> bool:
    """Return True if a JSON key is assigned and that derived value is asserted later."""
    derived_names: set[str] = set()

    for node in ast.walk(test_node):
        if not isinstance(node, ast.Assign):
            continue
        if any(_is_dict_key_access(child, var_name, key) for child in ast.walk(node.value)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    derived_names.add(target.id)

    if not derived_names:
        return False

    for node in ast.walk(test_node):
        if not isinstance(node, ast.Assert) or node is shape_assert:
            continue
        if any(_node_contains_name(node.test, name) for name in derived_names):
            return True
    return False


def _module_path_from_stem(module_stem: str) -> Path | None:
    """Return the source path for a ``vetinari.*`` module stem when present."""
    if not module_stem.startswith("vetinari."):
        return None

    parts = module_stem.split(".")[1:]
    if not parts:
        return None

    module_path = PROJECT_ROOT.joinpath("vetinari", *parts).with_suffix(".py")
    if module_path.exists():
        return module_path

    package_init = PROJECT_ROOT.joinpath("vetinari", *parts, "__init__.py")
    if package_init.exists():
        return package_init

    return None


@cache
def _module_symbol_kinds(module_stem: str) -> tuple[set[str], set[str]]:
    """Return top-level locally-defined and imported symbol roots for a module."""
    path = _module_path_from_stem(module_stem)
    if path is None:
        return set(), set()

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set(), set()

    local_defs: set[str] = set()
    imported_roots: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_defs.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_defs.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            local_defs.add(node.target.id)
        elif isinstance(node, ast.Import):
            imported_roots.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported_roots.add(alias.asname or alias.name)

    return local_defs, imported_roots


def _is_local_patch_target(module_stem: str, patched: str) -> bool:
    """Return True when a patch target refers to a locally-defined module symbol."""
    if patched == module_stem:
        return True
    if not patched.startswith(module_stem + "."):
        return False

    suffix = patched[len(module_stem) + 1 :]
    root = suffix.split(".", 1)[0]
    local_defs, imported_roots = _module_symbol_kinds(module_stem)

    if root in local_defs:
        return True
    if root in imported_roots:
        return False
    return False


def _iter_module_or_class_function_nodes(lines: list[str]) -> list[ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Parse source and return module-level or class-level function nodes."""
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError:
        return None

    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, function_types):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, function_types):
                    functions.append(item)
    return functions


def _iter_test_function_nodes(lines: list[str]) -> list[ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Parse source and return module-level or class-level test function nodes."""
    functions = _iter_module_or_class_function_nodes(lines)
    if functions is None:
        return None

    tests: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in functions:
        if node.name.startswith("test_") and not _is_pytest_fixture(node):
            tests.append(node)
    return tests


def _is_pytest_context_assertion_call(node: ast.Call) -> bool:
    """Return True for pytest assertion helpers used as context managers."""
    name = _call_name(node.func)
    return _is_pytest_assertion_helper_name(name, {"raises", "warns", "deprecated_call"})


def _is_pytest_direct_assertion_call(node: ast.Call) -> bool:
    """Return True for direct pytest helper calls that execute a callable."""
    name = _call_name(node.func)
    if _is_pytest_assertion_helper_name(name, {"raises", "warns"}):
        return len(node.args) >= 2
    if _is_pytest_assertion_helper_name(name, {"deprecated_call"}):
        return len(node.args) >= 1
    return _is_pytest_assertion_helper_name(name, {"fail"})


def _is_mock_assert_call(node: ast.Call) -> bool:
    """Return True for unittest.mock assertion methods."""
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr in {
        "assert_called",
        "assert_called_once",
        "assert_called_with",
        "assert_called_once_with",
        "assert_not_called",
        "assert_any_call",
        "assert_has_calls",
    }


class _AssertionVisitor(ast.NodeVisitor):
    """Detect executable assertion-equivalent operations inside one test function."""

    def __init__(self, root: ast.AST, assertion_helpers: set[str] | None = None) -> None:
        self.root = root
        self.assertion_helpers = assertion_helpers or set()
        self.has_assertion = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assert(self, node: ast.Assert) -> None:
        self.has_assertion = True

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            context = item.context_expr
            if isinstance(context, ast.Call) and _is_pytest_context_assertion_call(context):
                self.has_assertion = True
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            context = item.context_expr
            if isinstance(context, ast.Call) and _is_pytest_context_assertion_call(context):
                self.has_assertion = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            _is_pytest_direct_assertion_call(node)
            or _is_mock_assert_call(node)
            or _call_name(node.func) in self.assertion_helpers
        ):
            self.has_assertion = True
        self.generic_visit(node)


def _function_has_executable_assertion(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    assertion_helpers: set[str] | None = None,
) -> bool:
    """Return True when a parsed test function contains a real assertion."""
    visitor = _AssertionVisitor(node, assertion_helpers=assertion_helpers)
    visitor.visit(node)
    return visitor.has_assertion


def _collect_local_assertion_helpers(
    nodes: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    """Return local helper names that directly contain assertion operations."""
    helpers: set[str] = set()
    node_by_name = {node.name: node for node in nodes if not _is_pytest_fixture(node)}
    for node in nodes:
        if _is_pytest_fixture(node):
            continue
        if _function_has_executable_assertion(node):
            helpers.add(node.name)

    changed = True
    while changed:
        changed = False
        for name, node in node_by_name.items():
            if name in helpers:
                continue
            if not name.startswith(("_assert", "assert_")):
                continue
            if _function_has_executable_assertion(node, assertion_helpers=helpers):
                helpers.add(name)
                changed = True
    return helpers


# ---------------------------------------------------------------------------
# Per-file checkers
# ---------------------------------------------------------------------------


def _split_into_test_functions(
    lines: list[str],
) -> list[tuple[str, int, list[str], list[str]]]:
    """Split source lines into (name, start_line, body_lines, decorator_lines) per test function.

    Only captures top-level and class-method test functions (``def test_*``).
    Nested function definitions inside test bodies are included in the body
    and do NOT trigger a new function split.

    The heuristic used: a ``def test_*`` line counts as a new test function
    boundary only when it is at indentation level 0 (module-level) or exactly
    one level of indentation (class method, indented with 4 spaces or 1 tab).
    Any deeper indentation is treated as a nested helper defined inside a test.

    Args:
        lines: Source lines from a test file.

    Returns:
        List of (func_name, start_lineno, body_lines, decorator_lines) tuples.
        ``start_lineno`` is 1-based line of the ``def`` statement.
    """
    functions: list[tuple[str, int, list[str], list[str]]] = []
    current_name: str | None = None
    current_start: int = 0
    current_body: list[str] = []
    current_decorators: list[str] = []
    pending_decorators: list[str] = []

    def _indent_depth(line: str) -> int:
        """Return the number of leading spaces (tabs count as 4)."""
        count = 0
        for ch in line:
            if ch == " ":
                count += 1
            elif ch == "\t":
                count += 4
            else:
                break
        return count

    for i, line in enumerate(lines, 1):
        m = _TEST_DEF.match(line)
        if m:
            depth = _indent_depth(line)
            # Only treat as a new test boundary if it's at module or class level
            # (indent depth 0 or 4/8 — one level of class nesting).
            # Deeper = nested helper inside a test body.
            if depth <= 4:
                # Flush previous function
                if current_name is not None:
                    functions.append((current_name, current_start, current_body, current_decorators))
                current_name = m.group(2)
                current_start = i
                current_body = []
                current_decorators = list(pending_decorators)
                pending_decorators = []
                continue

        # Collect decorator lines before the def (only at boundary level)
        stripped = line.strip()
        if stripped.startswith("@") and current_name is None:
            pending_decorators.append(line)
        elif stripped and not stripped.startswith("@") and current_name is None:
            pending_decorators = []

        if current_name is not None:
            current_body.append(line)

    # Flush last function
    if current_name is not None:
        functions.append((current_name, current_start, current_body, current_decorators))

    return functions


def check_ambiguous_status(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find assert statements that check status truthiness without a value comparison.

    Detects patterns like ``assert result["status"]`` or ``assert resp.status_code``
    that pass for any truthy value — these should compare against a specific value.

    Args:
        filepath: Path to the test file.
        lines: Lines of the file.

    Returns:
        List of (line_number, code, severity, message) tuples.
    """
    violations: list[tuple[int, str, str, str]] = []
    test_nodes = _iter_test_function_nodes(lines)
    if test_nodes is not None:
        for test_node in test_nodes:
            for node in ast.walk(test_node):
                if isinstance(node, ast.Assert) and _is_bare_status_expr(node.test):
                    violations.append((
                        node.lineno,
                        "VET240",
                        WARNING,
                        "Ambiguous status assertion  -  compare to a specific value "
                        '(e.g. assert result["status"] == "completed")',
                    ))
        return violations

    # Join logical continuation lines (trailing backslash or open paren) so that
    # multiline assertions are checked as a single logical unit.
    joined_lines: list[tuple[int, str]] = []
    pending: list[str] = []
    pending_start = 0
    open_parens = 0
    for i, line in enumerate(lines, 1):
        if not pending:
            pending_start = i
        pending.append(line)
        open_parens += line.count("(") - line.count(")")
        if line.rstrip().endswith("\\") or open_parens > 0:
            continue
        open_parens = max(open_parens, 0)
        joined_lines.append((pending_start, " ".join(line_item.strip() for line_item in pending)))
        pending = []
        open_parens = 0
    if pending:
        joined_lines.append((pending_start, " ".join(line_item.strip() for line_item in pending)))

    for i, logical_line in joined_lines:
        if _AMBIGUOUS_STATUS.search(logical_line):
            violations.append((
                i,
                "VET240",
                WARNING,
                "Ambiguous status assertion — compare to a specific value "
                '(e.g. assert result["status"] == "completed")',
            ))
    return violations


def check_zero_assert(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find test functions that contain no assert statements at all.

    A test function without assertions always passes regardless of the system
    under test and provides no coverage signal.  The following are exempt and
    count as valid assertions:

    - ``pytest.raises`` / ``pytest.warns`` / ``pytest.deprecated_call``
      context managers, or direct ``pytest.raises(Exception, callable, ...)``
      calls. Bare helper references are not assertions.
    - ``mock.assert_called*`` / ``mock.assert_not_called`` / ``mock.assert_any_call``
      calls (mock assertion methods raise ``AssertionError`` on failure)

    Args:
        filepath: Path to the test file.
        lines: Lines of the file.

    Returns:
        List of (line_number, code, severity, message) tuples.
    """
    # pytest raises/warns/deprecated_call count only when used as a context
    # manager or as a direct callable assertion. A bare `pytest.raises`
    # reference is not executable proof.
    _PYTEST_HELPER_ASSERT_PATTERN = re.compile(
        r"\bwith\s+(?:_?pytest)\.(?:raises|warns|deprecated_call)\s*\("
        r"|\b(?:_?pytest)\.(?:raises|warns)\s*\([^)\n]+,\s*[^)\n]+\)"
    )
    # mock.assert_called* / mock.assert_not_called / mock.assert_any_call etc.
    # These mock methods raise AssertionError on failure and count as assertions.
    _MOCK_ASSERT_PATTERN = re.compile(r"\.\s*assert_(?:called|not_called|any_call|has_calls)")

    violations: list[tuple[int, str, str, str]] = []
    test_nodes = _iter_test_function_nodes(lines)
    if test_nodes is not None:
        function_nodes = _iter_module_or_class_function_nodes(lines) or test_nodes
        assertion_helpers = _collect_local_assertion_helpers(function_nodes)
        for node in test_nodes:
            if not _function_has_executable_assertion(node, assertion_helpers=assertion_helpers):
                violations.append((
                    node.lineno,
                    "VET241",
                    ERROR,
                    f"{node.name}() contains no assert statements — add an assertion or "
                    "use pytest.raises() if testing an exception path",
                ))
        return violations

    for name, start_line, body, _decorators in _split_into_test_functions(lines):
        body_text = "\n".join(body)
        # Skip docstrings and comments when scanning for real assert keywords.
        # Build a version with docstrings and comments stripped to avoid false
        # positives on asserts that appear only in docstrings or comments.
        in_triple_double = False
        in_triple_single = False
        stripped_body = []
        for ln in body:
            out_chars = []
            i = 0
            while i < len(ln):
                if i + 2 < len(ln):
                    if ln[i : i + 3] == '"""' and not in_triple_single:
                        in_triple_double = not in_triple_double
                        out_chars.append('"""')
                        i += 3
                        continue
                    elif ln[i : i + 3] == "'''" and not in_triple_double:
                        in_triple_single = not in_triple_single
                        out_chars.append("'''")
                        i += 3
                        continue
                if in_triple_double or in_triple_single:
                    out_chars.append(" ")
                elif ln[i] == "#":
                    # Stop processing at comment start
                    break
                else:
                    out_chars.append(ln[i])
                i += 1
            stripped_body.append("".join(out_chars))
        code_lines = stripped_body
        code_text = "\n".join(code_lines)
        has_assert = any(_ASSERT_STMT.search(ln) for ln in code_lines)
        # mock.assert_called* / .assert_any_call / etc. are assertion-equivalent (D2).
        has_mock_method_assert = bool(_MOCK_ASSERT_PATTERN.search(body_text))
        has_raises = bool(_PYTEST_HELPER_ASSERT_PATTERN.search(code_text))
        if not has_assert and not has_raises and not has_mock_method_assert:
            # ERROR not WARNING: a zero-assert test always passes regardless of
            # what the system under test does — it provides no defect signal.
            violations.append((
                start_line,
                "VET241",
                ERROR,
                f"{name}() contains no assert statements — add an assertion or "
                "use pytest.raises() if testing that no exception is raised",
            ))
    return violations


def check_weak_status_assertions(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find weak HTTP status assertions that do not prove route semantics."""
    violations: list[tuple[int, str, str, str]] = []
    test_nodes = _iter_test_function_nodes(lines)
    if test_nodes is None:
        return violations

    for test_node in test_nodes:
        for node in ast.walk(test_node):
            if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
                continue
            compare = node.test
            if len(compare.ops) != 1 or len(compare.comparators) != 1:
                continue

            op = compare.ops[0]
            right = compare.comparators[0]
            left = compare.left

            left_status = _is_status_code_expr(left)
            right_status = _is_status_code_expr(right)
            if isinstance(op, ast.NotEq) and (
                (left_status and isinstance(right, ast.Constant) and right.value == 500)
                or (right_status and isinstance(left, ast.Constant) and left.value == 500)
            ):
                violations.append((
                    node.lineno,
                    "VET244",
                    WARNING,
                    "Weak status assertion — replace 'status_code != 500' with exact expected status "
                    "and route-specific error/auth/mutation semantics",
                ))
                continue

            if isinstance(op, ast.In) and left_status:
                status_values = _constant_ints(right)
                if {200, 201}.issubset(status_values) or len({value for value in status_values if 200 <= value < 300}) > 1:
                    violations.append((
                        node.lineno,
                        "VET245",
                        WARNING,
                        "Broad success status assertion — assert the exact success status for this route "
                        "and pair it with a semantic body or side-effect invariant",
                    ))

    return violations


def check_response_shape_only_assertions(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find response JSON key checks that stop at shape instead of semantics."""
    violations: list[tuple[int, str, str, str]] = []
    test_nodes = _iter_test_function_nodes(lines)
    if test_nodes is None:
        return violations

    for test_node in test_nodes:
        json_vars: set[str] = set()
        for node in ast.walk(test_node):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call_name = _call_name(node.value.func)
                is_json_alias = False
                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "get":
                    owner = node.value.func.value
                    if (
                        isinstance(owner, ast.Name)
                        and owner.id in json_vars
                        and len(node.value.args) >= 2
                        and isinstance(node.value.args[1], ast.Name)
                        and node.value.args[1].id == owner.id
                    ):
                        # Treat body.get("data", body) as an alias for another JSON dict,
                        # but do not promote derived fields such as warnings/body lists.
                        is_json_alias = True
                if call_name.endswith(".json") or is_json_alias:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            json_vars.add(target.id)

        if not json_vars:
            continue

        for node in ast.walk(test_node):
            if not isinstance(node, ast.Assert):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.In)
                and isinstance(test.left, ast.Constant)
                and isinstance(test.left.value, str)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id in json_vars
            ):
                key = test.left.value
                var_name = test.comparators[0].id
                if _assert_has_semantic_key_check(test_node, node, var_name, key):
                    continue
                if _assert_has_key_flow_assertion(test_node, node, var_name, key):
                    continue
                violations.append((
                    node.lineno,
                    "VET246",
                    WARNING,
                    "Response-shape-only assertion — assert the field value, auth/error contract, "
                    "or mutation side effect instead of key presence only",
                ))

    return violations


def check_module_importorskip(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find module-level pytest.importorskip calls that skip entire suites."""
    violations: list[tuple[int, str, str, str]] = []
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError:
        return violations

    for node in tree.body:
        call: ast.Call | None = None
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)) or (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            call = node.value
        if call is not None and _call_name(call.func) == "pytest.importorskip":
            violations.append((
                node.lineno,
                "VET247",
                WARNING,
                "Module-level pytest.importorskip silently removes a whole suite — use an explicit "
                "tracked skip marker or dependency-gated fixture",
            ))

    return violations


def check_self_mocking(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find tests that patch the module they are themselves testing.

    For example, ``tests/test_foo.py`` patching ``vetinari.foo.some_func``
    means the test never exercises the real code in ``vetinari.foo``.

    Args:
        filepath: Path to the test file.
        lines: Lines of the file.

    Returns:
        List of (line_number, code, severity, message) tuples.
    """
    violations: list[tuple[int, str, str, str]] = []
    module_stem = _derive_module_stem(filepath)

    def _check_patch_target(i: int, line: str, patched: str) -> None:
        if "noqa: VET242" in line or "noqa:VET242" in line:
            return
        if _is_local_patch_target(module_stem, patched):
            violations.append((
                i,
                "VET242",
                WARNING,
                f'Self-mocking: {filepath.name} patches "{patched}" — '
                "the test should exercise the real code, not mock it",
            ))

    for i, line in enumerate(lines, 1):
        # D1: check both decorator-style @patch(...) and context-manager with patch(...)
        m = _PATCH_DECORATOR.search(line)
        if m:
            _check_patch_target(i, line, m.group(1))
            continue
        m2 = _PATCH_CTX_MANAGER.search(line)
        if m2:
            _check_patch_target(i, line, m2.group(1))
    return violations


def check_mock_only_assertion(
    filepath: Path,
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """Find test functions whose only assertions are mock.called checks.

    Asserting only that a mock was called (not what it returned or what
    side-effects occurred) provides little quality signal.

    Args:
        filepath: Path to the test file.
        lines: Lines of the file.

    Returns:
        List of (line_number, code, severity, message) tuples.
    """
    # D2: mock assertion method calls are assertion-equivalent even without `assert`
    _MOCK_METHOD_ASSERT = re.compile(r"\.\s*assert_(?:called|not_called|any_call|has_calls)")

    violations: list[tuple[int, str, str, str]] = []

    for name, start_line, body, _decorators in _split_into_test_functions(lines):
        # D3: skip comment lines — a `# assert something` line is not a real assertion
        code_lines = [ln for ln in body if not ln.lstrip().startswith("#")]
        # Collect all lines that are assertion-equivalent:
        # - lines with the `assert` keyword (D3: already filtered comments)
        # - lines with mock assertion method calls like mock.assert_called_once() (D2)
        assert_lines = [
            ln for ln in code_lines
            if _ASSERT_STMT.search(ln) or _MOCK_METHOD_ASSERT.search(ln)
        ]
        if not assert_lines:
            continue  # VET241 catches this

        # Check if ALL assertion lines are weak mock.called / assert_called count checks.
        # Exact argument assertions (assert_called_once_with, assert_any_call,
        # assert_has_calls) are interaction contracts, not bare call-count checks.
        all_weak_mock = all(_MOCK_ONLY_ASSERT.search(ln) for ln in assert_lines)
        if all_weak_mock:
            violations.append((
                start_line,
                "VET243",
                WARNING,
                f"{name}() only asserts mock.called / assert_called — "
                "add assertions about the actual output or side effects",
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
        description="Detect test quality issues in tests/ directory (VET240-VET243).",
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
    """Scan tests/ for test quality violations and report results.

    Returns:
        0 if no errors found, 1 if errors found.
    """
    args = _build_parser().parse_args()

    if not TESTS_DIR.exists():
        print(f"error: directory not found: {TESTS_DIR}", file=sys.stderr)
        return 1

    test_files = _iter_test_files(TESTS_DIR)
    all_violations: list[tuple[str, int, str, str, str]] = []
    files_checked = 0

    for filepath in test_files:
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

        for ln, code, sev, msg in check_ambiguous_status(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        for ln, code, sev, msg in check_weak_status_assertions(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        for ln, code, sev, msg in check_response_shape_only_assertions(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        for ln, code, sev, msg in check_module_importorskip(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        for ln, code, sev, msg in check_zero_assert(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        for ln, code, sev, msg in check_self_mocking(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

        for ln, code, sev, msg in check_mock_only_assertion(filepath, lines):
            all_violations.append((rel_str, ln, code, sev, msg))

    # Filter by severity if --errors-only
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
        print(f"\n{files_checked} file{'s' if files_checked != 1 else ''} checked — {status}")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
