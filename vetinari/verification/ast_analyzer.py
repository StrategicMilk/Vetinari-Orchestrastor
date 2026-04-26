"""AST-based code analysis — extracts structure, detects dead code, and identifies complexity hotspots.

Uses Python's stdlib ``ast`` module to parse source code and extract:
- Function and class definitions with signatures
- Call relationships (which functions call which)
- Import dependencies (internal and external)
- Dead code (defined but never referenced functions)
- Complexity hotspots (functions exceeding line/branch thresholds)

Part of the Inspector verification pipeline (US-015), supporting
claim-level verification (US-013) and consistency checking (US-014).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# -- Data types ---------------------------------------------------------------


class SymbolKind(Enum):
    """Kind of top-level symbol extracted from source."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE_VAR = "module_var"


@dataclass(frozen=True, slots=True)
class SymbolDef:
    """A symbol (function, class, variable) defined in source code.

    Attributes:
        name: Fully qualified name (e.g. "MyClass.my_method").
        kind: Whether this is a function, method, class, or module variable.
        line_start: 1-based starting line number.
        line_end: 1-based ending line number.
        args: Parameter names for functions/methods; empty for classes/vars.
        decorators: Decorator names applied to this symbol.
        docstring: First line of the docstring, or None if absent.
    """

    name: str
    kind: SymbolKind
    line_start: int
    line_end: int
    args: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    docstring: str | None = None

    def line_count(self) -> int:
        """Number of source lines this symbol spans.

        Returns:
            Positive integer line count (inclusive of start and end).
        """
        return self.line_end - self.line_start + 1

    def __repr__(self) -> str:
        """Show kind, name, and line range for debugging."""
        return f"SymbolDef({self.kind.value} {self.name!r}, lines {self.line_start}-{self.line_end})"


@dataclass(frozen=True, slots=True)
class CallRelation:
    """A call from one symbol to another detected in the AST.

    Attributes:
        caller: Name of the function/method that makes the call.
        callee: Name of the function/method being called.
        line: Line number where the call occurs.
    """

    caller: str
    callee: str
    line: int


@dataclass(frozen=True, slots=True)
class ImportDep:
    """An import dependency found in the source.

    Attributes:
        module: The module being imported (e.g. "os.path", "vetinari.types").
        names: Specific names imported (e.g. ("Path",)); empty for bare imports.
        is_internal: True if the import is from the vetinari package.
        line: Line number of the import statement.
    """

    module: str
    names: tuple[str, ...]
    is_internal: bool
    line: int

    def __repr__(self) -> str:
        return f"ImportDep(module={self.module!r}, is_internal={self.is_internal!r}, line={self.line})"


@dataclass(frozen=True, slots=True)
class ComplexityHotspot:
    """A function or method that exceeds complexity thresholds.

    Attributes:
        name: Symbol name of the complex function.
        line_count: Number of source lines the function spans.
        branch_count: Number of if/elif/for/while/try branches.
        line: Starting line number.
        reason: Human-readable description of why this is flagged.
    """

    name: str
    line_count: int
    branch_count: int
    line: int
    reason: str

    def __repr__(self) -> str:
        return f"ComplexityHotspot(name={self.name!r}, line_count={self.line_count}, branch_count={self.branch_count})"


@dataclass(frozen=True, slots=True)
class AstAnalysisResult:
    """Complete AST analysis result for a single source file.

    Attributes:
        file_path: Path to the analysed file.
        symbols: All defined symbols (functions, classes, variables).
        calls: Call relationships between symbols.
        imports: Import dependencies.
        dead_code: Symbols defined but never referenced within the file.
        hotspots: Complexity hotspots exceeding thresholds.
        total_lines: Total number of lines in the source file.
    """

    file_path: str
    symbols: tuple[SymbolDef, ...]
    calls: tuple[CallRelation, ...]
    imports: tuple[ImportDep, ...]
    dead_code: tuple[str, ...]
    hotspots: tuple[ComplexityHotspot, ...]
    total_lines: int

    def __repr__(self) -> str:
        """Show summary counts for debugging."""
        return (
            f"AnalysisResult({self.file_path!r}, "
            f"symbols={len(self.symbols)}, calls={len(self.calls)}, "
            f"dead={len(self.dead_code)}, hotspots={len(self.hotspots)})"
        )


# -- AST visitors -------------------------------------------------------------

# Threshold defaults
_MAX_FUNCTION_LINES = 80  # Functions longer than this are hotspots
_MAX_BRANCH_COUNT = 10  # Functions with more branches than this are hotspots


class _SymbolExtractor(ast.NodeVisitor):
    """Extract function, class, and variable definitions from the AST.

    Walks the tree top-down and records every function, method, class,
    and module-level assignment as a SymbolDef.

    Args:
        source_lines: Source code split by newlines, for docstring extraction.
    """

    def __init__(self, source_lines: list[str]) -> None:
        self._source_lines = source_lines
        self.symbols: list[SymbolDef] = []
        self._class_stack: list[str] = []

    def _decorator_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[str, ...]:
        """Extract decorator names from a node's decorator list.

        Args:
            node: AST node with a decorator_list attribute.

        Returns:
            Tuple of decorator name strings.
        """
        names: list[str] = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                names.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    names.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    names.append(dec.func.attr)
        return tuple(names)

    def _get_docstring(self, node: ast.AST) -> str | None:
        """Extract the first line of a node's docstring.

        Args:
            node: An AST node that might have a docstring body.

        Returns:
            First line of the docstring, or None if absent.
        """
        docstring = ast.get_docstring(node)
        if docstring:
            first_line = docstring.split("\n")[0].strip()
            return first_line or None
        return None

    def _func_args(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
        """Extract parameter names from a function definition.

        Args:
            node: A FunctionDef or AsyncFunctionDef AST node.

        Returns:
            Tuple of parameter name strings (excluding 'self' and 'cls').
        """
        args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
        return tuple(args)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record a function or method definition.

        Args:
            node: A FunctionDef AST node.
        """
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record an async function or method definition.

        Args:
            node: An AsyncFunctionDef AST node.
        """
        self._record_function(node)
        self.generic_visit(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Record a function/method definition as a SymbolDef.

        Args:
            node: A function definition AST node.
        """
        if self._class_stack:
            name = f"{'.'.join(self._class_stack)}.{node.name}"
            kind = SymbolKind.METHOD
        else:
            name = node.name
            kind = SymbolKind.FUNCTION

        self.symbols.append(
            SymbolDef(
                name=name,
                kind=kind,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                args=self._func_args(node),
                decorators=self._decorator_names(node),
                docstring=self._get_docstring(node),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record a class definition and visit its methods.

        Args:
            node: A ClassDef AST node.
        """
        name = f"{'.'.join(self._class_stack)}.{node.name}" if self._class_stack else node.name

        self.symbols.append(
            SymbolDef(
                name=name,
                kind=SymbolKind.CLASS,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                decorators=self._decorator_names(node),
                docstring=self._get_docstring(node),
            )
        )

        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Record module-level variable assignments.

        Only records assignments at the top level (not inside functions/classes).

        Args:
            node: An Assign AST node.
        """
        # Only top-level assignments (no class stack and visited from module)
        if not self._class_stack:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self.symbols.append(
                        SymbolDef(
                            name=target.id,
                            kind=SymbolKind.MODULE_VAR,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                        )
                    )
        self.generic_visit(node)


class _CallExtractor(ast.NodeVisitor):
    """Extract call relationships from within function/method bodies.

    For each function, records which other functions it calls by name.
    """

    def __init__(self) -> None:
        self.calls: list[CallRelation] = []
        self._current_func: str | None = None
        self._class_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Enter a function scope and record calls within it.

        Args:
            node: A FunctionDef AST node.
        """
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Enter an async function scope and record calls within it.

        Args:
            node: An AsyncFunctionDef AST node.
        """
        self._visit_func(node)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Record the current function and visit its body for calls.

        Args:
            node: A function definition AST node.
        """
        prev = self._current_func
        if self._class_stack:
            self._current_func = f"{'.'.join(self._class_stack)}.{node.name}"
        else:
            self._current_func = node.name
        self.generic_visit(node)
        self._current_func = prev

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class scope for method name qualification.

        Args:
            node: A ClassDef AST node.
        """
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Record a function call if we're inside a function body.

        Args:
            node: A Call AST node.
        """
        if self._current_func is None:
            self.generic_visit(node)
            return

        callee_name: str | None = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr

        if callee_name:
            self.calls.append(
                CallRelation(
                    caller=self._current_func,
                    callee=callee_name,
                    line=node.lineno,
                )
            )

        self.generic_visit(node)


# -- Import extraction --------------------------------------------------------


def _extract_imports(tree: ast.Module) -> list[ImportDep]:
    """Walk the AST to collect all import statements.

    Args:
        tree: Parsed AST module.

    Returns:
        List of ImportDep objects for each import statement.
    """
    imports: list[ImportDep] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                ImportDep(
                    module=alias.name,
                    names=(),
                    is_internal=alias.name.startswith("vetinari"),
                    line=node.lineno,
                )
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = tuple(alias.name for alias in node.names) if node.names else ()
            imports.append(
                ImportDep(
                    module=node.module,
                    names=names,
                    is_internal=node.module.startswith("vetinari"),
                    line=node.lineno,
                )
            )
    return imports


# -- Dead code detection ------------------------------------------------------


def _find_dead_code(symbols: list[SymbolDef], calls: list[CallRelation]) -> list[str]:
    """Identify symbols that are defined but never called within the file.

    Only considers private functions/methods (prefixed with ``_``) as dead
    code candidates, since public symbols may be called from external modules.

    Args:
        symbols: All symbols defined in the file.
        calls: All call relationships within the file.

    Returns:
        List of symbol names that appear to be dead code.
    """
    called_names = {c.callee for c in calls}

    dead: list[str] = []
    for sym in symbols:
        if sym.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            continue
        # Only flag private symbols — public symbols may have external callers
        bare_name = sym.name.split(".")[-1]
        if not bare_name.startswith("_"):
            continue
        # Skip dunder methods (always called implicitly)
        if bare_name.startswith("__") and bare_name.endswith("__"):
            continue
        # Check if the bare function name appears in any call
        if bare_name not in called_names:
            dead.append(sym.name)

    return dead


# -- Complexity detection -----------------------------------------------------


def _count_branches(node: ast.AST) -> int:
    """Count branching statements (if/elif/for/while/try) within a node.

    Args:
        node: An AST node (typically a function body) to count branches in.

    Returns:
        Total count of branching statements.
    """
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
            count += 1
    return count


def _find_hotspots(
    tree: ast.Module,
    symbols: list[SymbolDef],
    max_lines: int = _MAX_FUNCTION_LINES,
    max_branches: int = _MAX_BRANCH_COUNT,
) -> list[ComplexityHotspot]:
    """Identify functions that exceed complexity thresholds.

    Checks each function/method for line count and branch count against
    configurable thresholds.

    Args:
        tree: The parsed AST module.
        symbols: All symbols to check (only FUNCTION and METHOD are considered).
        max_lines: Maximum allowed lines per function.
        max_branches: Maximum allowed branches per function.

    Returns:
        List of ComplexityHotspot objects for functions exceeding thresholds.
    """
    hotspots: list[ComplexityHotspot] = []

    # Build a lookup from (lineno) to AST node for branch counting
    func_nodes: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_nodes[node.lineno] = node

    for sym in symbols:
        if sym.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            continue

        reasons: list[str] = []
        lines = sym.line_count()
        branches = 0

        if lines > max_lines:
            reasons.append(f"{lines} lines (max {max_lines})")

        func_node = func_nodes.get(sym.line_start)
        if func_node:
            branches = _count_branches(func_node)
            if branches > max_branches:
                reasons.append(f"{branches} branches (max {max_branches})")

        if reasons:
            hotspots.append(
                ComplexityHotspot(
                    name=sym.name,
                    line_count=lines,
                    branch_count=branches,
                    line=sym.line_start,
                    reason="; ".join(reasons),
                )
            )

    return hotspots


# -- Public API ---------------------------------------------------------------


def analyze_source(
    source_code: str,
    file_path: str = "<unknown>",
    max_function_lines: int = _MAX_FUNCTION_LINES,
    max_branch_count: int = _MAX_BRANCH_COUNT,
) -> AstAnalysisResult:
    """Perform full AST analysis on a Python source file.

    Extracts symbols (functions, classes, variables), call relationships,
    import dependencies, dead code candidates, and complexity hotspots.

    Args:
        source_code: Full Python source text to analyse.
        file_path: Path to the source file (used in result metadata).
        max_function_lines: Threshold for flagging long functions.
        max_branch_count: Threshold for flagging high-branch functions.

    Returns:
        AnalysisResult with all extracted data. Returns a result with empty
        collections if the source cannot be parsed.
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as exc:
        logger.warning("Cannot parse %s for AST analysis: %s", file_path, exc)
        return AstAnalysisResult(
            file_path=file_path,
            symbols=(),
            calls=(),
            imports=(),
            dead_code=(),
            hotspots=(),
            total_lines=source_code.count("\n") + 1,
        )

    source_lines = source_code.splitlines()

    # Extract symbols
    sym_visitor = _SymbolExtractor(source_lines)
    sym_visitor.visit(tree)
    symbols = sym_visitor.symbols

    # Extract calls
    call_visitor = _CallExtractor()
    call_visitor.visit(tree)
    calls = call_visitor.calls

    # Extract imports
    imports = _extract_imports(tree)

    # Detect dead code
    dead_code = _find_dead_code(symbols, calls)

    # Find complexity hotspots
    hotspots = _find_hotspots(tree, symbols, max_function_lines, max_branch_count)

    total_lines = len(source_lines)

    logger.debug(
        "analyze_source: %s -> %d symbols, %d calls, %d imports, %d dead, %d hotspots",
        file_path,
        len(symbols),
        len(calls),
        len(imports),
        len(dead_code),
        len(hotspots),
    )

    return AstAnalysisResult(
        file_path=file_path,
        symbols=tuple(symbols),
        calls=tuple(calls),
        imports=tuple(imports),
        dead_code=tuple(dead_code),
        hotspots=tuple(hotspots),
        total_lines=total_lines,
    )


def get_function_defs(source_code: str, file_path: str = "<unknown>") -> list[SymbolDef]:
    """Extract only function and method definitions from source code.

    Convenience wrapper around analyze_source() that returns just the
    FUNCTION and METHOD symbols, useful for targeted analysis.

    Args:
        source_code: Full Python source text.
        file_path: Path for reporting.

    Returns:
        List of SymbolDef objects for functions and methods only.
    """
    result = analyze_source(source_code, file_path)
    return [s for s in result.symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)]


def get_import_graph(source_code: str, file_path: str = "<unknown>") -> list[ImportDep]:
    """Extract the import dependency list from source code.

    Convenience wrapper around analyze_source() that returns just the
    import dependencies.

    Args:
        source_code: Full Python source text.
        file_path: Path for reporting.

    Returns:
        List of ImportDep objects.
    """
    result = analyze_source(source_code, file_path)
    return list(result.imports)


__all__ = [
    "AstAnalysisResult",
    "CallRelation",
    "ComplexityHotspot",
    "ImportDep",
    "SymbolDef",
    "SymbolKind",
    "analyze_source",
    "get_function_defs",
    "get_import_graph",
]
