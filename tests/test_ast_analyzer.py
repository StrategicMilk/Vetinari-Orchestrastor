"""Tests for vetinari.verification.ast_analyzer.

Verifies symbol extraction, call relationship detection, import parsing,
dead code detection, complexity hotspot identification, and convenience
wrappers.

Part of US-015: AST-Based Code Analysis.
"""

from __future__ import annotations

import textwrap

import pytest

from vetinari.verification.ast_analyzer import (
    AstAnalysisResult,
    CallRelation,
    ComplexityHotspot,
    ImportDep,
    SymbolDef,
    SymbolKind,
    analyze_source,
    get_function_defs,
    get_import_graph,
)

# -- Fixtures ----------------------------------------------------------------

SAMPLE_MODULE = textwrap.dedent("""\
    import os
    from pathlib import Path

    from vetinari.types import AgentType

    MAX_RETRIES = 3

    def public_func(x, y):
        \"\"\"Add two numbers.\"\"\"
        return _helper(x) + y

    def _helper(x):
        return x * 2

    def _unused_private():
        pass

    class MyClass:
        \"\"\"A sample class.\"\"\"

        def method_a(self, n):
            return self.method_b(n + 1)

        def method_b(self, n):
            return n * 2
""")

COMPLEX_FUNCTION = textwrap.dedent("""\
    def complex_func(data):
        result = []
        for item in data:
            if item > 0:
                if item > 100:
                    result.append("big")
                elif item > 50:
                    result.append("medium")
                else:
                    result.append("small")
            else:
                try:
                    val = process(item)
                    if val:
                        result.append(val)
                except ValueError:
                    pass
                while len(result) > 10:
                    result.pop(0)
                for sub in subdivide(item):
                    if sub > 0:
                        result.append(sub)
        return result
""")


# -- Symbol extraction -------------------------------------------------------


class TestSymbolExtraction:
    """Tests for extracting symbols (functions, classes, variables) from source."""

    def test_functions_extracted(self) -> None:
        """Top-level functions must be extracted as FUNCTION symbols."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        func_names = {s.name for s in result.symbols if s.kind == SymbolKind.FUNCTION}
        assert "public_func" in func_names
        assert "_helper" in func_names
        assert "_unused_private" in func_names

    def test_methods_extracted(self) -> None:
        """Class methods must be extracted as METHOD symbols with qualified names."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        method_names = {s.name for s in result.symbols if s.kind == SymbolKind.METHOD}
        assert "MyClass.method_a" in method_names
        assert "MyClass.method_b" in method_names

    def test_classes_extracted(self) -> None:
        """Classes must be extracted as CLASS symbols."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        class_names = {s.name for s in result.symbols if s.kind == SymbolKind.CLASS}
        assert "MyClass" in class_names

    def test_module_vars_extracted(self) -> None:
        """UPPER_CASE module-level assignments must be extracted as MODULE_VAR."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        var_names = {s.name for s in result.symbols if s.kind == SymbolKind.MODULE_VAR}
        assert "MAX_RETRIES" in var_names

    def test_function_args_captured(self) -> None:
        """Function parameter names must be captured in SymbolDef.args."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        public_func = next(s for s in result.symbols if s.name == "public_func")
        assert public_func.args == ("x", "y")

    def test_self_excluded_from_method_args(self) -> None:
        """'self' must be excluded from method args."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        method_a = next(s for s in result.symbols if s.name == "MyClass.method_a")
        assert "self" not in method_a.args
        assert "n" in method_a.args

    def test_docstring_extracted(self) -> None:
        """First line of docstring must be captured in SymbolDef.docstring."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        public_func = next(s for s in result.symbols if s.name == "public_func")
        assert public_func.docstring == "Add two numbers."

    def test_no_docstring_is_none(self) -> None:
        """Functions without docstrings must have docstring=None."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        helper = next(s for s in result.symbols if s.name == "_helper")
        assert helper.docstring is None

    def test_line_count_method(self) -> None:
        """SymbolDef.line_count() must return correct span length."""
        sym = SymbolDef(name="test", kind=SymbolKind.FUNCTION, line_start=10, line_end=20)
        assert sym.line_count() == 11

    def test_symbol_repr(self) -> None:
        """SymbolDef repr must show kind, name, and line range."""
        sym = SymbolDef(name="foo", kind=SymbolKind.FUNCTION, line_start=1, line_end=5)
        r = repr(sym)
        assert "function" in r
        assert "foo" in r
        assert "1" in r

    def test_decorator_extraction(self) -> None:
        """Decorators must be captured in SymbolDef.decorators."""
        source = "@property\ndef my_prop(self):\n    return self._x\n"
        result = analyze_source(source, "test.py")

        func = next(s for s in result.symbols if s.name == "my_prop")
        assert "property" in func.decorators


# -- Call relationship extraction ---------------------------------------------


class TestCallExtraction:
    """Tests for detecting call relationships between functions."""

    def test_direct_call_detected(self) -> None:
        """A call from public_func to _helper must be detected."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        caller_callee = {(c.caller, c.callee) for c in result.calls}
        assert ("public_func", "_helper") in caller_callee

    def test_method_call_detected(self) -> None:
        """A call from method_a to method_b must be detected."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        caller_callee = {(c.caller, c.callee) for c in result.calls}
        assert ("MyClass.method_a", "method_b") in caller_callee

    def test_call_has_line_number(self) -> None:
        """CallRelation must include the line number of the call."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        helper_calls = [c for c in result.calls if c.callee == "_helper"]
        assert len(helper_calls) >= 1
        assert helper_calls[0].line > 0


# -- Import extraction -------------------------------------------------------


class TestImportExtraction:
    """Tests for extracting import dependencies."""

    def test_stdlib_import_detected(self) -> None:
        """import os must be detected as an external import."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        os_imports = [i for i in result.imports if i.module == "os"]
        assert len(os_imports) == 1
        assert os_imports[0].is_internal is False

    def test_from_import_names_captured(self) -> None:
        """from pathlib import Path must capture 'Path' in names."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        pathlib_imports = [i for i in result.imports if i.module == "pathlib"]
        assert len(pathlib_imports) == 1
        assert "Path" in pathlib_imports[0].names

    def test_internal_import_flagged(self) -> None:
        """from vetinari.types import AgentType must be flagged as internal."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        internal = [i for i in result.imports if i.is_internal]
        assert len(internal) >= 1
        assert any(i.module == "vetinari.types" for i in internal)

    def test_import_has_line_number(self) -> None:
        """ImportDep must include the line number."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        for imp in result.imports:
            assert imp.line > 0


# -- Dead code detection ------------------------------------------------------


class TestDeadCodeDetection:
    """Tests for identifying unused private symbols."""

    def test_unused_private_detected(self) -> None:
        """_unused_private (defined but never called) must be flagged as dead code."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        assert "_unused_private" in result.dead_code

    def test_used_private_not_flagged(self) -> None:
        """_helper (called by public_func) must NOT be flagged as dead code."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        assert "_helper" not in result.dead_code

    def test_public_functions_not_flagged(self) -> None:
        """Public functions must never be flagged (may have external callers)."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")

        assert "public_func" not in result.dead_code

    def test_dunder_methods_not_flagged(self) -> None:
        """Dunder methods must never be flagged (called implicitly)."""
        source = textwrap.dedent("""\
            class Foo:
                def __repr__(self):
                    return "Foo"
                def __init__(self):
                    pass
        """)
        result = analyze_source(source, "test.py")

        dead_bare = [d.split(".")[-1] for d in result.dead_code]
        assert "__repr__" not in dead_bare
        assert "__init__" not in dead_bare


# -- Complexity hotspot detection ---------------------------------------------


class TestComplexityHotspots:
    """Tests for identifying functions that exceed complexity thresholds."""

    def test_high_branch_count_flagged(self) -> None:
        """A function with many branches must be flagged as a hotspot."""
        result = analyze_source(COMPLEX_FUNCTION, "test.py", max_branch_count=5)

        assert len(result.hotspots) >= 1
        assert result.hotspots[0].name == "complex_func"
        assert "branch" in result.hotspots[0].reason.lower()

    def test_long_function_flagged(self) -> None:
        """A function exceeding max_function_lines must be flagged."""
        # Generate a long function
        lines = ["def long_func():\n"] + [f"    x_{i} = {i}\n" for i in range(100)]
        source = "".join(lines)
        result = analyze_source(source, "test.py", max_function_lines=50)

        assert len(result.hotspots) >= 1
        assert result.hotspots[0].name == "long_func"
        assert "lines" in result.hotspots[0].reason.lower()

    def test_simple_function_not_flagged(self) -> None:
        """A short, simple function must not be flagged."""
        source = "def simple():\n    return 42\n"
        result = analyze_source(source, "test.py")

        assert len(result.hotspots) == 0

    def test_hotspot_has_counts(self) -> None:
        """ComplexityHotspot must include line_count and branch_count."""
        result = analyze_source(COMPLEX_FUNCTION, "test.py", max_branch_count=5)

        if result.hotspots:
            hotspot = result.hotspots[0]
            assert hotspot.line_count > 0
            assert hotspot.branch_count > 5


# -- AnalysisResult ----------------------------------------------------------


class TestAnalysisResult:
    """Tests for the AnalysisResult aggregate."""

    def test_total_lines_counted(self) -> None:
        """total_lines must match the number of source lines."""
        source = "x = 1\ny = 2\nz = 3\n"
        result = analyze_source(source, "test.py")
        assert result.total_lines == 3

    def test_file_path_stored(self) -> None:
        """file_path must be stored in the result."""
        result = analyze_source("x = 1\n", "my_module.py")
        assert result.file_path == "my_module.py"

    def test_syntax_error_returns_empty_result(self) -> None:
        """Unparseable source must return an AnalysisResult with empty collections."""
        result = analyze_source("def broken(\n", "bad.py")

        assert isinstance(result, AstAnalysisResult)
        assert len(result.symbols) == 0
        assert len(result.calls) == 0
        assert len(result.imports) == 0
        assert len(result.dead_code) == 0
        assert len(result.hotspots) == 0
        assert result.total_lines > 0

    def test_result_repr(self) -> None:
        """AnalysisResult repr must show summary counts."""
        result = analyze_source(SAMPLE_MODULE, "sample.py")
        r = repr(result)
        assert "sample.py" in r
        assert "symbols=" in r
        assert "calls=" in r


# -- Convenience wrappers ----------------------------------------------------


class TestConvenienceWrappers:
    """Tests for get_function_defs() and get_import_graph()."""

    def test_get_function_defs_returns_only_functions_and_methods(self) -> None:
        """get_function_defs() must return only FUNCTION and METHOD symbols."""
        defs = get_function_defs(SAMPLE_MODULE, "sample.py")

        for d in defs:
            assert d.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)

        names = {d.name for d in defs}
        assert "public_func" in names
        assert "MyClass.method_a" in names
        # Should NOT include classes or module vars
        assert "MyClass" not in names
        assert "MAX_RETRIES" not in names

    def test_get_import_graph_returns_imports(self) -> None:
        """get_import_graph() must return import dependencies."""
        imports = get_import_graph(SAMPLE_MODULE, "sample.py")

        modules = {i.module for i in imports}
        assert "os" in modules
        assert "pathlib" in modules
        assert "vetinari.types" in modules


# -- Async function support ---------------------------------------------------


class TestAsyncSupport:
    """Tests for async function and method handling."""

    def test_async_function_extracted(self) -> None:
        """Async functions must be extracted as FUNCTION symbols."""
        source = "async def fetch_data(url):\n    return await get(url)\n"
        result = analyze_source(source, "test.py")

        func_names = {s.name for s in result.symbols if s.kind == SymbolKind.FUNCTION}
        assert "fetch_data" in func_names

    def test_async_method_extracted(self) -> None:
        """Async methods must be extracted as METHOD symbols."""
        source = textwrap.dedent("""\
            class Client:
                async def fetch(self, url):
                    return await self.get(url)
        """)
        result = analyze_source(source, "test.py")

        method_names = {s.name for s in result.symbols if s.kind == SymbolKind.METHOD}
        assert "Client.fetch" in method_names

    def test_async_function_args_captured(self) -> None:
        """Async function args must be captured correctly."""
        source = "async def process(data, timeout):\n    pass\n"
        result = analyze_source(source, "test.py")

        func = next(s for s in result.symbols if s.name == "process")
        assert func.args == ("data", "timeout")


# -- Module wiring -----------------------------------------------------------


class TestModuleWiring:
    """Verify the module is importable and exports are correct."""

    def test_imports_from_ast_analyzer(self) -> None:
        """Key types must be importable from vetinari.verification.ast_analyzer."""
        from vetinari.verification.ast_analyzer import (
            AstAnalysisResult,
            SymbolDef,
            SymbolKind,
            analyze_source,
            get_function_defs,
            get_import_graph,
        )

        assert AstAnalysisResult is not None
        assert SymbolDef is not None
        assert SymbolKind is not None
        assert analyze_source is not None
        assert get_function_defs is not None
        assert get_import_graph is not None

    def test_imports_from_verification_init(self) -> None:
        """Key types must be importable from vetinari.verification."""
        from vetinari.verification import (
            AstAnalysisResult,
            SymbolKind,
            analyze_source,
            get_function_defs,
        )

        assert AstAnalysisResult is not None
        assert SymbolKind is not None
        assert analyze_source is not None
        assert get_function_defs is not None
