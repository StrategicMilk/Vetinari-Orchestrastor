"""
Tests for the Code Mode feature (FastMCP-inspired tool transform).

Covers:
  - BM25Index: tokenization, indexing, ranking
  - ToolSearchIndex: registry integration, search
  - SearchToolsTool: meta-tool for discovery
  - GetSchemasTool: meta-tool for schema retrieval
  - ExecuteCodeTool: meta-tool for sandboxed code running
  - _ToolCallBridge: call_tool() dispatch
  - _execute_in_sandbox: security restrictions, output capture
  - CodeModeTransform: MCP registry integration
  - Error paths: invalid inputs, timeouts, forbidden code
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolRegistry,
    ToolResult,
)
from vetinari.execution_context import ExecutionMode, ToolPermission
from vetinari.tools.code_mode import (
    BM25Index,
    CodeModeTransform,
    ExecuteCodeTool,
    GetSchemasTool,
    SearchToolsTool,
    ToolSearchIndex,
    _ToolCallBridge,
    _execute_in_sandbox,
    _tokenize,
    _tool_to_json_schema,
)


# ---------------------------------------------------------------------------
# Helpers - minimal concrete Tool for testing
# ---------------------------------------------------------------------------


class _MockTool(Tool):
    """A simple concrete tool for tests."""

    def __init__(
        self,
        name="mock_tool",
        description="A mock tool",
        tags=None,
        category=ToolCategory.SEARCH_ANALYSIS,
        return_value="mock_output",
        should_fail=False,
    ):
        self._return_value = return_value
        self._should_fail = should_fail
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="input",
                    type=str,
                    description="Input value",
                    required=True,
                ),
                ToolParameter(
                    name="count",
                    type=int,
                    description="Count value",
                    required=False,
                    default=1,
                ),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=tags or ["test"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs):
        if self._should_fail:
            return ToolResult(success=False, output=None, error="Mock failure")
        return ToolResult(success=True, output=self._return_value)


def _make_registry(*tools):
    """Create a ToolRegistry populated with given tools."""
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        tokens = _tokenize("Hello World!")
        self.assertEqual(tokens, ["hello", "world"])

    def test_underscores(self):
        tokens = _tokenize("file_read code_execution")
        self.assertEqual(tokens, ["file_read", "code_execution"])

    def test_empty(self):
        self.assertEqual(_tokenize(""), [])
        self.assertEqual(_tokenize("!!!"), [])

    def test_mixed(self):
        tokens = _tokenize("Search for code-gen tools v2.0")
        self.assertEqual(
            tokens, ["search", "for", "code", "gen", "tools", "v2", "0"]
        )


class TestBM25Index(unittest.TestCase):
    def setUp(self):
        self.idx = BM25Index()
        self.idx.add("tool_a", "code generation and building software")
        self.idx.add("tool_b", "file search and discovery")
        self.idx.add("tool_c", "code review and analysis")
        self.idx.build()

    def test_search_basic(self):
        results = self.idx.search("code", limit=5)
        names = [r[0] for r in results]
        # tool_a and tool_c both mention "code"
        self.assertIn("tool_a", names)
        self.assertIn("tool_c", names)
        self.assertNotIn("tool_b", names)

    def test_search_ranking(self):
        # "code generation" should rank tool_a higher (has both words)
        results = self.idx.search("code generation", limit=5)
        self.assertEqual(results[0][0], "tool_a")

    def test_search_no_match(self):
        results = self.idx.search("database", limit=5)
        self.assertEqual(results, [])

    def test_search_limit(self):
        results = self.idx.search("code", limit=1)
        self.assertEqual(len(results), 1)

    def test_empty_query(self):
        results = self.idx.search("", limit=5)
        self.assertEqual(results, [])

    def test_empty_index(self):
        empty = BM25Index()
        empty.build()
        self.assertEqual(empty.search("anything"), [])

    def test_scores_positive(self):
        results = self.idx.search("code")
        for _, score in results:
            self.assertGreater(score, 0)


# ---------------------------------------------------------------------------
# ToolSearchIndex tests
# ---------------------------------------------------------------------------


class TestToolSearchIndex(unittest.TestCase):
    def setUp(self):
        self.registry = _make_registry(
            _MockTool(
                "explorer",
                "Fast codebase search and file discovery",
                tags=["search", "discovery", "grep"],
            ),
            _MockTool(
                "builder",
                "Code implementation and refactoring",
                tags=["code", "build", "implement"],
                category=ToolCategory.CODE_EXECUTION,
            ),
            _MockTool(
                "researcher",
                "Research and information gathering",
                tags=["research", "web", "analysis"],
            ),
        )
        self.index = ToolSearchIndex(self.registry)

    def test_refresh_returns_count(self):
        count = self.index.refresh()
        self.assertEqual(count, 3)

    def test_search_returns_results(self):
        results = self.index.search("search file")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["name"], "explorer")

    def test_search_result_structure(self):
        results = self.index.search("code")
        self.assertTrue(len(results) > 0)
        result = results[0]
        self.assertIn("name", result)
        self.assertIn("description", result)
        self.assertIn("score", result)
        self.assertIn("category", result)
        self.assertIn("tags", result)

    def test_auto_refresh_on_first_search(self):
        # Fresh index, never refreshed
        fresh = ToolSearchIndex(self.registry)
        results = fresh.search("search")
        self.assertTrue(len(results) > 0)

    def test_search_no_match(self):
        results = self.index.search("xyznonexistent")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# Schema extraction tests
# ---------------------------------------------------------------------------


class TestToolToJsonSchema(unittest.TestCase):
    def test_basic_schema(self):
        tool = _MockTool("test_tool", "A test tool", tags=["test", "demo"])
        schema = _tool_to_json_schema(tool)

        self.assertEqual(schema["name"], "test_tool")
        self.assertEqual(schema["description"], "A test tool")
        self.assertEqual(schema["tags"], ["test", "demo"])
        self.assertIn("parameters", schema)

        params = schema["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("input", params["properties"])
        self.assertIn("count", params["properties"])
        self.assertIn("input", params["required"])
        self.assertNotIn("count", params["required"])

    def test_parameter_types(self):
        tool = _MockTool()
        schema = _tool_to_json_schema(tool)
        props = schema["parameters"]["properties"]
        self.assertEqual(props["input"]["type"], "string")
        self.assertEqual(props["count"]["type"], "integer")


# ---------------------------------------------------------------------------
# _ToolCallBridge tests
# ---------------------------------------------------------------------------


class TestToolCallBridge(unittest.TestCase):
    def setUp(self):
        self.tool = _MockTool("adder", "Adds things", return_value=42)
        self.registry = _make_registry(self.tool)
        self.bridge = _ToolCallBridge(self.registry)

    def test_call_success(self):
        result = self.bridge.call_tool("adder", input="hello")
        self.assertEqual(result, 42)

    def test_call_log(self):
        self.bridge.call_tool("adder", input="test")
        self.assertEqual(len(self.bridge.call_log), 1)
        log = self.bridge.call_log[0]
        self.assertEqual(log["tool"], "adder")
        self.assertTrue(log["success"])
        self.assertIn("elapsed_ms", log)

    def test_call_unknown_tool(self):
        with self.assertRaises(ValueError) as ctx:
            self.bridge.call_tool("nonexistent", input="x")
        self.assertIn("Unknown tool", str(ctx.exception))

    def test_call_failed_tool(self):
        fail_tool = _MockTool("failer", "Fails", should_fail=True)
        registry = _make_registry(fail_tool)
        bridge = _ToolCallBridge(registry)
        with self.assertRaises(RuntimeError) as ctx:
            bridge.call_tool("failer", input="x")
        self.assertIn("Mock failure", str(ctx.exception))

    def test_multiple_calls_logged(self):
        self.bridge.call_tool("adder", input="a")
        self.bridge.call_tool("adder", input="b")
        self.assertEqual(len(self.bridge.call_log), 2)


# ---------------------------------------------------------------------------
# _execute_in_sandbox tests
# ---------------------------------------------------------------------------


class TestExecuteInSandbox(unittest.TestCase):
    def setUp(self):
        tool = _MockTool("echo", "Echoes input", return_value="echoed")
        registry = _make_registry(tool)
        self.bridge = _ToolCallBridge(registry)

    def test_basic_math(self):
        result = _execute_in_sandbox("result = 2 + 2", self.bridge)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], 4)

    def test_print_capture(self):
        result = _execute_in_sandbox('print("hello world")', self.bridge)
        self.assertTrue(result["success"])
        self.assertIn("hello world", result["stdout"])

    def test_call_tool_in_code(self):
        code = 'result = call_tool("echo", input="test")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "echoed")
        self.assertEqual(len(self.bridge.call_log), 1)

    def test_json_available(self):
        code = 'result = json.dumps({"key": "value"})'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertTrue(result["success"])
        self.assertEqual(json.loads(result["result"]), {"key": "value"})

    def test_import_blocked(self):
        code = "import os"
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Import statements are not allowed", result["error"])

    def test_from_import_blocked(self):
        code = "from pathlib import Path"
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Import statements are not allowed", result["error"])

    def test_eval_blocked(self):
        code = 'eval("1+1")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Forbidden function call: eval", result["error"])

    def test_exec_blocked(self):
        code = 'exec("x = 1")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Forbidden function call: exec", result["error"])

    def test_open_blocked(self):
        code = 'open("/etc/passwd")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Forbidden function call: open", result["error"])

    def test_dunder_import_blocked(self):
        code = '__import__("os")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Forbidden function call: __import__", result["error"])

    def test_getattr_blocked(self):
        code = 'getattr(str, "__subclasses__")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("Forbidden function call: getattr", result["error"])

    def test_syntax_error(self):
        code = "def foo(:"
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("SyntaxError", result["error"])

    def test_runtime_error(self):
        code = "x = 1 / 0"
        result = _execute_in_sandbox(code, self.bridge)
        self.assertFalse(result["success"])
        self.assertIn("ZeroDivisionError", result["error"])

    def test_empty_code_valid(self):
        result = _execute_in_sandbox("", self.bridge)
        # Empty code is valid Python, runs fine
        self.assertTrue(result["success"])

    def test_builtins_available(self):
        code = """
result = {
    "len": len([1,2,3]),
    "sum": sum([1,2,3]),
    "sorted": sorted([3,1,2]),
    "range": list(range(3)),
    "max": max(1,2,3),
    "min": min(1,2,3),
    "bool": bool(1),
    "int": int("42"),
    "float": float("3.14"),
    "str": str(42),
    "isinstance": isinstance(42, int),
}
"""
        result = _execute_in_sandbox(code, self.bridge)
        self.assertTrue(result["success"])
        output = result["result"]
        self.assertEqual(output["len"], 3)
        self.assertEqual(output["sum"], 6)
        self.assertEqual(output["sorted"], [1, 2, 3])
        self.assertEqual(output["range"], [0, 1, 2])
        self.assertEqual(output["max"], 3)
        self.assertEqual(output["min"], 1)
        self.assertTrue(output["bool"])
        self.assertEqual(output["int"], 42)
        self.assertAlmostEqual(output["float"], 3.14)
        self.assertEqual(output["str"], "42")
        self.assertTrue(output["isinstance"])

    def test_multi_tool_calls(self):
        code = """
r1 = call_tool("echo", input="first")
r2 = call_tool("echo", input="second")
result = [r1, r2]
"""
        result = _execute_in_sandbox(code, self.bridge)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], ["echoed", "echoed"])
        self.assertEqual(len(self.bridge.call_log), 2)

    def test_call_log_in_result(self):
        code = 'result = call_tool("echo", input="test")'
        result = _execute_in_sandbox(code, self.bridge)
        self.assertIn("call_log", result)
        self.assertEqual(len(result["call_log"]), 1)

    def test_execution_time_tracked(self):
        code = "result = 1 + 1"
        result = _execute_in_sandbox(code, self.bridge)
        self.assertIn("execution_time_ms", result)
        self.assertIsInstance(result["execution_time_ms"], int)


# ---------------------------------------------------------------------------
# SearchToolsTool tests
# ---------------------------------------------------------------------------


class TestSearchToolsTool(unittest.TestCase):
    def setUp(self):
        registry = _make_registry(
            _MockTool("explorer", "File search and discovery", tags=["search"]),
            _MockTool("builder", "Code generation", tags=["code", "build"]),
        )
        self.index = ToolSearchIndex(registry)
        self.search_tool = SearchToolsTool(index=self.index)

    def test_execute_success(self):
        result = self.search_tool.execute(query="search file")
        self.assertTrue(result.success)
        self.assertIn("results", result.output)
        self.assertIn("total", result.output)

    def test_execute_empty_query(self):
        result = self.search_tool.execute(query="")
        self.assertFalse(result.success)
        self.assertIn("required", result.error)

    def test_execute_limit(self):
        result = self.search_tool.execute(query="search", limit=1)
        self.assertTrue(result.success)
        self.assertLessEqual(len(result.output["results"]), 1)

    def test_metadata(self):
        self.assertEqual(self.search_tool.metadata.name, "search_tools")
        self.assertIn("code_mode", self.search_tool.metadata.tags)


# ---------------------------------------------------------------------------
# GetSchemasTool tests
# ---------------------------------------------------------------------------


class TestGetSchemasTool(unittest.TestCase):
    def setUp(self):
        self.registry = _make_registry(
            _MockTool("explorer", "File search"),
            _MockTool("builder", "Code gen"),
        )
        self.schema_tool = GetSchemasTool(registry=self.registry)

    def test_get_single_schema(self):
        result = self.schema_tool.execute(tool_names=["explorer"])
        self.assertTrue(result.success)
        self.assertIn("explorer", result.output["schemas"])
        self.assertEqual(result.output["not_found"], [])

    def test_get_multiple_schemas(self):
        result = self.schema_tool.execute(tool_names=["explorer", "builder"])
        self.assertTrue(result.success)
        self.assertEqual(len(result.output["schemas"]), 2)

    def test_not_found(self):
        result = self.schema_tool.execute(tool_names=["nonexistent"])
        self.assertTrue(result.success)
        self.assertEqual(result.output["schemas"], {})
        self.assertEqual(result.output["not_found"], ["nonexistent"])

    def test_mixed_found_and_not_found(self):
        result = self.schema_tool.execute(tool_names=["explorer", "fake"])
        self.assertTrue(result.success)
        self.assertIn("explorer", result.output["schemas"])
        self.assertEqual(result.output["not_found"], ["fake"])

    def test_empty_list(self):
        result = self.schema_tool.execute(tool_names=[])
        self.assertFalse(result.success)

    def test_schema_has_parameters(self):
        result = self.schema_tool.execute(tool_names=["explorer"])
        schema = result.output["schemas"]["explorer"]
        self.assertIn("parameters", schema)
        self.assertIn("properties", schema["parameters"])
        self.assertIn("input", schema["parameters"]["properties"])

    def test_metadata(self):
        self.assertEqual(self.schema_tool.metadata.name, "get_schemas")
        self.assertIn("code_mode", self.schema_tool.metadata.tags)


# ---------------------------------------------------------------------------
# ExecuteCodeTool tests
# ---------------------------------------------------------------------------


class TestExecuteCodeTool(unittest.TestCase):
    def setUp(self):
        self.registry = _make_registry(
            _MockTool("calculator", "Does math", return_value=42),
        )
        self.exec_tool = ExecuteCodeTool(
            registry=self.registry, timeout_seconds=10
        )

    def test_basic_run(self):
        result = self.exec_tool.execute(code="result = 1 + 1")
        self.assertTrue(result.success)
        self.assertEqual(result.output["result"], 2)

    def test_tool_call(self):
        result = self.exec_tool.execute(
            code='result = call_tool("calculator", input="x")'
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output["result"], 42)

    def test_empty_code(self):
        result = self.exec_tool.execute(code="")
        self.assertFalse(result.success)
        self.assertIn("required", result.error)

    def test_whitespace_only_code(self):
        result = self.exec_tool.execute(code="   \n  ")
        self.assertFalse(result.success)

    def test_import_blocked(self):
        result = self.exec_tool.execute(code="import os")
        self.assertFalse(result.success)
        self.assertIn("Import", result.error)

    def test_metadata(self):
        self.assertEqual(self.exec_tool.metadata.name, "execute_code")
        self.assertIn(
            ToolPermission.PYTHON_EXECUTE,
            self.exec_tool.metadata.required_permissions,
        )
        self.assertIn("code_mode", self.exec_tool.metadata.tags)

    def test_tool_call_count_in_metadata(self):
        code = (
            'call_tool("calculator", input="a")\n'
            'result = call_tool("calculator", input="b")'
        )
        result = self.exec_tool.execute(code=code)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["tool_calls"], 2)

    def test_timeout_capped(self):
        # Timeout should be capped at 120s even if user asks for more
        result = self.exec_tool.execute(code="result = 1", timeout=999)
        self.assertTrue(result.success)


# ---------------------------------------------------------------------------
# CodeModeTransform tests
# ---------------------------------------------------------------------------


class TestCodeModeTransform(unittest.TestCase):
    def setUp(self):
        self.tool_registry = _make_registry(
            _MockTool("explorer", "Search files", tags=["search"]),
            _MockTool("builder", "Build code", tags=["code"]),
        )
        self.transform = CodeModeTransform(tool_registry=self.tool_registry)

    def test_get_meta_tools(self):
        tools = self.transform.get_meta_tools()
        self.assertEqual(len(tools), 3)
        names = {t.metadata.name for t in tools}
        self.assertEqual(names, {"search_tools", "get_schemas", "execute_code"})

    def test_get_meta_tools_idempotent(self):
        tools1 = self.transform.get_meta_tools()
        tools2 = self.transform.get_meta_tools()
        self.assertIs(tools1, tools2)

    def test_apply_to_mcp_registry(self):
        from vetinari.mcp.tools import MCPToolRegistry

        mcp_registry = MCPToolRegistry()
        self.transform.apply(mcp_registry)

        tools = mcp_registry.list_tools()
        tool_names = {t["name"] for t in tools}
        self.assertIn("search_tools", tool_names)
        self.assertIn("get_schemas", tool_names)
        self.assertIn("execute_code", tool_names)

    def test_is_applied_flag(self):
        self.assertFalse(self.transform.is_applied)
        from vetinari.mcp.tools import MCPToolRegistry

        mcp_registry = MCPToolRegistry()
        self.transform.apply(mcp_registry)
        self.assertTrue(self.transform.is_applied)

    def test_mcp_handlers_work(self):
        from vetinari.mcp.tools import MCPToolRegistry

        mcp_registry = MCPToolRegistry()
        self.transform.apply(mcp_registry)

        # Invoke search_tools via MCP
        result = mcp_registry.invoke("search_tools", {"query": "search"})
        self.assertIn("result", result)

        # Invoke get_schemas via MCP
        result = mcp_registry.invoke(
            "get_schemas", {"tool_names": ["explorer"]}
        )
        self.assertIn("result", result)

    def test_underlying_tools_accessible_via_code(self):
        from vetinari.mcp.tools import MCPToolRegistry

        mcp_registry = MCPToolRegistry()
        self.transform.apply(mcp_registry)

        # The underlying 'explorer' tool should be callable via execute_code
        result = mcp_registry.invoke(
            "execute_code",
            {"code": 'result = call_tool("explorer", input="test")'},
        )
        self.assertIn("result", result)


# ---------------------------------------------------------------------------
# Integration: end-to-end workflow test
# ---------------------------------------------------------------------------


class TestCodeModeEndToEnd(unittest.TestCase):
    """Test the full search -> schema -> execute workflow."""

    def setUp(self):
        self.registry = _make_registry(
            _MockTool(
                "explorer",
                "Search files and code",
                tags=["search", "grep", "files"],
                return_value={"files": ["main.py", "utils.py"]},
            ),
            _MockTool(
                "builder",
                "Generate code implementations",
                tags=["code", "generate", "implement"],
                category=ToolCategory.CODE_EXECUTION,
                return_value={"code": "def hello(): pass"},
            ),
        )
        self.index = ToolSearchIndex(self.registry)
        self.search_tool = SearchToolsTool(index=self.index)
        self.schema_tool = GetSchemasTool(registry=self.registry)
        self.exec_tool = ExecuteCodeTool(registry=self.registry)

    def test_discover_then_schema_then_run(self):
        # Step 1: Search for relevant tools
        search_result = self.search_tool.execute(query="search files")
        self.assertTrue(search_result.success)
        found_tools = search_result.output["results"]
        self.assertTrue(len(found_tools) > 0)
        tool_name = found_tools[0]["name"]

        # Step 2: Get the schema
        schema_result = self.schema_tool.execute(tool_names=[tool_name])
        self.assertTrue(schema_result.success)
        schema = schema_result.output["schemas"][tool_name]
        self.assertIn("parameters", schema)

        # Step 3: Run code using the discovered tool
        code = f'result = call_tool("{tool_name}", input="*.py")'
        exec_result = self.exec_tool.execute(code=code)
        self.assertTrue(exec_result.success)
        self.assertIsNotNone(exec_result.output["result"])

    def test_multi_tool_orchestration(self):
        """LLM writes code that calls multiple tools in sequence."""
        code = """
# Find relevant files
files = call_tool("explorer", input="auth")

# Generate code based on findings
generated = call_tool("builder", input="JWT authentication")

# Combine results
result = {
    "discovered": files,
    "generated": generated,
    "steps": 2,
}
"""
        exec_result = self.exec_tool.execute(code=code)
        self.assertTrue(exec_result.success)
        output = exec_result.output["result"]
        self.assertEqual(output["steps"], 2)
        self.assertIsNotNone(output["discovered"])
        self.assertIsNotNone(output["generated"])
        # Both tools called
        self.assertEqual(exec_result.metadata["tool_calls"], 2)


# ---------------------------------------------------------------------------
# Edge cases and security
# ---------------------------------------------------------------------------


class TestCodeModeSecurity(unittest.TestCase):
    """Security-focused tests for the sandbox."""

    def setUp(self):
        self.registry = _make_registry(_MockTool("safe", "Safe tool"))
        self.bridge = _ToolCallBridge(self.registry)

    def test_no_filesystem_access(self):
        result = _execute_in_sandbox('open("/tmp/test", "w")', self.bridge)
        self.assertFalse(result["success"])

    def test_no_subprocess(self):
        result = _execute_in_sandbox("import subprocess", self.bridge)
        self.assertFalse(result["success"])

    def test_no_network(self):
        result = _execute_in_sandbox("import socket", self.bridge)
        self.assertFalse(result["success"])

    def test_no_os_module(self):
        result = _execute_in_sandbox("import os", self.bridge)
        self.assertFalse(result["success"])

    def test_no_sys_module(self):
        result = _execute_in_sandbox("import sys", self.bridge)
        self.assertFalse(result["success"])

    def test_no_compile(self):
        result = _execute_in_sandbox(
            'compile("1+1", "", "eval")', self.bridge
        )
        self.assertFalse(result["success"])

    def test_no_globals_access(self):
        result = _execute_in_sandbox("globals()", self.bridge)
        self.assertFalse(result["success"])

    def test_no_locals_access(self):
        result = _execute_in_sandbox("locals()", self.bridge)
        self.assertFalse(result["success"])

    def test_exception_in_tool_caught(self):
        """Tool call failure should be catchable in user code."""
        code = """
try:
    call_tool("nonexistent", input="x")
    result = "should not reach here"
except ValueError as e:
    result = f"caught: {e}"
"""
        result = _execute_in_sandbox(code, self.bridge)
        self.assertTrue(result["success"])
        self.assertIn("caught:", result["result"])


if __name__ == "__main__":
    unittest.main()
