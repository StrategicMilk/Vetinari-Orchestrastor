"""
Code Mode - FastMCP-inspired tool exposure transform for Vetinari.

Instead of exposing N individual tools directly to the LLM, Code Mode
exposes three meta-tools:

  1. **search_tools** - BM25-ranked keyword search over all registered
     tool names, descriptions, tags, and parameter info.
  2. **get_schemas** - Return full JSON schemas for one or more tools
     by exact name.
  3. **execute_code** - Run a Python snippet in a sandbox that has
     access to a ``call_tool(name, **params)`` bridge function.

This dramatically reduces the number of tools in the LLM's context
window while still giving it full access to all capabilities via
programmatic composition.

Reference: FastMCP 3.1 Code Mode
  https://www.jlowin.dev/blog/fastmcp-3-1-code-mode
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from vetinari.execution_context import ExecutionMode, ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 search index
# ---------------------------------------------------------------------------


@dataclass
class _ToolDoc:
    """A searchable document representing a single tool."""

    tool_name: str
    text: str  # concatenated searchable text
    tokens: List[str] = field(default_factory=list)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """
    Okapi BM25 ranking index over tool documents.

    Lightweight, zero-dependency implementation suitable for the small
    corpus of tool descriptions (typically 10-50 documents).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[_ToolDoc] = []
        self._df: Dict[str, int] = defaultdict(int)  # document frequency
        self._avgdl: float = 0.0

    # -- Building ----------------------------------------------------------

    def add(self, tool_name: str, searchable_text: str) -> None:
        """Add a tool document to the index."""
        tokens = _tokenize(searchable_text)
        self._docs.append(
            _ToolDoc(tool_name=tool_name, text=searchable_text, tokens=tokens)
        )

    def build(self) -> None:
        """Compute IDF and average doc length after all docs are added."""
        if not self._docs:
            return
        self._df.clear()
        total_len = 0
        for doc in self._docs:
            total_len += len(doc.tokens)
            seen: Set[str] = set()
            for tok in doc.tokens:
                if tok not in seen:
                    self._df[tok] += 1
                    seen.add(tok)
        self._avgdl = total_len / len(self._docs) if self._docs else 1.0

    # -- Querying ----------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> List[tuple]:
        """
        Return ``(tool_name, score)`` pairs ranked by BM25 relevance.

        Args:
            query: Free-text search query.
            limit: Maximum results to return.

        Returns:
            List of (tool_name, bm25_score) tuples, descending by score.
        """
        if not self._docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self._docs)
        scores: List[tuple] = []

        for doc in self._docs:
            score = 0.0
            dl = len(doc.tokens)

            # Build term frequency map for this doc
            tf_map: Dict[str, int] = defaultdict(int)
            for tok in doc.tokens:
                tf_map[tok] += 1

            for qt in query_tokens:
                if qt not in self._df:
                    continue
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                df = self._df[qt]
                # IDF with smoothing
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                # BM25 TF normalization
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
                score += idf * tf_norm

            if score > 0:
                scores.append((doc.tool_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]


# ---------------------------------------------------------------------------
# Tool index - builds searchable text from ToolMetadata
# ---------------------------------------------------------------------------


class ToolSearchIndex:
    """
    Maintains a BM25 search index over all tools registered in a
    ``ToolRegistry``.  Rebuilt on demand via ``refresh()``.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or get_tool_registry()
        self._bm25 = BM25Index()
        self._built = False

    def refresh(self) -> int:
        """(Re)build the index from the current registry state.

        Returns:
            Number of tools indexed.
        """
        self._bm25 = BM25Index()
        tools = self._registry.list_tools()

        for tool in tools:
            meta = tool.metadata
            parts = [
                meta.name,
                meta.description,
                meta.category.value.replace("_", " "),
                " ".join(meta.tags),
            ]
            for p in meta.parameters:
                parts.append(p.name)
                parts.append(p.description)
            searchable = " ".join(parts)
            self._bm25.add(meta.name, searchable)

        self._bm25.build()
        self._built = True
        logger.info("ToolSearchIndex: indexed %d tools", len(tools))
        return len(tools)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tools by query, returning ranked results with summaries.

        Args:
            query: Free-text search query.
            limit: Max results.

        Returns:
            List of dicts with ``name``, ``description``, ``score``,
            ``category``, ``tags``.
        """
        if not self._built:
            self.refresh()

        hits = self._bm25.search(query, limit=limit)
        results = []
        for tool_name, score in hits:
            tool = self._registry.get(tool_name)
            if tool is None:
                continue
            meta = tool.metadata
            results.append(
                {
                    "name": meta.name,
                    "description": meta.description,
                    "score": round(score, 3),
                    "category": meta.category.value,
                    "tags": meta.tags,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def _tool_to_json_schema(tool: Tool) -> Dict[str, Any]:
    """Convert a Tool's metadata into a JSON-Schema-style description."""
    meta = tool.metadata
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for p in meta.parameters:
        prop: Dict[str, Any] = {
            "description": p.description,
        }
        # Map Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        prop["type"] = type_map.get(p.type, "string")

        if p.allowed_values:
            prop["enum"] = p.allowed_values
        if p.default is not None:
            prop["default"] = p.default

        properties[p.name] = prop
        if p.required:
            required.append(p.name)

    return {
        "name": meta.name,
        "description": meta.description,
        "category": meta.category.value,
        "version": meta.version,
        "tags": meta.tags,
        "required_permissions": [p.value for p in meta.required_permissions],
        "allowed_modes": [m.value for m in meta.allowed_modes],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# Sandboxed call_tool bridge
# ---------------------------------------------------------------------------


class _ToolCallBridge:
    """
    Provides the ``call_tool(name, **params)`` function that user code
    invokes inside the sandbox.  Each call is dispatched to the real tool
    via ``Tool.run()``.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self.call_log: List[Dict[str, Any]] = []

    def call_tool(self, name: str, **params: Any) -> Any:
        """Call a registered tool by name with given parameters.

        Args:
            name: Exact tool name.
            **params: Keyword arguments matching the tool's parameter schema.

        Returns:
            The tool's output on success.

        Raises:
            ValueError: If tool not found.
            RuntimeError: If tool run fails.
        """
        tool = self._registry.get(name)
        if tool is None:
            available = [t.metadata.name for t in self._registry.list_tools()]
            raise ValueError(
                f"Unknown tool: '{name}'. Available tools: {available}"
            )

        start = time.time()
        result = tool.run(**params)
        elapsed = int((time.time() - start) * 1000)

        self.call_log.append(
            {
                "tool": name,
                "params": params,
                "success": result.success,
                "elapsed_ms": elapsed,
            }
        )

        if not result.success:
            raise RuntimeError(f"Tool '{name}' failed: {result.error}")

        return result.output


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------

# Restricted builtins - no file/network/import access
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
    # Exception types (needed for try/except blocks)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "OverflowError": OverflowError,
    "NotImplementedError": NotImplementedError,
}

# Functions forbidden in the sandbox AST
_FORBIDDEN_CALLS = frozenset({
    "exec", "eval", "compile", "open", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals",
})


def _check_ast_safety(code: str) -> Optional[str]:
    """Check AST for forbidden constructs. Returns error string or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return (
                "Import statements are not allowed in Code Mode. "
                "Use call_tool() to interact with Vetinari tools."
            )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                return (
                    f"Forbidden function call: {func.id}(). "
                    f"Use call_tool() to interact with Vetinari tools."
                )
    return None


def _execute_in_sandbox(
    code: str,
    bridge: _ToolCallBridge,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Run user-supplied Python code in a restricted namespace that
    has ``call_tool()`` available.

    Args:
        code: Python source code.
        bridge: The tool-call bridge instance.
        timeout_seconds: Max execution time.

    Returns:
        Dict with ``success``, ``result``, ``stdout``, ``error``,
        ``call_log``, ``execution_time_ms``.
    """
    start = time.time()

    # AST safety check
    error = _check_ast_safety(code)
    if error is not None:
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "error": error,
            "call_log": bridge.call_log,
            "execution_time_ms": int((time.time() - start) * 1000),
        }

    # Capture printed output
    output_lines: List[str] = []

    def _print(*args: Any, **kwargs: Any) -> None:
        output_lines.append(" ".join(str(a) for a in args))

    # Build execution namespace
    namespace: Dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "call_tool": bridge.call_tool,
        "json": json,
        "print": _print,
    }

    try:
        compiled = compile(code, "<code_mode>", "exec")
    except SyntaxError as e:
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "error": f"SyntaxError: {e}",
            "call_log": bridge.call_log,
            "execution_time_ms": int((time.time() - start) * 1000),
        }

    # Execute with timeout via threading
    exec_error: Dict[str, Any] = {"exception": None}

    def _run():
        try:
            exec(compiled, namespace)  # noqa: S102 - intentional restricted sandbox
        except Exception as e:
            exec_error["exception"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return {
            "success": False,
            "result": None,
            "stdout": "\n".join(output_lines),
            "error": f"Execution timed out after {timeout_seconds}s",
            "call_log": bridge.call_log,
            "execution_time_ms": timeout_seconds * 1000,
        }

    elapsed_ms = int((time.time() - start) * 1000)

    if exec_error["exception"] is not None:
        exc = exec_error["exception"]
        return {
            "success": False,
            "result": None,
            "stdout": "\n".join(output_lines),
            "error": f"{type(exc).__name__}: {exc}",
            "call_log": bridge.call_log,
            "execution_time_ms": elapsed_ms,
        }

    # Try to extract a 'result' variable from the namespace
    final_result = namespace.get("result", None)
    if final_result is None and output_lines:
        final_result = "\n".join(output_lines)

    return {
        "success": True,
        "result": final_result,
        "stdout": "\n".join(output_lines),
        "error": None,
        "call_log": bridge.call_log,
        "execution_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Meta-Tool implementations
# ---------------------------------------------------------------------------


class SearchToolsTool(Tool):
    """
    Meta-tool: discover tools by keyword search.

    Returns ranked list of tool names, descriptions, and relevance
    scores using BM25 ranking.
    """

    def __init__(self, index: Optional[ToolSearchIndex] = None):
        self._index = index or ToolSearchIndex()
        metadata = ToolMetadata(
            name="search_tools",
            description=(
                "Search for available tools by keyword. Returns ranked "
                "results with tool names and descriptions. Use this to "
                "discover which tools can help with your task before "
                "calling get_schemas or execute_code."
            ),
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="query",
                    type=str,
                    description="Keywords to search for (e.g., 'code generation', 'file search')",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type=int,
                    description="Maximum number of results to return",
                    required=False,
                    default=5,
                ),
            ],
            required_permissions=[],
            allowed_modes=[
                ExecutionMode.EXECUTION,
                ExecutionMode.PLANNING,
                ExecutionMode.SANDBOX,
            ],
            tags=["meta", "search", "discovery", "code_mode"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)

        if not query:
            return ToolResult(
                success=False,
                output=None,
                error="Query parameter is required",
            )

        results = self._index.search(query, limit=limit)
        return ToolResult(
            success=True,
            output={
                "query": query,
                "results": results,
                "total": len(results),
            },
        )


class GetSchemasTool(Tool):
    """
    Meta-tool: retrieve full JSON schemas for tools by name.

    Returns parameter schemas so the LLM knows exactly how to call
    each tool via ``call_tool()`` in the ``execute_code`` sandbox.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or get_tool_registry()
        metadata = ToolMetadata(
            name="get_schemas",
            description=(
                "Get the full parameter schemas for one or more tools by name. "
                "Use after search_tools to understand exactly what parameters "
                "a tool accepts before writing code to call it."
            ),
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="tool_names",
                    type=list,
                    description="List of tool names to get schemas for",
                    required=True,
                ),
            ],
            required_permissions=[],
            allowed_modes=[
                ExecutionMode.EXECUTION,
                ExecutionMode.PLANNING,
                ExecutionMode.SANDBOX,
            ],
            tags=["meta", "schema", "introspection", "code_mode"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        tool_names = kwargs.get("tool_names", [])

        if not tool_names:
            return ToolResult(
                success=False,
                output=None,
                error="tool_names parameter is required (list of tool names)",
            )

        schemas: Dict[str, Any] = {}
        not_found: List[str] = []

        for name in tool_names:
            tool = self._registry.get(name)
            if tool is None:
                not_found.append(name)
            else:
                schemas[name] = _tool_to_json_schema(tool)

        return ToolResult(
            success=True,
            output={
                "schemas": schemas,
                "not_found": not_found,
            },
        )


class ExecuteCodeTool(Tool):
    """
    Meta-tool: run Python code with access to ``call_tool()``.

    The code runs in a restricted sandbox. It cannot import modules,
    access the filesystem, or make network calls. The only way to
    interact with the system is through the ``call_tool(name, **params)``
    bridge function.

    Example code the LLM might write::

        # Search the codebase for authentication
        results = call_tool("explorer",
            capability="grep_search",
            query="authentication",
            max_results=5,
        )
        print(json.dumps(results, indent=2))

        # Build on results
        result = call_tool("builder",
            action="implement",
            description="Add JWT auth middleware",
        )
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        timeout_seconds: int = 30,
    ):
        self._registry = registry or get_tool_registry()
        self._timeout = timeout_seconds
        metadata = ToolMetadata(
            name="execute_code",
            description=(
                "Run Python code that can call Vetinari tools via "
                "call_tool(name, **params). The code runs in a sandbox with "
                "no file/network/import access. Use json module for data "
                "manipulation. Set a 'result' variable to return output, "
                "or use print() statements."
            ),
            category=ToolCategory.CODE_EXECUTION,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="code",
                    type=str,
                    description=(
                        "Python code to run. Has access to call_tool(name, **params) "
                        "and the json module. Set a 'result' variable for the return value."
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    type=int,
                    description="Timeout in seconds (default: 30, max: 120)",
                    required=False,
                    default=30,
                ),
            ],
            required_permissions=[
                ToolPermission.PYTHON_EXECUTE,
            ],
            allowed_modes=[
                ExecutionMode.EXECUTION,
                ExecutionMode.SANDBOX,
            ],
            tags=["meta", "execution", "code", "sandbox", "code_mode"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "")
        timeout = min(kwargs.get("timeout", self._timeout), 120)

        if not code or not code.strip():
            return ToolResult(
                success=False,
                output=None,
                error="Code parameter is required",
            )

        bridge = _ToolCallBridge(self._registry)
        sandbox_result = _execute_in_sandbox(code, bridge, timeout_seconds=timeout)

        return ToolResult(
            success=sandbox_result["success"],
            output=sandbox_result,
            error=sandbox_result.get("error"),
            metadata={
                "tool_calls": len(bridge.call_log),
                "execution_time_ms": sandbox_result["execution_time_ms"],
            },
        )


# ---------------------------------------------------------------------------
# Code Mode Transform - replaces direct tool exposure
# ---------------------------------------------------------------------------


class CodeModeTransform:
    """
    Transforms the tool exposure strategy from direct to Code Mode.

    Instead of registering all N tools with the MCP registry, this
    transform registers only the three meta-tools (search_tools,
    get_schemas, execute_code) while keeping the underlying tools
    accessible via ``call_tool()`` in the sandbox.

    Usage::

        from vetinari.tools.code_mode import CodeModeTransform

        transform = CodeModeTransform()
        transform.apply(mcp_registry)

    After ``apply()``, the MCP registry contains only the three
    meta-tools. The LLM workflow becomes:

    1. ``search_tools(query="code search")`` -> discover relevant tools
    2. ``get_schemas(tool_names=["explorer"])`` -> get parameter details
    3. ``execute_code(code="result = call_tool('explorer', ...)")`` -> run
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: int = 30,
    ):
        self._tool_registry = tool_registry or get_tool_registry()
        self._timeout = timeout_seconds
        self._index = ToolSearchIndex(self._tool_registry)
        self._meta_tools: List[Tool] = []
        self._applied = False

    def get_meta_tools(self) -> List[Tool]:
        """Return the three Code Mode meta-tools (creating them if needed)."""
        if not self._meta_tools:
            self._meta_tools = [
                SearchToolsTool(index=self._index),
                GetSchemasTool(registry=self._tool_registry),
                ExecuteCodeTool(
                    registry=self._tool_registry,
                    timeout_seconds=self._timeout,
                ),
            ]
        return self._meta_tools

    def apply(self, mcp_registry: Any) -> None:
        """
        Apply Code Mode to an MCP tool registry.

        Registers the three meta-tools as MCP tools. Does NOT remove
        existing tools - the caller decides whether to clear first.

        Args:
            mcp_registry: An ``MCPToolRegistry`` instance.
        """
        from vetinari.mcp.tools import MCPTool, MCPToolParameter

        # Ensure search index is built
        self._index.refresh()

        for tool in self.get_meta_tools():
            mcp_params = []
            for p in tool.metadata.parameters:
                type_map = {
                    str: "string",
                    int: "number",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                mcp_params.append(
                    MCPToolParameter(
                        name=p.name,
                        type=type_map.get(p.type, "string"),
                        description=p.description,
                        required=p.required,
                        default=p.default,
                    )
                )

            # Create handler closure
            def _make_handler(t: Tool) -> Callable:
                def handler(**kwargs: Any) -> Dict[str, Any]:
                    result = t.run(**kwargs)
                    return result.to_dict()

                return handler

            mcp_registry.register(
                MCPTool(
                    name=tool.metadata.name,
                    description=tool.metadata.description,
                    parameters=mcp_params,
                    handler=_make_handler(tool),
                )
            )

        self._applied = True
        logger.info(
            "CodeModeTransform applied: %d meta-tools registered, "
            "%d underlying tools searchable",
            len(self._meta_tools),
            len(self._tool_registry.list_tools()),
        )

    @property
    def is_applied(self) -> bool:
        return self._applied


# ---------------------------------------------------------------------------
# Convenience - module-level helpers
# ---------------------------------------------------------------------------

_code_mode_transform: Optional[CodeModeTransform] = None


def get_code_mode_transform() -> CodeModeTransform:
    """Get or create the global Code Mode transform singleton."""
    global _code_mode_transform
    if _code_mode_transform is None:
        _code_mode_transform = CodeModeTransform()
    return _code_mode_transform


def enable_code_mode(mcp_registry: Any) -> CodeModeTransform:
    """
    Enable Code Mode on an MCP registry (convenience function).

    Args:
        mcp_registry: The MCP tool registry to apply Code Mode to.

    Returns:
        The CodeModeTransform instance.
    """
    transform = get_code_mode_transform()
    if not transform.is_applied:
        transform.apply(mcp_registry)
    return transform
