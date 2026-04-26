"""Comprehensive tests for the Vetinari MCP (Model Context Protocol) module.

Covers MCPServer, MCPToolRegistry, MCPTool, tool handlers, MCPClient,
and StdioTransport.  All external subsystems are mocked --
no real API calls or subprocesses are created during the test run.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from vetinari.constants import MCP_PROCESS_TIMEOUT
from vetinari.exceptions import MCPError
from vetinari.mcp.client import MCPClient
from vetinari.mcp.config import MCPServerConfig, validate_mcp_launch_command
from vetinari.mcp.server import MCPServer, get_mcp_server
from vetinari.mcp.tools import MCPTool, MCPToolParameter, MCPToolRegistry
from vetinari.mcp.transport import StdioTransport
from vetinari.mcp.worker_bridge import WorkerMCPBridge

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def registry() -> MCPToolRegistry:
    """Return a fresh empty MCPToolRegistry."""
    return MCPToolRegistry()


@pytest.fixture
def registry_with_defaults() -> MCPToolRegistry:
    """Return a registry with the five default tools registered."""
    reg = MCPToolRegistry()
    reg.register_defaults()
    return reg


@pytest.fixture
def server() -> MCPServer:
    """Return a fresh MCPServer instance (does not share singleton state)."""
    return MCPServer()


@pytest.fixture
def simple_tool() -> MCPTool:
    """Return a minimal MCPTool with a working handler."""
    return MCPTool(
        name="echo_tool",
        description="Echoes the input back",
        parameters=[MCPToolParameter("text", "string", "Text to echo")],
        handler=lambda text: {"echo": text},
    )


# ===========================================================================
# TestMCPServer
# ===========================================================================


class TestMCPServer:
    """Tests for MCPServer.handle_message() and its internal handlers."""

    def test_protocol_version_constant(self, server: MCPServer) -> None:
        assert server.PROTOCOL_VERSION == "2024-11-05"

    def test_initialize_sets_initialized_flag(self, server: MCPServer) -> None:
        assert server._initialized is False
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
        server.handle_message(msg)
        assert server._initialized is True

    def test_initialize_returns_protocol_version(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
        response = server.handle_message(msg)
        assert response["result"]["protocolVersion"] == "2024-11-05"

    def test_initialize_returns_server_info(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
        response = server.handle_message(msg)
        info = response["result"]["serverInfo"]
        assert info["name"] == "vetinari"
        assert "version" in info

    def test_initialize_returns_capabilities(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
        response = server.handle_message(msg)
        assert "capabilities" in response["result"]

    def test_tools_list_requires_initialize(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}}
        response = server.handle_message(msg)
        assert response["error"]["code"] == -32002
        assert "not initialized" in response["error"]["message"]

    def test_tools_call_requires_initialize(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "vetinari_plan"}}
        response = server.handle_message(msg)
        assert response["error"]["code"] == -32002
        assert "not initialized" in response["error"]["message"]

    def test_tools_list_returns_tools_key(self, server: MCPServer) -> None:
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        msg = {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}}
        response = server.handle_message(msg)
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

    def test_tools_list_contains_default_tools(self, server: MCPServer) -> None:
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        msg = {"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}}
        response = server.handle_message(msg)
        names = [t["name"] for t in response["result"]["tools"]]
        assert "vetinari_plan" in names
        assert "vetinari_search" in names
        assert "vetinari_execute" in names
        assert "vetinari_memory" in names
        assert "vetinari_benchmark" in names

    def test_tools_call_dispatches_to_registry(self, server: MCPServer) -> None:
        """tools/call should proxy to the registry and wrap result in content block."""
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        server.registry.register(
            MCPTool(
                name="my_tool",
                description="test",
                handler=lambda: {"ok": True},
            )
        )
        msg = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "my_tool", "arguments": {}},
        }
        response = server.handle_message(msg)
        assert response["result"]["content"][0]["type"] == "text"

    def test_tools_call_error_result_sets_is_error(self, server: MCPServer) -> None:
        """When the invoked tool returns an error dict, isError must be True."""
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        msg = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        response = server.handle_message(msg)
        assert response["result"]["isError"] is True

    def test_ping_returns_empty_result(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 8, "method": "ping", "params": {}}
        response = server.handle_message(msg)
        assert response["result"] == {}

    def test_unknown_method_returns_minus_32601(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 9, "method": "no_such_method", "params": {}}
        response = server.handle_message(msg)
        assert response["error"]["code"] == -32601
        assert "no_such_method" in response["error"]["message"]

    def test_unknown_method_preserves_message_id(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 42, "method": "ghost", "params": {}}
        response = server.handle_message(msg)
        assert response["id"] == 42

    def test_handler_exception_returns_minus_32603(self, server: MCPServer) -> None:
        """A handler that raises must produce a -32603 Internal Error response.

        The error message must be a generic string — exception details must not
        be leaked to the caller (they are logged server-side via logger.exception).
        """

        def _bad_handler(_params):
            raise RuntimeError("boom")

        server._handle_ping = _bad_handler  # monkeypatch one handler
        msg = {"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}}
        response = server.handle_message(msg)
        assert response["error"]["code"] == -32603
        assert response["error"]["message"] == "Internal server error"
        assert "boom" not in response["error"]["message"]

    def test_get_mcp_server_returns_singleton(self) -> None:
        a = get_mcp_server()
        b = get_mcp_server()
        assert a is b

    def test_handle_message_includes_jsonrpc_version(self, server: MCPServer) -> None:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
        response = server.handle_message(msg)
        assert response["jsonrpc"] == "2.0"


# ===========================================================================
# TestMCPToolRegistry
# ===========================================================================


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry register/get/list/invoke and defaults."""

    def test_register_and_get(self, registry: MCPToolRegistry, simple_tool: MCPTool) -> None:
        registry.register(simple_tool)
        assert registry.get("echo_tool") is simple_tool

    def test_get_unknown_returns_none(self, registry: MCPToolRegistry) -> None:
        assert registry.get("phantom") is None

    def test_register_overwrites_existing(self, registry: MCPToolRegistry) -> None:
        tool_v1 = MCPTool(name="t", description="v1", handler=dict)
        tool_v2 = MCPTool(name="t", description="v2", handler=dict)
        registry.register(tool_v1)
        registry.register(tool_v2)
        assert registry.get("t").description == "v2"

    def test_list_tools_empty(self, registry: MCPToolRegistry) -> None:
        assert registry.list_tools() == []

    def test_list_tools_returns_schemas(self, registry: MCPToolRegistry, simple_tool: MCPTool) -> None:
        registry.register(simple_tool)
        schemas = registry.list_tools()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo_tool"

    def test_unregister_external_server_removes_only_matching_namespace(self, registry: MCPToolRegistry) -> None:
        registry.register_external_tool("alpha", MCPTool(name="read", description="alpha read"))
        registry.register_external_tool("alpha", MCPTool(name="write", description="alpha write"))
        registry.register_external_tool("beta", MCPTool(name="read", description="beta read"))

        removed = registry.unregister_external_server("alpha")

        assert removed == 2
        assert registry.get("mcp__alpha__read") is None
        assert registry.get("mcp__alpha__write") is None
        assert registry.get("mcp__beta__read") is not None

    def test_invoke_known_tool(self, registry: MCPToolRegistry, simple_tool: MCPTool) -> None:
        registry.register(simple_tool)
        result = registry.invoke("echo_tool", {"text": "hello"})
        assert result == {"result": {"echo": "hello"}}

    def test_invoke_unknown_tool_returns_error(self, registry: MCPToolRegistry) -> None:
        result = registry.invoke("ghost", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_invoke_tool_without_handler_returns_error(self, registry: MCPToolRegistry) -> None:
        no_handler = MCPTool(name="bare", description="no handler", handler=None)
        registry.register(no_handler)
        result = registry.invoke("bare", {})
        assert "error" in result
        assert "no handler" in result["error"]

    def test_invoke_handler_exception_returns_error(self, registry: MCPToolRegistry) -> None:
        """Handler exceptions must return a generic error message without leaking details."""

        def _exploding(**_kwargs):
            raise ValueError("deliberate failure")

        broken = MCPTool(name="broken", description="fails", handler=_exploding)
        registry.register(broken)
        result = registry.invoke("broken", {})
        assert "error" in result
        assert "deliberate failure" not in result["error"]
        assert "broken" in result["error"]

    def test_register_defaults_creates_five_tools(self, registry_with_defaults: MCPToolRegistry) -> None:
        schemas = registry_with_defaults.list_tools()
        assert len(schemas) == 5

    def test_register_defaults_tool_names(self, registry_with_defaults: MCPToolRegistry) -> None:
        names = {s["name"] for s in registry_with_defaults.list_tools()}
        expected = {
            "vetinari_plan",
            "vetinari_search",
            "vetinari_execute",
            "vetinari_memory",
            "vetinari_benchmark",
        }
        assert names == expected


# ===========================================================================
# TestMCPTool
# ===========================================================================


class TestMCPTool:
    """Tests for MCPTool.to_schema()."""

    def test_to_schema_name_and_description(self) -> None:
        tool = MCPTool(name="my_tool", description="A tool")
        schema = tool.to_schema()
        assert schema["name"] == "my_tool"
        assert schema["description"] == "A tool"

    def test_to_schema_input_schema_type(self) -> None:
        tool = MCPTool(name="t", description="d")
        schema = tool.to_schema()
        assert schema["inputSchema"]["type"] == "object"

    def test_to_schema_required_parameter_listed(self) -> None:
        param = MCPToolParameter("query", "string", "The query", required=True)
        tool = MCPTool(name="t", description="d", parameters=[param])
        schema = tool.to_schema()
        assert "query" in schema["inputSchema"]["required"]

    def test_to_schema_optional_parameter_not_in_required(self) -> None:
        param = MCPToolParameter("limit", "number", "Max results", required=False)
        tool = MCPTool(name="t", description="d", parameters=[param])
        schema = tool.to_schema()
        assert "limit" not in schema["inputSchema"]["required"]

    def test_to_schema_property_type_and_description(self) -> None:
        param = MCPToolParameter("text", "string", "Some text")
        tool = MCPTool(name="t", description="d", parameters=[param])
        schema = tool.to_schema()
        prop = schema["inputSchema"]["properties"]["text"]
        assert prop["type"] == "string"
        assert prop["description"] == "Some text"

    def test_to_schema_no_parameters(self) -> None:
        tool = MCPTool(name="t", description="d", parameters=[])
        schema = tool.to_schema()
        assert schema["inputSchema"]["properties"] == {}
        assert schema["inputSchema"]["required"] == []


# ===========================================================================
# TestToolHandlers
# ===========================================================================


class TestToolHandlers:
    """Tests for the private tool handler methods on MCPToolRegistry."""

    # ── vetinari_plan ────────────────────────────────────────────────────────

    def test_handle_plan_returns_plan_id_goal_steps(self) -> None:
        mock_node = MagicMock()
        mock_node.description = "step one"
        mock_node.depends_on = set()

        mock_graph = MagicMock()
        mock_graph.plan_id = "plan-abc"
        mock_graph.nodes = {"node1": mock_node}

        mock_orch = MagicMock()
        mock_orch.generate_plan_only.return_value = mock_graph

        reg = MCPToolRegistry()
        with patch("vetinari.orchestration.two_layer.get_two_layer_orchestrator", return_value=mock_orch):
            result = reg._handle_plan(goal="build a thing")

        assert result["plan_id"] == "plan-abc"
        assert result["goal"] == "build a thing"
        assert isinstance(result["steps"], list)
        assert result["steps"][0]["id"] == "node1"
        assert result["steps"][0]["description"] == "step one"

    def test_handle_plan_passes_constraints_when_context_given(self) -> None:
        mock_graph = MagicMock()
        mock_graph.plan_id = "p1"
        mock_graph.nodes = {}

        mock_orch = MagicMock()
        mock_orch.generate_plan_only.return_value = mock_graph

        reg = MCPToolRegistry()
        with patch("vetinari.orchestration.two_layer.get_two_layer_orchestrator", return_value=mock_orch):
            reg._handle_plan(goal="do x", context="extra info")

        call_kwargs = mock_orch.generate_plan_only.call_args
        assert call_kwargs.kwargs["constraints"] == {"context": "extra info"}

    def test_handle_plan_exception_returns_error_dict(self) -> None:
        reg = MCPToolRegistry()
        with patch(
            "vetinari.orchestration.two_layer.get_two_layer_orchestrator",
            side_effect=RuntimeError("orch down"),
        ):
            result = reg._handle_plan(goal="anything")

        assert "error" in result
        assert result["tool"] == "vetinari_plan"

    # ── vetinari_execute ─────────────────────────────────────────────────────

    def test_handle_execute_returns_orchestrator_result(self) -> None:
        expected = {"plan_id": "p2", "completed": 3, "failed": 0, "outputs": {}}
        mock_orch = MagicMock()
        mock_orch.execute_with_agent_graph.return_value = expected

        reg = MCPToolRegistry()
        with patch("vetinari.orchestration.two_layer.get_two_layer_orchestrator", return_value=mock_orch):
            result = reg._handle_execute(task="run the tests")

        assert result == expected
        mock_orch.execute_with_agent_graph.assert_called_once_with("run the tests")

    def test_handle_execute_exception_returns_error_dict(self) -> None:
        reg = MCPToolRegistry()
        with patch(
            "vetinari.orchestration.two_layer.get_two_layer_orchestrator",
            side_effect=Exception("network error"),
        ):
            result = reg._handle_execute(task="do something")

        assert "error" in result
        assert result["tool"] == "vetinari_execute"

    # ── vetinari_memory ──────────────────────────────────────────────────────

    def test_handle_memory_store_calls_remember(self) -> None:
        mock_store = MagicMock()
        mock_store.remember.return_value = 7

        mock_entry_cls = MagicMock(return_value=MagicMock())

        reg = MCPToolRegistry()
        with (
            patch("vetinari.memory.get_unified_memory_store", return_value=mock_store),
            patch("vetinari.memory.interfaces.MemoryEntry", mock_entry_cls),
        ):
            result = reg._handle_memory(action="store", content="important fact")

        assert result["status"] == "stored"
        assert result["id"] == 7
        mock_store.remember.assert_called_once()

    def test_handle_memory_recall_calls_search(self) -> None:
        mock_result = MagicMock()
        mock_result.content = "remembered text"
        mock_store = MagicMock()
        mock_store.search.return_value = [mock_result]

        reg = MCPToolRegistry()
        with (
            patch("vetinari.memory.get_unified_memory_store", return_value=mock_store),
            patch("vetinari.memory.interfaces.MemoryEntry", MagicMock()),
        ):
            result = reg._handle_memory(action="recall", content="what do I know")

        assert "entries" in result
        assert "remembered text" in result["entries"]
        mock_store.search.assert_called_once_with("what do I know", limit=5)

    def test_handle_memory_query_is_alias_for_recall(self) -> None:
        mock_store = MagicMock()
        mock_store.search.return_value = []

        reg = MCPToolRegistry()
        with (
            patch("vetinari.memory.get_unified_memory_store", return_value=mock_store),
            patch("vetinari.memory.interfaces.MemoryEntry", MagicMock()),
        ):
            result = reg._handle_memory(action="query", content="find this")

        assert "entries" in result

    def test_handle_memory_unknown_action_returns_error(self) -> None:
        mock_store = MagicMock()

        reg = MCPToolRegistry()
        with (
            patch("vetinari.memory.get_unified_memory_store", return_value=mock_store),
            patch("vetinari.memory.interfaces.MemoryEntry", MagicMock()),
        ):
            result = reg._handle_memory(action="delete_all", content="")

        assert "error" in result
        assert "Unknown action" in result["error"]
        assert result["tool"] == "vetinari_memory"

    def test_handle_memory_exception_returns_error_dict(self) -> None:
        reg = MCPToolRegistry()
        with patch(
            "vetinari.memory.get_unified_memory_store",
            side_effect=ImportError("no memory module"),
        ):
            result = reg._handle_memory(action="recall", content="anything")

        assert "error" in result
        assert result["tool"] == "vetinari_memory"

    # ── vetinari_search ──────────────────────────────────────────────────────

    def test_handle_search_returns_results_list(self) -> None:
        mock_adapter = MagicMock()
        mock_adapter.search.return_value = [{"file": "foo.py", "line": 1}]

        reg = MCPToolRegistry()
        with patch("vetinari.code_search.CocoIndexAdapter", return_value=mock_adapter):
            result = reg._handle_search(query="find me something", limit=5)

        assert "results" in result
        assert len(result["results"]) == 1
        mock_adapter.search.assert_called_once_with("find me something", limit=5)

    def test_handle_search_exception_returns_degraded_response(self) -> None:
        reg = MCPToolRegistry()
        with patch("vetinari.code_search.CocoIndexAdapter", side_effect=ImportError("no coco")):
            result = reg._handle_search(query="something")

        # Should degrade gracefully: return empty results, not raise
        assert "results" in result
        assert result["results"] == []
        assert "error" in result
        assert result["tool"] == "vetinari_search"

    # ── vetinari_benchmark ───────────────────────────────────────────────────

    def test_handle_benchmark_returns_summary_dict(self) -> None:
        mock_report = MagicMock()
        mock_report.summary_dict.return_value = {"passed": 10, "failed": 0}

        mock_runner = MagicMock()
        mock_runner.run_suite.return_value = mock_report

        reg = MCPToolRegistry()
        with patch("vetinari.benchmarks.get_default_runner", return_value=mock_runner):
            result = reg._handle_benchmark(suite="reasoning", limit=5)

        assert result == {"passed": 10, "failed": 0}
        mock_runner.run_suite.assert_called_once_with("reasoning", limit=5)

    def test_handle_benchmark_exception_returns_error_dict(self) -> None:
        reg = MCPToolRegistry()
        with patch(
            "vetinari.benchmarks.get_default_runner",
            side_effect=FileNotFoundError("suite not found"),
        ):
            result = reg._handle_benchmark(suite="missing_suite")

        assert "error" in result
        assert result["tool"] == "vetinari_benchmark"


# ===========================================================================
# TestMCPConfig
# ===========================================================================


class TestMCPConfig:
    """Tests for MCP server config loading and cache isolation."""

    def test_load_mcp_server_configs_caches_per_resolved_path(self, tmp_path: Path) -> None:
        from vetinari.mcp.config import load_mcp_server_configs, reset_mcp_config_cache

        reset_mcp_config_cache()
        first_path = tmp_path / "first.yaml"
        second_path = tmp_path / "second.yaml"
        first_path.write_text(
            "mcp_servers:\n"
            "  - name: first\n"
            "    command: cmd-first\n",
            encoding="utf-8",
        )
        second_path.write_text(
            "mcp_servers:\n"
            "  - name: second\n"
            "    command: cmd-second\n",
            encoding="utf-8",
        )

        try:
            first = load_mcp_server_configs(config_path=first_path)
            second = load_mcp_server_configs(config_path=second_path)
            first_again = load_mcp_server_configs(config_path=first_path)
        finally:
            reset_mcp_config_cache()

        assert [cfg.name for cfg in first] == ["first"]
        assert [cfg.name for cfg in second] == ["second"]
        assert first_again is first

    def test_enabled_unpinned_npx_server_is_skipped(self, tmp_path: Path) -> None:
        from vetinari.mcp.config import load_mcp_server_configs, reset_mcp_config_cache

        config_path = tmp_path / "mcp.yaml"
        config_path.write_text(
            "mcp_servers:\n"
            "  - name: moving\n"
            "    command: npx\n"
            "    args: ['-y', '@modelcontextprotocol/server-memory']\n"
            "    enabled: true\n",
            encoding="utf-8",
        )

        try:
            configs = load_mcp_server_configs(config_path=config_path)
        finally:
            reset_mcp_config_cache()

        assert configs == []

    def test_validate_mcp_launch_command_rejects_shell_wrappers(self) -> None:
        with pytest.raises(ValueError, match="directly"):
            validate_mcp_launch_command(["cmd", "/c", "server"])

    def test_validate_mcp_launch_command_rejects_unpinned_remote_runner(self) -> None:
        with pytest.raises(ValueError, match="unpinned"):
            validate_mcp_launch_command(["npx", "-y", "@modelcontextprotocol/server-memory"])


# ===========================================================================
# TestWorkerMCPBridge
# ===========================================================================


class TestWorkerMCPBridge:
    """Tests for the WorkerMCPBridge singleton and its public API.

    Each test resets the module-level singleton so tests are independent
    and do not inherit state from prior test runs.
    """

    @pytest.fixture(autouse=True)
    def reset_bridge(self) -> None:
        """Reset the WorkerMCPBridge singleton before and after each test.

        Calls reset_worker_mcp_bridge() so tests start with a clean slate
        and do not leak subprocess state between cases.
        """
        from vetinari.mcp.worker_bridge import reset_worker_mcp_bridge

        reset_worker_mcp_bridge()
        yield
        reset_worker_mcp_bridge()

    def test_get_available_mcp_tools_returns_list_when_no_servers_configured(self) -> None:
        """Bridge returns an empty list when no external MCP servers are configured."""
        with patch("vetinari.mcp.worker_bridge.load_mcp_server_configs", return_value=[]):
            from vetinari.mcp.worker_bridge import get_available_mcp_tools

            tools = get_available_mcp_tools()

        assert isinstance(tools, list)
        assert tools == []

    def test_invoke_mcp_tool_rejects_non_namespaced_name(self) -> None:
        """Invoking a tool without the mcp__ prefix returns an error dict."""
        with patch("vetinari.mcp.worker_bridge.load_mcp_server_configs", return_value=[]):
            from vetinari.mcp.worker_bridge import invoke_mcp_tool

            result = invoke_mcp_tool("bare_tool_name", {})

        assert "error" in result
        assert "bare_tool_name" in result["error"]

    def test_reset_worker_mcp_bridge_clears_singleton(self) -> None:
        """After reset, the next call to get_worker_mcp_bridge creates a fresh instance."""
        from vetinari.mcp import worker_bridge

        with patch("vetinari.mcp.worker_bridge.load_mcp_server_configs", return_value=[]):
            bridge_first = worker_bridge.get_worker_mcp_bridge()

        worker_bridge.reset_worker_mcp_bridge()

        with patch("vetinari.mcp.worker_bridge.load_mcp_server_configs", return_value=[]):
            bridge_second = worker_bridge.get_worker_mcp_bridge()

        assert bridge_first is not bridge_second

    def test_shutdown_retracts_tools_for_stopped_external_clients(self) -> None:
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        registry = MCPToolRegistry()
        registry.register_external_tool("fs", MCPTool(name="read_file", description="Read file"))
        bridge = WorkerMCPBridge(registry=registry)
        bridge._active_clients["fs"] = MagicMock()

        bridge.shutdown()

        assert registry.get("mcp__fs__read_file") is None

    def test_shutdown_retracts_tools_even_when_client_stop_fails(self) -> None:
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        registry = MCPToolRegistry()
        registry.register_external_tool("fs", MCPTool(name="read_file", description="Read file"))
        client = MagicMock()
        client.stop.side_effect = RuntimeError("already gone")
        bridge = WorkerMCPBridge(registry=registry)
        bridge._active_clients["fs"] = client

        bridge.shutdown()

        assert registry.get("mcp__fs__read_file") is None


# ===========================================================================
# TestMCPClient
# ===========================================================================


@pytest.fixture
def make_client_with_process():
    """Return a factory that creates a started MCPClient with a mocked process.

    The factory accepts a ``response_payload`` dict and wires up mock stdin/stdout
    so that the client's ``_send_request`` reads back the payload as a JSON line.
    """

    def _factory(response_payload: dict[str, Any]) -> MCPClient:
        raw = json.dumps(response_payload).encode() + b"\n"

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline.return_value = raw

        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        return client

    return _factory


class TestMCPClient:
    """Tests for MCPClient lifecycle, JSON-RPC, and MCP protocol methods."""

    # ── Constructor ──────────────────────────────────────────────────────────

    def test_constructor_stores_command(self) -> None:
        client = MCPClient(command=["echo", "hello"])
        assert client._command == ["echo", "hello"]

    def test_constructor_stores_env(self) -> None:
        env = {"FOO": "bar"}
        client = MCPClient(command=["echo"], env=env)
        assert client._env == env

    def test_constructor_process_starts_as_none(self) -> None:
        client = MCPClient(command=["echo"])
        assert client._process is None

    def test_constructor_scrubs_parent_secret_env_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "secret")
        monkeypatch.setenv("PATH", "safe-path")
        client = MCPClient(command=["echo"])
        assert client._env is not None
        assert client._env.get("PATH") == "safe-path"
        assert "OPENAI_API_KEY" not in client._env

    # ── start / stop ────────────────────────────────────────────────────────

    def test_start_launches_subprocess(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            client = MCPClient(command=["my_server"])
            client.start()
        mock_popen.assert_called_once()
        assert client._process is mock_proc

    def test_start_passes_scrubbed_env_to_popen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "secret")
        monkeypatch.setenv("PATH", "safe-path")
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            client = MCPClient(command=["my_server"])
            client.start()
        env = mock_popen.call_args.kwargs["env"]
        assert env.get("PATH") == "safe-path"
        assert "HF_TOKEN" not in env

    def test_start_raises_mcp_error_on_os_error(self) -> None:
        with patch("subprocess.Popen", side_effect=OSError("file not found")):
            client = MCPClient(command=["bad_cmd"])
            with pytest.raises(MCPError, match="Failed to start"):
                client.start()

    def test_start_rejects_unsafe_launch_command_before_popen(self) -> None:
        with patch("subprocess.Popen") as mock_popen:
            client = MCPClient(command=["npx", "-y", "@modelcontextprotocol/server-memory"])
            with pytest.raises(MCPError, match="Unsafe MCP server command"):
                client.start()

        mock_popen.assert_not_called()

    def test_stop_terminates_process(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        client = MCPClient(command=["srv"])
        client._process = mock_proc
        client.stop()
        mock_proc.terminate.assert_called_once()
        assert client._process is None

    def test_stop_when_not_started_is_noop(self) -> None:
        client = MCPClient(command=["srv"])
        client.stop()  # should not raise
        assert client._process is None  # process must still be None

    def test_stop_kills_on_timeout(self) -> None:
        import subprocess as _sp

        mock_proc = MagicMock()
        mock_proc.pid = 5678
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.closed = False
        # First wait (with timeout=5) raises; second wait (after kill) succeeds
        mock_proc.wait.side_effect = [_sp.TimeoutExpired(cmd="srv", timeout=5), None]

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        client.stop()

        mock_proc.stdin.close.assert_called_once()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_args_list[0].kwargs["timeout"] == MCP_PROCESS_TIMEOUT
        assert mock_proc.wait.call_args_list[1].args == ()
        assert client._process is None

    # ── Context manager ──────────────────────────────────────────────────────

    def test_context_manager_calls_start_and_stop(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 111
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.closed = False
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            client = MCPClient(command=["srv"])
            with client as active_client:
                assert active_client is client
                assert client._process is mock_proc
        mock_popen.assert_called_once()
        mock_proc.terminate.assert_called_once()
        mock_proc.stdin.close.assert_called_once()
        assert client._process is None

    # ── _require_started ─────────────────────────────────────────────────────

    def test_require_started_raises_when_no_process(self) -> None:
        client = MCPClient(command=["srv"])
        with pytest.raises(MCPError, match="not started"):
            client._require_started()

    def test_require_started_returns_process_when_running(self) -> None:
        mock_proc = MagicMock()
        client = MCPClient(command=["srv"])
        client._process = mock_proc
        assert client._require_started() is mock_proc

    # ── _send_request ────────────────────────────────────────────────────────

    def test_send_request_writes_json_rpc_to_stdin(self, make_client_with_process) -> None:
        response = {"jsonrpc": "2.0", "id": 0, "result": {"ok": True}}
        client = make_client_with_process(response)
        client._send_request("ping")
        written = client._process.stdin.write.call_args[0][0]
        parsed = json.loads(written.decode())
        assert parsed["method"] == "ping"
        assert parsed["jsonrpc"] == "2.0"

    def test_send_request_returns_result_field(self, make_client_with_process) -> None:
        response = {"jsonrpc": "2.0", "id": 0, "result": {"hello": "world"}}
        client = make_client_with_process(response)
        result = client._send_request("any_method")
        assert result == {"hello": "world"}

    def test_send_request_raises_mcp_error_on_error_response(self, make_client_with_process) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 0,
            "error": {"code": -32601, "message": "Method not found"},
        }
        client = make_client_with_process(response)
        with pytest.raises(MCPError, match="Method not found"):
            client._send_request("ghost_method")

    def test_send_request_raises_on_empty_stdout(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = b""  # empty = server closed

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        with pytest.raises(MCPError, match="closed stdout"):
            client._send_request("ping")

    def test_send_request_raises_on_invalid_json(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = b"not json\n"

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        with pytest.raises(MCPError, match="Invalid JSON"):
            client._send_request("ping")

    def test_send_request_increments_id(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            (json.dumps({"jsonrpc": "2.0", "id": 0, "result": {}}) + "\n").encode(),
            (json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n").encode(),
        ]

        # Capture written bytes to inspect IDs
        written_payloads: list[dict] = []

        def _write(data):
            written_payloads.append(json.loads(data.decode()))

        mock_proc.stdin.write.side_effect = _write

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        client._send_request("m1")
        client._send_request("m2")

        ids = [p["id"] for p in written_payloads]
        assert ids == [0, 1]

    def test_send_request_skips_notifications_until_matching_response(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            (json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"pct": 50}}) + "\n").encode(),
            (json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"ok": True}}) + "\n").encode(),
        ]

        client = MCPClient(command=["srv"])
        client._process = mock_proc

        assert client._send_request("work") == {"ok": True}

    def test_send_request_skips_mismatched_ids_until_matching_response(self) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            (json.dumps({"jsonrpc": "2.0", "id": 99, "result": {"wrong": True}}) + "\n").encode(),
            (json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"ok": True}}) + "\n").encode(),
        ]

        client = MCPClient(command=["srv"])
        client._process = mock_proc

        assert client._send_request("work") == {"ok": True}

    # ── initialize ───────────────────────────────────────────────────────────

    def test_initialize_sends_correct_params(self) -> None:
        init_response = {
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "0.1"},
            },
        }
        # initialize sends a request then a notification (no response read for notification)
        mock_proc = MagicMock()
        mock_proc.pid = 77
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = (json.dumps(init_response) + "\n").encode()

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        result = client.initialize()

        assert result["protocolVersion"] == "2024-11-05"

    def test_initialize_stores_capabilities(self) -> None:
        caps = {"tools": {"listChanged": True}}
        init_response = {
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": caps,
                "serverInfo": {"name": "srv", "version": "1.0"},
            },
        }
        mock_proc = MagicMock()
        mock_proc.pid = 88
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = (json.dumps(init_response) + "\n").encode()

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        client.initialize()

        assert client._capabilities == caps

    # ── list_tools ───────────────────────────────────────────────────────────

    def test_list_tools_returns_tools_array(self, make_client_with_process) -> None:
        tools_payload = [{"name": "read_file"}, {"name": "write_file"}]
        response = {"jsonrpc": "2.0", "id": 0, "result": {"tools": tools_payload}}
        client = make_client_with_process(response)
        tools = client.list_tools()
        assert tools == tools_payload

    def test_list_tools_returns_empty_list_when_no_tools_key(self, make_client_with_process) -> None:
        response = {"jsonrpc": "2.0", "id": 0, "result": {}}
        client = make_client_with_process(response)
        tools = client.list_tools()
        assert tools == []

    # ── call_tool ────────────────────────────────────────────────────────────

    def test_call_tool_returns_text_from_first_content_block(self, make_client_with_process) -> None:
        result_payload = {"content": [{"type": "text", "text": "the answer"}]}
        response = {"jsonrpc": "2.0", "id": 0, "result": result_payload}
        client = make_client_with_process(response)
        text = client.call_tool("my_tool", {"key": "val"})
        assert text == "the answer"

    def test_call_tool_sends_correct_params(self) -> None:
        result_payload = {"content": [{"type": "text", "text": "ok"}]}
        response = {"jsonrpc": "2.0", "id": 0, "result": result_payload}

        written_data: list[dict] = []

        mock_proc = MagicMock()
        mock_proc.pid = 55
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write.side_effect = lambda b: written_data.append(json.loads(b.decode()))
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = (json.dumps(response) + "\n").encode()

        client = MCPClient(command=["srv"])
        client._process = mock_proc
        client.call_tool("search", {"query": "hello"})

        payload = written_data[0]
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "search"
        assert payload["params"]["arguments"] == {"query": "hello"}

    def test_call_tool_raises_mcp_error_when_no_content_blocks(self, make_client_with_process) -> None:
        response = {"jsonrpc": "2.0", "id": 0, "result": {"content": []}}
        client = make_client_with_process(response)
        with pytest.raises(MCPError, match="no content blocks"):
            client.call_tool("empty_tool")

    def test_call_tool_raises_on_mcp_is_error_result(self, make_client_with_process) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "{\"error\":\"backend failed\"}"}],
            },
        }
        client = make_client_with_process(response)
        with pytest.raises(MCPError, match="backend failed"):
            client.call_tool("broken_tool")

    # ── ping ─────────────────────────────────────────────────────────────────

    def test_ping_returns_true_on_success(self, make_client_with_process) -> None:
        response = {"jsonrpc": "2.0", "id": 0, "result": {}}
        client = make_client_with_process(response)
        assert client.ping() is True

    def test_ping_returns_false_on_mcp_error(self, make_client_with_process) -> None:
        response = {"jsonrpc": "2.0", "id": 0, "error": {"code": -32603, "message": "Internal error"}}
        client = make_client_with_process(response)
        assert client.ping() is False

    # ── stopped client raises ────────────────────────────────────────────────

    def test_list_tools_raises_when_not_started(self) -> None:
        client = MCPClient(command=["srv"])
        with pytest.raises(MCPError):
            client.list_tools()

    def test_call_tool_raises_when_not_started(self) -> None:
        client = MCPClient(command=["srv"])
        with pytest.raises(MCPError):
            client.call_tool("any")


# ===========================================================================
# TestTransports
# ===========================================================================


class TestWorkerMCPBridgeEnvironment:
    def test_empty_config_env_does_not_become_parent_inheritance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        monkeypatch.setenv("PATH", "safe-path")
        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
                captured["command"] = command
                captured["env"] = env

            def start(self) -> None:
                return None

            def initialize(self) -> None:
                return None

            def stop(self) -> None:
                return None

        registry = MagicMock()
        registry.register_external_server.return_value = []
        cfg = MCPServerConfig(name="local", command="tool", args=[], env={})
        with patch("vetinari.mcp.client.MCPClient", FakeClient):
            WorkerMCPBridge(registry=registry).load_servers([cfg])

        assert captured["env"] is not None
        assert captured["env"].get("PATH") == "safe-path"
        assert "ANTHROPIC_API_KEY" not in captured["env"]


class TestTransports:
    """Tests for StdioTransport."""

    # ── StdioTransport ────────────────────────────────────────────────────────

    def test_stdio_transport_routes_message_to_server(self) -> None:
        mock_server = MagicMock()
        mock_server.handle_message.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {},
        }

        message = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
        line = json.dumps(message) + "\n"

        fake_stdin = io.StringIO(line)
        fake_stdout = io.StringIO()

        transport = StdioTransport(mock_server)
        with patch("sys.stdin", fake_stdin), patch("sys.stdout", fake_stdout):
            transport.run()

        mock_server.handle_message.assert_called_once_with(message)
        output = fake_stdout.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["id"] == 1

    def test_stdio_transport_skips_empty_lines(self) -> None:
        mock_server = MagicMock()
        mock_server.handle_message.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}

        message = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        fake_stdin = io.StringIO("\n\n" + json.dumps(message) + "\n")
        fake_stdout = io.StringIO()

        transport = StdioTransport(mock_server)
        with patch("sys.stdin", fake_stdin), patch("sys.stdout", fake_stdout):
            transport.run()

        mock_server.handle_message.assert_called_once_with(message)
        assert json.loads(fake_stdout.getvalue().strip()) == {"jsonrpc": "2.0", "id": 1, "result": {}}

    def test_stdio_transport_returns_parse_error_on_invalid_json(self) -> None:
        mock_server = MagicMock()
        fake_stdin = io.StringIO("this is not json\n")
        fake_stdout = io.StringIO()

        transport = StdioTransport(mock_server)
        with patch("sys.stdin", fake_stdin), patch("sys.stdout", fake_stdout):
            transport.run()

        mock_server.handle_message.assert_not_called()
        output = fake_stdout.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["error"]["code"] == -32700

    def test_stdio_transport_suppresses_notifications_no_id(self) -> None:
        """Responses without an id (notifications) should NOT be written to stdout."""
        mock_server = MagicMock()
        # Return a response with id=None to simulate a notification acknowledgement
        mock_server.handle_message.return_value = {
            "jsonrpc": "2.0",
            "id": None,
            "result": {},
        }

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        fake_stdin = io.StringIO(json.dumps(notif) + "\n")
        fake_stdout = io.StringIO()

        transport = StdioTransport(mock_server)
        with patch("sys.stdin", fake_stdin), patch("sys.stdout", fake_stdout):
            transport.run()

        assert fake_stdout.getvalue() == ""
