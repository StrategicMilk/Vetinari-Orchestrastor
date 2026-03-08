"""Tests for MCP server implementation (Task 25)."""

import json

import pytest

from vetinari.mcp.server import MCPServer, get_mcp_server
from vetinari.mcp.tools import MCPTool, MCPToolParameter, MCPToolRegistry
from vetinari.mcp.transport import StdioTransport, SSETransport


# ---------------------------------------------------------------------------
# MCPToolParameter
# ---------------------------------------------------------------------------

class TestMCPToolParameter:
    def test_required_param(self):
        p = MCPToolParameter("name", "string", "A name")
        assert p.name == "name"
        assert p.type == "string"
        assert p.required is True

    def test_optional_param(self):
        p = MCPToolParameter("count", "number", "A count", required=False, default=5)
        assert p.required is False
        assert p.default == 5


# ---------------------------------------------------------------------------
# MCPTool schema generation
# ---------------------------------------------------------------------------

class TestMCPToolSchema:
    def test_schema_basic(self):
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            parameters=[
                MCPToolParameter("arg1", "string", "First arg"),
                MCPToolParameter("arg2", "number", "Second arg", required=False),
            ],
        )
        schema = tool.to_schema()
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"
        assert "arg1" in schema["inputSchema"]["properties"]
        assert "arg2" in schema["inputSchema"]["properties"]
        assert "arg1" in schema["inputSchema"]["required"]
        assert "arg2" not in schema["inputSchema"]["required"]

    def test_schema_no_params(self):
        tool = MCPTool(name="empty", description="No params")
        schema = tool.to_schema()
        assert schema["inputSchema"]["properties"] == {}
        assert schema["inputSchema"]["required"] == []


# ---------------------------------------------------------------------------
# MCPToolRegistry register/list/invoke
# ---------------------------------------------------------------------------

class TestMCPToolRegistry:
    def test_register_and_get(self):
        reg = MCPToolRegistry()
        tool = MCPTool(name="t1", description="Tool 1")
        reg.register(tool)
        assert reg.get("t1") is tool
        assert reg.get("nonexistent") is None

    def test_list_tools(self):
        reg = MCPToolRegistry()
        reg.register(MCPTool(name="a", description="A"))
        reg.register(MCPTool(name="b", description="B"))
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"a", "b"}

    def test_invoke_success(self):
        def handler(x, y=0):
            return {"sum": x + y}

        reg = MCPToolRegistry()
        reg.register(MCPTool(name="add", description="Add", handler=handler))
        result = reg.invoke("add", {"x": 3, "y": 4})
        assert result == {"result": {"sum": 7}}

    def test_invoke_unknown_tool(self):
        reg = MCPToolRegistry()
        result = reg.invoke("missing", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_invoke_no_handler(self):
        reg = MCPToolRegistry()
        reg.register(MCPTool(name="stub", description="No handler"))
        result = reg.invoke("stub", {})
        assert "error" in result
        assert "no handler" in result["error"]

    def test_invoke_handler_exception(self):
        def bad_handler():
            raise ValueError("boom")

        reg = MCPToolRegistry()
        reg.register(MCPTool(name="bad", description="Fails", handler=bad_handler))
        result = reg.invoke("bad", {})
        assert "error" in result
        assert "boom" in result["error"]

    def test_register_defaults(self):
        reg = MCPToolRegistry()
        reg.register_defaults()
        tools = reg.list_tools()
        names = {t["name"] for t in tools}
        assert "vetinari_plan" in names
        assert "vetinari_search" in names
        assert "vetinari_execute" in names
        assert "vetinari_memory" in names
        assert "vetinari_benchmark" in names

    def test_default_plan_handler(self):
        reg = MCPToolRegistry()
        reg.register_defaults()
        result = reg.invoke("vetinari_plan", {"goal": "test goal"})
        assert "result" in result
        assert result["result"]["status"] == "planned"

    def test_default_execute_handler(self):
        reg = MCPToolRegistry()
        reg.register_defaults()
        result = reg.invoke("vetinari_execute", {"task": "do something"})
        assert "result" in result
        assert result["result"]["status"] == "queued"

    def test_default_search_handler_graceful(self):
        """Search handler returns gracefully when subsystem unavailable."""
        reg = MCPToolRegistry()
        reg.register_defaults()
        result = reg.invoke("vetinari_search", {"query": "test"})
        assert "result" in result

    def test_default_memory_handler_graceful(self):
        """Memory handler returns gracefully when subsystem unavailable."""
        reg = MCPToolRegistry()
        reg.register_defaults()
        result = reg.invoke("vetinari_memory", {"action": "recall", "content": "test"})
        assert "result" in result


# ---------------------------------------------------------------------------
# MCPServer creation
# ---------------------------------------------------------------------------

class TestMCPServerCreation:
    def test_create(self):
        server = MCPServer()
        assert server._initialized is False
        assert server.registry is not None

    def test_has_default_tools(self):
        server = MCPServer()
        tools = server.registry.list_tools()
        assert len(tools) >= 5


# ---------------------------------------------------------------------------
# Initialize handler
# ---------------------------------------------------------------------------

class TestMCPServerInitialize:
    def test_initialize(self):
        server = MCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert "protocolVersion" in result
        assert result["serverInfo"]["name"] == "vetinari"
        assert server._initialized is True


# ---------------------------------------------------------------------------
# Tools list
# ---------------------------------------------------------------------------

class TestMCPServerToolsList:
    def test_tools_list(self):
        server = MCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert resp["id"] == 2
        tools = resp["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) >= 5


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------

class TestMCPServerToolCall:
    def test_call_plan(self):
        server = MCPServer()
        resp = server.handle_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "vetinari_plan", "arguments": {"goal": "test"}},
        })
        assert resp["id"] == 3
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        data = json.loads(content[0]["text"])
        assert "result" in data


# ---------------------------------------------------------------------------
# Unknown tool error
# ---------------------------------------------------------------------------

class TestMCPServerUnknownTool:
    def test_unknown_tool(self):
        server = MCPServer()
        resp = server.handle_message({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        content = resp["result"]["content"]
        assert resp["result"]["isError"] is True
        data = json.loads(content[0]["text"])
        assert "error" in data


# ---------------------------------------------------------------------------
# Unknown method error
# ---------------------------------------------------------------------------

class TestMCPServerUnknownMethod:
    def test_unknown_method(self):
        server = MCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 5, "method": "unknown/method", "params": {}})
        assert "error" in resp
        assert resp["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

class TestMCPServerPing:
    def test_ping(self):
        server = MCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 6, "method": "ping", "params": {}})
        assert resp["id"] == 6
        assert resp["result"] == {}


# ---------------------------------------------------------------------------
# StdioTransport existence
# ---------------------------------------------------------------------------

class TestStdioTransport:
    def test_creation(self):
        server = MCPServer()
        transport = StdioTransport(server)
        assert transport._server is server
        assert hasattr(transport, "run")


# ---------------------------------------------------------------------------
# SSETransport blueprint creation
# ---------------------------------------------------------------------------

class TestSSETransport:
    def test_creation(self):
        server = MCPServer()
        transport = SSETransport(server)
        assert transport._server is server

    def test_create_flask_routes(self):
        """SSETransport can create a Flask blueprint (requires flask)."""
        pytest.importorskip("flask")
        server = MCPServer()
        transport = SSETransport(server)
        bp = transport.create_flask_routes()
        assert bp.name == "mcp"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

class TestGetMCPServer:
    def test_returns_instance(self):
        import vetinari.mcp.server as mod
        mod._mcp_server = None
        s = get_mcp_server()
        assert isinstance(s, MCPServer)

    def test_returns_same_instance(self):
        import vetinari.mcp.server as mod
        mod._mcp_server = None
        a = get_mcp_server()
        b = get_mcp_server()
        assert a is b
