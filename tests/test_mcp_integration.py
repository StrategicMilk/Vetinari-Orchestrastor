"""Integration tests for the MCP subsystem.

Tests cover:
- MCPToolRegistry.register_external_tool() namespacing
- MCPToolRegistry.register_external_server() tool discovery
- MCPServer.handle_message() with valid JSON-RPC objects
- POST /mcp/message error responses for non-object JSON bodies
- POST /mcp/message initialized session preservation
- MCPClient bounded readline timeout (raises MCPError within timeout window)
- MCPClient interleaved-notification skipping (notification returns {} not RuntimeError)
- MCP config loading from YAML
- WorkerMCPBridge.get_available_mcp_tools() returns tool schemas
"""

from __future__ import annotations

import json
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(name: str = "test_tool", description: str = "A test tool") -> Any:
    """Build a minimal MCPTool for use in tests."""
    from vetinari.mcp.tools import MCPTool, MCPToolParameter

    return MCPTool(
        name=name,
        description=description,
        parameters=[
            MCPToolParameter("input", "string", "Input value", required=True),
        ],
        handler=lambda input: f"echo:{input}",
    )


# ---------------------------------------------------------------------------
# MCPToolRegistry — external tool namespacing
# ---------------------------------------------------------------------------


class TestMCPToolRegistryExternalTool:
    """Tests for register_external_tool() namespacing behaviour."""

    def test_register_external_tool_creates_namespaced_name(self) -> None:
        """register_external_tool prepends mcp__{server}__ to the tool name."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        tool = _make_mcp_tool("read_file")
        registry.register_external_tool("filesystem", tool)

        keys = registry.list_keys()
        assert "mcp__filesystem__read_file" in keys

    def test_register_external_tool_does_not_mutate_original(self) -> None:
        """The original MCPTool object's name must not be changed."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        tool = _make_mcp_tool("original_name")
        registry.register_external_tool("server", tool)

        assert tool.name == "original_name", "register_external_tool must not mutate the original tool"

    def test_namespaced_tool_is_invokable(self) -> None:
        """A tool registered via register_external_tool can be invoked by namespaced name."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        registry.register_external_tool("fs", _make_mcp_tool("echo"))

        result = registry.invoke("mcp__fs__echo", {"input": "hello"})

        assert "result" in result, f"Expected result key; got: {result}"
        assert result["result"] == "echo:hello"

    def test_namespaced_tool_schema_has_correct_name(self) -> None:
        """list_tools() returns the namespaced name in each schema dict."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        registry.register_external_tool("myserver", _make_mcp_tool("greet"))

        schemas = registry.list_tools()
        names = [s["name"] for s in schemas]
        assert "mcp__myserver__greet" in names

    def test_multiple_servers_no_collision(self) -> None:
        """Tools from different servers with the same base name do not collide."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        registry.register_external_tool("server_a", _make_mcp_tool("search"))
        registry.register_external_tool("server_b", _make_mcp_tool("search"))

        assert len(registry) == 2
        assert "mcp__server_a__search" in registry.list_keys()
        assert "mcp__server_b__search" in registry.list_keys()

    def test_register_external_server_uses_client_list_tools(self) -> None:
        """register_external_server calls client.list_tools() to discover tools."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        mock_client = MagicMock()
        mock_client.list_tools.return_value = [
            {
                "name": "write_file",
                "description": "Write a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
            }
        ]
        mock_client.call_tool.return_value = "OK"

        registered = registry.register_external_server("myfs", mock_client)

        assert registered == ["mcp__myfs__write_file"]
        assert "mcp__myfs__write_file" in registry.list_keys()

    def test_register_external_server_invokes_via_client(self) -> None:
        """Invoking a tool discovered via register_external_server calls client.call_tool."""
        from vetinari.mcp.tools import MCPToolRegistry

        registry = MCPToolRegistry()
        mock_client = MagicMock()
        mock_client.list_tools.return_value = [
            {
                "name": "ping",
                "description": "Ping the server",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            }
        ]
        mock_client.call_tool.return_value = "pong"

        registry.register_external_server("remote", mock_client)
        result = registry.invoke("mcp__remote__ping", {})

        mock_client.call_tool.assert_called_once_with("ping", {})
        assert result == {"result": {"text": "pong"}}


# ---------------------------------------------------------------------------
# MCPServer — handle_message
# ---------------------------------------------------------------------------


class TestMCPServerHandleMessage:
    """Tests for MCPServer.handle_message() dispatching."""

    @pytest.fixture
    def server(self) -> Any:
        """Fresh MCPServer with default tools (no singleton pollution)."""
        from vetinari.mcp.server import MCPServer

        return MCPServer()

    def test_handle_initialize_returns_protocol_version(self, server: Any) -> None:
        """initialize returns the expected protocolVersion."""
        response = server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })

        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert response["result"]["serverInfo"]["name"] == "vetinari"

    def test_handle_tools_list_returns_tools(self, server: Any) -> None:
        """tools/list returns a list with at least the 5 default tools."""
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        response = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

        tools = response["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) >= 5
        tool_names = [t["name"] for t in tools]
        assert "vetinari_plan" in tool_names
        assert "vetinari_search" in tool_names

    def test_handle_ping_returns_empty_result(self, server: Any) -> None:
        """ping returns an empty result dict."""
        response = server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}})

        assert response["result"] == {}

    def test_unknown_method_returns_error(self, server: Any) -> None:
        """An unrecognized method returns a JSON-RPC -32601 error."""
        response = server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}})

        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_response_preserves_request_id(self, server: Any) -> None:
        """The response id always matches the request id."""
        response = server.handle_message({"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}})

        assert response["id"] == 99


# ---------------------------------------------------------------------------
# POST /mcp/message — HTTP error handling via Litestar test client
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_app() -> Any:
    """Minimal Litestar app with only the MCP transport routes registered.

    Patches the admin_guard to always pass so tests focus on body validation,
    not authentication.
    """
    from litestar import Litestar

    def _pass_guard(connection: Any, _handler: Any = None, **kwargs: Any) -> None:
        return None

    with patch("vetinari.web.litestar_guards.admin_guard", _pass_guard):
        from vetinari.web.litestar_mcp_transport import create_mcp_transport_handlers

        handlers = create_mcp_transport_handlers()

    return Litestar(route_handlers=handlers)


class TestMCPMessageEndpointErrors:
    """POST /mcp/message error-handling contract tests."""

    def test_non_object_string_body_returns_400(self, mcp_app: Any) -> None:
        """A JSON string body (not an object) must return HTTP 400."""
        from litestar.testing import TestClient

        with TestClient(app=mcp_app) as client:
            response = client.post(
                "/mcp/message",
                content=json.dumps("just a string"),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "object" in body["error"].lower()

    def test_non_object_array_body_returns_400(self, mcp_app: Any) -> None:
        """A JSON array body must return HTTP 400."""
        from litestar.testing import TestClient

        with TestClient(app=mcp_app) as client:
            response = client.post(
                "/mcp/message",
                content=json.dumps([1, 2, 3]),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400

    def test_non_object_number_body_returns_400(self, mcp_app: Any) -> None:
        """A JSON number body must return HTTP 400."""
        from litestar.testing import TestClient

        with TestClient(app=mcp_app) as client:
            response = client.post(
                "/mcp/message",
                content=json.dumps(42),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400

    def test_malformed_json_returns_400(self, mcp_app: Any) -> None:
        """A body that is not valid JSON must return HTTP 400."""
        from litestar.testing import TestClient

        with TestClient(app=mcp_app) as client:
            response = client.post(
                "/mcp/message",
                content=b"{not valid json{{",
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400
        body = response.json()
        assert "Invalid JSON" in body.get("error", "") or "invalid" in body.get("error", "").lower()

    def test_valid_json_rpc_object_returns_200(self, mcp_app: Any) -> None:
        """A valid JSON-RPC object must be dispatched and return 200."""
        from litestar.testing import TestClient

        with TestClient(app=mcp_app) as client:
            response = client.post(
                "/mcp/message",
                content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == {}
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 1

    def test_empty_body_returns_400(self, mcp_app: Any) -> None:
        """An empty request body must return HTTP 400."""
        from litestar.testing import TestClient

        with TestClient(app=mcp_app) as client:
            response = client.post(
                "/mcp/message",
                content=b"",
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# HTTP MCP initialized session preservation
# ---------------------------------------------------------------------------


class TestMCPHTTPInitializedSession:
    """Prove that HTTP MCP requests preserve initialized protocol state.

    The HTTP transport must share a server session across sequential frames so
    an ``initialize`` request is visible to the next JSON-RPC request.
    """

    def test_second_request_uses_initialized_session(self, mcp_app: Any) -> None:
        """A later tools/list frame must observe the initialize frame.

        The first request sends an ``initialize`` call.  The second request
        immediately sends a ``tools/list`` call without repeating initialize.
        """
        from litestar.testing import TestClient

        import vetinari.mcp.server as server_mod

        server_mod._mcp_server = None

        init_payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }
        )
        tools_list_payload = json.dumps(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )

        try:
            with TestClient(app=mcp_app) as client:
                first = client.post(
                    "/mcp/message",
                    content=init_payload,
                    headers={"Content-Type": "application/json"},
                )
                second = client.post(
                    "/mcp/message",
                    content=tools_list_payload,
                    headers={"Content-Type": "application/json"},
                )
        finally:
            server_mod._mcp_server = None

        assert first.status_code == 200
        first_body = first.json()
        assert first_body["result"]["protocolVersion"] == "2024-11-05", f"initialize returned error: {first_body}"
        assert first_body["result"]["serverInfo"]["name"] == "vetinari", f"Expected server name in initialize result: {first_body}"

        assert second.status_code == 200
        second_body = second.json()
        assert isinstance(second_body["result"]["tools"], list), (
            f"Expected tools/list result after initialize, got: {second_body}"
        )
        assert "tools" in second_body["result"]


# ---------------------------------------------------------------------------
# MCPClient bounded timeout (Defect 4 proof)
# ---------------------------------------------------------------------------


class TestMCPClientBoundedTimeout:
    """Prove that MCPClient._send_request raises MCPError when the server
    does not respond within the timeout window.

    Before the Defect 4 fix, the blocking ``readline()`` path could keep the
    caller stuck indefinitely.  The fix uses a bounded daemon reader and stops
    the subprocess when the read deadline expires.
    """

    def test_readline_timeout_raises_mcp_error(self) -> None:
        """A stalled readline must raise MCPError within the timeout window.

        We patch ``stdout.readline`` to block indefinitely (gated by an event
        that is never set) and set the module-level timeout to a very small
        value so the test completes quickly.
        """
        from vetinari.exceptions import MCPError
        from vetinari.mcp.client import MCPClient

        # Gate that keeps the mock readline blocked until explicitly released.
        # In this test it is never set so readline blocks forever until the
        # bounded reader timeout fires.
        _block = threading.Event()

        def _blocking_readline() -> bytes:
            _block.wait()  # blocks until the event is set
            return b""

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.flush = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = _blocking_readline

        client = MCPClient(command=["echo", "dummy"])
        client._process = mock_process

        # Shrink the timeout to 0.1 s so the test completes quickly.
        with patch("vetinari.mcp.client._READLINE_TIMEOUT", 0.1):
            with pytest.raises(MCPError, match="did not respond within"):
                client._send_request("ping", {})

        # Unblock the daemon thread so it can exit cleanly after the test.
        _block.set()


# ---------------------------------------------------------------------------
# MCPClient interleaved-notification skipping (Defect 5 proof)
# ---------------------------------------------------------------------------


class TestMCPClientNotificationSkipping:
    """Prove that MCPClient._send_request loops past JSON-RPC notifications.

    JSON-RPC 2.0 notifications have a ``method`` key but no ``id``.  Before the
    fix, such a message caused the code to return ``{}`` immediately, silently
    discarding the real response.  The fix loops on notification frames and
    returns the result from the next matching response frame.
    """

    def test_notification_skipped_and_real_response_returned(self) -> None:
        """Notification (method, no id) is skipped; the following response is returned."""
        from vetinari.mcp.client import MCPClient

        # First readline returns a notification; second returns the real response.
        notification = json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        ).encode("utf-8") + b"\n"
        # _next_id starts at 0, so request_id for the first call is 0.
        real_response = json.dumps(
            {"jsonrpc": "2.0", "id": 0, "result": {}}
        ).encode("utf-8") + b"\n"

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.flush = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = MagicMock(side_effect=[notification, real_response])

        client = MCPClient(command=["echo", "dummy"])
        client._process = mock_process

        result = client._send_request("ping", {})

        # The notification is skipped; the real response result dict is returned.
        assert result == {}, f"Expected real response result dict, got: {result}"
        # readline must have been called twice: once for the notification, once for
        # the real response.
        assert mock_process.stdout.readline.call_count == 2

    def test_notification_does_not_raise_runtime_error(self) -> None:
        """The pre-fix behaviour (RuntimeError on ID mismatch) must NOT occur."""
        from vetinari.mcp.client import MCPClient

        notification = json.dumps(
            {"jsonrpc": "2.0", "method": "$/progress", "params": {"token": "tok1"}}
        ).encode("utf-8") + b"\n"
        # _next_id starts at 0, so request_id for the first call is 0.
        real_response = json.dumps(
            {"jsonrpc": "2.0", "id": 0, "result": {"tools": []}}
        ).encode("utf-8") + b"\n"

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.flush = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = MagicMock(side_effect=[notification, real_response])

        client = MCPClient(command=["echo", "dummy"])
        client._process = mock_process

        # Must not raise RuntimeError — the pre-fix code would have raised here.
        try:
            result = client._send_request("tools/list", {})
        except RuntimeError as exc:
            pytest.fail(
                f"_send_request raised RuntimeError for a notification: {exc}"
            )

        # The real response result is returned, not {}.
        assert result == {"tools": []}


# ---------------------------------------------------------------------------
# MCP config loading
# ---------------------------------------------------------------------------


class TestMCPConfigLoading:
    """Tests for load_mcp_server_configs() and MCPServerConfig."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self) -> Any:
        """Clear the module-level config cache before and after each test."""
        from vetinari.mcp.config import reset_mcp_config_cache

        reset_mcp_config_cache()
        yield
        reset_mcp_config_cache()

    def test_load_returns_list_from_yaml(self, tmp_path: Path) -> None:
        """load_mcp_server_configs returns MCPServerConfig list from a valid YAML file."""
        from vetinari.mcp.config import load_mcp_server_configs

        yaml_content = textwrap.dedent("""\
            mcp_servers:
              - name: myserver
                command: python
                args: ["-m", "some_mcp_pkg"]
                env: {}
                enabled: true
        """)
        config_file = tmp_path / "mcp_servers.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        configs = load_mcp_server_configs(config_path=config_file)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.name == "myserver"
        assert cfg.command == "python"
        assert cfg.args == ["-m", "some_mcp_pkg"]
        assert cfg.enabled is True

    def test_load_filters_disabled_entries(self, tmp_path: Path) -> None:
        """Disabled entries are still returned (filtering is the caller's job)."""
        from vetinari.mcp.config import load_mcp_server_configs

        yaml_content = textwrap.dedent("""\
            mcp_servers:
              - name: enabled_srv
                command: cmd1
                args: []
                env: {}
                enabled: true
              - name: disabled_srv
                command: cmd2
                args: []
                env: {}
                enabled: false
        """)
        config_file = tmp_path / "mcp_servers.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        configs = load_mcp_server_configs(config_path=config_file)

        assert len(configs) == 2
        enabled = [c for c in configs if c.enabled]
        disabled = [c for c in configs if not c.enabled]
        assert len(enabled) == 1
        assert len(disabled) == 1
        assert enabled[0].name == "enabled_srv"

    def test_load_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        """load_mcp_server_configs returns [] when the config file does not exist."""
        from vetinari.mcp.config import load_mcp_server_configs

        missing_path = tmp_path / "nonexistent.yaml"
        configs = load_mcp_server_configs(config_path=missing_path)

        assert configs == []

    def test_load_skips_entry_without_name(self, tmp_path: Path) -> None:
        """Entries missing 'name' are skipped with a warning, not an exception."""
        from vetinari.mcp.config import load_mcp_server_configs

        yaml_content = textwrap.dedent("""\
            mcp_servers:
              - command: npx
                args: []
                env: {}
                enabled: true
        """)
        config_file = tmp_path / "mcp_servers.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        configs = load_mcp_server_configs(config_path=config_file)

        assert configs == []

    def test_mcp_server_config_to_command(self) -> None:
        """MCPServerConfig.to_command() returns command + args as a flat list."""
        from vetinari.mcp.config import MCPServerConfig

        cfg = MCPServerConfig(
            name="fs",
            command="npx",
            args=["-y", "@mcp/server", "/tmp"],
            env={},
            enabled=True,
        )

        assert cfg.to_command() == ["npx", "-y", "@mcp/server", "/tmp"]

    def test_load_caches_result(self, tmp_path: Path) -> None:
        """Calling load_mcp_server_configs twice returns the same list object."""
        from vetinari.mcp.config import load_mcp_server_configs

        yaml_content = textwrap.dedent("""\
            mcp_servers:
              - name: cached_srv
                command: cmd
                args: []
                env: {}
                enabled: true
        """)
        config_file = tmp_path / "mcp_servers.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        first = load_mcp_server_configs(config_path=config_file)
        second = load_mcp_server_configs(config_path=config_file)

        # Must be the same list object — cached, no re-read
        assert first is second


# ---------------------------------------------------------------------------
# WorkerMCPBridge
# ---------------------------------------------------------------------------


class TestWorkerMCPBridge:
    """Tests for WorkerMCPBridge.get_available_mcp_tools()."""

    def test_get_available_mcp_tools_returns_only_namespaced(self) -> None:
        """get_available_mcp_tools filters to mcp__* tools only."""
        from vetinari.mcp.tools import MCPToolRegistry
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        registry = MCPToolRegistry()
        # Register one built-in tool (no prefix) and one external tool (prefixed)
        from vetinari.mcp.tools import MCPTool

        registry.register(MCPTool(name="vetinari_plan", description="Plan", handler=dict))
        registry.register_external_tool("fs", _make_mcp_tool("read"))

        bridge = WorkerMCPBridge(registry=registry)
        tools = bridge.get_available_mcp_tools()

        names = [t["name"] for t in tools]
        assert "mcp__fs__read" in names
        assert "vetinari_plan" not in names

    def test_get_available_mcp_tools_empty_when_no_external(self) -> None:
        """get_available_mcp_tools returns [] when only built-in tools are registered."""
        from vetinari.mcp.tools import MCPTool, MCPToolRegistry
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        registry = MCPToolRegistry()
        registry.register(MCPTool(name="vetinari_plan", description="Plan", handler=dict))

        bridge = WorkerMCPBridge(registry=registry)

        assert bridge.get_available_mcp_tools() == []

    def test_invoke_mcp_tool_dispatches_through_registry(self) -> None:
        """invoke_mcp_tool routes namespaced calls to the correct handler."""
        from vetinari.mcp.tools import MCPToolRegistry
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        registry = MCPToolRegistry()
        registry.register_external_tool("test_srv", _make_mcp_tool("echo"))

        bridge = WorkerMCPBridge(registry=registry)
        result = bridge.invoke_mcp_tool("mcp__test_srv__echo", {"input": "ping"})

        assert "result" in result
        assert result["result"] == "echo:ping"

    def test_invoke_mcp_tool_rejects_non_namespaced(self) -> None:
        """invoke_mcp_tool returns an error for tool names without mcp__ prefix."""
        from vetinari.mcp.tools import MCPToolRegistry
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        bridge = WorkerMCPBridge(registry=MCPToolRegistry())
        result = bridge.invoke_mcp_tool("vetinari_plan", {})

        assert "error" in result

    def test_load_servers_skips_disabled(self, tmp_path: Path) -> None:
        """load_servers skips disabled server configs and returns 0 registered tools."""
        from vetinari.mcp.config import MCPServerConfig
        from vetinari.mcp.worker_bridge import WorkerMCPBridge

        bridge = WorkerMCPBridge()
        configs = [
            MCPServerConfig(
                name="disabled_srv",
                command="npx",
                args=[],
                env={},
                enabled=False,
            )
        ]

        count = bridge.load_servers(configs=configs)

        assert count == 0
        assert bridge.get_available_mcp_tools() == []
