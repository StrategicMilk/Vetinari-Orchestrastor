"""Proof tests for SESSION-32A protocol-truth fixes.

Each test class exercises a specific contract enforced by the fixes in this
session.  Tests are written so they would FAIL against the pre-fix code and
PASS against the fixed code.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vetinari.a2a.ag_ui import AGUIEventEmitter, AGUIEventType
from vetinari.a2a.transport import A2ATransport
from vetinari.mcp import MCPToolParameter, StdioTransport, reset_worker_mcp_bridge
from vetinari.mcp.server import MCPServer
from vetinari.mcp.tools import MCPToolRegistry
from vetinari.mcp.worker_bridge import WorkerMCPBridge

# ── Helper ────────────────────────────────────────────────────────────────────


def _fresh_server() -> MCPServer:
    """Return a new, uninitialized MCPServer with no shared state."""
    return MCPServer()


# ── TestMCPServerProtocol ─────────────────────────────────────────────────────


class TestMCPServerProtocol:
    """Verify jsonrpc version gate, protocol-version rejection, and the
    _initialized guard on all post-handshake methods."""

    def test_handle_message_rejects_missing_jsonrpc(self) -> None:
        """Message with no jsonrpc field must return error code -32600."""
        server = _fresh_server()
        response = server.handle_message({"method": "tools/list", "id": 1})
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32600

    def test_handle_message_rejects_wrong_jsonrpc_version(self) -> None:
        """Message with jsonrpc='1.0' must return error code -32600."""
        server = _fresh_server()
        response = server.handle_message({"jsonrpc": "1.0", "method": "tools/list", "id": 1})
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32600

    def test_initialize_rejects_incompatible_version(self) -> None:
        """_handle_initialize with a mismatched protocolVersion must return an
        error and leave _initialized False so subsequent calls are still gated."""
        server = _fresh_server()
        assert server._initialized is False

        # Call through handle_message so the full dispatch chain runs.
        response = server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2023-01-01"},
        })
        assert response is not None
        assert "error" in response
        # The server uses -32002 for protocol-version mismatch.
        assert response["error"]["code"] == -32002
        # State must not have been flipped — incompatible client was rejected.
        assert server._initialized is False

    def test_tools_list_requires_initialization(self) -> None:
        """tools/list before initialize must return error -32002."""
        server = _fresh_server()
        response = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32002

    def test_notifications_initialized_returns_none(self) -> None:
        """notifications/initialized is a notification (no id) — handle_message
        must return None, not an error dict."""
        server = _fresh_server()
        response = server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert response is None

    def test_handle_message_rejects_non_dict_params(self) -> None:
        """params that is a list (not an object) must return error -32602."""
        server = _fresh_server()
        response = server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": [1, 2, 3]})
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32602


# ── TestMCPClientValidation ───────────────────────────────────────────────────


class TestMCPClientValidation:
    """Verify the client-side response validation fixes:
    all content blocks joined, non-dict response raises RuntimeError,
    and mismatched response ID raises RuntimeError."""

    def test_call_tool_returns_all_content_blocks(self) -> None:
        """call_tool must concatenate ALL text content blocks with newline separators."""
        from vetinari.mcp.client import MCPClient

        client = MCPClient(["dummy"])
        # Inject a fake process with preconfigured stdout response.
        fake_result = {
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "content": [
                    {"type": "text", "text": "block_one"},
                    {"type": "text", "text": "block_two"},
                    {"type": "text", "text": "block_three"},
                ]
            },
        }
        raw_response = (json.dumps(fake_result) + "\n").encode("utf-8")

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.closed = False
        mock_process.stdout.readline.return_value = raw_response
        client._process = mock_process

        result = client.call_tool("my_tool", {"arg": "val"})
        assert result == "block_one\nblock_two\nblock_three"

    def test_send_request_raises_on_non_dict_response(self) -> None:
        """_send_request must raise RuntimeError when the server returns a JSON
        array instead of a JSON object."""
        from vetinari.mcp.client import MCPClient

        client = MCPClient(["dummy"])
        raw_response = (json.dumps([1, 2, 3]) + "\n").encode("utf-8")

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.closed = False
        mock_process.stdout.readline.return_value = raw_response
        client._process = mock_process

        with pytest.raises(RuntimeError, match="Expected JSON object"):
            client._send_request("tools/list")

    def test_send_request_skips_id_mismatch_until_matching_response(self) -> None:
        """_send_request must ignore unrelated frames until the matching id arrives."""
        from vetinari.mcp.client import MCPClient

        client = MCPClient(["dummy"])
        wrong_response = (json.dumps({"jsonrpc": "2.0", "id": 99, "result": {}}) + "\n").encode("utf-8")
        right_response = (json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"ok": True}}) + "\n").encode("utf-8")

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.closed = False
        mock_process.stdout.readline = MagicMock(side_effect=[wrong_response, right_response])
        client._process = mock_process

        assert client._send_request("tools/list") == {"ok": True}


# ── TestStdioTransportFraming ─────────────────────────────────────────────────


class TestStdioTransportFraming:
    """Verify that StdioTransport writes a -32600 error and continues when it
    receives a JSON array instead of a JSON object."""

    def test_run_returns_error_on_non_dict_json(self) -> None:
        """Feed '[1,2,3]\\n' to transport.run() — must write a -32600 error to
        stdout and not raise an exception."""
        server = _fresh_server()
        transport = StdioTransport(server)

        stdin_data = "[1,2,3]\n"
        captured_output: list[str] = []

        mock_stdin = io.StringIO(stdin_data)
        mock_stdout = MagicMock()
        mock_stdout.write = lambda s: captured_output.append(s)
        mock_stdout.flush = MagicMock()

        with patch.object(sys, "stdin", mock_stdin), patch.object(sys, "stdout", mock_stdout):
            transport.run()  # must not raise

        # At least one response should have been written.
        assert len(captured_output) >= 1
        written_json = json.loads(captured_output[0])
        assert "error" in written_json
        assert written_json["error"]["code"] == -32600


# ── TestWorkerBridgeRegistry ──────────────────────────────────────────────────


class TestWorkerBridgeRegistry:
    """Verify that WorkerMCPBridge does not replace a valid empty registry with
    a new one (the None-vs-falsy bug that was fixed)."""

    def test_registry_injection_not_replaced_when_empty(self) -> None:
        """Passing an empty MCPToolRegistry must result in bridge._registry being
        the SAME object, not a freshly-created replacement."""
        injected = MCPToolRegistry()
        bridge = WorkerMCPBridge(registry=injected)
        # Identity check: must be the exact same object we passed in.
        assert bridge._registry is injected


# ── TestA2ATransport ──────────────────────────────────────────────────────────


class TestA2ATransport:
    """Verify that A2ATransport fails closed on non-dict inputs at both entry
    points: handle_request_json and handle_request."""

    def test_handle_request_json_rejects_array_body(self) -> None:
        """handle_request_json('[1,2,3]') must return a JSON error response,
        not crash with AttributeError."""
        transport = A2ATransport()
        raw_response = transport.handle_request_json("[1,2,3]")
        response = json.loads(raw_response)
        assert "error" in response
        # The error code must be the invalid-request sentinel, not an internal error.
        assert response["error"]["code"] in (-32600, -32700)

    def test_handle_request_rejects_non_dict(self) -> None:
        """handle_request([1,2,3]) (a list, not a dict) must return an error
        response, not raise AttributeError."""
        transport = A2ATransport()
        response = transport.handle_request([1, 2, 3])  # type: ignore[arg-type]
        assert "error" in response
        assert response["error"]["code"] == -32600


# ── TestAGUIEventEmitter ──────────────────────────────────────────────────────


class TestAGUIEventEmitter:
    """Verify the half-open lifecycle fix in AGUIEventEmitter.emit_tool_call."""

    def test_emit_tool_call_no_half_open_on_bad_args(self) -> None:
        """emit_tool_call with non-serialisable args must emit exactly 2 events
        (START + END with error), never leaving the lifecycle half-open."""
        emitter = AGUIEventEmitter()
        results = emitter.emit_tool_call("my_tool", {"key": object()})  # type: ignore[dict-item]

        # Must be exactly 2 SSE strings: START + END(error).
        assert len(results) == 2
        events = emitter.get_events()
        assert len(events) == 2
        assert events[0].event_type == AGUIEventType.TOOL_CALL_START
        assert events[1].event_type == AGUIEventType.TOOL_CALL_END
        # The END event must carry an error field, not a clean close.
        assert "error" in events[1].data

    def test_emit_tool_call_happy_path_returns_three_events(self) -> None:
        """emit_tool_call with serialisable args must emit exactly 3 events:
        TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_END."""
        emitter = AGUIEventEmitter()
        results = emitter.emit_tool_call("my_tool", {"k": "v"})

        assert len(results) == 3
        events = emitter.get_events()
        assert len(events) == 3
        assert events[0].event_type == AGUIEventType.TOOL_CALL_START
        assert events[1].event_type == AGUIEventType.TOOL_CALL_ARGS
        assert events[2].event_type == AGUIEventType.TOOL_CALL_END
        # Happy-path END must not carry an error field.
        assert "error" not in events[2].data


# ── TestMCPPackageFacade ──────────────────────────────────────────────────────


class TestMCPPackageFacade:
    """Verify that the vetinari.mcp package __init__ exports the three
    symbols that were added in SESSION-32A."""

    def test_mcp_package_exports_stdio_transport(self) -> None:
        """from vetinari.mcp import StdioTransport must succeed."""
        from vetinari.mcp import StdioTransport as _ST

        assert _ST is StdioTransport

    def test_mcp_package_exports_mcp_tool_parameter(self) -> None:
        """from vetinari.mcp import MCPToolParameter must succeed."""
        from vetinari.mcp import MCPToolParameter as _MTP

        assert _MTP is MCPToolParameter

    def test_mcp_package_exports_reset_worker_mcp_bridge(self) -> None:
        """from vetinari.mcp import reset_worker_mcp_bridge must succeed."""
        from vetinari.mcp import reset_worker_mcp_bridge as _rwb

        assert callable(_rwb)


# ---------------------------------------------------------------------------
# Per-caller MCPServer isolation (Defect 3 proof — protocol layer)
# ---------------------------------------------------------------------------


class TestMCPPerCallerIsolation:
    """Prove two MCPServer() instances are fully independent.

    This is the protocol-layer companion to
    ``TestMCPPerCallerSessionIsolation`` in ``test_mcp_integration.py``.
    That test exercises isolation through the HTTP transport; this test drives
    ``MCPServer.handle_message()`` directly to confirm the invariant holds at
    the server object level independently of the transport.
    """

    def test_two_instances_are_independent(self) -> None:
        """Initializing server_a must not affect the state of server_b.

        server_a is initialized via the ``initialize`` handshake.  server_b is
        never initialized.  A ``tools/list`` call on server_b must return a
        JSON-RPC error (not a tools list), proving the two instances share no
        state.
        """
        server_a = _fresh_server()
        server_b = _fresh_server()

        # Initialize server_a so it enters the "ready" state.
        init_msg: dict = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }
        response_a = server_a.handle_message(init_msg)
        assert "result" in response_a, (
            f"initialize on server_a returned error: {response_a}"
        )

        # server_b is still uninitialized — tools/list must return an error,
        # not a tools list, because it has never seen an initialize message.
        list_msg: dict = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        response_b = server_b.handle_message(list_msg)
        assert "error" in response_b, (
            "Expected a JSON-RPC error for tools/list on an uninitialized "
            f"server_b (state leaked from server_a?), got: {response_b}"
        )
