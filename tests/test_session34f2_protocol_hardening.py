"""Regression tests for Session 34F2 protocol and SSE hardening."""

from __future__ import annotations

import asyncio
import io
import json
import queue
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vetinari.a2a.executor import VetinariA2AExecutor
from vetinari.a2a.transport import A2ATransport
from vetinari.exceptions import MCPError
from vetinari.mcp.client import MCPClient
from vetinari.mcp.server import MCPServer
from vetinari.mcp.tools import MCPTool, MCPToolParameter
from vetinari.mcp.transport import StdioTransport
from vetinari.web.visualization import _get_viz_queue, _remove_viz_queue, _viz_streams, _viz_streams_lock


def _allowing_guardrails() -> MagicMock:
    verdict = MagicMock()
    verdict.allowed = True
    guardrails = MagicMock()
    guardrails.check_input.return_value = verdict
    guardrails.check_both.return_value = (verdict, verdict)
    return guardrails


class TestA2AProtocolValidation:
    def test_non_string_method_returns_jsonrpc_error(self) -> None:
        transport = A2ATransport(executor=MagicMock())

        response = transport.handle_request({"jsonrpc": "2.0", "id": 1, "method": ["a2a.getAgentCard"]})

        assert response["error"]["code"] == -32600
        assert response["id"] == 1

    def test_invalid_task_send_params_skip_guardrails_and_executor(self) -> None:
        executor = MagicMock()
        transport = A2ATransport(executor=executor)

        with patch("vetinari.safety.guardrails.get_guardrails") as get_guardrails:
            response = transport.handle_request({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "a2a.taskSend",
                "params": {"taskType": 123, "inputData": "raw", "metadata": []},
            })

        assert response["error"]["code"] == -32602
        get_guardrails.assert_not_called()
        executor.execute.assert_not_called()

    def test_duplicate_task_id_is_rejected_before_execution(self) -> None:
        executor = MagicMock()
        executor.execute.return_value.to_dict.return_value = {
            "taskId": "dup",
            "status": "completed",
            "outputData": {},
            "error": "",
        }
        transport = A2ATransport(executor=executor)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "a2a.taskSend",
            "params": {"taskType": "build", "taskId": "dup", "inputData": {}},
        }

        with patch("vetinari.safety.guardrails.get_guardrails", return_value=_allowing_guardrails()):
            first = transport.handle_request(request)
            second = transport.handle_request({**request, "id": 2})

        assert "result" in first
        assert second["error"]["code"] == -32602
        assert executor.execute.call_count == 1

    def test_status_uses_persisted_state_after_transport_restart(self) -> None:
        conn = sqlite3.connect(":memory:")
        with patch("vetinari.database.get_connection", return_value=conn):
            executor = VetinariA2AExecutor(recover_on_init=False)
            original = A2ATransport(executor=executor)
            with (
                patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None),
                patch("vetinari.safety.guardrails.get_guardrails", return_value=_allowing_guardrails()),
            ):
                sent = original.handle_request({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "a2a.taskSend",
                    "params": {"taskType": "build", "taskId": "restart-task", "inputData": {}},
                })
            restarted = A2ATransport(executor=executor)
            status = restarted.handle_request({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "a2a.taskStatus",
                "params": {"taskId": "restart-task"},
            })

        assert sent["result"]["taskId"] == "restart-task"
        assert status["result"]["status"] == sent["result"]["status"]


class TestMCPProtocolValidation:
    def test_method_and_id_shapes_fail_closed(self) -> None:
        server = MCPServer()

        bad_method = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": ["ping"], "params": {}})
        bad_id = server.handle_message({"jsonrpc": "2.0", "id": {"bad": True}, "method": "ping", "params": {}})

        assert bad_method["error"]["code"] == -32600
        assert bad_id["id"] is None
        assert bad_id["error"]["code"] == -32600

    def test_explicit_null_id_gets_response_but_notification_does_not(self) -> None:
        server = MCPServer()

        explicit_null = server.handle_message({"jsonrpc": "2.0", "id": None, "method": "ping", "params": {}})
        notification = server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})

        assert explicit_null == {"jsonrpc": "2.0", "id": None, "result": {}}
        assert notification is None

    def test_tools_call_name_and_arguments_are_validated(self) -> None:
        server = MCPServer()
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        })

        missing_name = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}})
        bad_args = server.handle_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "vetinari_plan", "arguments": []},
        })

        assert missing_name["error"]["code"] == -32602
        assert bad_args["error"]["code"] == -32602

    def test_input_schema_rejects_before_handler_dispatch(self) -> None:
        server = MCPServer()
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        })
        handler = MagicMock(return_value={"ok": True})
        server.registry.register(
            MCPTool(
                name="numeric_tool",
                description="Needs a number",
                parameters=[MCPToolParameter("count", "number", "Count")],
                handler=handler,
            )
        )

        response = server.handle_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "numeric_tool", "arguments": {"count": "not-a-number"}},
        })

        assert response["result"]["isError"] is True
        assert "count" in response["result"]["content"][0]["text"]
        handler.assert_not_called()

    def test_sensitive_builtins_require_explicit_permissions(self) -> None:
        server = MCPServer()
        server.handle_message({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        })

        response = server.handle_message({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "vetinari_execute", "arguments": {"task": "run"}},
        })

        assert response["error"]["code"] == -32003

    def test_stdio_writes_response_for_explicit_null_id(self) -> None:
        server = MCPServer()
        transport = StdioTransport(server)
        stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": None, "method": "ping"}) + "\n")
        stdout = io.StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            transport.run()

        assert json.loads(stdout.getvalue()) == {"jsonrpc": "2.0", "id": None, "result": {}}


class TestMCPHTTPAndClient:
    def test_http_message_preserves_initialized_session(self) -> None:
        litestar = pytest.importorskip("litestar")
        from litestar.testing import TestClient

        def _pass_guard(connection, handler=None, **kwargs):
            return None

        with patch("vetinari.web.litestar_guards.admin_guard", _pass_guard):
            import vetinari.mcp.server as server_mod
            from vetinari.web.litestar_mcp_transport import create_mcp_transport_handlers

            server_mod._mcp_server = None
            app = litestar.Litestar(route_handlers=create_mcp_transport_handlers())

        with TestClient(app=app) as client:
            first = client.post(
                "/mcp/message",
                content=json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
                }),
                headers={"Content-Type": "application/json"},
            )
            second = client.post(
                "/mcp/message",
                content=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
                headers={"Content-Type": "application/json"},
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert "tools" in second.json()["result"]

    def test_client_timeout_stops_process(self) -> None:
        block = threading.Event()

        def _blocking_readline() -> bytes:
            block.wait()
            return b""

        mock_process = MagicMock()
        mock_process.pid = 123
        mock_process.stdin = MagicMock()
        mock_process.stdin.closed = False
        mock_process.stdout = SimpleNamespace(readline=_blocking_readline)
        mock_process.wait.return_value = None

        client = MCPClient(command=["dummy"])
        client._process = mock_process

        try:
            with patch("vetinari.mcp.client._READLINE_TIMEOUT", 0.01), pytest.raises(MCPError):
                client._send_request("ping", {})
            mock_process.terminate.assert_called_once()
            mock_process.stdin.close.assert_called_once_with()
            mock_process.wait.assert_called_once_with(timeout=5)
            assert client._process is None
        finally:
            block.set()


class TestSSELifecycle:
    def test_log_stream_queue_poll_does_not_use_executor(self) -> None:
        import inspect

        import vetinari.web.litestar_log_stream as mod

        assert "run_in_executor" not in inspect.getsource(mod._generate_log_events)

    def test_visualization_queue_cleanup_removes_only_matching_queue(self) -> None:
        first = _get_viz_queue("plan-a")
        replacement: queue.Queue = queue.Queue()
        with _viz_streams_lock:
            _viz_streams["plan-a"] = replacement

        _remove_viz_queue("plan-a", first)
        with _viz_streams_lock:
            assert _viz_streams["plan-a"] is replacement

        _remove_viz_queue("plan-a", replacement)
        with _viz_streams_lock:
            assert "plan-a" not in _viz_streams

    def test_log_queue_timeout_helper_yields_without_blocking_loop(self) -> None:
        from vetinari.web.litestar_log_stream import _queue_get_with_timeout

        async def _probe() -> str:
            q: queue.Queue = queue.Queue()
            q.put("ready")
            return await _queue_get_with_timeout(q, 1)

        assert asyncio.run(_probe()) == "ready"
