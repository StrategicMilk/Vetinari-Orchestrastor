"""MCP Server -- JSON-RPC based protocol handler."""

from __future__ import annotations

import json
import logging
from typing import Any

from vetinari.mcp.tools import MCPToolRegistry

logger = logging.getLogger(__name__)


class MCPServer:
    """Model Context Protocol server for Vetinari."""

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_NAME = "vetinari"
    SERVER_VERSION = "1.0.0"

    def __init__(self):
        self._registry = MCPToolRegistry()
        self._registry.register_defaults()
        self._initialized = False

    @property
    def registry(self) -> MCPToolRegistry:
        return self._registry

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming JSON-RPC message."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
        }

        handler = handlers.get(method)
        if handler is None:
            return self._error_response(msg_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as e:
            return self._error_response(msg_id, -32603, str(e))

    def _handle_initialize(self, params: dict) -> dict:
        self._initialized = True
        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.SERVER_NAME, "version": self.SERVER_VERSION},
        }

    def _handle_tools_list(self, params: dict) -> dict:
        return {"tools": self._registry.list_tools()}

    def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = self._registry.invoke(name, arguments)
        if "error" in result:
            return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    def _handle_ping(self, params: dict) -> dict:
        return {}

    def _error_response(self, msg_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


_mcp_server: MCPServer | None = None


def get_mcp_server() -> MCPServer:
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server
