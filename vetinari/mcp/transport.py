"""MCP Transport layer -- stdio and SSE."""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)


class StdioTransport:
    """stdio-based MCP transport."""

    def __init__(self, server):
        self._server = server

    def run(self) -> None:
        """Read JSON-RPC messages from stdin, write responses to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                response = self._server.handle_message(message)
                if response.get("id") is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                error = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                sys.stdout.write(json.dumps(error) + "\n")
                sys.stdout.flush()


class SSETransport:
    """Server-Sent Events transport for MCP."""

    def __init__(self, server):
        self._server = server

    def create_flask_routes(self):
        """Create Flask Blueprint for SSE-based MCP transport."""
        from flask import Blueprint, jsonify, request

        bp = Blueprint("mcp", __name__)

        @bp.route("/mcp/message", methods=["POST"])
        def handle_message():
            message = request.get_json(silent=True)
            if not message:
                return jsonify({"error": "Invalid JSON"}), 400
            response = self._server.handle_message(message)
            return jsonify(response)

        @bp.route("/mcp/tools", methods=["GET"])
        def list_tools():
            return jsonify({"tools": self._server.registry.list_tools()})

        return bp
