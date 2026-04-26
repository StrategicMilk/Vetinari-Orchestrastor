"""MCP transport — native Litestar handlers for the Model Context Protocol.

Migrated from ``vetinari.mcp.transport.SSETransport.create_flask_routes``.
Native Litestar equivalents (ADR-0066). URL paths identical to Flask.

Endpoints
---------
    POST /mcp/message   — accept a JSON-RPC request and return the MCP server response
    GET  /mcp/tools     — list all tools registered with the MCP server

Non-object JSON bodies (strings, arrays, numbers) are rejected with 400.
Litestar parses the raw body as ``bytes`` when the declared type is ``Any``,
so manual JSON parsing and dict-type validation is done in the handler.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Request, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_mcp_transport_handlers() -> list[Any]:
    """Create all Litestar route handlers for the MCP SSE transport.

    Wraps the singleton ``MCPServer`` instance for JSON-RPC message dispatch
    and tool listing. Falls back to an empty list when Litestar is not installed
    so the module remains importable in non-Litestar environments.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — MCP transport handlers not registered")
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- POST /mcp/message ----------------------------------------------------

    @post("/mcp/message", media_type=MediaType.JSON, guards=[admin_guard], status_code=200)
    async def handle_mcp_message(request: Request) -> Any:
        """Accept a JSON-RPC request and return the MCP server response.

        Reads the raw request body and parses it manually so that non-object
        JSON (strings, arrays, numbers) can be rejected with a structured 400
        rather than letting Litestar's dict-deserialization raise an unhandled
        500 for those inputs.

        Returns 400 for missing body, malformed JSON, or non-object JSON.
        Dispatch errors are converted to JSON-RPC internal errors.

        Args:
            request: The Litestar Request object, used to read the raw body.

        Returns:
            JSON-RPC response dict from the MCP server on success; Litestar
            Response with 400 status for request-shape failures.
        """
        from vetinari.web.responses import litestar_error_response

        # Read raw body — avoids Litestar's dict-only coercion for /mcp/message
        raw_body = await request.body()
        if not raw_body:
            return litestar_error_response("Request body must be a JSON object", 400)

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            logger.warning("MCP transport received invalid JSON body — rejecting with 400")
            return litestar_error_response("Invalid JSON", 400)

        if not isinstance(parsed, dict):
            return litestar_error_response("Request body must be a JSON object", 400)

        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(parsed):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)
        if body_has_oversized_key(parsed):
            return litestar_error_response("Request body contains oversized key", 400)

        try:
            from vetinari.mcp.server import get_mcp_server

            server = get_mcp_server()
            response = await asyncio.to_thread(server.handle_message, parsed)
            if response is None:
                return Response(content=None, status_code=202)
            return response

        except Exception:
            logger.exception("MCP message dispatch failed")
            raw_id = parsed.get("id")
            msg_id = raw_id if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": "Internal server error"},
            }

    # -- GET /mcp/tools -------------------------------------------------------

    @get("/mcp/tools", media_type=MediaType.JSON, guards=[admin_guard])
    async def list_mcp_tools() -> dict[str, Any]:
        """Return the list of tools registered with the MCP server.

        Queries the tool registry on the singleton ``MCPServer`` instance.
        Returns 503 when the MCP server is unavailable.

        Returns:
            JSON object with ``tools`` key containing the list of registered
            tool descriptors; 503 when the MCP server is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.mcp.server import get_mcp_server

            server = get_mcp_server()
            return {"tools": server.registry.list_tools()}

        except Exception:
            logger.exception("Failed to list MCP tools — returning 503")
            return litestar_error_response("MCP server unavailable", 503)  # type: ignore[return-value]

    return [handle_mcp_message, list_mcp_tools]
