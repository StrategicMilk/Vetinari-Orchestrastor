"""Vetinari MCP (Model Context Protocol) Server.

Exposes Vetinari agents as MCP tools for integration with
Claude Code, VS Code, and other MCP-compatible editors.
"""

from __future__ import annotations

from vetinari.mcp.server import MCPServer, get_mcp_server
from vetinari.mcp.tools import MCPTool, MCPToolRegistry

__all__ = ["MCPServer", "MCPTool", "MCPToolRegistry", "get_mcp_server"]
