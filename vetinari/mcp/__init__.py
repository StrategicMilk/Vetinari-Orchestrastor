"""Vetinari MCP (Model Context Protocol) Server.

Exposes Vetinari agents as MCP tools for integration with
Claude Code, VS Code, and other MCP-compatible editors.
"""
from vetinari.mcp.server import MCPServer, get_mcp_server
from vetinari.mcp.tools import MCPTool, MCPToolRegistry

__all__ = ["MCPServer", "get_mcp_server", "MCPTool", "MCPToolRegistry"]
