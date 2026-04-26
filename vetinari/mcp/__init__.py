"""Vetinari MCP (Model Context Protocol) Server.

Exposes Vetinari agents as MCP tools for integration with
Claude Code, VS Code, and other MCP-compatible editors.

Sub-modules
-----------
- ``server``: MCPServer singleton (JSON-RPC dispatcher)
- ``tools``: MCPToolRegistry with built-in and external tool support
- ``client``: MCPClient for connecting to external MCP servers
- ``config``: Config loader for ``config/mcp_servers.yaml``
- ``worker_bridge``: WorkerMCPBridge — wires external tools into the Worker agent
"""

from __future__ import annotations

from vetinari.mcp.client import MCPClient
from vetinari.mcp.config import MCPServerConfig, load_mcp_server_configs
from vetinari.mcp.server import MCPServer, get_mcp_server
from vetinari.mcp.tools import MCPTool, MCPToolParameter, MCPToolRegistry
from vetinari.mcp.transport import StdioTransport
from vetinari.mcp.worker_bridge import (
    WorkerMCPBridge,
    get_available_mcp_tools,
    invoke_mcp_tool,
    reset_worker_mcp_bridge,
)

__all__ = [
    "MCPClient",
    "MCPServer",
    "MCPServerConfig",
    "MCPTool",
    "MCPToolParameter",
    "MCPToolRegistry",
    "StdioTransport",
    "WorkerMCPBridge",
    "get_available_mcp_tools",
    "get_mcp_server",
    "invoke_mcp_tool",
    "load_mcp_server_configs",
    "reset_worker_mcp_bridge",
]
