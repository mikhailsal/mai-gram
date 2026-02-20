"""MCP servers, registry, and bridge utilities."""

from mai_companion.mcp_servers.bridge import (
    MCPToolCall,
    mcp_result_to_openai,
    mcp_tools_to_openai,
    openai_tool_call_to_mcp,
    run_with_tools,
)
from mai_companion.mcp_servers.manager import MCPManager, MCPServer, RegisteredTool
from mai_companion.mcp_servers.messages_server import MCPToolSpec, MessagesMCPServer
from mai_companion.mcp_servers.sleep_server import SleepMCPServer
from mai_companion.mcp_servers.wiki_server import WikiMCPServer

__all__ = [
    "MCPManager",
    "MCPServer",
    "MCPToolCall",
    "MCPToolSpec",
    "MessagesMCPServer",
    "RegisteredTool",
    "SleepMCPServer",
    "WikiMCPServer",
    "mcp_result_to_openai",
    "mcp_tools_to_openai",
    "openai_tool_call_to_mcp",
    "run_with_tools",
]
