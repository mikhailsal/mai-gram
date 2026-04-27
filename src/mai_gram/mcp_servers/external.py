"""External MCP server bridge using stdio JSON-RPC transport.

Launches MCP-compatible servers as subprocesses and communicates via
JSON-RPC 2.0 over stdin/stdout, following the MCP stdio transport spec.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any

from mai_gram.mcp_servers.messages_server import MCPToolSpec

logger = logging.getLogger(__name__)

_JSONRPC_VERSION = "2.0"


class ExternalMCPServer:
    """MCP server backed by an external subprocess (stdio transport).

    Implements the same protocol as WikiMCPServer/MessagesMCPServer
    so it can be registered with MCPManager.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._name = name
        self._command = command
        self._args = args or []
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._started = False
        self._tools_cache: list[MCPToolSpec] | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        """Launch the subprocess and perform the MCP initialize handshake."""
        if self._started:
            return

        async with self._lock:
            if self._started:
                return

            merged_env = {**os.environ}
            if self._env:
                merged_env.update(self._env)

            logger.info(
                "Starting external MCP server '%s': %s %s",
                self._name,
                self._command,
                self._args,
            )
            self._process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )

            init_result = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mai-gram", "version": "1.0"},
                },
            )
            logger.info(
                "MCP server '%s' initialized: %s",
                self._name,
                init_result.get("serverInfo", {}).get("name", "unknown"),
            )

            await self._send_notification("notifications/initialized", {})
            self._started = True

    async def stop(self) -> None:
        """Terminate the subprocess."""
        if self._process is None:
            return

        try:
            if self._process.stdin and not self._process.stdin.is_closing():
                self._process.stdin.close()
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
        except ProcessLookupError:
            pass
        finally:
            self._process = None
            self._started = False
            self._tools_cache = None
            logger.info("External MCP server '%s' stopped", self._name)

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return tools exposed by this external server."""
        if self._tools_cache is not None:
            return self._tools_cache

        await self.start()

        result = await self._send_request("tools/list", {})
        tools_raw = result.get("tools", [])

        specs = []
        for t in tools_raw:
            specs.append(
                MCPToolSpec(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
            )

        self._tools_cache = specs
        logger.info("MCP server '%s' provides %d tool(s)", self._name, len(specs))
        return specs

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call on the external server."""
        await self.start()

        result = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

        content_items = result.get("content", [])
        if not content_items:
            return ""

        text_parts = []
        for item in content_items:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(json.dumps(item, ensure_ascii=False))
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts) if text_parts else ""

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        stdin, stdout = self._require_request_pipes()
        request_id = self._next_request_id()
        stdin.write(self._serialize_request(request_id, method, params))
        await stdin.drain()

        while True:
            raw_line = await asyncio.wait_for(stdout.readline(), timeout=timeout)
            if not raw_line:
                raise RuntimeError(
                    f"MCP server '{self._name}' closed stdout unexpectedly. "
                    f"stderr: {await self._read_stderr_excerpt()}"
                )

            response = self._decode_matching_response(raw_line, request_id)
            if response is None:
                continue
            return self._extract_result(response)

    def _require_request_pipes(
        self,
    ) -> tuple[asyncio.StreamWriter, asyncio.StreamReader]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(f"MCP server '{self._name}' is not running")
        return self._process.stdin, self._process.stdout

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @staticmethod
    def _serialize_request(request_id: int, method: str, params: dict[str, Any]) -> bytes:
        return (
            json.dumps(
                {
                    "jsonrpc": _JSONRPC_VERSION,
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")

    async def _read_stderr_excerpt(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        with contextlib.suppress(asyncio.TimeoutError):
            stderr_bytes = await asyncio.wait_for(self._process.stderr.read(4096), timeout=1.0)
            return stderr_bytes.decode("utf-8", errors="replace")[:500]
        return ""

    @staticmethod
    def _decode_matching_response(raw_line: bytes, request_id: int) -> dict[str, Any] | None:
        try:
            response = json.loads(raw_line)
        except json.JSONDecodeError:
            return None
        if not isinstance(response, dict) or "id" not in response:
            return None
        if response.get("id") != request_id:
            return None
        return response

    def _extract_result(self, response: dict[str, Any]) -> dict[str, Any]:
        if "error" in response:
            err = response["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"MCP server '{self._name}' error: {msg}")
        result = response.get("result", {})
        return result if isinstance(result, dict) else {}

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            return

        notification = {
            "jsonrpc": _JSONRPC_VERSION,
            "method": method,
            "params": params,
        }

        line = json.dumps(notification, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()


class ExternalMCPPool:
    """Manages a pool of external MCP servers based on config."""

    def __init__(self, server_configs: dict[str, dict[str, Any]]) -> None:
        self._servers: dict[str, ExternalMCPServer] = {}
        for name, config in server_configs.items():
            command = config.get("command", "")
            args = config.get("args", [])
            env = config.get("env")
            if command:
                self._servers[name] = ExternalMCPServer(
                    name=name,
                    command=command,
                    args=args,
                    env=env,
                )

    @property
    def server_names(self) -> list[str]:
        return list(self._servers.keys())

    def get_server(self, name: str) -> ExternalMCPServer | None:
        return self._servers.get(name)

    def get_all_servers(self) -> dict[str, ExternalMCPServer]:
        return dict(self._servers)

    async def stop_all(self) -> None:
        """Stop all running external servers."""
        for server in self._servers.values():
            try:
                await server.stop()
            except Exception:
                logger.exception("Error stopping MCP server '%s'", server.name)
