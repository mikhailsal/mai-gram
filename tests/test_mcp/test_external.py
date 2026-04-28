"""Tests for the external MCP server bridge."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram.mcp_servers.external import ExternalMCPPool, ExternalMCPServer
from mai_gram.mcp_servers.messages_server import MCPToolSpec


class _FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed


class _FakeReader:
    def __init__(self, lines: list[bytes] | None = None, read_data: bytes = b"") -> None:
        self._lines = list(lines or [])
        self._read_data = read_data

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""

    async def read(self, _size: int = -1) -> bytes:
        return self._read_data


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[bytes] | None = None,
        stderr_data: bytes = b"",
        wait_side_effect: Exception | None = None,
        terminate_side_effect: Exception | None = None,
    ) -> None:
        self.stdin = _FakeWriter()
        self.stdout = _FakeReader(stdout_lines)
        self.stderr = _FakeReader(read_data=stderr_data)
        self.wait = AsyncMock(side_effect=wait_side_effect)
        self.terminate = MagicMock(side_effect=terminate_side_effect)
        self.kill = MagicMock()


def _attach_process(
    server: ExternalMCPServer,
    *,
    stdout_lines: list[bytes],
    stderr_data: bytes = b"",
) -> _FakeWriter:
    writer = _FakeWriter()
    server._process = cast(
        "Any",
        SimpleNamespace(
            stdin=writer,
            stdout=_FakeReader(stdout_lines),
            stderr=_FakeReader(read_data=stderr_data),
        ),
    )
    return writer


@pytest.mark.asyncio
async def test_send_request_skips_unrelated_lines_until_matching_response() -> None:
    server = ExternalMCPServer("test", "server")
    writer = _attach_process(
        server,
        stdout_lines=[
            b"not-json\n",
            json.dumps({"jsonrpc": "2.0", "method": "log"}).encode("utf-8") + b"\n",
            json.dumps({"jsonrpc": "2.0", "id": 999, "result": {"ignored": True}}).encode("utf-8")
            + b"\n",
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode("utf-8") + b"\n",
        ],
    )

    result = await server._send_request("tools/list", {"page": 1})

    assert result == {"ok": True}
    assert json.loads(writer.writes[0].decode("utf-8")) == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {"page": 1},
    }


@pytest.mark.asyncio
async def test_send_request_raises_with_stderr_excerpt_on_closed_stdout() -> None:
    server = ExternalMCPServer("test", "server")
    _attach_process(
        server,
        stdout_lines=[b""],
        stderr_data=b"traceback details",
    )

    with pytest.raises(RuntimeError, match="traceback details"):
        await server._send_request("tools/list", {})


@pytest.mark.asyncio
async def test_send_request_raises_on_jsonrpc_error() -> None:
    server = ExternalMCPServer("test", "server")
    _attach_process(
        server,
        stdout_lines=[
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"message": "no such tool"},
                }
            ).encode("utf-8")
            + b"\n"
        ],
    )

    with pytest.raises(RuntimeError, match="no such tool"):
        await server._send_request("tools/call", {"name": "missing"})


@pytest.mark.asyncio
async def test_start_initializes_once_and_merges_env(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()
    create_process = AsyncMock(return_value=process)
    monkeypatch.setattr(
        "mai_gram.mcp_servers.external.asyncio.create_subprocess_exec",
        create_process,
    )

    server = ExternalMCPServer("test", "server", args=["--flag"], env={"EXTRA_ENV": "1"})
    server._send_request = AsyncMock(return_value={"serverInfo": {"name": "srv"}})
    server._send_notification = AsyncMock()

    await server.start()
    await server.start()

    create_process.assert_awaited_once()
    assert create_process.await_args.kwargs["env"]["EXTRA_ENV"] == "1"
    server._send_request.assert_awaited_once()
    server._send_notification.assert_awaited_once_with("notifications/initialized", {})
    assert server._started is True


@pytest.mark.asyncio
async def test_stop_handles_timeout_and_process_lookup_error() -> None:
    timeout_server = ExternalMCPServer("test", "server")
    timeout_server._tools_cache = [MCPToolSpec(name="tool", description="desc", input_schema={})]
    timeout_server._started = True
    timeout_server._process = cast("Any", _FakeProcess(wait_side_effect=asyncio.TimeoutError()))

    await timeout_server.stop()

    assert timeout_server._process is None
    assert timeout_server._started is False
    assert timeout_server._tools_cache is None

    lookup_server = ExternalMCPServer("test", "server")
    lookup_server._process = cast("Any", _FakeProcess(terminate_side_effect=ProcessLookupError()))

    await lookup_server.stop()

    assert lookup_server._process is None


@pytest.mark.asyncio
async def test_list_tools_caches_specs_and_call_tool_formats_content() -> None:
    server = ExternalMCPServer("test", "server")
    server.start = AsyncMock()
    server._send_request = AsyncMock(
        side_effect=[
            {
                "tools": [
                    {
                        "name": "wiki_search",
                        "description": "Search the wiki",
                        "inputSchema": {"type": "object"},
                    }
                ]
            },
            {
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "json", "value": 2},
                    "last",
                ]
            },
            {"content": []},
        ]
    )

    tools = await server.list_tools()
    cached_tools = await server.list_tools()
    rendered = await server.call_tool("wiki_search", {"query": "hello"})
    empty = await server.call_tool("wiki_search", {"query": "empty"})

    assert tools[0].name == "wiki_search"
    assert cached_tools is tools
    assert rendered == 'first\n{"type": "json", "value": 2}\nlast'
    assert empty == ""


@pytest.mark.asyncio
async def test_external_helper_methods_and_pool_management(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = ExternalMCPServer("test", "server")

    with pytest.raises(RuntimeError, match="is not running"):
        server._require_request_pipes()

    assert await server._read_stderr_excerpt() == ""
    assert server._decode_matching_response(b"not-json", 1) is None
    assert server._decode_matching_response(json.dumps([1]).encode("utf-8"), 1) is None
    assert server._decode_matching_response(json.dumps({"id": 2}).encode("utf-8"), 1) is None
    assert server._extract_result({"result": "not-a-dict"}) == {}

    with pytest.raises(RuntimeError, match="boom"):
        server._extract_result({"error": {"message": "boom"}})

    await server._send_notification("notifications/test", {})

    writer = _attach_process(server, stdout_lines=[])
    await server._send_notification("notifications/test", {"value": 1})
    assert json.loads(writer.writes[0].decode("utf-8")) == {
        "jsonrpc": "2.0",
        "method": "notifications/test",
        "params": {"value": 1},
    }

    pool = ExternalMCPPool(
        {
            "one": {"command": "cmd", "args": ["--ok"], "env": {"ENV": "1"}},
            "two": {"command": ""},
        }
    )
    server_one = pool.get_server("one")
    assert server_one is not None
    server_one.stop = AsyncMock(side_effect=RuntimeError("boom"))

    await pool.stop_all()

    assert pool.server_names == ["one"]
    assert pool.get_all_servers()["one"] is server_one
    assert "Error stopping MCP server 'one'" in caplog.text
