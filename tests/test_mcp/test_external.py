"""Tests for the external MCP server bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mai_gram.mcp_servers.external import ExternalMCPServer


class _FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


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
