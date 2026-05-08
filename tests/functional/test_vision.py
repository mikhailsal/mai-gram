"""Integration test for vision/image support.

Sends a programmatically generated solid-colour image to a vision-capable
model via OpenRouter and verifies the model correctly identifies the colour.
"""

from __future__ import annotations

import base64
import os

import pytest

from mai_gram.llm.openrouter import OpenRouterProvider
from mai_gram.llm.provider import ChatMessage, MessageRole

pytestmark = pytest.mark.functional


def _make_solid_colour_png(r: int, g: int, b: int, size: int = 64) -> bytes:
    """Create a minimal solid-colour PNG entirely in pure Python (no Pillow)."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    raw_rows = b""
    row_data = bytes([r, g, b]) * size
    for _ in range(size):
        raw_rows += b"\x00" + row_data
    compressed = zlib.compress(raw_rows)
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


async def test_vision_model_describes_solid_red_image() -> None:
    """Send a solid red image and verify the model mentions 'red'."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is required for vision integration test")

    red_png = _make_solid_colour_png(255, 0, 0)
    b64 = base64.b64encode(red_png).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    provider = OpenRouterProvider(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a concise assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What colour is this image? Reply with ONLY the colour name, nothing else.",
            image_urls=[data_uri],
        ),
    ]

    max_attempts = 3
    last_answer = ""
    try:
        for attempt in range(1, max_attempts + 1):
            parts: list[str] = []
            async for chunk in provider.generate_stream(
                messages,
                model="openrouter/free",
                temperature=0.0,
                max_tokens=32,
            ):
                if chunk.content:
                    parts.append(chunk.content)

            last_answer = "".join(parts).strip().lower()
            if "red" in last_answer:
                break
            if attempt < max_attempts:
                import asyncio

                await asyncio.sleep(2.0)
    finally:
        await provider.close()

    assert "red" in last_answer, (
        f"Expected 'red' in model response after {max_attempts} attempts, got: {last_answer!r}"
    )


async def test_vision_serialization_produces_multimodal_content() -> None:
    """Verify that ChatMessage with image_urls serializes to the OpenAI
    multimodal content array format."""
    from mai_gram.llm.openrouter_support import serialize_message

    msg = ChatMessage(
        role=MessageRole.USER,
        content="Describe this",
        image_urls=["data:image/png;base64,abc123"],
    )
    payload = serialize_message(msg)

    assert isinstance(payload["content"], list)
    assert payload["content"][0] == {"type": "text", "text": "Describe this"}
    assert payload["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }


async def test_vision_serialization_plain_message_unchanged() -> None:
    """A message without image_urls still has plain string content."""
    from mai_gram.llm.openrouter_support import serialize_message

    msg = ChatMessage(role=MessageRole.USER, content="Hello")
    payload = serialize_message(msg)

    assert payload["content"] == "Hello"
    assert isinstance(payload["content"], str)
