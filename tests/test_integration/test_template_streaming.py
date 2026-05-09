"""Integration tests for template-aware streaming display.

Verifies that during streaming, the live display renders structured fields
cleanly (no raw XML/JSON/markdown tags visible to the user) when a template
is active.  Uses mock LLM providers with controlled chunk sequences and the
same _build_handler pattern as test_llm_error_delivery.py.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from mai_gram.llm.provider import LLMProvider, LLMResponse, StreamChunk, TokenUsage

pytestmark = pytest.mark.integration

CHAT_ID = "test-tpl-stream"
USER_ID = "test-user-tpl-stream"


def _make_chunks(text: str, *, chunk_size: int = 20) -> list[str]:
    """Split text into chunks of approximately chunk_size characters."""
    parts = []
    for i in range(0, len(text), chunk_size):
        parts.append(text[i : i + chunk_size])
    return parts


class _StreamingLLM(LLMProvider):
    """Mock LLM that yields pre-defined content chunks."""

    def __init__(self, full_text: str, *, chunk_size: int = 20) -> None:
        self._chunks = _make_chunks(full_text, chunk_size=chunk_size)

    async def generate(self, messages: list[Any], **kw: Any) -> LLMResponse:
        return LLMResponse(
            content="".join(self._chunks),
            model="test-model",
            usage=TokenUsage(10, 20, 30),
        )

    async def generate_stream(self, messages: list[Any], **kw: Any):
        for chunk_text in self._chunks:
            yield StreamChunk(content=chunk_text)
        yield StreamChunk(
            content="",
            finish_reason="stop",
            usage=TokenUsage(10, 20, 30),
        )

    async def count_tokens(self, messages: list[Any], **kw: Any) -> int:
        return 0

    async def close(self) -> None:
        pass


async def _build_handler(tmp_path, llm, *, template: str = "empty", stream_debug: bool = False):
    """Build a BotHandler with a live DB and specific template."""
    import mai_gram.config as _cfg
    from mai_gram.bot.handler import BotHandler
    from mai_gram.config import Settings, reset_settings
    from mai_gram.db import init_db, reset_db_state, run_migrations
    from mai_gram.db.database import get_session
    from mai_gram.db.models import Chat
    from mai_gram.messenger.console import ConsoleMessenger

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.joinpath("default.txt").write_text("You are a test assistant.\n", encoding="utf-8")

    reset_settings()
    reset_db_state()

    settings = Settings(
        telegram_bot_token="fake-token",  # noqa: S106
        openrouter_api_key="fake-key",
        database_url=db_url,
        memory_data_dir=str(data_dir),
        allowed_users="",
        log_level="DEBUG",
    )
    _cfg._settings_instance = settings

    engine = await init_db(db_url, echo=False)
    await run_migrations(engine)

    async with get_session() as session:
        chat = Chat(
            id=CHAT_ID,
            user_id=USER_ID,
            bot_id="console",
            llm_model="test-model",
            system_prompt="You are a test assistant.",
            prompt_name="default",
            response_template=template,
        )
        session.add(chat)
        await session.commit()

    output_buf = io.StringIO()
    messenger = ConsoleMessenger(output=output_buf, stream_debug=stream_debug)

    handler = BotHandler(
        messenger,
        llm,
        memory_data_dir=settings.memory_data_dir,
        test_mode=True,
    )
    return handler, messenger, output_buf


async def _teardown():
    from mai_gram.config import reset_settings
    from mai_gram.db import close_db, reset_db_state

    await close_db()
    reset_db_state()
    reset_settings()


async def _send_and_collect(messenger, output_buf, text="Hello"):
    escaped: BaseException | None = None
    try:
        await messenger.dispatch_text(
            chat_id=CHAT_ID,
            user_id=USER_ID,
            text=text,
        )
    except Exception as exc:
        escaped = exc
    messenger.flush_edits()
    return output_buf.getvalue(), escaped


# ──────────────────────────────────────────────────────────────────
# XML template streaming
# ──────────────────────────────────────────────────────────────────


async def test_xml_template_stream_hides_tags_from_user(tmp_path) -> None:
    """With the XML template, the final output should not contain raw XML tags."""
    full_text = (
        "<thought>\nThe user wants a simple answer.\n</thought>\n"
        "<content>\nThe answer is 42.\n</content>"
    )
    llm = _StreamingLLM(full_text, chunk_size=15)
    _handler, messenger, output_buf = await _build_handler(tmp_path, llm, template="xml")

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"
        assert "42" in output, f"Expected '42' in output, got:\n{output}"
        last_message = output.strip().split("\n")[-1] if output.strip() else ""
        assert "<thought>" not in last_message
        assert "</thought>" not in last_message
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# JSON template streaming
# ──────────────────────────────────────────────────────────────────


async def test_json_template_stream_produces_response(tmp_path) -> None:
    """With the JSON template, the final output should contain the content."""
    full_text = '{"thought": "Quick math check.", "content": "2 + 2 = 4."}'
    llm = _StreamingLLM(full_text, chunk_size=10)
    _handler, messenger, output_buf = await _build_handler(tmp_path, llm, template="json")

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"
        assert "2 + 2 = 4" in output or "4" in output, f"Expected content in output, got:\n{output}"
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Markdown headers template streaming
# ──────────────────────────────────────────────────────────────────


async def test_markdown_headers_stream_produces_response(tmp_path) -> None:
    """With the markdown headers template, the final output should be clean."""
    full_text = "## Thought\nReasoning about math.\n\n## Content\nThe answer is 7."
    llm = _StreamingLLM(full_text, chunk_size=12)
    _handler, messenger, output_buf = await _build_handler(
        tmp_path, llm, template="markdown_headers"
    )

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"
        assert "7" in output, f"Expected '7' in output, got:\n{output}"
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Empty template (regression check)
# ──────────────────────────────────────────────────────────────────


async def test_empty_template_stream_preserves_raw_output(tmp_path) -> None:
    """With the empty template, output should be passed through unchanged."""
    full_text = "The quick brown fox jumps over the lazy dog."
    llm = _StreamingLLM(full_text, chunk_size=10)
    _handler, messenger, output_buf = await _build_handler(tmp_path, llm, template="empty")

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"
        assert "quick brown fox" in output, f"Expected content in output, got:\n{output}"
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# XML prefill template streaming
# ──────────────────────────────────────────────────────────────────


async def test_xml_prefill_stream_works(tmp_path) -> None:
    """With XML prefill, response starts mid-tag but final output should be clean."""
    response_after_prefill = (
        "\nThinking about this.\n</thought>\n<content>\nHere is the answer.\n</content>"
    )
    llm = _StreamingLLM(response_after_prefill, chunk_size=15)
    _handler, messenger, output_buf = await _build_handler(tmp_path, llm, template="xml_prefill")

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"
        assert "answer" in output.lower(), f"Expected 'answer' in output, got:\n{output}"
    finally:
        await _teardown()


async def test_xml_prefill_stream_thought_in_blockquote(tmp_path) -> None:
    """With XML prefill, the thought field must be rendered in a blockquote during streaming.

    The prefill primes the LLM with ``<thought>``, so the streamed chunks
    start *inside* the thought tag (no opening ``<thought>`` in the stream).
    The streaming parser must still recognise the thought field and render
    it inside an HTML ``<blockquote>`` — just like the non-prefill XML
    template does.

    Uses ``stream_debug=True`` so every intermediate streaming edit is
    captured.  The thought section is deliberately long (>60 chars) to
    ensure at least one streaming edit fires while the response still
    contains the ``</thought>`` close tag.  We verify that the live edits
    do NOT expose raw ``</thought>`` tag text to the user.
    """
    long_reasoning = (
        "Let me carefully think through this problem step by step. "
        "The user wants a simple greeting, so I should respond politely."
    )
    response_after_prefill = (
        f"\n{long_reasoning}\n</thought>\n<content>\nHello there, nice to meet you!\n</content>"
    )
    llm = _StreamingLLM(response_after_prefill, chunk_size=8)
    _handler, messenger, output_buf = await _build_handler(
        tmp_path,
        llm,
        template="xml_prefill",
        stream_debug=True,
    )

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"
        assert "Hello" in output, f"Expected 'Hello' in output, got:\n{output}"

        streaming_edits = [
            section
            for section in output.split("--- Edited AI Response")
            if "replaces" in section and "final edit" not in section
        ]
        assert streaming_edits, (
            f"Expected at least one intermediate streaming edit, got none. Full output:\n{output}"
        )
        for edit in streaming_edits:
            assert "&lt;/thought&gt;" not in edit, (
                "Raw </thought> close tag leaked into a streaming edit "
                f"(should be inside a blockquote). Edit:\n{edit}"
            )
            assert "&lt;content&gt;" not in edit, (
                "Raw <content> open tag leaked into a streaming edit "
                f"(should not be visible). Edit:\n{edit}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: streaming display alternates between parsed and raw thought
# ──────────────────────────────────────────────────────────────────


async def test_xml_stream_thought_always_in_blockquote(tmp_path) -> None:
    """The thought field must ALWAYS be rendered inside a blockquote during streaming.

    Bug: while the thought field is actively streaming (before ``</thought>``
    arrives), ``_render_template_live_text`` shows the thought content as
    plain body text (no blockquote wrapper).  Once ``</thought>`` arrives in
    a later chunk, the thought suddenly appears inside a ``<blockquote>``.
    The user sees the display *alternating* between two layouts:

    Edit N:   raw thought text as body (no blockquote)
    Edit N+1: thought in blockquote + content as body

    This test captures every intermediate edit via ``stream_debug=True``.
    Any edit that contains recognisable thought text (fragments like
    "think carefully", "factual answer") MUST have that text inside a
    ``<blockquote>`` -- never as bare body text.
    """
    full_text = (
        "<thought>\n"
        "Let me think carefully about this problem step by step. "
        "The user wants a simple factual answer, I should be concise.\n"
        "</thought>\n"
        "<content>\nThe answer is 42.\n</content>"
    )
    llm = _StreamingLLM(full_text, chunk_size=8)
    _handler, messenger, output_buf = await _build_handler(
        tmp_path,
        llm,
        template="xml",
        stream_debug=True,
    )

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"

        streaming_edits = [
            section
            for section in output.split("--- Edited AI Response")
            if "replaces" in section and "final edit" not in section
        ]
        assert streaming_edits, (
            f"Expected at least one intermediate streaming edit, got none. Full output:\n{output}"
        )

        thought_fragments = ["think carefully", "factual answer", "step by step"]
        for i, edit in enumerate(streaming_edits):
            contains_thought = any(frag in edit.lower() for frag in thought_fragments)
            if not contains_thought:
                continue
            assert "<blockquote" in edit, (
                f"Streaming edit #{i + 1} shows thought content as bare body text "
                f"(no blockquote). The thought field must always be wrapped in a "
                f"<blockquote> during streaming, even while it is still actively "
                f"being generated. Edit:\n{edit}"
            )
    finally:
        await _teardown()


async def test_xml_stream_display_no_layout_flip(tmp_path) -> None:
    """The layout must not flip between 'thought as body' and 'thought as blockquote'.

    Bug: early streaming edits show the thought content as the main body text
    (because ``active_field=thought`` makes it the ``active_text``).  Once
    the ``</thought>`` close tag arrives, the thought moves into a blockquote
    header and the content field becomes the body.  The user sees a jarring
    layout change: the text they were reading suddenly jumps into a collapsed
    blockquote and different text replaces it in the body area.

    This test verifies that the rendering approach is consistent: if thought
    content appears as body text in one edit, it must NOT later appear inside
    a blockquote (or vice versa).
    """
    full_text = (
        "<thought>\n"
        "Analyzing the user's question about mathematics. "
        "This requires careful step-by-step reasoning to arrive at the correct answer.\n"
        "</thought>\n"
        "<content>\n2 + 2 = 4.\n</content>"
    )
    llm = _StreamingLLM(full_text, chunk_size=8)
    _handler, messenger, output_buf = await _build_handler(
        tmp_path,
        llm,
        template="xml",
        stream_debug=True,
    )

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"

        streaming_edits = [
            section
            for section in output.split("--- Edited AI Response")
            if "replaces" in section and "final edit" not in section
        ]
        assert streaming_edits, (
            f"Expected intermediate streaming edits, got none. Full output:\n{output}"
        )

        thought_fragments = ["analyzing", "mathematics", "step-by-step reasoning"]
        thought_in_body: list[int] = []
        thought_in_blockquote: list[int] = []

        for i, edit in enumerate(streaming_edits):
            contains_thought = any(frag in edit.lower() for frag in thought_fragments)
            if not contains_thought:
                continue
            if "<blockquote" in edit:
                thought_in_blockquote.append(i + 1)
            else:
                thought_in_body.append(i + 1)

        if thought_in_body and thought_in_blockquote:
            pytest.fail(
                f"Streaming display alternates between two layouts for the thought. "
                f"Edits {thought_in_body} show thought as body text; "
                f"edits {thought_in_blockquote} show it in a blockquote. "
                f"The user sees a jarring layout flip. Full output:\n{output}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: thought blockquote is expandable (collapsed) during streaming
# ──────────────────────────────────────────────────────────────────


def _intermediate_streaming_edits(output: str) -> list[str]:
    """Extract only intermediate streaming edits, excluding the final turn-completion edit.

    The final edit contains usage footer (e.g. "10/20 tokens") and action
    buttons ("Regenerate"), so we exclude any section that has those markers.
    """
    return [
        section
        for section in output.split("--- Edited AI Response")
        if "replaces" in section
        and "final edit" not in section
        and "tokens" not in section
        and "Regenerate" not in section
    ]


async def test_xml_stream_thought_blockquote_not_expandable(tmp_path) -> None:
    """During streaming, the thought blockquote must NOT be expandable (collapsed).

    Bug: ``_build_template_header`` renders the thought field with
    ``expandable=descriptor.expandable``.  The XML template sets
    ``expandable=True`` on the thought field descriptor, so the streaming
    display wraps it in ``<blockquote expandable>``.  Telegram renders this
    as a collapsed block requiring a tap to expand, making the thought
    completely unreadable during live streaming (and it re-collapses on
    every edit).

    During streaming, the blockquote should use plain ``<blockquote>``
    (``expandable=False``) so the user can read the reasoning as it arrives.
    The ``expandable=True`` style is appropriate only for the final message.
    """
    full_text = (
        "<thought>\n"
        "Let me carefully analyze this question in great detail. "
        "I need to consider multiple angles and perspectives before answering. "
        "The user seems to want a thorough explanation.\n"
        "</thought>\n"
        "<content>\nHere is a detailed answer for you.\n</content>"
    )
    llm = _StreamingLLM(full_text, chunk_size=8)
    _handler, messenger, output_buf = await _build_handler(
        tmp_path,
        llm,
        template="xml",
        stream_debug=True,
    )

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"

        streaming_edits = _intermediate_streaming_edits(output)
        assert streaming_edits, (
            f"Expected intermediate streaming edits, got none. Full output:\n{output}"
        )

        edits_with_blockquote = [
            (i, edit) for i, edit in enumerate(streaming_edits) if "<blockquote" in edit
        ]
        assert edits_with_blockquote, (
            f"Expected at least one streaming edit with a blockquote, got none. "
            f"Full output:\n{output}"
        )

        for i, edit in edits_with_blockquote:
            assert "<blockquote expandable>" not in edit, (
                f"Streaming edit #{i + 1} uses <blockquote expandable> for the "
                f"thought block. During streaming this makes the thought "
                f"unreadable (collapsed in Telegram, re-collapses on every edit). "
                f"Should use plain <blockquote> during streaming. Edit:\n{edit}"
            )
    finally:
        await _teardown()


async def test_xml_prefill_stream_thought_blockquote_not_expandable(tmp_path) -> None:
    """Same expandable bug as above, but for the prefill variant.

    With prefill, the thought field completes faster (the ``<thought>`` open
    tag is in the prefill), so the blockquote appears earlier in streaming.
    Verify it uses plain ``<blockquote>`` rather than ``<blockquote expandable>``.
    """
    long_reasoning = (
        "Let me carefully think through this problem step by step. "
        "The user wants a simple greeting, so I should respond politely. "
        "I need to consider the context and provide a warm welcome."
    )
    response_after_prefill = f"\n{long_reasoning}\n</thought>\n<content>\nHello there!\n</content>"
    llm = _StreamingLLM(response_after_prefill, chunk_size=8)
    _handler, messenger, output_buf = await _build_handler(
        tmp_path,
        llm,
        template="xml_prefill",
        stream_debug=True,
    )

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, f"Unexpected exception: {escaped}"

        streaming_edits = _intermediate_streaming_edits(output)
        assert streaming_edits, (
            f"Expected intermediate streaming edits, got none. Full output:\n{output}"
        )

        edits_with_blockquote = [
            (i, edit) for i, edit in enumerate(streaming_edits) if "<blockquote" in edit
        ]
        assert edits_with_blockquote, (
            f"Expected at least one streaming edit with a blockquote, got none. "
            f"Full output:\n{output}"
        )

        for i, edit in edits_with_blockquote:
            assert "<blockquote expandable>" not in edit, (
                f"Streaming edit #{i + 1} uses <blockquote expandable> for the "
                f"thought block in prefill mode. During streaming this makes the "
                f"thought unreadable (collapsed). Should use plain <blockquote>. "
                f"Edit:\n{edit}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Edge case: incomplete XML tags gracefully handled
# ──────────────────────────────────────────────────────────────────


async def test_xml_template_handles_malformed_gracefully(tmp_path) -> None:
    """A malformed XML response should still be handled without crashing."""
    full_text = "<thought>reasoning here</thought><content>The answer is 5."
    llm = _StreamingLLM(full_text, chunk_size=15)
    _handler, messenger, output_buf = await _build_handler(tmp_path, llm, template="xml")

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        if escaped is not None:
            assert "format error" in str(escaped).lower() or "empty" in str(escaped).lower()
        else:
            assert output.strip()
    finally:
        await _teardown()
