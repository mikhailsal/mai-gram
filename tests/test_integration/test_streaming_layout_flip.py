"""Integration tests for the streaming layout flip bug.

Reproduces the bug where Telegram rejects complex HTML during streaming,
causing the fallback path to send raw LLM output (with visible XML tags)
to the user.  The display alternates between correctly parsed blockquote
and raw text on consecutive edits.

The core mechanism:
1. As thought content grows, the HTML sent to Telegram becomes complex
   enough that Telegram's strict HTML parser rejects the edit_message call.
2. The fallback in _edit_existing sends ``current_content`` (raw LLM output
   including literal <thought>, </thought>, <content> tags) WITHOUT
   parse_mode="html", so XML tags render as visible text.
3. The next edit may succeed (the HTML is valid again), producing the
   alternation.

ConsoleMessenger always succeeds, so the fallback is never triggered in
normal tests.  These tests use a FailingHtmlMessenger that simulates
Telegram rejection to exercise the fallback path.
"""

from __future__ import annotations

import io
import re
from typing import Any

import pytest

from mai_gram.llm.provider import LLMProvider, LLMResponse, StreamChunk, TokenUsage
from mai_gram.messenger.base import SendResult
from mai_gram.messenger.console import ConsoleMessenger

pytestmark = pytest.mark.integration

CHAT_ID = "test-layout-flip"
USER_ID = "test-user-layout-flip"


def _make_chunks(text: str, *, chunk_size: int = 20) -> list[str]:
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


class FailingHtmlMessenger(ConsoleMessenger):
    """ConsoleMessenger that simulates Telegram HTML rejection.

    When ``parse_mode="html"`` is passed to ``edit_message`` and the
    text exceeds ``fail_above_len`` characters, the edit returns
    ``success=False`` — just like Telegram would when it rejects
    complex/malformed HTML.  The caller then falls back to sending
    the fallback text (without parse_mode), which is where the bug
    manifests.
    """

    def __init__(
        self,
        *,
        output: io.StringIO,
        stream_debug: bool = True,
        fail_above_len: int = 200,
        fail_every_n: int = 1,
    ) -> None:
        super().__init__(output=output, stream_debug=stream_debug)
        self._fail_above_len = fail_above_len
        self._fail_every_n = fail_every_n
        self._html_edit_count = 0
        self.rejected_edits: list[str] = []
        self.fallback_edits: list[str] = []

    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        parse_mode = kwargs.get("parse_mode")

        if parse_mode == "html" and len(new_text) > self._fail_above_len:
            self._html_edit_count += 1
            if self._html_edit_count % self._fail_every_n == 0:
                self.rejected_edits.append(new_text)
                return SendResult(success=False, error="Simulated Telegram HTML rejection")

        if parse_mode is None and self._html_edit_count > 0:
            self.fallback_edits.append(new_text)

        return await super().edit_message(chat_id, message_id, new_text, **kwargs)


async def _build_handler(
    tmp_path,
    llm,
    *,
    template: str = "xml",
    messenger: ConsoleMessenger | None = None,
):
    """Build a BotHandler with a live DB and custom messenger."""
    import mai_gram.config as _cfg
    from mai_gram.bot.handler import BotHandler
    from mai_gram.config import Settings, reset_settings
    from mai_gram.db import init_db, reset_db_state, run_migrations
    from mai_gram.db.database import get_session
    from mai_gram.db.models import Chat

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

    if messenger is None:
        messenger = ConsoleMessenger(output=io.StringIO(), stream_debug=True)

    handler = BotHandler(
        messenger,
        llm,
        memory_data_dir=settings.memory_data_dir,
        test_mode=True,
    )
    return handler, messenger


async def _teardown():
    from mai_gram.config import reset_settings
    from mai_gram.db import close_db, reset_db_state

    await close_db()
    reset_db_state()
    reset_settings()


async def _send_and_collect(messenger, text="Hello"):
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
    return escaped


_RAW_XML_TAGS = re.compile(r"</?(?:thought|content|feelings|think|scratchpad|reasoning)\b[^>]*>")


def _contains_raw_xml(text: str) -> list[str]:
    """Find raw XML template tags in text that should never be shown to user."""
    return _RAW_XML_TAGS.findall(text)


# ──────────────────────────────────────────────────────────────────
# Core bug: fallback exposes raw XML when HTML edit is rejected
# ──────────────────────────────────────────────────────────────────


async def test_fallback_never_contains_raw_xml_tags(tmp_path) -> None:
    """When Telegram rejects an HTML edit, the fallback must not expose raw XML.

    This is the primary reproduction of the streaming layout flip bug.
    When ``_edit_existing`` gets ``success=False`` from the HTML edit, it
    falls back to sending ``(remaining or current_content)[:max_len]``
    without parse_mode.  Since ``remaining`` is empty for non-content
    active fields (thought), this degrades to ``current_content`` — the
    raw LLM output with literal XML tags.

    The user sees: ``<thought>The user wants something...</thought>``
    as plain text in the chat.  This must never happen.
    """
    thought_text = (
        "Let me think carefully about this problem step by step. "
        "The user wants a simple factual answer, I should be concise. "
        "I need to consider multiple angles and perspectives before answering. "
        "Let me also think about edge cases and potential misunderstandings. "
        "This requires thorough analysis of the question at hand."
    )
    full_text = f"<thought>\n{thought_text}\n</thought>\n<content>\nThe answer is 42.\n</content>"
    llm = _StreamingLLM(full_text, chunk_size=8)
    output_buf = io.StringIO()
    messenger = FailingHtmlMessenger(
        output=output_buf,
        stream_debug=True,
        fail_above_len=100,
    )
    _handler, _ = await _build_handler(tmp_path, llm, template="xml", messenger=messenger)

    try:
        exc = await _send_and_collect(messenger)
        assert exc is None, f"Unexpected exception: {exc}"

        assert messenger.rejected_edits, (
            "Expected at least one HTML edit to be rejected (simulating Telegram), "
            "but none were. The FailingHtmlMessenger threshold may be too high."
        )

        for i, fallback_text in enumerate(messenger.fallback_edits):
            raw_tags = _contains_raw_xml(fallback_text)
            assert not raw_tags, (
                f"Fallback edit #{i + 1} contains raw XML tags that are visible "
                f"to the user: {raw_tags}\n"
                f"Full fallback text:\n{fallback_text}\n\n"
                "The fallback path must strip or re-render XML structure. "
                "Currently it sends raw current_content when remaining is empty."
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: layout oscillates between blockquote and raw text
# ──────────────────────────────────────────────────────────────────


async def test_no_oscillation_between_parsed_and_raw_display(tmp_path) -> None:
    """Display must not alternate between blockquote-rendered and raw-text states.

    The user sees: edit N is a nicely formatted blockquote, edit N+1 is
    raw ``<thought>...</thought>`` text, edit N+2 is blockquote again.
    This oscillation is caused by HTML edits alternately succeeding and
    failing, with the fallback showing raw LLM output.

    Simulates every-other HTML edit failing (fail_every_n=2) to trigger
    the alternation pattern.
    """
    thought_text = (
        "Let me carefully analyze this question in great detail. "
        "I need to consider multiple angles and perspectives before answering. "
        "The user seems to want a thorough explanation of the topic. "
        "I should also consider relevant examples and counterexamples. "
        "Let me structure my reasoning in a clear and logical way."
    )
    full_text = (
        f"<thought>\n{thought_text}\n</thought>\n"
        "<content>\nHere is a detailed answer for you.\n</content>"
    )
    llm = _StreamingLLM(full_text, chunk_size=8)
    output_buf = io.StringIO()
    messenger = FailingHtmlMessenger(
        output=output_buf,
        stream_debug=True,
        fail_above_len=100,
        fail_every_n=2,
    )
    _handler, _ = await _build_handler(tmp_path, llm, template="xml", messenger=messenger)

    try:
        exc = await _send_and_collect(messenger)
        assert exc is None, f"Unexpected exception: {exc}"

        output = output_buf.getvalue()
        edits = [
            section
            for section in output.split("--- Edited AI Response")
            if "replaces" in section and "final edit" not in section
        ]

        has_blockquote = []
        has_raw_xml = []
        for i, edit in enumerate(edits):
            if "<blockquote" in edit:
                has_blockquote.append(i)
            if _contains_raw_xml(edit):
                has_raw_xml.append(i)

        if has_blockquote and has_raw_xml:
            pytest.fail(
                f"Display oscillates between parsed and raw states. "
                f"Edits with blockquote: {has_blockquote}; "
                f"edits with raw XML tags: {has_raw_xml}. "
                f"The user sees the layout flipping back and forth.\n"
                f"Full output:\n{output}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: long thought pushes message over Telegram 4096 char limit
# ──────────────────────────────────────────────────────────────────


async def test_long_thought_fallback_is_clean(tmp_path) -> None:
    """A very long thought that approaches the 4096-char Telegram limit.

    When the thought is long, the HTML overhead (blockquote tags, label,
    markdown-rendered formatting) can push the total message length over
    Telegram's 4096-char hard limit, causing rejection.  The fallback
    must still present clean, readable text — not raw XML.
    """
    long_thought = ("I need to carefully analyze this complex mathematical problem. " * 30).strip()
    full_text = f"<thought>\n{long_thought}\n</thought>\n<content>\nThe answer is 42.\n</content>"
    llm = _StreamingLLM(full_text, chunk_size=8)
    output_buf = io.StringIO()
    messenger = FailingHtmlMessenger(
        output=output_buf,
        stream_debug=True,
        fail_above_len=150,
    )
    _handler, _ = await _build_handler(tmp_path, llm, template="xml", messenger=messenger)

    try:
        exc = await _send_and_collect(messenger)
        assert exc is None, f"Unexpected exception: {exc}"

        assert messenger.rejected_edits, (
            "Expected HTML edits to be rejected for long thought content."
        )

        for i, fallback_text in enumerate(messenger.fallback_edits):
            raw_tags = _contains_raw_xml(fallback_text)
            assert not raw_tags, (
                f"Long-thought fallback #{i + 1} exposes raw XML: {raw_tags}\n"
                f"Fallback text (first 500 chars):\n{fallback_text[:500]}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: markdown > quotes create nested <blockquote> (rejected by TG)
# ──────────────────────────────────────────────────────────────────


async def test_thought_with_markdown_quotes_no_nested_blockquote(tmp_path) -> None:
    """Thought content with ``> quoted`` markdown must not produce nested blockquotes.

    If the LLM writes ``> some text`` inside ``<thought>``, the
    ``markdown_to_html`` converter wraps it in ``<blockquote>``.  This
    sits inside the outer ``<blockquote>`` from the thought field
    rendering, creating nested ``<blockquote>`` tags.  Telegram may
    reject such nesting, triggering the fallback path.

    This test verifies that the streamed HTML never contains nested
    ``<blockquote>`` within ``<blockquote>``.
    """
    thought_with_quotes = (
        "The user's question raises several points:\n"
        "> First, we need to consider the mathematical foundations\n"
        "> Second, the practical implications are significant\n"
        "After considering these quoted perspectives, I believe the answer is clear."
    )
    full_text = (
        f"<thought>\n{thought_with_quotes}\n</thought>\n<content>\nThe answer is 42.\n</content>"
    )
    llm = _StreamingLLM(full_text, chunk_size=8)
    output_buf = io.StringIO()
    messenger = ConsoleMessenger(output=output_buf, stream_debug=True)
    _handler, _ = await _build_handler(tmp_path, llm, template="xml", messenger=messenger)

    try:
        exc = await _send_and_collect(messenger)
        assert exc is None, f"Unexpected exception: {exc}"

        output = output_buf.getvalue()
        edits = [
            section
            for section in output.split("--- Edited AI Response")
            if "replaces" in section and "final edit" not in section
        ]

        for i, edit in enumerate(edits):
            nested = re.findall(
                r"<blockquote[^>]*>.*?<blockquote[^>]*>",
                edit,
                re.DOTALL,
            )
            assert not nested, (
                f"Streaming edit #{i + 1} contains nested <blockquote> tags. "
                f"Telegram rejects nested blockquotes, which triggers the "
                f"fallback path and exposes raw XML to the user.\n"
                f"Nested match: {nested[0][:200]}\n"
                f"Full edit:\n{edit}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: prefill variant has the same fallback-exposes-raw-XML issue
# ──────────────────────────────────────────────────────────────────


async def test_prefill_fallback_never_contains_raw_xml_tags(tmp_path) -> None:
    """Prefill fallback must not expose raw XML tags.

    With prefill, the ``<thought>`` open tag is in the assistant prefill
    (not in the streamed content), so the opening tag won't appear.
    However, ``</thought>`` and ``<content>`` ARE in the streamed text.
    Before the fix, the fallback degraded to ``current_content`` (the raw
    stream) which could expose these close/open tags as visible text.

    After the fix, the fallback uses ``fallback_source`` which is the
    parsed field content (``result.active_content``) — clean reasoning
    text without XML tags.  This test verifies that none of the fallback
    edits contain raw XML template tags.
    """
    long_reasoning = (
        "Let me carefully think through this problem step by step. "
        "The user wants a simple greeting, so I should respond politely. "
        "I need to consider the context and provide a warm welcome. "
        "I should also think about cultural sensitivity and tone. "
        "Let me also consider if there are any hidden implications. "
        "Additionally, I should verify the greeting is appropriate."
    )
    response_after_prefill = (
        f"\n{long_reasoning}\n</thought>\n<content>\nHello there, nice to meet you!\n</content>"
    )
    llm = _StreamingLLM(response_after_prefill, chunk_size=8)
    output_buf = io.StringIO()
    messenger = FailingHtmlMessenger(
        output=output_buf,
        stream_debug=True,
        fail_above_len=100,
    )
    _handler, _ = await _build_handler(tmp_path, llm, template="xml_prefill", messenger=messenger)

    try:
        exc = await _send_and_collect(messenger)
        assert exc is None, f"Unexpected exception: {exc}"

        assert messenger.rejected_edits, (
            "Expected at least one HTML edit to be rejected for prefill variant."
        )

        for i, fallback_text in enumerate(messenger.fallback_edits):
            raw_tags = _contains_raw_xml(fallback_text)
            assert not raw_tags, (
                f"Prefill fallback #{i + 1} contains raw XML tags: {raw_tags}\n"
                f"Full fallback text:\n{fallback_text}\n\n"
                "The fallback must not expose template structure in prefill mode."
            )
    finally:
        await _teardown()
