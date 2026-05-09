"""Integration tests for the streaming layout flip bug.

Reproduces the bug where Telegram rejects complex HTML during streaming.
The fix strategy: when an HTML ``edit_message`` call fails, the system
**skips the update entirely** and keeps the last successfully rendered
message visible.  The next streaming tick retries with fresh HTML.

This means:
- No fallback text is ever sent for existing messages (only for initial send).
- No blank/cursor-only messages appear.
- No raw XML tags are ever shown to the user.
- The display never oscillates between blockquote and degraded states.

ConsoleMessenger always succeeds, so the rejection is never triggered in
normal tests.  These tests use a FailingHtmlMessenger that simulates
Telegram rejection to verify the skip-on-failure behaviour.
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
    complex/malformed HTML.

    After the fix, the caller should **skip the update** (keep the
    previous message) rather than sending a fallback.  The
    ``fallback_edits`` list tracks any non-HTML edits that happen
    after a rejection — it should remain empty if the fix works.
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


async def test_rejected_html_edit_skips_update(tmp_path) -> None:
    """When Telegram rejects an HTML edit, no fallback is sent.

    The system must keep the last successfully rendered message visible
    instead of replacing it with a degraded fallback.  This eliminates
    all flicker — the user either sees the latest successful render or
    a newer one, never a blank or raw-text message.
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

        assert not messenger.fallback_edits, (
            f"Expected zero fallback edits (rejected HTML edits should be skipped), "
            f"but {len(messenger.fallback_edits)} fallback(s) were sent:\n"
            + "\n---\n".join(messenger.fallback_edits[:3])
        )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: layout oscillates between blockquote and raw text
# ──────────────────────────────────────────────────────────────────


async def test_no_oscillation_with_alternating_failures(tmp_path) -> None:
    """Alternating HTML failures must not produce visible display changes.

    Simulates every-other HTML edit failing (fail_every_n=2).  Since
    rejected edits are now skipped entirely, the user should only see
    the successful HTML edits — no degraded fallback messages, no
    oscillation between blockquote and raw text.
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

        assert messenger.rejected_edits, "Expected at least one rejected HTML edit."

        assert not messenger.fallback_edits, (
            f"Alternating failures produced {len(messenger.fallback_edits)} fallback "
            f"edit(s), but rejected edits should be silently skipped:\n"
            + "\n---\n".join(messenger.fallback_edits[:3])
        )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: long thought pushes message over Telegram 4096 char limit
# ──────────────────────────────────────────────────────────────────


async def test_long_thought_rejection_skips_silently(tmp_path) -> None:
    """A very long thought that approaches the 4096-char Telegram limit.

    When the thought is long, the HTML overhead (blockquote tags, label,
    markdown-rendered formatting) can push the total message length over
    Telegram's 4096-char hard limit, causing rejection.  The system must
    silently skip the failed edit, preserving the last successful render.
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

        assert not messenger.fallback_edits, (
            f"Long thought rejection produced {len(messenger.fallback_edits)} "
            f"fallback edit(s) instead of skipping:\n"
            + "\n---\n".join(messenger.fallback_edits[:3])
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
# Bug: blank cursor-only message flicker (the "emptiness" bug)
# ──────────────────────────────────────────────────────────────────


async def test_cursor_only_blank_message_never_sent(tmp_path) -> None:
    """The user must never see a blank message with only the typing cursor.

    When the thought field is active, ``remaining`` is empty and
    ``fallback_source`` may also be empty (or the same as thought text).
    Before the skip-on-failure fix, this produced fallback messages like
    `` ▍`` (just the cursor character) — visible as a blank message flicker.
    """
    thought_text = (
        "Let me think carefully about this problem step by step. "
        "The user wants a simple factual answer, I should be concise. "
        "I need to consider multiple angles and perspectives. "
        "Let me also think about edge cases and potential misunderstandings."
    )
    full_text = f"<thought>\n{thought_text}\n</thought>\n<content>\nThe answer is 42.\n</content>"
    llm = _StreamingLLM(full_text, chunk_size=8)
    output_buf = io.StringIO()
    messenger = FailingHtmlMessenger(
        output=output_buf,
        stream_debug=True,
        fail_above_len=80,
    )
    _handler, _ = await _build_handler(tmp_path, llm, template="xml", messenger=messenger)

    try:
        exc = await _send_and_collect(messenger)
        assert exc is None, f"Unexpected exception: {exc}"

        assert messenger.rejected_edits, "Expected HTML edit rejections."

        for i, fallback_text in enumerate(messenger.fallback_edits):
            stripped = fallback_text.replace("▍", "").strip()
            assert stripped, (
                f"Fallback edit #{i + 1} is blank (cursor-only): {fallback_text!r}\n"
                "The user sees an empty message flicker."
            )

        assert not messenger.fallback_edits, (
            f"Expected zero fallback edits, got {len(messenger.fallback_edits)}."
        )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Verify: all displayed edits contain valid HTML (no degraded output)
# ──────────────────────────────────────────────────────────────────


async def test_all_displayed_edits_are_valid_html(tmp_path) -> None:
    """Every edit the user sees must be properly formatted HTML.

    When some HTML edits are rejected, the output stream should only
    contain successfully rendered edits — never plain-text fallbacks
    or partial renders.
    """
    thought_text = (
        "Let me carefully analyze this question in great detail. "
        "I need to consider multiple angles and perspectives before answering. "
        "The user seems to want a thorough explanation of the topic."
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

        for i, edit in enumerate(edits):
            raw_tags = _contains_raw_xml(edit)
            assert not raw_tags, (
                f"Displayed edit #{i + 1} contains raw XML tags: {raw_tags}\nFull edit:\n{edit}"
            )
    finally:
        await _teardown()


# ──────────────────────────────────────────────────────────────────
# Bug: prefill variant has the same fallback-exposes-raw-XML issue
# ──────────────────────────────────────────────────────────────────


async def test_prefill_rejection_skips_update(tmp_path) -> None:
    """Prefill variant: rejected HTML edits must be skipped, not replaced.

    With prefill, the ``<thought>`` open tag is in the assistant prefill
    (not in the streamed content).  Regardless, when an HTML edit is
    rejected, the system must skip the update entirely — keeping the
    last successful render visible.
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

        assert not messenger.fallback_edits, (
            f"Prefill rejection produced {len(messenger.fallback_edits)} "
            f"fallback edit(s) instead of skipping:\n"
            + "\n---\n".join(messenger.fallback_edits[:3])
        )
    finally:
        await _teardown()
