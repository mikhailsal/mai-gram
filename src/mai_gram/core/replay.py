"""Rate-limited message replay engine for imported conversations.

Sends imported messages to a Telegram chat with proper throttling
to stay within Telegram Bot API rate limits.  Handles message splitting
for content that exceeds Telegram's 4096-char limit, automatic retry
with backoff on flood-control errors, and strict ordering guarantees.
"""

from __future__ import annotations

import asyncio
import html as _html
import logging
from typing import TYPE_CHECKING, Any

from mai_gram.core.telegram_limits import (
    MAX_CONTENT_LENGTH_FOR_TRUNCATION,
    SAFE_MAX_LENGTH,
    TELEGRAM_MAX_LENGTH,
    split_html_safe,
)
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.db.models import Message
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)

DELAY_SECONDS = 1.5
PROGRESS_INTERVAL = 25
MAX_SEND_RETRIES = 10
FLOOD_EXTRA_BUFFER = 5


def _format_user_message(content: str) -> list[str]:
    """Format a user message, splitting if necessary."""
    chunks = split_html_safe(content, MAX_CONTENT_LENGTH_FOR_TRUNCATION)
    result: list[str] = []
    for i, chunk in enumerate(chunks):
        escaped = _html.escape(chunk)
        if i == 0:
            result.append(f"\U0001f464 <b>[You]</b>\n<i>{escaped}</i>")
        else:
            result.append(f"<i>{escaped}</i>")
    return result


def _format_assistant_message(content: str, reasoning: str | None = None) -> list[str]:
    """Format an assistant message, splitting if necessary."""
    from mai_gram.core.md_to_telegram import markdown_to_html

    header_parts: list[str] = ["\U0001f916 <b>[AI]</b>\n"]

    if reasoning and reasoning.strip():
        reasoning_truncated = reasoning[:2000]
        if len(reasoning) > 2000:
            reasoning_truncated += "..."
        escaped_reasoning = _html.escape(reasoning_truncated.strip())
        header_parts.append(
            f"<blockquote expandable>\U0001f4ad Reasoning\n{escaped_reasoning}</blockquote>\n\n"
        )

    header = "".join(header_parts)

    if not content.strip():
        return [header.rstrip()]

    content_chunks = split_html_safe(content, MAX_CONTENT_LENGTH_FOR_TRUNCATION)
    result: list[str] = []
    for i, chunk in enumerate(content_chunks):
        html_chunk = markdown_to_html(chunk)
        if i == 0:
            result.append(header + html_chunk)
        else:
            result.append(html_chunk)

    if result and len(result[0]) > TELEGRAM_MAX_LENGTH:
        first = result[0]
        result = [header.rstrip(), markdown_to_html(content_chunks[0])] + result[1:]
        if len(result[0]) > TELEGRAM_MAX_LENGTH:
            result[0] = result[0][:SAFE_MAX_LENGTH] + "..."
        logger.debug("First chunk was %d chars after HTML; split header from body", len(first))

    return result


def _format_tool_message(content: str, tool_call_id: str | None = None) -> str:
    """Format a tool result message for display (compact)."""
    preview = content[:200]
    if len(content) > 200:
        preview += "..."
    escaped = _html.escape(preview)
    label = f"tool:{tool_call_id}" if tool_call_id else "tool"
    return f"\U0001f527 <i>[{label}]</i> {escaped}"


def _build_cut_keyboard(db_message_id: int) -> list[list[tuple[str, str]]]:
    """Build an inline keyboard with a Cut button for an assistant message."""
    return [[("\u2702 Cut this & above", f"cut:{db_message_id}")]]


async def _send_with_retry(
    messenger: Messenger,
    msg: OutgoingMessage,
    *,
    max_retries: int = MAX_SEND_RETRIES,
    base_delay: float = DELAY_SECONDS,
) -> bool:
    """Send a message with unlimited flood-control retry to guarantee ordering.

    Returns True if the message was delivered, False only on permanent failure.
    """
    for attempt in range(1, max_retries + 1):
        result = await messenger.send_message(msg)
        if result.success:
            return True

        error = (result.error or "").lower()

        if "too long" in error or "message is too long" in error:
            logger.error("Message too long even after splitting (%d chars)", len(msg.text))
            if len(msg.text) > SAFE_MAX_LENGTH:
                msg = OutgoingMessage(
                    text=msg.text[:SAFE_MAX_LENGTH] + "...",
                    chat_id=msg.chat_id,
                    parse_mode=msg.parse_mode,
                    keyboard=msg.keyboard,
                )
                continue
            return False

        if "flood control" in error or "too many requests" in error or "429" in error:
            import re

            match = re.search(r"retry in (\d+)", error)
            wait = int(match.group(1)) + FLOOD_EXTRA_BUFFER if match else 30
            logger.warning(
                "Flood control on replay (attempt %d/%d): waiting %ds",
                attempt,
                max_retries,
                wait,
            )
            await asyncio.sleep(wait)
            continue

        if "timed out" in error or "network" in error:
            wait = min(2**attempt, 60)
            logger.warning(
                "Transient error on replay (attempt %d/%d): %s - waiting %ds",
                attempt,
                max_retries,
                result.error,
                wait,
            )
            await asyncio.sleep(wait)
            continue

        logger.error("Permanent send failure: %s", result.error)
        return False

    logger.error("Exhausted %d retries for message send", max_retries)
    return False


async def replay_imported_messages(
    messenger: Messenger,
    tg_chat_id: str,
    messages: list[Message],
    *,
    delay_seconds: float = DELAY_SECONDS,
    progress_interval: int = PROGRESS_INTERVAL,
    show_tool_calls: bool = False,
) -> int:
    """Replay imported messages to a Telegram chat with rate limiting.

    Sends each message with appropriate formatting and delays between sends.
    Messages exceeding Telegram's limit are split into multiple parts.
    Flood control errors trigger automatic retry with backoff.
    Assistant messages get a "Cut this & above" button (on last part).

    Returns the number of messages actually sent to Telegram.
    """
    total = len(messages)
    if total == 0:
        return 0

    sent_count = 0
    displayable_count = sum(
        1
        for m in messages
        if m.role in ("user", "assistant") or (m.role == "tool" and show_tool_calls)
    )

    est_minutes = int(displayable_count * delay_seconds / 60) + 1
    await _send_with_retry(
        messenger,
        OutgoingMessage(
            text=(
                f"\U0001f4e5 Replaying {displayable_count} messages from imported history...\n"
                f"This will take about {est_minutes} minute(s)."
            ),
            chat_id=tg_chat_id,
        ),
    )
    await asyncio.sleep(delay_seconds)

    for msg in messages:
        if msg.role == "system":
            continue

        if msg.role == "user":
            parts = _format_user_message(msg.content)
            for part in parts:
                ok = await _send_with_retry(
                    messenger,
                    OutgoingMessage(
                        text=part,
                        chat_id=tg_chat_id,
                        parse_mode="html",
                    ),
                )
                if ok and part is parts[-1]:
                    sent_count += 1
                await asyncio.sleep(delay_seconds)

        elif msg.role == "assistant":
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None)
            has_tool_calls = bool(msg.tool_calls)

            if not content.strip() and has_tool_calls:
                if show_tool_calls:
                    text = "\U0001f916 <i>[AI tool call]</i>"
                    kb: Any = _build_cut_keyboard(msg.id)
                    ok = await _send_with_retry(
                        messenger,
                        OutgoingMessage(
                            text=text,
                            chat_id=tg_chat_id,
                            parse_mode="html",
                            keyboard=kb,
                        ),
                    )
                    if ok:
                        sent_count += 1
                    await asyncio.sleep(delay_seconds)
                continue

            parts = _format_assistant_message(content, reasoning)
            kb = _build_cut_keyboard(msg.id)
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1
                ok = await _send_with_retry(
                    messenger,
                    OutgoingMessage(
                        text=part,
                        chat_id=tg_chat_id,
                        parse_mode="html",
                        keyboard=kb if is_last else None,
                    ),
                )
                if ok and is_last:
                    sent_count += 1
                await asyncio.sleep(delay_seconds)

        elif msg.role == "tool":
            if not show_tool_calls:
                continue
            text = _format_tool_message(msg.content, msg.tool_call_id)
            ok = await _send_with_retry(
                messenger,
                OutgoingMessage(
                    text=text,
                    chat_id=tg_chat_id,
                    parse_mode="html",
                ),
            )
            if ok:
                sent_count += 1
            await asyncio.sleep(delay_seconds)
        else:
            continue

        if sent_count > 0 and sent_count % progress_interval == 0:
            await _send_with_retry(
                messenger,
                OutgoingMessage(
                    text=f"\u23f3 Replay progress: {sent_count}/{displayable_count} messages...",
                    chat_id=tg_chat_id,
                ),
            )
            await asyncio.sleep(delay_seconds)

    last_assistant_id: int | None = None
    for msg in reversed(messages):
        if msg.role == "assistant" and (msg.content or "").strip():
            last_assistant_id = msg.id
            break

    summary_parts = [
        f"\u2705 Import complete! {sent_count} messages replayed.",
    ]
    if last_assistant_id is not None:
        summary_parts.append(
            'Use the "\u2702 Cut this & above" button on any AI message to trim history.'
        )
    summary_parts.append("Send a message to continue the conversation.")

    await _send_with_retry(
        messenger,
        OutgoingMessage(
            text="\n".join(summary_parts),
            chat_id=tg_chat_id,
        ),
    )

    return sent_count
