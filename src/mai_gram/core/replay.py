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
from typing import TYPE_CHECKING

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


def _truncate_oversized_message(msg: OutgoingMessage, error: str) -> OutgoingMessage | None:
    """Return a truncated retry message when Telegram rejects the payload as too large."""
    if "too long" not in error and "message is too long" not in error:
        return None

    logger.error("Message too long even after splitting (%d chars)", len(msg.text))
    if len(msg.text) <= SAFE_MAX_LENGTH:
        return None

    return OutgoingMessage(
        text=msg.text[:SAFE_MAX_LENGTH] + "...",
        chat_id=msg.chat_id,
        parse_mode=msg.parse_mode,
        keyboard=msg.keyboard,
    )


def _retry_delay_for_send_error(error: str, attempt: int) -> tuple[int, str] | None:
    """Classify replay send failures into retryable delays."""
    if "flood control" in error or "too many requests" in error or "429" in error:
        import re

        match = re.search(r"retry in (\d+)", error)
        wait_seconds = int(match.group(1)) + FLOOD_EXTRA_BUFFER if match else 30
        return wait_seconds, "Flood control on replay"

    if "timed out" in error or "network" in error:
        return min(2**attempt, 60), "Transient error on replay"

    return None


async def _send_replay_message(
    messenger: Messenger,
    msg: OutgoingMessage,
    *,
    delay_seconds: float,
    parse_mode: str | None = None,
    keyboard: list[list[tuple[str, str]]] | None = None,
) -> bool:
    """Send a replay message and apply the standard post-send pacing delay."""
    ok = await _send_with_retry(
        messenger,
        OutgoingMessage(
            text=msg.text,
            chat_id=msg.chat_id,
            parse_mode=parse_mode if parse_mode is not None else msg.parse_mode,
            keyboard=keyboard if keyboard is not None else msg.keyboard,
        ),
    )
    await asyncio.sleep(delay_seconds)
    return ok


async def _send_replay_parts(
    messenger: Messenger,
    tg_chat_id: str,
    parts: list[str],
    *,
    delay_seconds: float,
    keyboard: list[list[tuple[str, str]]] | None = None,
) -> bool:
    """Send one or more replay parts, attaching the keyboard only to the last part."""
    delivered_last_part = False
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        delivered_last_part = await _send_replay_message(
            messenger,
            OutgoingMessage(text=part, chat_id=tg_chat_id),
            delay_seconds=delay_seconds,
            parse_mode="html",
            keyboard=keyboard if is_last else None,
        )
    return delivered_last_part


async def _replay_user_message(
    messenger: Messenger,
    tg_chat_id: str,
    content: str,
    *,
    delay_seconds: float,
) -> int:
    parts = _format_user_message(content)
    return int(
        await _send_replay_parts(
            messenger,
            tg_chat_id,
            parts,
            delay_seconds=delay_seconds,
        )
    )


async def _replay_assistant_message(
    messenger: Messenger,
    tg_chat_id: str,
    msg: Message,
    *,
    delay_seconds: float,
    show_tool_calls: bool,
) -> int:
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning", None)
    has_tool_calls = bool(msg.tool_calls)

    if not content.strip() and has_tool_calls:
        if not show_tool_calls:
            return 0
        return int(
            await _send_replay_parts(
                messenger,
                tg_chat_id,
                ["\U0001f916 <i>[AI tool call]</i>"],
                delay_seconds=delay_seconds,
                keyboard=_build_cut_keyboard(msg.id),
            )
        )

    parts = _format_assistant_message(content, reasoning)
    return int(
        await _send_replay_parts(
            messenger,
            tg_chat_id,
            parts,
            delay_seconds=delay_seconds,
            keyboard=_build_cut_keyboard(msg.id),
        )
    )


async def _replay_tool_message(
    messenger: Messenger,
    tg_chat_id: str,
    msg: Message,
    *,
    delay_seconds: float,
    show_tool_calls: bool,
) -> int:
    if not show_tool_calls:
        return 0

    return int(
        await _send_replay_parts(
            messenger,
            tg_chat_id,
            [_format_tool_message(msg.content, msg.tool_call_id)],
            delay_seconds=delay_seconds,
        )
    )


async def _replay_message(
    messenger: Messenger,
    tg_chat_id: str,
    msg: Message,
    *,
    delay_seconds: float,
    show_tool_calls: bool,
) -> int:
    if msg.role == "system":
        return 0
    if msg.role == "user":
        return await _replay_user_message(
            messenger,
            tg_chat_id,
            msg.content,
            delay_seconds=delay_seconds,
        )
    if msg.role == "assistant":
        return await _replay_assistant_message(
            messenger,
            tg_chat_id,
            msg,
            delay_seconds=delay_seconds,
            show_tool_calls=show_tool_calls,
        )
    if msg.role == "tool":
        return await _replay_tool_message(
            messenger,
            tg_chat_id,
            msg,
            delay_seconds=delay_seconds,
            show_tool_calls=show_tool_calls,
        )
    return 0


def _count_displayable_messages(messages: list[Message], *, show_tool_calls: bool) -> int:
    return sum(
        1
        for msg in messages
        if msg.role in ("user", "assistant") or (msg.role == "tool" and show_tool_calls)
    )


async def _send_replay_intro(
    messenger: Messenger,
    tg_chat_id: str,
    *,
    displayable_count: int,
    delay_seconds: float,
) -> None:
    estimated_minutes = int(displayable_count * delay_seconds / 60) + 1
    await _send_replay_message(
        messenger,
        OutgoingMessage(
            text=(
                f"\U0001f4e5 Replaying {displayable_count} messages from imported history...\n"
                f"This will take about {estimated_minutes} minute(s)."
            ),
            chat_id=tg_chat_id,
        ),
        delay_seconds=delay_seconds,
    )


async def _send_replay_progress(
    messenger: Messenger,
    tg_chat_id: str,
    *,
    sent_count: int,
    displayable_count: int,
    delay_seconds: float,
) -> None:
    await _send_replay_message(
        messenger,
        OutgoingMessage(
            text=f"\u23f3 Replay progress: {sent_count}/{displayable_count} messages...",
            chat_id=tg_chat_id,
        ),
        delay_seconds=delay_seconds,
    )


def _find_last_assistant_id(messages: list[Message]) -> int | None:
    for msg in reversed(messages):
        if msg.role == "assistant" and (msg.content or "").strip():
            return msg.id
    return None


def _build_replay_summary(sent_count: int, *, last_assistant_id: int | None) -> str:
    summary_parts = [f"\u2705 Import complete! {sent_count} messages replayed."]
    if last_assistant_id is not None:
        summary_parts.append(
            'Use the "\u2702 Cut this & above" button on any AI message to trim history.'
        )
    summary_parts.append("Send a message to continue the conversation.")
    return "\n".join(summary_parts)


async def _send_replay_summary(
    messenger: Messenger,
    tg_chat_id: str,
    *,
    sent_count: int,
    last_assistant_id: int | None,
) -> None:
    await _send_with_retry(
        messenger,
        OutgoingMessage(
            text=_build_replay_summary(sent_count, last_assistant_id=last_assistant_id),
            chat_id=tg_chat_id,
        ),
    )


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
        result = [header.rstrip(), markdown_to_html(content_chunks[0]), *result[1:]]
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

        truncated = _truncate_oversized_message(msg, error)
        if truncated is not None:
            msg = truncated
            continue
        if "too long" in error or "message is too long" in error:
            return False

        retry = _retry_delay_for_send_error(error, attempt)
        if retry is not None:
            wait, reason = retry
            logger.warning(
                "%s (attempt %d/%d): %s - waiting %ds",
                reason,
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
    displayable_count = _count_displayable_messages(messages, show_tool_calls=show_tool_calls)
    await _send_replay_intro(
        messenger,
        tg_chat_id,
        displayable_count=displayable_count,
        delay_seconds=delay_seconds,
    )

    for msg in messages:
        sent_count += await _replay_message(
            messenger,
            tg_chat_id,
            msg,
            delay_seconds=delay_seconds,
            show_tool_calls=show_tool_calls,
        )

        if sent_count > 0 and sent_count % progress_interval == 0:
            await _send_replay_progress(
                messenger,
                tg_chat_id,
                sent_count=sent_count,
                displayable_count=displayable_count,
                delay_seconds=delay_seconds,
            )

    await _send_replay_summary(
        messenger,
        tg_chat_id,
        sent_count=sent_count,
        last_assistant_id=_find_last_assistant_id(messages),
    )

    return sent_count
