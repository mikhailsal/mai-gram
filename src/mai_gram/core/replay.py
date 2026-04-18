"""Rate-limited message replay engine for imported conversations.

Sends imported messages to a Telegram chat with proper throttling
to stay within Telegram Bot API rate limits (~1 msg/sec per chat).
"""

from __future__ import annotations

import asyncio
import html as _html
import logging
from typing import TYPE_CHECKING, Any

from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.db.models import Message
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)

DELAY_SECONDS = 1.2
PROGRESS_INTERVAL = 25
MAX_MESSAGE_LENGTH = 4000


def _format_user_message(content: str) -> str:
    """Format a user message for display (sent from bot on behalf of user)."""
    truncated = content[:MAX_MESSAGE_LENGTH]
    if len(content) > MAX_MESSAGE_LENGTH:
        truncated += "..."
    escaped = _html.escape(truncated)
    return f"\U0001f464 <b>[You]</b>\n<i>{escaped}</i>"


def _format_assistant_message(content: str, reasoning: str | None = None) -> str:
    """Format an assistant message using the same style as the regular chat handler."""
    from mai_gram.core.md_to_telegram import markdown_to_html

    parts: list[str] = ["\U0001f916 <b>[AI]</b>\n"]

    if reasoning and reasoning.strip():
        reasoning_truncated = reasoning[:2000]
        if len(reasoning) > 2000:
            reasoning_truncated += "..."
        escaped_reasoning = _html.escape(reasoning_truncated.strip())
        parts.append(
            f"<blockquote expandable>\U0001f4ad Reasoning\n{escaped_reasoning}</blockquote>\n\n"
        )

    if content.strip():
        truncated = content[:MAX_MESSAGE_LENGTH]
        if len(content) > MAX_MESSAGE_LENGTH:
            truncated += "..."
        parts.append(markdown_to_html(truncated))

    return "".join(parts)


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
    Assistant messages get a "Cut this & above" button.

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

    await messenger.send_message(
        OutgoingMessage(
            text=f"\U0001f4e5 Replaying {displayable_count} messages from imported history...\n"
            f"This will take about {int(displayable_count * delay_seconds / 60) + 1} minute(s).",
            chat_id=tg_chat_id,
        )
    )
    await asyncio.sleep(delay_seconds)

    for msg in messages:
        if msg.role == "system":
            continue

        if msg.role == "user":
            text = _format_user_message(msg.content)
            result = await messenger.send_message(
                OutgoingMessage(
                    text=text,
                    chat_id=tg_chat_id,
                    parse_mode="html",
                )
            )
            if result.success:
                sent_count += 1

        elif msg.role == "assistant":
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None)
            has_tool_calls = bool(msg.tool_calls)

            if not content.strip() and has_tool_calls:
                if show_tool_calls:
                    text = "\U0001f916 <i>[AI tool call]</i>"
                    kb: Any = _build_cut_keyboard(msg.id)
                    result = await messenger.send_message(
                        OutgoingMessage(
                            text=text,
                            chat_id=tg_chat_id,
                            parse_mode="html",
                            keyboard=kb,
                        )
                    )
                    if result.success:
                        sent_count += 1
                continue

            text = _format_assistant_message(content, reasoning)
            kb = _build_cut_keyboard(msg.id)
            result = await messenger.send_message(
                OutgoingMessage(
                    text=text,
                    chat_id=tg_chat_id,
                    parse_mode="html",
                    keyboard=kb,
                )
            )
            if result.success:
                sent_count += 1

        elif msg.role == "tool":
            if not show_tool_calls:
                continue
            text = _format_tool_message(msg.content, msg.tool_call_id)
            result = await messenger.send_message(
                OutgoingMessage(
                    text=text,
                    chat_id=tg_chat_id,
                    parse_mode="html",
                )
            )
            if result.success:
                sent_count += 1
        else:
            continue

        if sent_count > 0 and sent_count % progress_interval == 0:
            await messenger.send_message(
                OutgoingMessage(
                    text=f"\u23f3 Replay progress: {sent_count}/{displayable_count} messages...",
                    chat_id=tg_chat_id,
                )
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

    await messenger.send_message(
        OutgoingMessage(
            text="\n".join(summary_parts),
            chat_id=tg_chat_id,
        )
    )

    return sent_count
