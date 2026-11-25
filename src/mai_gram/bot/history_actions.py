"""History-related actions such as previewing and cutting prompt context."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Select, select
from sqlalchemy.orm import aliased

from mai_gram.core.md_to_telegram import markdown_to_html
from mai_gram.db.database import get_session
from mai_gram.db.history import assistant_message_target, cut_history
from mai_gram.db.models import Message
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from mai_gram.messenger.base import CallbackSourceMessage, Messenger


HISTORY_ACTION_UNAVAILABLE = "This history action is no longer available."


_CUT_BADGE = "✂️ [this and above are hidden from the AI]"


@dataclass(frozen=True, slots=True)
class HistoryPreview:
    token: str
    timestamp: datetime
    question: str | None
    content: str
    role: str = "assistant"

    @property
    def timestamp_utc(self) -> datetime:
        return _utc(self.timestamp)

    def describe(self) -> str:
        timestamp = self.timestamp_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [f"Stored turn ({timestamp})", f"Question: {_excerpt(self.question)}"]
        if self.role == "assistant":
            lines.append(f"Response: {_excerpt(self.content)}")
        elif self.role == "tool":
            lines.append(f"Last tool result: {_excerpt(self.content)}")
        return "\n".join(lines)


def _excerpt(text: str | None, limit: int = 220) -> str:
    if text is None:
        return "(no preceding question in history)"
    text = " ".join(text.split())
    return text[:limit] + "…" if len(text) > limit else text or "(empty)"


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def source_time(timestamp: datetime | None) -> datetime | None:
    if timestamp is None or timestamp.year <= 1970:
        return None
    return _utc(timestamp).replace(microsecond=0)


def _plain_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", "", text))


def legacy_source_matches(source: CallbackSourceMessage, preview: HistoryPreview) -> bool | None:
    """Check evidence from an old response, allowing partial and rendered text."""
    # The old keyboard could have been attached by editing a streaming placeholder.
    edited_at = source_time(source.edited_at)
    if edited_at and preview.timestamp_utc >= edited_at + timedelta(seconds=1):
        return False

    text = source.plain_text
    if text is None:
        text = _plain_html(source.text) if source.parse_mode == "html" else source.text
    text = text.removeprefix(_CUT_BADGE).strip()
    text = re.sub(r"(?:^|\n\s*)\d+/\d+ tokens(?: \| \$[\d.]+)?$", "", text)
    source_text = " ".join(text.split())
    if not source_text:
        return None
    candidates = [preview.content, _plain_html(markdown_to_html(preview.content))]
    for content in candidates:
        candidate = " ".join(content.split())
        if candidate and (source_text in candidate or candidate in source_text):
            return True

    # Long, structured and visibly truncated output can differ from its stored form.
    if (
        len(preview.content) > 1000
        or len(source_text) > 1000
        or source_text.startswith(("…", "...", "💭"))
        or source_text.endswith(("…", "...", "▍"))
        or preview.content.lstrip().startswith(("{", "[", "<", "#"))
    ):
        return None
    return False


def _preview_query() -> Select[tuple[str, datetime, str | None, str, str]]:
    question = aliased(Message)
    preceding_question = (
        select(question.content)
        .where(
            question.chat_id == Message.chat_id, question.role == "user", question.id <= Message.id
        )
        .order_by(question.id.desc())
        .limit(1)
        .correlate(Message)
        .scalar_subquery()
        .label("question")
    )
    return select(
        Message.history_ref, Message.timestamp, preceding_question, Message.content, Message.role
    )


class HistoryActions:
    """Own history preview and cut-above behavior."""

    def __init__(
        self,
        messenger: Messenger,
        *,
        resolve_chat_id: Callable[[IncomingMessage], str],
    ) -> None:
        self._messenger = messenger
        self._resolve_chat_id = resolve_chat_id

    async def get_message_preview(
        self, message: IncomingMessage, target: int | str
    ) -> HistoryPreview | None:
        """Read the assistant response and preceding question together."""
        chat_id = self._resolve_chat_id(message)
        async with get_session() as session:
            row = (
                await session.execute(
                    _preview_query().where(assistant_message_target(chat_id, target))
                )
            ).one_or_none()
        if row is None:
            return None
        return HistoryPreview(row.history_ref, row.timestamp, row.question, row.content, row.role)

    async def get_retry_preview(self, message: IncomingMessage) -> HistoryPreview | None:
        """Return the latest row only when it is an eligible pending turn."""
        chat_id = self._resolve_chat_id(message)
        async with get_session() as session:
            row = (
                await session.execute(
                    _preview_query()
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.id.desc())
                    .limit(1)
                )
            ).one_or_none()
        if row is None or row.role not in ("user", "tool"):
            return None
        return HistoryPreview(row.history_ref, row.timestamp, row.question, row.content, row.role)

    async def handle_cut_above(
        self,
        message: IncomingMessage,
        target: int | str,
        *,
        original_tg_msg_id: str = "",
        cached_original: tuple[str, str | None] | None = None,
        confirmation_message_id: str = "",
    ) -> None:
        """Set the cut-above point so prior messages are excluded from prompts."""
        chat_id = self._resolve_chat_id(message)

        async with get_session() as session:
            cut_count = await cut_history(
                session,
                chat_id,
                target,
                confirmation=(message.chat_id, confirmation_message_id)
                if confirmation_message_id
                else None,
            )

        if confirmation_message_id:
            await self._messenger.delete_message(message.chat_id, confirmation_message_id)

        if cut_count is None:
            await self._messenger.send_message(
                OutgoingMessage(text=HISTORY_ACTION_UNAVAILABLE, chat_id=message.chat_id)
            )
            return

        if original_tg_msg_id and cached_original is not None:
            original_html, original_parse = cached_original
            badge = "✂️ <i>[this and above are hidden from the AI]</i>"
            if original_parse == "html":
                marked_text = f"{badge}\n\n{original_html}"
            else:
                marked_text = f"{badge}\n\n{html.escape(original_html)}"
            if len(marked_text) > 4000:
                marked_text = marked_text[:4000] + "..."
            await self._messenger.edit_message(
                message.chat_id,
                original_tg_msg_id,
                marked_text,
                parse_mode="html",
            )

        footer_lines = ["\u2500" * 20, "\u2702\ufe0f History cut applied"]
        if cut_count > 0:
            footer_lines.append(f"\U0001f4e6 {cut_count} message(s) hidden from AI")
        footer_lines.append("\u2139\ufe0f Hidden messages are still searchable via tools")
        footer_lines.append("\u2500" * 20)
        await self._messenger.send_message(
            OutgoingMessage(
                text="\n".join(footer_lines),
                chat_id=message.chat_id,
            )
        )
