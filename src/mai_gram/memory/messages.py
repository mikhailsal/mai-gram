"""Message store for conversation history."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import and_, asc, desc, func, select

from mai_gram.db.models import Message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _escape_like_pattern(query: str) -> str:
    """Escape SQL LIKE wildcard characters (%, _) in user input."""
    return query.replace("%", r"\%").replace("_", r"\_")


def _to_utc_naive(dt: datetime) -> datetime:
    """Normalize datetime values for safe chronological comparisons."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


class MessageStore:
    """Data access layer for conversation messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime | None = None,
        tool_calls: str | None = None,
        tool_call_id: str | None = None,
        reasoning: str | None = None,
        timezone_name: str = "UTC",
    ) -> Message:
        """Persist a message and return the saved ORM object."""
        if timestamp is not None:
            result = await self._session.execute(
                select(Message.timestamp)
                .where(Message.chat_id == chat_id)
                .order_by(desc(Message.timestamp), desc(Message.id))
                .limit(1)
            )
            last_timestamp = result.scalar_one_or_none()
            if isinstance(last_timestamp, datetime) and _to_utc_naive(timestamp) <= _to_utc_naive(
                last_timestamp
            ):
                raise ValueError(
                    f"Error: timestamp {timestamp.isoformat()} is not after "
                    f"the last message at {last_timestamp.isoformat()}. "
                    "Messages must be chronologically ordered."
                )

        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            reasoning=reasoning,
            timezone=timezone_name,
        )
        if timestamp is not None:
            message.timestamp = timestamp

        self._session.add(message)
        await self._session.flush()
        return message

    async def get_recent(
        self,
        chat_id: str,
        *,
        limit: int = 50,
        after_message_id: int | None = None,
    ) -> list[Message]:
        """Return the most recent messages for a chat (newest first).

        If *after_message_id* is set, only messages with id > that value
        are returned (used by the "cut this & above" feature).
        """
        conditions = [Message.chat_id == chat_id]
        if after_message_id is not None:
            conditions.append(Message.id > after_message_id)
        result = await self._session.execute(
            select(Message).where(and_(*conditions)).order_by(desc(Message.id)).limit(limit)
        )
        return list(result.scalars().all())

    async def get_all(
        self,
        chat_id: str,
    ) -> list[Message]:
        """Return all messages for a chat ordered by id ascending."""
        result = await self._session.execute(
            select(Message).where(Message.chat_id == chat_id).order_by(asc(Message.id))
        )
        return list(result.scalars().all())

    async def search(
        self,
        chat_id: str,
        query: str,
        *,
        limit: int = 20,
        oldest_first: bool = False,
    ) -> list[Message]:
        """Search messages by content using SQL LIKE."""
        escaped = _escape_like_pattern(query)
        pattern = f"%{escaped}%"
        order = asc(Message.id) if oldest_first else desc(Message.id)
        result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.chat_id == chat_id,
                    Message.content.like(pattern, escape="\\"),
                )
            )
            .order_by(order)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_messages_for_date(
        self,
        chat_id: str,
        target_date: date,
    ) -> list[Message]:
        """Return messages for one date in chronological order."""
        start = datetime.combine(target_date, time.min)
        end = start + timedelta(days=1)
        result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.chat_id == chat_id,
                    Message.timestamp >= start,
                    Message.timestamp < end,
                )
            )
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        return list(result.scalars().all())

    async def get_dates_with_messages(
        self,
        chat_id: str,
        *,
        before_date: date | None = None,
    ) -> list[date]:
        """Return distinct dates that have messages, ordered chronologically."""
        date_col = func.date(Message.timestamp)
        query = (
            select(date_col).where(Message.chat_id == chat_id).group_by(date_col).order_by(date_col)
        )
        if before_date is not None:
            query = query.where(Message.timestamp < datetime.combine(before_date, time.min))
        result = await self._session.execute(query)
        return [date.fromisoformat(row[0]) for row in result.all()]

    async def get_message_by_id(
        self,
        chat_id: str,
        message_id: int,
    ) -> Message | None:
        """Return a single message by ID, or None if not found."""
        result = await self._session.execute(
            select(Message).where(
                and_(
                    Message.chat_id == chat_id,
                    Message.id == message_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_message_context(
        self,
        chat_id: str,
        message_id: int,
        *,
        before: int = 5,
        after: int = 5,
    ) -> tuple[list[Message], Message | None, list[Message]]:
        """Return messages surrounding a specific message for context."""
        target = await self.get_message_by_id(chat_id, message_id)
        if target is None:
            return [], None, []

        before_result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.chat_id == chat_id,
                    Message.id < message_id,
                )
            )
            .order_by(desc(Message.id))
            .limit(before)
        )
        messages_before = list(before_result.scalars().all())
        messages_before.reverse()

        after_result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.chat_id == chat_id,
                    Message.id > message_id,
                )
            )
            .order_by(asc(Message.id))
            .limit(after)
        )
        messages_after = list(after_result.scalars().all())

        return messages_before, target, messages_after

    async def get_messages_paginated(
        self,
        chat_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        oldest_first: bool = True,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[list[Message], int]:
        """Return paginated messages with optional date filtering."""
        conditions = [Message.chat_id == chat_id]

        if start_date is not None:
            start = datetime.combine(start_date, time.min)
            conditions.append(Message.timestamp >= start)

        if end_date is not None:
            end = datetime.combine(end_date + timedelta(days=1), time.min)
            conditions.append(Message.timestamp < end)

        count_result = await self._session.execute(select(Message.id).where(and_(*conditions)))
        total_count = len(count_result.all())

        order = asc(Message.id) if oldest_first else desc(Message.id)
        result = await self._session.execute(
            select(Message)
            .where(and_(*conditions))
            .order_by(order)
            .limit(min(limit, 50))
            .offset(offset)
        )
        messages = list(result.scalars().all())

        return messages, total_count
