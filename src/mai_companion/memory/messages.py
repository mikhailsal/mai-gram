"""Message store for short-term memory and history retrieval."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import and_, asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from mai_companion.db.models import Message


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
        companion_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime | None = None,
        is_proactive: bool = False,
        tool_calls: str | None = None,
        tool_call_id: str | None = None,
    ) -> Message:
        """Persist a message and return the saved ORM object.

        Parameters
        ----------
        tool_calls:
            JSON-serialized list of tool calls for assistant messages.
            Format: [{"id": "...", "name": "...", "arguments": "..."}, ...]
        tool_call_id:
            ID of the tool call this message responds to (for role='tool').
        """
        if timestamp is not None:
            result = await self._session.execute(
                select(Message.timestamp)
                .where(Message.companion_id == companion_id)
                .order_by(desc(Message.timestamp), desc(Message.id))
                .limit(1)
            )
            last_timestamp = result.scalar_one_or_none()
            if (
                isinstance(last_timestamp, datetime)
                and _to_utc_naive(timestamp) <= _to_utc_naive(last_timestamp)
            ):
                raise ValueError(
                    "Error: target date "
                    f"{timestamp.date().isoformat()} would place this message before "
                    f"the last message at {last_timestamp.isoformat()}. "
                    "Messages must be chronologically ordered."
                )

        message = Message(
            companion_id=companion_id,
            role=role,
            content=content,
            is_proactive=is_proactive,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        if timestamp is not None:
            message.timestamp = timestamp

        self._session.add(message)
        await self._session.flush()
        return message

    async def get_short_term(
        self,
        companion_id: str,
        *,
        limit: int = 30,
        now: datetime | None = None,
    ) -> list[Message]:
        """Return recent messages + all messages from today, deduplicated.

        The final result is ordered by message id descending (newest first).
        """
        current = now or datetime.now(timezone.utc)
        today_start = datetime.combine(current.date(), time.min)
        tomorrow_start = today_start + timedelta(days=1)

        recent_result = await self._session.execute(
            select(Message)
            .where(Message.companion_id == companion_id)
            .order_by(desc(Message.id))
            .limit(limit)
        )
        recent_messages = list(recent_result.scalars().all())

        today_result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
                    Message.timestamp >= today_start,
                    Message.timestamp < tomorrow_start,
                )
            )
            .order_by(desc(Message.id))
        )
        today_messages = list(today_result.scalars().all())

        by_id: dict[int, Message] = {}
        for message in recent_messages:
            by_id[message.id] = message
        for message in today_messages:
            by_id[message.id] = message

        return sorted(by_id.values(), key=lambda message: message.id, reverse=True)

    async def search(
        self,
        companion_id: str,
        query: str,
        *,
        limit: int = 20,
        oldest_first: bool = False,
    ) -> list[Message]:
        """Search messages by content using SQL LIKE.

        Parameters
        ----------
        oldest_first:
            If True, return oldest matching messages first.
            If False (default), return newest matching messages first.
        """
        escaped = _escape_like_pattern(query)
        pattern = f"%{escaped}%"
        order = asc(Message.id) if oldest_first else desc(Message.id)
        result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
                    Message.content.like(pattern, escape="\\"),
                )
            )
            .order_by(order)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_messages_for_date(
        self,
        companion_id: str,
        target_date: date,
    ) -> list[Message]:
        """Return messages for one date in chronological order."""
        start = datetime.combine(target_date, time.min)
        end = start + timedelta(days=1)
        result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
                    Message.timestamp >= start,
                    Message.timestamp < end,
                )
            )
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        return list(result.scalars().all())

    async def get_messages_in_range(
        self,
        companion_id: str,
        start_date: date,
        end_date: date,
    ) -> list[Message]:
        """Return messages for an inclusive date range in chronological order."""
        start = datetime.combine(start_date, time.min)
        end = datetime.combine(end_date + timedelta(days=1), time.min)
        result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
                    Message.timestamp >= start,
                    Message.timestamp < end,
                )
            )
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        return list(result.scalars().all())

    async def get_message_by_id(
        self,
        companion_id: str,
        message_id: int,
    ) -> Message | None:
        """Return a single message by ID, or None if not found."""
        result = await self._session.execute(
            select(Message).where(
                and_(
                    Message.companion_id == companion_id,
                    Message.id == message_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_message_context(
        self,
        companion_id: str,
        message_id: int,
        *,
        before: int = 5,
        after: int = 5,
    ) -> tuple[list[Message], Message | None, list[Message]]:
        """Return messages surrounding a specific message for context.

        Parameters
        ----------
        message_id:
            The ID of the target message to get context for.
        before:
            Number of messages to retrieve before the target (default 5).
        after:
            Number of messages to retrieve after the target (default 5).

        Returns
        -------
        A tuple of (messages_before, target_message, messages_after).
        All lists are in chronological order (oldest first).
        target_message is None if the message ID doesn't exist.
        """
        # Get the target message
        target = await self.get_message_by_id(companion_id, message_id)
        if target is None:
            return [], None, []

        # Get messages before (smaller IDs)
        before_result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
                    Message.id < message_id,
                )
            )
            .order_by(desc(Message.id))
            .limit(before)
        )
        messages_before = list(before_result.scalars().all())
        # Reverse to get chronological order
        messages_before.reverse()

        # Get messages after (larger IDs)
        after_result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
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
        companion_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        oldest_first: bool = True,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[list[Message], int]:
        """Return paginated messages with optional date filtering.

        Parameters
        ----------
        limit:
            Maximum number of messages to return (default 20, max 50).
        offset:
            Number of messages to skip (for pagination).
        oldest_first:
            If True (default), return oldest messages first.
            If False, return newest messages first.
        start_date:
            Optional start date filter (inclusive).
        end_date:
            Optional end date filter (inclusive).

        Returns
        -------
        A tuple of (messages, total_count).
        total_count is the total number of messages matching the filter
        (useful for pagination UI, though AI won't need it often).
        """
        # Build base conditions
        conditions = [Message.companion_id == companion_id]

        if start_date is not None:
            start = datetime.combine(start_date, time.min)
            conditions.append(Message.timestamp >= start)

        if end_date is not None:
            end = datetime.combine(end_date + timedelta(days=1), time.min)
            conditions.append(Message.timestamp < end)

        # Get total count
        count_result = await self._session.execute(
            select(Message.id).where(and_(*conditions))
        )
        total_count = len(count_result.all())

        # Get paginated results
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
