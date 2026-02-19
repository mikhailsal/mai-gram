"""Message store for short-term memory and history retrieval."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import and_, desc, select
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
    ) -> Message:
        """Persist a message and return the saved ORM object."""
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
    ) -> list[Message]:
        """Search messages by content using SQL LIKE."""
        escaped = _escape_like_pattern(query)
        pattern = f"%{escaped}%"
        result = await self._session.execute(
            select(Message)
            .where(
                and_(
                    Message.companion_id == companion_id,
                    Message.content.like(pattern, escape="\\"),
                )
            )
            .order_by(desc(Message.id))
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
