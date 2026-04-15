"""Bot middleware for rate limiting, logging, and request tracking.

This module provides middleware components that wrap message handlers
to add cross-cutting concerns like rate limiting and logging.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mai_gram.messenger.base import IncomingMessage

logger = logging.getLogger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, None]]


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes
    ----------
    messages_per_minute:
        Maximum messages allowed per minute per user.
    messages_per_hour:
        Maximum messages allowed per hour per user.
    cooldown_seconds:
        How long to wait before allowing messages again after hitting the limit.
    """

    messages_per_minute: int = 20
    messages_per_hour: int = 200
    cooldown_seconds: int = 60


@dataclass
class UserRateState:
    """Tracks rate limiting state for a single user.

    Attributes
    ----------
    minute_timestamps:
        Timestamps of messages in the current minute window.
    hour_timestamps:
        Timestamps of messages in the current hour window.
    cooldown_until:
        If set, the user is rate-limited until this time.
    """

    minute_timestamps: list[float] = field(default_factory=list)
    hour_timestamps: list[float] = field(default_factory=list)
    cooldown_until: float | None = None


class RateLimiter:
    """Rate limiter for incoming messages.

    Uses a sliding window algorithm to track message frequency
    per user and enforce rate limits.

    Parameters
    ----------
    config:
        Rate limiting configuration.
    on_rate_limited:
        Optional callback when a user is rate limited.
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        on_rate_limited: Callable[[str, str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._config = config or RateLimitConfig()
        self._on_rate_limited = on_rate_limited
        self._user_states: dict[str, UserRateState] = defaultdict(UserRateState)

    def _cleanup_old_timestamps(self, state: UserRateState, now: float) -> None:
        """Remove timestamps outside the current windows."""
        minute_ago = now - 60
        hour_ago = now - 3600

        state.minute_timestamps = [ts for ts in state.minute_timestamps if ts > minute_ago]
        state.hour_timestamps = [ts for ts in state.hour_timestamps if ts > hour_ago]

    async def check_rate_limit(self, user_id: str, chat_id: str) -> bool:
        """Check if a user is within rate limits.

        Parameters
        ----------
        user_id:
            The user's identifier.
        chat_id:
            The chat identifier (for the callback).

        Returns
        -------
        bool
            True if the message should be allowed, False if rate limited.
        """
        now = time.time()
        state = self._user_states[user_id]

        # Check if user is in cooldown
        if state.cooldown_until and now < state.cooldown_until:
            logger.debug("User %s is in cooldown until %s", user_id, state.cooldown_until)
            return False

        # Clear expired cooldown
        state.cooldown_until = None

        # Cleanup old timestamps
        self._cleanup_old_timestamps(state, now)

        # Check minute limit
        if len(state.minute_timestamps) >= self._config.messages_per_minute:
            logger.warning(
                "User %s hit minute rate limit (%d messages)",
                user_id,
                len(state.minute_timestamps),
            )
            state.cooldown_until = now + self._config.cooldown_seconds
            if self._on_rate_limited:
                await self._on_rate_limited(user_id, chat_id)
            return False

        # Check hour limit
        if len(state.hour_timestamps) >= self._config.messages_per_hour:
            logger.warning(
                "User %s hit hour rate limit (%d messages)",
                user_id,
                len(state.hour_timestamps),
            )
            state.cooldown_until = now + self._config.cooldown_seconds * 5
            if self._on_rate_limited:
                await self._on_rate_limited(user_id, chat_id)
            return False

        # Record this message
        state.minute_timestamps.append(now)
        state.hour_timestamps.append(now)

        return True

    def reset_user(self, user_id: str) -> None:
        """Reset rate limiting state for a user.

        Parameters
        ----------
        user_id:
            The user's identifier.
        """
        if user_id in self._user_states:
            del self._user_states[user_id]


class MessageLogger:
    """Logs incoming and outgoing messages for debugging.

    Parameters
    ----------
    log_content:
        If True, log the full message content. If False, only log metadata.
    """

    def __init__(self, *, log_content: bool = False) -> None:
        self._log_content = log_content

    def log_incoming(self, message: IncomingMessage) -> None:
        """Log an incoming message.

        Parameters
        ----------
        message:
            The incoming message to log.
        """
        content_preview = ""
        if self._log_content and message.text:
            content_preview = f" content={message.text[:50]!r}..."

        logger.info(
            "Incoming message: platform=%s chat=%s user=%s type=%s%s",
            message.platform,
            message.chat_id,
            message.user_id,
            message.message_type.value,
            content_preview,
        )

    def log_outgoing(
        self, chat_id: str, text: str, *, success: bool, message_id: str | None = None
    ) -> None:
        """Log an outgoing message.

        Parameters
        ----------
        chat_id:
            The target chat.
        text:
            The message text.
        success:
            Whether the send was successful.
        message_id:
            The ID of the sent message, if successful.
        """
        content_preview = ""
        if self._log_content:
            content_preview = f" content={text[:50]!r}..."

        if success:
            logger.info(
                "Outgoing message: chat=%s msg_id=%s%s",
                chat_id,
                message_id,
                content_preview,
            )
        else:
            logger.warning(
                "Failed to send message: chat=%s%s",
                chat_id,
                content_preview,
            )


@dataclass
class RequestContext:
    """Context information for a single request/message.

    Attributes
    ----------
    message:
        The incoming message.
    start_time:
        When processing started.
    user_id:
        The user's identifier.
    chat_id:
        The chat identifier.
    companion_id:
        The companion's ID (if known).
    """

    message: IncomingMessage
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = ""
    chat_id: str = ""
    companion_id: str | None = None

    def __post_init__(self) -> None:
        self.user_id = self.message.user_id
        self.chat_id = self.message.chat_id

    @property
    def elapsed_ms(self) -> float:
        """Return milliseconds elapsed since request started."""
        now = datetime.now(timezone.utc)
        return (now - self.start_time).total_seconds() * 1000


def with_rate_limit(
    rate_limiter: RateLimiter,
) -> Callable[[MessageHandler], MessageHandler]:
    """Decorator that applies rate limiting to a message handler.

    Parameters
    ----------
    rate_limiter:
        The rate limiter instance to use.

    Returns
    -------
    Callable
        A decorator function.
    """

    def decorator(handler: MessageHandler) -> MessageHandler:
        async def wrapper(message: IncomingMessage) -> None:
            if await rate_limiter.check_rate_limit(message.user_id, message.chat_id):
                await handler(message)

        return wrapper

    return decorator


def with_logging(
    message_logger: MessageLogger,
) -> Callable[[MessageHandler], MessageHandler]:
    """Decorator that adds logging to a message handler.

    Parameters
    ----------
    message_logger:
        The message logger instance to use.

    Returns
    -------
    Callable
        A decorator function.
    """

    def decorator(handler: MessageHandler) -> MessageHandler:
        async def wrapper(message: IncomingMessage) -> None:
            message_logger.log_incoming(message)
            await handler(message)

        return wrapper

    return decorator
