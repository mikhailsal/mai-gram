"""Transport-facing access control for bot handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.messenger.base import IncomingMessage, Messenger

logger = logging.getLogger(__name__)


class AccessControl:
    """Enforce user access rules and rate-limit notifications."""

    def __init__(self, messenger: Messenger, *, allowed_users: set[str]) -> None:
        self._messenger = messenger
        self._allowed_users = allowed_users
        if self._allowed_users:
            logger.info(
                "Access control enabled: %d user(s) allowed",
                len(self._allowed_users),
            )

    async def handle_rate_limited(self, user_id: str, chat_id: str) -> None:
        del user_id
        await self._messenger.send_message(
            OutgoingMessage(
                text="Slow down! Too many messages. Wait a moment and try again.",
                chat_id=chat_id,
            )
        )

    async def check_access(self, message: IncomingMessage) -> bool:
        if not self._allowed_users:
            return True
        if message.user_id in self._allowed_users:
            return True

        logger.warning("Access denied for user_id=%s", message.user_id)
        await self._messenger.send_message(
            OutgoingMessage(
                text=(f"Access denied. This is a private bot. Your user ID: {message.user_id}"),
                chat_id=message.chat_id,
            )
        )
        return False
