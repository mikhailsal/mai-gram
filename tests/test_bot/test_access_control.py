"""Unit tests for transport-facing bot access control."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

from mai_gram.bot.access_control import AccessControl
from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage, SendResult


def _make_message(*, user_id: str = "blocked-user") -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id="test-chat",
        message_id="msg-1",
        message_type=MessageType.TEXT,
        text="hello",
    )


def _make_control(*, allowed_users: set[str]) -> tuple[AccessControl, MagicMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    return AccessControl(messenger, allowed_users=allowed_users), messenger


class TestAccessControl:
    async def test_allows_all_users_when_whitelist_is_empty(self) -> None:
        control, messenger = _make_control(allowed_users=set())

        allowed = await control.check_access(_make_message())

        assert allowed is True
        cast("AsyncMock", messenger.send_message).assert_not_awaited()

    async def test_denies_unknown_user_and_sends_message(self) -> None:
        control, messenger = _make_control(allowed_users={"allowed-user"})

        allowed = await control.check_access(_make_message())

        assert allowed is False
        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        outgoing = cast("OutgoingMessage", await_args.args[0])
        assert "access denied" in outgoing.text.lower()

    async def test_rate_limit_callback_sends_slow_down_message(self) -> None:
        control, messenger = _make_control(allowed_users=set())

        await control.handle_rate_limited("user-1", "test-chat")

        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        outgoing = cast("OutgoingMessage", await_args.args[0])
        assert "slow down" in outgoing.text.lower()
