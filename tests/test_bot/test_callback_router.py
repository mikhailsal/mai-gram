"""Tests for the callback router extraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from mai_gram.bot.callback_router import CallbackRouter
from mai_gram.messenger.base import IncomingMessage, MessageType, SendResult


def _make_message(
    callback_data: str,
    *,
    chat_id: str = "tg-chat",
    user_id: str = "test-user",
    raw: object | None = None,
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        message_id="msg-1",
        message_type=MessageType.CALLBACK,
        callback_data=callback_data,
        raw=raw,
    )


def _make_router() -> tuple[
    CallbackRouter, dict[str, list[str]], dict[str, tuple[str, str | None]]
]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))

    import_workflow = MagicMock()
    import_workflow.is_in_import.return_value = False
    import_workflow.handle_import_callback = AsyncMock()

    setup_workflow = MagicMock()
    setup_workflow.is_in_setup.return_value = False
    setup_workflow.handle_setup_callback = AsyncMock()

    reset_workflow = MagicMock()
    reset_workflow.execute_reset = AsyncMock()

    history_actions = MagicMock()
    history_actions.get_message_preview = AsyncMock(return_value="preview text")
    history_actions.handle_cut_above = AsyncMock()

    regenerate_service = MagicMock()
    regenerate_service.handle_regenerate = AsyncMock(return_value=["new-msg"])

    show_confirmation = AsyncMock()
    delete_callback_message = AsyncMock()
    cut_original_html: dict[str, tuple[str, str | None]] = {}
    response_message_ids: dict[str, list[str]] = {"tg-chat": ["old-msg"]}

    router = CallbackRouter(
        messenger,
        import_workflow=import_workflow,
        setup_workflow=setup_workflow,
        reset_workflow=reset_workflow,
        history_actions=history_actions,
        regenerate_service=regenerate_service,
        show_confirmation=show_confirmation,
        delete_callback_message=delete_callback_message,
        cut_original_html=cut_original_html,
        response_message_ids=response_message_ids,
    )
    return router, response_message_ids, cut_original_html


class TestCallbackRouter:
    async def test_ignores_stale_setup_callback(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("model:openrouter/free"))

        send_message = cast("AsyncMock", router._messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        assert "ignored" in await_args.args[0].text

    async def test_shows_regen_confirmation(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("regen"))

        show_confirmation = cast("AsyncMock", router._show_confirmation)
        await_args = show_confirmation.await_args
        assert await_args is not None
        assert await_args.args[1] == "Regenerate this response?"
        assert await_args.kwargs["confirm_data"] == "confirm_regen"

    async def test_cut_callback_caches_original_message_and_confirms(self) -> None:
        router, _, cut_original_html = _make_router()
        raw = SimpleNamespace(
            callback_query=SimpleNamespace(
                message=SimpleNamespace(message_id=77, text_html=None, text="Original text")
            )
        )

        await router.handle_callback(_make_message("cut:12", raw=raw))

        assert cut_original_html["tg-chat:77"] == ("Original text", None)
        show_confirmation = cast("AsyncMock", router._show_confirmation)
        await_args = show_confirmation.await_args
        assert await_args is not None
        assert await_args.kwargs["confirm_data"] == "confirm_cut:12:77"

    async def test_confirm_regen_deletes_callback_and_updates_response_ids(self) -> None:
        router, response_message_ids, _ = _make_router()

        await router.handle_callback(_make_message("confirm_regen"))

        cast("AsyncMock", router._delete_callback_message).assert_awaited_once()
        cast("AsyncMock", router._regenerate_service.handle_regenerate).assert_awaited_once()
        assert response_message_ids["tg-chat"] == ["new-msg"]

    async def test_confirm_cut_deletes_callback_and_runs_history_action(self) -> None:
        router, _, cut_original_html = _make_router()
        cut_original_html["tg-chat:77"] = ("Original text", None)

        await router.handle_callback(_make_message("confirm_cut:12:77"))

        cast("AsyncMock", router._delete_callback_message).assert_awaited_once()
        history_actions = cast("AsyncMock", router._history_actions.handle_cut_above)
        await_args = history_actions.await_args
        assert await_args is not None
        assert await_args.args[1] == 12
        assert await_args.kwargs["cached_original"] == ("Original text", None)

    async def test_confirm_reset_deletes_callback_and_runs_reset(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("confirm_reset:test-user@test-bot"))

        cast("AsyncMock", router._delete_callback_message).assert_awaited_once()
        reset_workflow = cast("AsyncMock", router._reset_workflow.execute_reset)
        await_args = reset_workflow.await_args
        assert await_args is not None
        assert await_args.args[1] == "test-user@test-bot"
