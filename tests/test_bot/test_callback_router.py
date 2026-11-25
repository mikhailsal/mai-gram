"""Tests for the callback router extraction."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram.bot.callback_router import CallbackRouter
from mai_gram.bot.history_actions import HistoryPreview
from mai_gram.messenger.base import CallbackSourceMessage, IncomingMessage, MessageType, SendResult

_TOKEN = "a" * 32


def _make_message(
    callback_data: str,
    *,
    chat_id: str = "tg-chat",
    user_id: str = "test-user",
    raw: object | None = None,
    platform: str = "telegram",
) -> IncomingMessage:
    return IncomingMessage(
        platform=platform,
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
    setup_workflow.handle_model_change = AsyncMock()

    reset_workflow = MagicMock()
    reset_workflow.execute_reset = AsyncMock()

    history_actions = MagicMock()
    history_actions.get_message_preview = AsyncMock(
        return_value=HistoryPreview(
            _TOKEN, datetime(2026, 4, 26, tzinfo=timezone.utc), "Question", "Original text"
        )
    )
    history_actions.get_retry_preview = AsyncMock(
        return_value=HistoryPreview(
            _TOKEN, datetime(2026, 4, 26, tzinfo=timezone.utc), "Question", "Question", "user"
        )
    )
    history_actions.handle_cut_above = AsyncMock()

    regenerate_service = MagicMock()
    regenerate_service.handle_regenerate = AsyncMock(return_value=["new-msg"])

    show_confirmation = AsyncMock()
    cut_original_html: dict[str, tuple[str, str | None]] = {}
    response_message_ids: dict[str, list[str]] = {"tg-chat": ["old-msg"]}

    messenger.get_callback_source_message.return_value = None
    messenger.delete_callback_source_message = AsyncMock(return_value=True)

    router = CallbackRouter(
        messenger,
        import_workflow=import_workflow,
        setup_workflow=setup_workflow,
        reset_workflow=reset_workflow,
        history_actions=history_actions,
        regenerate_service=regenerate_service,
        show_confirmation=show_confirmation,
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

    async def test_setmodel_callback_delegates_to_setup_workflow(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("setmodel:google/gemini-2.5-flash"))

        handle_model_change = cast("AsyncMock", router._setup_workflow.handle_model_change)
        handle_model_change.assert_awaited_once()
        await_args = handle_model_change.await_args
        assert await_args is not None
        assert await_args.args[1] == "google/gemini-2.5-flash"
        # A live model switch must never be treated as a stale setup callback.
        cast("AsyncMock", router._messenger.send_message).assert_not_awaited()

    async def test_cancel_action_deletes_source_message(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("cancel_action"))

        cast("AsyncMock", router._messenger.delete_callback_source_message).assert_awaited_once()
        router._history_actions.handle_cut_above.assert_not_awaited()
        router._regenerate_service.handle_regenerate.assert_not_awaited()

    async def test_shows_regen_confirmation(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("regen", platform="console"))

        show_confirmation = cast("AsyncMock", router._show_confirmation)
        await_args = show_confirmation.await_args
        assert await_args is not None
        assert await_args.args[1] == "Regenerate this response?"
        assert await_args.kwargs["confirm_data"] == "confirm_regen"

    @pytest.mark.parametrize(
        ("sibling", "expected"), [("cut:12", f"confirm_regen:{_TOKEN}"), ("cut:", None)]
    )
    async def test_legacy_regen_uses_sibling_target_when_present(
        self, sibling: str, expected: str | None
    ) -> None:
        router, _, _ = _make_router()
        markup = SimpleNamespace(inline_keyboard=[[SimpleNamespace(callback_data=sibling)]])
        raw = SimpleNamespace(
            callback_query=SimpleNamespace(message=SimpleNamespace(reply_markup=markup))
        )
        message = _make_message("regen", raw=raw)

        await router.handle_callback(message)

        confirmation = cast("AsyncMock", router._show_confirmation)
        if expected is None:
            confirmation.assert_not_awaited()
            router._messenger.send_message.assert_awaited_once()
        else:
            router._history_actions.get_message_preview.assert_awaited_once_with(message, 12)
            assert confirmation.await_args is not None
            assert "Question: Question" in confirmation.await_args.args[1]
            assert "Response: Original text" in confirmation.await_args.args[1]
            assert confirmation.await_args.kwargs["confirm_data"] == expected

    @pytest.mark.parametrize("action", ["cut", "regen", "confirm_cut", "confirm_regen"])
    @pytest.mark.parametrize("target", ["", "invalid", "-2", "0", "9223372036854775808"])
    async def test_malformed_targets_never_dispatch(self, action: str, target: str) -> None:
        router, response_ids, _ = _make_router()

        await router.handle_callback(_make_message(f"{action}:{target}"))

        cast("AsyncMock", router._show_confirmation).assert_not_awaited()
        router._history_actions.get_message_preview.assert_not_awaited()
        router._history_actions.handle_cut_above.assert_not_awaited()
        router._regenerate_service.handle_regenerate.assert_not_awaited()
        router._messenger.send_message.assert_awaited_once()
        assert response_ids["tg-chat"] == ["old-msg"]

    @pytest.mark.parametrize("action", ["cut", "regen"])
    async def test_unavailable_target_does_not_open_confirmation(self, action: str) -> None:
        router, _, _ = _make_router()
        router._history_actions.get_message_preview.return_value = None

        await router.handle_callback(_make_message(f"{action}:12"))

        cast("AsyncMock", router._show_confirmation).assert_not_awaited()
        router._messenger.send_message.assert_awaited_once()

    async def test_cut_callback_caches_original_message_and_confirms(self) -> None:
        router, _, cut_original_html = _make_router()
        router._messenger.get_callback_source_message.return_value = CallbackSourceMessage(
            message_id="77",
            text="Original text",
            parse_mode=None,
        )

        await router.handle_callback(_make_message("cut:12"))

        assert cut_original_html[f"tg-chat:77:{_TOKEN}"] == ("Original text", None)
        show_confirmation = cast("AsyncMock", router._show_confirmation)
        await_args = show_confirmation.await_args
        assert await_args is not None
        assert await_args.kwargs["confirm_data"] == f"confirm_cut:{_TOKEN}:77"

    async def test_confirm_regen_deletes_callback_and_updates_response_ids(self) -> None:
        router, response_message_ids, _ = _make_router()

        await router.handle_callback(_make_message("confirm_regen", platform="console"))

        cast("AsyncMock", router._messenger.delete_callback_source_message).assert_awaited_once()
        cast("AsyncMock", router._regenerate_service.handle_regenerate).assert_awaited_once()
        assert response_message_ids["tg-chat"] == ["new-msg"]

    async def test_rejected_targeted_regen_preserves_response_ids(self) -> None:
        router, response_ids, _ = _make_router()
        router._regenerate_service.handle_regenerate.return_value = None
        message = _make_message(f"confirm_regen:{_TOKEN}")

        await router.handle_callback(message)

        router._regenerate_service.handle_regenerate.assert_awaited_once_with(
            message,
            previous_response_ids=["old-msg"],
            target_assistant_msg_id=None,
            target_ref=_TOKEN,
            retry_ref=None,
        )
        assert response_ids["tg-chat"] == ["old-msg"]

    @pytest.mark.parametrize(
        ("data", "replacement"),
        [
            ("confirm_cut:12:77", f"confirm_cut:{_TOKEN}"),
            ("confirm_regen:12", f"confirm_regen:{_TOKEN}"),
            ("confirm_regen", f"confirm_retry:{_TOKEN}"),
        ],
    )
    async def test_old_telegram_yes_requires_a_bound_confirmation(
        self, data: str, replacement: str
    ) -> None:
        router, response_ids, _ = _make_router()
        router._messenger.delete_callback_source_message.return_value = False

        await router.handle_callback(_make_message(data))

        assert router._show_confirmation.await_args.kwargs["confirm_data"] == replacement
        router._history_actions.handle_cut_above.assert_not_awaited()
        router._regenerate_service.handle_regenerate.assert_not_awaited()
        assert response_ids["tg-chat"] == ["old-msg"]

    async def test_confirm_cut_passes_source_for_transactional_cleanup(self) -> None:
        router, _, cut_original_html = _make_router()
        cut_original_html[f"tg-chat:77:{_TOKEN}"] = ("Original text", None)
        router._messenger.get_callback_source_message.return_value = CallbackSourceMessage(
            message_id="88", text="Confirmation"
        )
        message = _make_message(f"confirm_cut:{_TOKEN}:77")

        await router.handle_callback(message)

        router._messenger.delete_callback_source_message.assert_not_awaited()
        router._history_actions.handle_cut_above.assert_awaited_once_with(
            message,
            _TOKEN,
            original_tg_msg_id="77",
            cached_original=("Original text", None),
            confirmation_message_id="88",
        )
        assert not cut_original_html

    async def test_confirm_reset_deletes_callback_and_runs_reset(self) -> None:
        router, _, _ = _make_router()

        await router.handle_callback(_make_message("confirm_reset:test-user@test-bot"))

        cast("AsyncMock", router._messenger.delete_callback_source_message).assert_awaited_once()
        reset_workflow = cast("AsyncMock", router._reset_workflow.execute_reset)
        await_args = reset_workflow.await_args
        assert await_args is not None
        assert await_args.args[1] == "test-user@test-bot"
