"""Unit tests for the shared assistant-turn request builder."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
from mai_gram.db.models import Chat, Message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _make_builder(
    *,
    build_mcp_manager: MagicMock | None = None,
    model_params: dict | None = None,
    max_output_tokens: int = 0,
) -> AssistantTurnBuilder:
    llm = MagicMock()
    settings = MagicMock()
    settings.get_model_params.return_value = dict(model_params or {"temperature": 0.2})
    settings.get_model_id.return_value = "resolved/model-id"
    settings.get_max_output_tokens.return_value = max_output_tokens
    return AssistantTurnBuilder(
        llm,
        settings,
        build_mcp_manager=build_mcp_manager or MagicMock(return_value=MagicMock()),
        memory_data_dir="./data",
        wiki_context_limit=20,
        short_term_limit=500,
        test_mode=True,
    )


def _make_chat() -> Chat:
    return Chat(
        id="test-user@test-bot",
        user_id="test-user",
        bot_id="test-bot",
        llm_model="test-model",
        system_prompt="test prompt",
        timezone="UTC",
        send_datetime=True,
        show_reasoning=True,
        show_tool_calls=True,
    )


class TestAssistantTurnBuilder:
    async def test_save_user_message_and_build_request_persists_user_turn(
        self, session: AsyncSession
    ) -> None:
        build_mcp_manager = MagicMock(return_value=MagicMock())
        builder = _make_builder(build_mcp_manager=build_mcp_manager)
        chat = _make_chat()
        session.add(chat)
        await session.commit()
        now = datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc)
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch("mai_gram.bot.assistant_turn_builder.PromptBuilder") as prompt_builder_cls,
            patch("mai_gram.bot.assistant_turn_builder.WikiStore", return_value=wiki_store),
        ):
            prompt_builder_cls.return_value.build_context = AsyncMock(return_value=["ctx"])

            request = await builder.save_user_message_and_build_request(
                session,
                chat=chat,
                user_text="Hello there",
                telegram_chat_id="telegram-chat",
                failure_log_message="Failed to generate response",
                current_time=now,
            )

        saved_messages = list(
            (
                await session.execute(
                    select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
                )
            ).scalars()
        )
        assert [item.role for item in saved_messages] == ["user"]
        assert saved_messages[0].content == "Hello there"
        assert request.chat.id == chat.id
        assert request.llm_messages == ["ctx"]
        assert request.telegram_chat_id == "telegram-chat"
        assert request.extra_params == {"temperature": 0.2}
        assert request.resolved_model == "resolved/model-id"
        assert request.model_for_api == "resolved/model-id"
        build_mcp_manager.assert_called_once()
        prompt_builder_cls.return_value.build_context.assert_awaited_once_with(
            chat,
            current_time=now,
            send_datetime=True,
            chat_timezone="UTC",
            cut_above_message_id=None,
            image_urls=None,
        )
        assert build_mcp_manager.call_args.args[2] is wiki_store
        wiki_store.sync_from_disk.assert_awaited_once_with(chat.id)

    async def test_build_request_reuses_existing_chat_state(self, session: AsyncSession) -> None:
        builder = _make_builder()
        chat = _make_chat()
        session.add(chat)
        session.add(Message(chat_id=chat.id, role="user", content="Stored question"))
        await session.commit()
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch("mai_gram.bot.assistant_turn_builder.PromptBuilder") as prompt_builder_cls,
            patch("mai_gram.bot.assistant_turn_builder.WikiStore", return_value=wiki_store),
        ):
            prompt_builder_cls.return_value.build_context = AsyncMock(
                return_value=[SimpleNamespace(role="user", content="Stored question")]
            )

            request = await builder.build_request(
                session,
                chat=chat,
                telegram_chat_id="telegram-chat",
                failure_log_message="Failed to regenerate response",
            )

        assert request.failure_log_message == "Failed to regenerate response"
        assert request.timezone_name == "UTC"
        assert request.show_reasoning is True
        assert request.show_tool_calls is True
        prompt_builder_cls.return_value.build_context.assert_awaited_once()
        assert builder._build_mcp_manager.call_args.args[2] is wiki_store
        wiki_store.sync_from_disk.assert_awaited_once_with(chat.id)


class TestMaxOutputTokensInjection:
    """Verify max_output_tokens is applied as max_tokens in extra_params."""

    async def test_injects_max_tokens_when_max_output_tokens_set(
        self, session: AsyncSession
    ) -> None:
        builder = _make_builder(max_output_tokens=16384)
        chat = _make_chat()
        session.add(chat)
        await session.commit()
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch("mai_gram.bot.assistant_turn_builder.PromptBuilder") as pb_cls,
            patch("mai_gram.bot.assistant_turn_builder.WikiStore", return_value=wiki_store),
        ):
            pb_cls.return_value.build_context = AsyncMock(return_value=[])

            request = await builder.build_request(
                session,
                chat=chat,
                telegram_chat_id="tg-chat",
                failure_log_message="fail",
            )

        assert request.extra_params["max_tokens"] == 16384
        assert request.extra_params["temperature"] == 0.2

    async def test_explicit_max_tokens_takes_precedence(self, session: AsyncSession) -> None:
        builder = _make_builder(
            model_params={"temperature": 0.5, "max_tokens": 65536},
            max_output_tokens=16384,
        )
        chat = _make_chat()
        session.add(chat)
        await session.commit()
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch("mai_gram.bot.assistant_turn_builder.PromptBuilder") as pb_cls,
            patch("mai_gram.bot.assistant_turn_builder.WikiStore", return_value=wiki_store),
        ):
            pb_cls.return_value.build_context = AsyncMock(return_value=[])

            request = await builder.build_request(
                session,
                chat=chat,
                telegram_chat_id="tg-chat",
                failure_log_message="fail",
            )

        assert request.extra_params["max_tokens"] == 65536

    async def test_zero_max_output_tokens_does_not_inject(self, session: AsyncSession) -> None:
        builder = _make_builder(max_output_tokens=0)
        chat = _make_chat()
        session.add(chat)
        await session.commit()
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch("mai_gram.bot.assistant_turn_builder.PromptBuilder") as pb_cls,
            patch("mai_gram.bot.assistant_turn_builder.WikiStore", return_value=wiki_store),
        ):
            pb_cls.return_value.build_context = AsyncMock(return_value=[])

            request = await builder.build_request(
                session,
                chat=chat,
                telegram_chat_id="tg-chat",
                failure_log_message="fail",
            )

        assert "max_tokens" not in request.extra_params
