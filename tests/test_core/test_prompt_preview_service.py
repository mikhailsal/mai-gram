"""Tests for the prompt preview service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mai_gram.core.prompt_preview_service import PromptPreviewService
from mai_gram.db.models import Chat


def _make_chat() -> Chat:
    return Chat(
        id="test-user@test-bot",
        user_id="test-user",
        bot_id="test-bot",
        llm_model="test-model",
        system_prompt="test prompt",
        prompt_name="default",
    )


class TestPromptPreviewService:
    async def test_build_preview_reuses_prompt_builder_and_mcp_factory(self) -> None:
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=42)
        settings = MagicMock()
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=_make_chat()))
        )

        prompt_builder = MagicMock()
        prompt_builder.build_context = AsyncMock(
            return_value=[SimpleNamespace(content="system prompt")]
        )
        mcp_manager = MagicMock()
        mcp_manager.list_all_tools = AsyncMock(
            return_value=[SimpleNamespace(name="wiki_search", description="Search wiki")]
        )
        factory = MagicMock(build_manager=MagicMock(return_value=mcp_manager))

        with patch(
            "mai_gram.core.prompt_preview_service.PromptBuilder",
            return_value=prompt_builder,
        ) as prompt_builder_cls:
            service = PromptPreviewService(
                llm,
                settings,
                memory_data_dir="data",
                test_mode=False,
                mcp_manager_factory=factory,
            )
            preview = await service.build_preview(session, chat_id="test-user@test-bot")

        prompt_builder_cls.assert_called_once()
        factory.build_manager.assert_called_once()
        llm.count_tokens.assert_awaited_once_with(preview.context)
        assert preview.token_count == 42
        assert preview.context[0].content == "system prompt"
        assert preview.tools[0].name == "wiki_search"

    async def test_build_preview_raises_lookup_error_for_missing_chat(self) -> None:
        service = PromptPreviewService(
            MagicMock(count_tokens=AsyncMock(return_value=0)),
            MagicMock(),
            memory_data_dir="data",
        )
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        )

        with pytest.raises(LookupError, match="missing-chat"):
            await service.build_preview(session, chat_id="missing-chat")
