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
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch(
                "mai_gram.core.prompt_preview_service.PromptBuilder",
                return_value=prompt_builder,
            ) as prompt_builder_cls,
            patch("mai_gram.core.prompt_preview_service.WikiStore", return_value=wiki_store),
        ):
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
        assert factory.build_manager.call_args.args[2] is wiki_store
        wiki_store.sync_from_disk.assert_awaited_once_with("test-user@test-bot")
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

    async def test_init_builds_mcp_factory_with_external_pool(self) -> None:
        llm = MagicMock()
        llm.count_tokens = AsyncMock(return_value=1)
        settings = MagicMock()
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=_make_chat()))
        )
        prompt_builder = MagicMock()
        prompt_builder.build_context = AsyncMock(return_value=[SimpleNamespace(content="system")])
        mcp_manager = MagicMock()
        mcp_manager.list_all_tools = AsyncMock(return_value=[])
        external_mcp_pool = MagicMock()
        wiki_store = MagicMock(sync_from_disk=AsyncMock())

        with (
            patch(
                "mai_gram.core.prompt_preview_service.PromptBuilder",
                return_value=prompt_builder,
            ),
            patch(
                "mai_gram.core.prompt_preview_service.MCPManagerFactory",
            ) as factory_cls,
            patch("mai_gram.core.prompt_preview_service.WikiStore", return_value=wiki_store),
        ):
            factory = MagicMock(build_manager=MagicMock(return_value=mcp_manager))
            factory_cls.return_value = factory

            service = PromptPreviewService(
                llm,
                settings,
                memory_data_dir="data",
                external_mcp_pool=external_mcp_pool,
            )
            await service.build_preview(session, chat_id="test-user@test-bot")

        factory_cls.assert_called_once_with(settings, external_mcp_pool=external_mcp_pool)
        assert factory.build_manager.call_args.args[2] is wiki_store
        wiki_store.sync_from_disk.assert_awaited_once_with("test-user@test-bot")
