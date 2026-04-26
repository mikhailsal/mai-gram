"""Unit tests for MCP manager composition and filtering."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mai_gram.bot.mcp_manager_factory import MCPManagerFactory
from mai_gram.config import PromptConfig
from mai_gram.db.models import Chat
from mai_gram.mcp_servers.messages_server import MCPToolSpec


class _ExternalServer:
    async def list_tools(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                name="ext_tool",
                description="External tool",
                input_schema={"type": "object"},
            )
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        del tool_name, arguments
        return {"ok": True}


def _make_chat(*, prompt_name: str | None = "helper") -> Chat:
    return Chat(
        id="test-user@test-bot",
        user_id="test-user",
        bot_id="test-bot",
        llm_model="test-model",
        system_prompt="test prompt",
        prompt_name=prompt_name,
    )


class TestMCPManagerFactory:
    async def test_uses_prompt_server_filter_and_global_tool_whitelist(self) -> None:
        settings = MagicMock()
        settings.get_tool_filter.return_value = (["wiki_read", "ext_tool"], None)
        settings.get_prompt_config.return_value = PromptConfig(
            mcp_servers_disabled=["messages"],
        )
        external_pool = MagicMock()
        external_pool.get_all_servers.return_value = {"extsrv": _ExternalServer()}
        factory = MCPManagerFactory(settings, external_mcp_pool=external_pool)

        manager = factory.build_manager(
            _make_chat(),
            message_store=MagicMock(),
            wiki_store=MagicMock(),
        )

        assert set(manager._servers) == {"wiki", "ext:extsrv"}
        tools = await manager.list_all_tools()
        assert sorted(tool.name for tool in tools) == ["ext_tool", "wiki_read"]

    async def test_prompt_tool_override_takes_precedence_over_global_blacklist(self) -> None:
        settings = MagicMock()
        settings.get_tool_filter.return_value = (None, ["search_messages"])
        settings.get_prompt_config.return_value = PromptConfig(
            tools_enabled=["search_messages"],
            mcp_servers_enabled=["messages"],
        )
        factory = MCPManagerFactory(settings)

        manager = factory.build_manager(
            _make_chat(),
            message_store=MagicMock(),
            wiki_store=MagicMock(),
        )

        assert set(manager._servers) == {"messages"}
        tools = await manager.list_all_tools()
        assert [tool.name for tool in tools] == ["search_messages"]
