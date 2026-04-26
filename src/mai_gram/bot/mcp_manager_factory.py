"""Build MCP managers with global and per-prompt filtering applied."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mai_gram.mcp_servers.manager import MCPManager
from mai_gram.mcp_servers.messages_server import MessagesMCPServer
from mai_gram.mcp_servers.wiki_server import WikiMCPServer

if TYPE_CHECKING:
    from mai_gram.config import PromptConfig, Settings
    from mai_gram.db.models import Chat
    from mai_gram.mcp_servers.external import ExternalMCPPool
    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore


class MCPManagerFactory:
    """Create `MCPManager` instances for one chat turn."""

    def __init__(
        self,
        settings: Settings,
        *,
        external_mcp_pool: ExternalMCPPool | None = None,
    ) -> None:
        self._settings = settings
        self._external_mcp_pool = external_mcp_pool

    def build_manager(
        self,
        chat: Chat,
        message_store: MessageStore,
        wiki_store: WikiStore,
    ) -> MCPManager:
        prompt_cfg = self._prompt_config(chat)
        enabled_tools, disabled_tools = self._tool_filters(prompt_cfg)
        mcp_manager = MCPManager(
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
        )

        if self._is_server_allowed("messages", prompt_cfg):
            mcp_manager.register_server(
                "messages",
                MessagesMCPServer(message_store, chat.id),
            )
        if self._is_server_allowed("wiki", prompt_cfg):
            mcp_manager.register_server(
                "wiki",
                WikiMCPServer(wiki_store, chat.id),
            )
        if self._external_mcp_pool is not None:
            for server_name, server in self._external_mcp_pool.get_all_servers().items():
                if self._is_server_allowed(server_name, prompt_cfg):
                    mcp_manager.register_server(f"ext:{server_name}", server)

        return mcp_manager

    def _prompt_config(self, chat: Chat) -> PromptConfig | None:
        if not chat.prompt_name:
            return None
        return self._settings.get_prompt_config(chat.prompt_name)

    def _tool_filters(
        self,
        prompt_cfg: PromptConfig | None,
    ) -> tuple[list[str] | None, list[str] | None]:
        enabled_tools, disabled_tools = self._settings.get_tool_filter()
        if prompt_cfg is not None and (
            prompt_cfg.tools_enabled is not None or prompt_cfg.tools_disabled is not None
        ):
            enabled_tools = prompt_cfg.tools_enabled
            disabled_tools = prompt_cfg.tools_disabled
        return enabled_tools, disabled_tools

    @staticmethod
    def _is_server_allowed(name: str, prompt_cfg: PromptConfig | None) -> bool:
        if prompt_cfg is None:
            return True
        if prompt_cfg.mcp_servers_enabled is not None:
            return name in prompt_cfg.mcp_servers_enabled
        if prompt_cfg.mcp_servers_disabled is not None:
            return name not in prompt_cfg.mcp_servers_disabled
        return True
