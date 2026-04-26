"""Service assembly helpers for `BotHandler`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
from mai_gram.bot.callback_router import CallbackRouter
from mai_gram.bot.conversation_executor import ConversationExecutor
from mai_gram.bot.conversation_service import ConversationService
from mai_gram.bot.history_actions import HistoryActions
from mai_gram.bot.import_workflow import ImportWorkflow
from mai_gram.bot.mcp_manager_factory import MCPManagerFactory
from mai_gram.bot.regenerate_service import RegenerateService
from mai_gram.bot.resend_service import ResendService
from mai_gram.bot.reset_workflow import ResetWorkflow
from mai_gram.bot.response_renderer import ResponseRenderer
from mai_gram.bot.setup_workflow import SetupWorkflow

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.bot.middleware import MessageLogger
    from mai_gram.bot.reset_workflow import ResetPresenter
    from mai_gram.config import BotConfig, Settings
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool
    from mai_gram.messenger.base import IncomingMessage, Messenger


@dataclass(frozen=True, slots=True)
class HandlerServices:
    """Concrete services wired into `BotHandler`."""

    response_renderer: ResponseRenderer
    mcp_manager_factory: MCPManagerFactory
    assistant_turn_builder: AssistantTurnBuilder
    conversation_executor: ConversationExecutor
    conversation_service: ConversationService
    import_workflow: ImportWorkflow
    history_actions: HistoryActions
    regenerate_service: RegenerateService
    resend_service: ResendService
    reset_workflow: ResetWorkflow
    setup_workflow: SetupWorkflow
    callback_router: CallbackRouter


def build_handler_services(
    messenger: Messenger,
    llm_provider: LLMProvider,
    *,
    settings: Settings,
    message_logger: MessageLogger,
    presenter: ResetPresenter,
    resolve_chat_id: Callable[[IncomingMessage], str],
    clear_setup_session: Callable[[str], None],
    show_confirmation: Callable[..., Awaitable[None]],
    delete_callback_message: Callable[[IncomingMessage], Awaitable[None]],
    cut_original_html: dict[str, tuple[str, str | None]],
    response_message_ids: dict[str, list[str]],
    memory_data_dir: str,
    wiki_context_limit: int,
    short_term_limit: int,
    tool_max_iterations: int,
    test_mode: bool,
    bot_config: BotConfig | None,
    external_mcp_pool: ExternalMCPPool | None,
) -> HandlerServices:
    response_renderer = ResponseRenderer(
        messenger,
        message_logger=message_logger,
    )
    mcp_manager_factory = MCPManagerFactory(
        settings,
        external_mcp_pool=external_mcp_pool,
    )
    assistant_turn_builder = AssistantTurnBuilder(
        llm_provider,
        settings,
        build_mcp_manager=mcp_manager_factory.build_manager,
        memory_data_dir=memory_data_dir,
        wiki_context_limit=wiki_context_limit,
        short_term_limit=short_term_limit,
        test_mode=test_mode,
    )
    conversation_executor = ConversationExecutor(
        messenger,
        llm_provider,
        tool_max_iterations=tool_max_iterations,
        renderer=response_renderer,
    )
    conversation_service = ConversationService(
        messenger,
        conversation_executor,
        turn_builder=assistant_turn_builder,
        resolve_chat_id=resolve_chat_id,
    )
    import_workflow = ImportWorkflow(
        messenger,
        settings,
        get_allowed_models=lambda: _allowed_models(settings, bot_config),
        resolve_chat_id=resolve_chat_id,
    )
    history_actions = HistoryActions(
        messenger,
        resolve_chat_id=resolve_chat_id,
    )
    regenerate_service = RegenerateService(
        messenger,
        conversation_executor,
        turn_builder=assistant_turn_builder,
        resolve_chat_id=resolve_chat_id,
    )
    resend_service = ResendService(
        messenger,
        renderer=response_renderer,
        resolve_chat_id=resolve_chat_id,
    )
    reset_workflow = ResetWorkflow(
        messenger,
        presenter=presenter,
        resolve_chat_id=resolve_chat_id,
        clear_setup_session=clear_setup_session,
        memory_data_dir=memory_data_dir,
    )
    setup_workflow = SetupWorkflow(
        messenger,
        settings,
        bot_config=bot_config,
        resolve_chat_id=resolve_chat_id,
    )
    callback_router = CallbackRouter(
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
    return HandlerServices(
        response_renderer=response_renderer,
        mcp_manager_factory=mcp_manager_factory,
        assistant_turn_builder=assistant_turn_builder,
        conversation_executor=conversation_executor,
        conversation_service=conversation_service,
        import_workflow=import_workflow,
        history_actions=history_actions,
        regenerate_service=regenerate_service,
        resend_service=resend_service,
        reset_workflow=reset_workflow,
        setup_workflow=setup_workflow,
        callback_router=callback_router,
    )


def _allowed_models(settings: Settings, bot_config: BotConfig | None) -> list[str]:
    global_models = settings.get_allowed_models()
    if bot_config and bot_config.allowed_models:
        bot_set = set(bot_config.allowed_models)
        return [model for model in global_models if model in bot_set]
    return global_models
