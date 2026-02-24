"""Telegram bot message handlers.

This module contains the main message handling logic that connects
incoming Telegram messages to the companion's conversation engine.
It handles:

- /start command (triggers onboarding)
- /reset command (resets the companion)
- /mood command (shows current mood)
- Regular messages (conversation)
- Callback queries (button presses)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable

from sqlalchemy import select

from mai_companion.bot.middleware import MessageLogger, RateLimiter, RateLimitConfig
from mai_companion.bot.onboarding import OnboardingManager, OnboardingResult, OnboardingState
from mai_companion.clock import Clock
from mai_companion.config import get_settings
from mai_companion.core.prompt_builder import PromptBuilder
from mai_companion.db.database import get_session
from mai_companion.db.models import Companion, Message
from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_companion.llm.translation import TranslationService
from mai_companion.mcp_servers.bridge import run_with_tools
from mai_companion.mcp_servers.manager import MCPManager
from mai_companion.mcp_servers.messages_server import MessagesMCPServer
from mai_companion.mcp_servers.sleep_server import SleepMCPServer
from mai_companion.mcp_servers.wiki_server import WikiMCPServer
from mai_companion.memory.forgetting import ForgettingEngine
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.manager import MemoryManager
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore
from mai_companion.memory.summarizer import MemorySummarizer
from mai_companion.messenger.base import IncomingMessage, MessageType, OutgoingMessage
from mai_companion.personality.character import CharacterBuilder, CharacterConfig
from mai_companion.personality.mood import MoodManager
from mai_companion.personality.temperature import compute_temperature

if TYPE_CHECKING:
    from mai_companion.messenger.base import Messenger

logger = logging.getLogger(__name__)


class BotHandler:
    """Main handler for bot messages and commands.

    Coordinates between the messenger, database, LLM provider,
    and onboarding flow.

    Parameters
    ----------
    messenger:
        The messenger instance for sending/receiving messages.
    llm_provider:
        The LLM provider for generating responses.
    rate_limit_config:
        Optional rate limiting configuration.
    """

    def __init__(
        self,
        messenger: Messenger,
        llm_provider: LLMProvider,
        *,
        rate_limit_config: RateLimitConfig | None = None,
        memory_data_dir: str | None = None,
        summary_threshold: int | None = None,
        wiki_context_limit: int | None = None,
        short_term_limit: int | None = None,
        tool_max_iterations: int | None = None,
        clock_provider: Callable[[str], Clock] | None = None,
        test_mode: bool = False,
    ) -> None:
        self._messenger = messenger
        self._llm = llm_provider
        self._translation_service = TranslationService(llm_provider)
        self._onboarding = OnboardingManager(messenger, self._translation_service)
        self._rate_limiter = RateLimiter(
            rate_limit_config,
            on_rate_limited=self._handle_rate_limited,
        )
        self._message_logger = MessageLogger(log_content=False)
        self._test_mode = test_mode

        # Load allowed users from config
        settings = get_settings()
        self._memory_data_dir = memory_data_dir or settings.memory_data_dir
        self._summary_threshold = summary_threshold or settings.summary_threshold
        self._wiki_context_limit = wiki_context_limit or settings.wiki_context_limit
        self._short_term_limit = short_term_limit or settings.short_term_limit
        self._tool_max_iterations = tool_max_iterations or settings.tool_max_iterations
        self._clock_provider = clock_provider or (lambda _chat_id: Clock())
        self._allowed_users = settings.get_allowed_user_ids()
        if self._allowed_users:
            logger.info(
                "Access control enabled: %d user(s) allowed",
                len(self._allowed_users),
            )
        else:
            logger.warning(
                "Access control DISABLED: anyone can use this bot. "
                "Set ALLOWED_USERS in .env to restrict access."
            )

        # Register handlers
        messenger.register_command_handler("start", self._handle_start)
        messenger.register_command_handler("reset", self._handle_reset)
        messenger.register_command_handler("mood", self._handle_mood)
        messenger.register_command_handler("help", self._handle_help)
        messenger.register_message_handler(self._handle_message)
        messenger.register_callback_handler(self._handle_callback)

    async def _handle_rate_limited(self, user_id: str, chat_id: str) -> None:
        """Handle a rate-limited user.

        Parameters
        ----------
        user_id:
            The user's identifier.
        chat_id:
            The chat identifier.
        """
        # Get the companion to respond in character
        async with get_session() as session:
            companion = await self._get_companion_by_chat(session, chat_id)
            if companion:
                language = companion.human_language
                name = companion.name
            else:
                language = "English"
                name = "I"

        if language.lower() == "english":
            msg = (
                f"Whoa, slow down! {name} need{'s' if name != 'I' else ''} a moment "
                "to catch up. Let's take a short break."
            )
        else:
            msg = await self._translation_service.translate(
                f"Whoa, slow down! I need a moment to catch up. Let's take a short break.",
                language,
            )

        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=chat_id)
        )

    async def _check_access(self, message: IncomingMessage) -> bool:
        """Check if the user is allowed to use the bot.

        Parameters
        ----------
        message:
            The incoming message.

        Returns
        -------
        bool
            True if the user is allowed, False otherwise.
        """
        # If no allowed users configured, allow everyone
        if not self._allowed_users:
            return True

        # Check if user is in the allowed list
        if message.user_id in self._allowed_users:
            return True

        # User not allowed - log and send rejection message
        logger.warning(
            "Access denied for user_id=%s (not in ALLOWED_USERS)",
            message.user_id,
        )
        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    "🚫 Access denied.\n\n"
                    "This is a private bot. If you believe you should have access, "
                    "please contact the bot owner and provide your user ID: "
                    f"`{message.user_id}`"
                ),
                chat_id=message.chat_id,
            )
        )
        return False

    async def _handle_start(self, message: IncomingMessage) -> None:
        """Handle the /start command.

        Parameters
        ----------
        message:
            The incoming message.
        """
        self._message_logger.log_incoming(message)

        # Check access control
        if not await self._check_access(message):
            return

        # Check if a companion already exists for this chat
        async with get_session() as session:
            companion = await self._get_companion_by_chat(session, message.chat_id)

        if companion:
            # Companion already exists
            msg = f"We've already met! I'm {companion.name}, remember?\n\n"
            msg += "If you want to start over with a new companion, use /reset."

            if companion.human_language.lower() != "english":
                msg = await self._translation_service.translate(msg, companion.human_language)

            await self._messenger.send_message(
                OutgoingMessage(text=msg, chat_id=message.chat_id)
            )
            return

        # Start onboarding
        await self._onboarding.start_onboarding(message.user_id, message.chat_id)

    async def _handle_reset(self, message: IncomingMessage) -> None:
        """Handle the /reset command.

        Parameters
        ----------
        message:
            The incoming message.
        """
        self._message_logger.log_incoming(message)

        # Check access control
        if not await self._check_access(message):
            return

        async with get_session() as session:
            companion = await self._get_companion_by_chat(session, message.chat_id)

            if companion:
                # Delete the companion (cascade deletes messages, etc.)
                await session.delete(companion)
                await session.commit()

                msg = (
                    "I've been reset. Our conversation history is gone, "
                    "and I no longer remember who I was.\n\n"
                    "Use /start to create a new companion."
                )
            else:
                msg = "There's no companion to reset. Use /start to create one."

        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=message.chat_id)
        )

        # Clear any onboarding session
        self._onboarding.clear_session(message.user_id)

    async def _handle_mood(self, message: IncomingMessage) -> None:
        """Handle the /mood command.

        Parameters
        ----------
        message:
            The incoming message.
        """
        self._message_logger.log_incoming(message)

        # Check access control
        if not await self._check_access(message):
            return

        async with get_session() as session:
            companion = await self._get_companion_by_chat(session, message.chat_id)

            if not companion:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="No companion exists yet. Use /start to create one.",
                        chat_id=message.chat_id,
                    )
                )
                return

            traits = json.loads(companion.personality_traits)
            mood_manager = MoodManager(session)
            mood = await mood_manager.get_current_mood(companion.id, traits)

            # Build mood display message
            msg = (
                f"**{companion.name}'s current mood:**\n\n"
                f"🎭 Feeling: {mood.label.title()}\n"
                f"📊 Valence: {mood.coordinates.valence:.2f} "
                f"({'positive' if mood.coordinates.valence > 0 else 'negative' if mood.coordinates.valence < 0 else 'neutral'})\n"
                f"⚡ Arousal: {mood.coordinates.arousal:.2f} "
                f"({'energetic' if mood.coordinates.arousal > 0 else 'calm' if mood.coordinates.arousal < 0 else 'neutral'})\n"
            )
            if mood.cause:
                msg += f"💭 Cause: {mood.cause}\n"

            if companion.human_language.lower() != "english":
                msg = await self._translation_service.translate(msg, companion.human_language)

        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=message.chat_id)
        )

    async def _handle_help(self, message: IncomingMessage) -> None:
        """Handle the /help command.

        Parameters
        ----------
        message:
            The incoming message.
        """
        self._message_logger.log_incoming(message)

        # Check access control
        if not await self._check_access(message):
            return

        msg = (
            "**Available commands:**\n\n"
            "/start - Create a new AI companion\n"
            "/reset - Reset and delete the current companion\n"
            "/mood - Check your companion's current mood\n"
            "/help - Show this help message\n\n"
            "Just send a message to chat with your companion!"
        )

        async with get_session() as session:
            companion = await self._get_companion_by_chat(session, message.chat_id)
            if companion and companion.human_language.lower() != "english":
                msg = await self._translation_service.translate(msg, companion.human_language)

        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=message.chat_id)
        )

    async def _handle_message(self, message: IncomingMessage) -> None:
        """Handle a regular text message.

        Parameters
        ----------
        message:
            The incoming message.
        """
        self._message_logger.log_incoming(message)

        # Check access control
        if not await self._check_access(message):
            return

        # Check rate limit
        if not await self._rate_limiter.check_rate_limit(message.user_id, message.chat_id):
            return

        # Check if user is in onboarding
        if self._onboarding.is_onboarding(message.user_id):
            result = await self._onboarding.handle_message(message)
            if result:
                await self._create_companion(message.chat_id, result.config)
                self._onboarding.clear_session(message.user_id)
            return

        # Regular conversation
        await self._handle_conversation(message)

    async def _handle_callback(self, message: IncomingMessage) -> None:
        """Handle a callback query (button press).

        Parameters
        ----------
        message:
            The incoming message with callback data.
        """
        self._message_logger.log_incoming(message)

        # Check access control
        if not await self._check_access(message):
            return

        # Check if user is in onboarding
        if self._onboarding.is_onboarding(message.user_id):
            result = await self._onboarding.handle_message(message)
            if result:
                await self._create_companion(message.chat_id, result.config)
                self._onboarding.clear_session(message.user_id)
            return

        # Other callbacks (future features)
        logger.debug("Unhandled callback: %s", message.callback_data)

    async def _handle_conversation(self, message: IncomingMessage) -> None:
        """Handle a conversation message.

        Parameters
        ----------
        message:
            The incoming message.
        """
        async with get_session() as session:
            companion = await self._get_companion_by_chat(session, message.chat_id)

            if not companion:
                # No companion exists
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="I don't exist yet! Use /start to create me.",
                        chat_id=message.chat_id,
                    )
                )
                return

            # Show typing indicator
            await self._messenger.send_typing_indicator(message.chat_id)

            message_store = MessageStore(session)
            wiki_store = WikiStore(session, data_dir=self._memory_data_dir)
            summary_store = SummaryStore(data_dir=self._memory_data_dir)
            summarizer = MemorySummarizer(
                message_store,
                summary_store,
                self._llm,
                summary_threshold=self._summary_threshold,
                wiki_store=wiki_store,
                companion_name=companion.name,
                companion_model=companion.llm_model,
            )
            forgetting_engine = ForgettingEngine(summary_store, summarizer)
            memory_manager = MemoryManager(
                message_store,
                summary_store,
                wiki_store,
                summarizer,
                forgetting_engine,
            )
            prompt_builder = PromptBuilder(
                self._llm,
                message_store,
                wiki_store,
                summary_store,
                wiki_context_limit=self._wiki_context_limit,
                short_term_limit=self._short_term_limit,
                test_mode=self._test_mode,
            )

            mcp_manager = MCPManager()
            clock = self._clock_provider(message.chat_id)
            mcp_manager.register_server(
                "messages",
                MessagesMCPServer(message_store, companion.id),
            )
            mcp_manager.register_server(
                "wiki",
                WikiMCPServer(wiki_store, companion.id, clock=clock),
            )
            mcp_manager.register_server("sleep", SleepMCPServer())

            # Save the human's message
            await memory_manager.save_message(
                companion.id,
                "user",
                message.text,
                clock=clock,
                is_proactive=False,
                trigger_summary=True,
            )

            # Get current mood
            traits = json.loads(companion.personality_traits)
            mood_manager = MoodManager(session)
            mood = await mood_manager.get_current_mood(companion.id, traits)

            llm_messages = await prompt_builder.build_context(companion, mood, clock=clock)
            debug_tool_observer = getattr(self._llm, "record_tool_execution", None)
            if not callable(debug_tool_observer):
                debug_tool_observer = None

            # Callback to deliver intermediate messages (multi-message support).
            # When the LLM produces text alongside a tool call (e.g. sleep),
            # this sends the text immediately so the human sees it as a
            # separate message.
            intermediate_parts: list[str] = []

            async def _send_intermediate(text: str) -> None:
                intermediate_parts.append(text)
                await self._messenger.send_message(
                    OutgoingMessage(text=text, chat_id=message.chat_id)
                )
                await self._messenger.send_typing_indicator(message.chat_id)

            # Callback to save assistant messages with tool calls to the database.
            # This ensures the AI "remembers" that it used tools in past conversations.
            async def _save_assistant_tool_call(content: str, tool_calls_json: str) -> None:
                await memory_manager.save_message(
                    companion.id,
                    "assistant",
                    content,
                    clock=clock,
                    is_proactive=False,
                    tool_calls=tool_calls_json,
                )

            # Callback to save tool results to the database and optionally log for debug.
            async def _save_tool_result(
                tool_call_id: str,
                tool_name: str,
                arguments: str,
                result: str | None,
                error: str | None,
                server_name: str | None,
            ) -> None:
                # Save tool result message to database
                result_content = error if error else (result or "")
                await memory_manager.save_message(
                    companion.id,
                    "tool",
                    result_content,
                    clock=clock,
                    is_proactive=False,
                    tool_call_id=tool_call_id,
                )
                # Also call debug logger if available
                if debug_tool_observer is not None:
                    debug_tool_observer(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result,
                        error=error,
                        server_name=server_name,
                    )

            # Generate response using the companion's bound model.
            # The model is the companion's "soul" -- captured at creation time
            # to protect their identity from casual model changes in .env.
            try:
                response = await run_with_tools(
                    self._llm,
                    mcp_manager,
                    llm_messages,
                    model=companion.llm_model,
                    temperature=companion.temperature,
                    max_iterations=self._tool_max_iterations,
                    on_tool_result=_save_tool_result,
                    on_intermediate_content=_send_intermediate,
                    on_assistant_tool_call=_save_assistant_tool_call,
                )
                response_text = response.content

                # NOTE: Intermediate messages that came with tool calls are now
                # saved via _save_assistant_tool_call (with tool_calls field set).
                # We no longer save them separately here to avoid duplicates.
                # The intermediate_parts list is still used for UI delivery only.

                # Save the companion's final response (may be empty if
                # everything was delivered via intermediate messages).
                # This is the response WITHOUT tool calls (the final answer).
                if response_text and response_text.strip():
                    await memory_manager.save_message(
                        companion.id,
                        "assistant",
                        response_text,
                        clock=clock,
                        is_proactive=False,
                    )
                await memory_manager.run_forgetting_cycle(companion.id, clock=clock)

                # Update mood based on conversation
                from mai_companion.personality.mood import evaluate_message_sentiment
                sentiment, intensity = evaluate_message_sentiment(message.text)
                if abs(sentiment) > 0.1 or intensity > 0.2:
                    await mood_manager.apply_reactive_shift(
                        companion.id,
                        sentiment,
                        intensity,
                        f"conversation: {message.text[:50]}",
                        traits,
                    )

            except Exception as e:
                logger.exception("Failed to generate response")
                response_text = (
                    "I'm having trouble thinking right now. "
                    "Could you try again in a moment?"
                )
                if companion.human_language.lower() != "english":
                    response_text = await self._translation_service.translate(
                        response_text, companion.human_language
                    )

        # Send the final response (skip if empty — everything was already
        # delivered through intermediate messages).
        if response_text and response_text.strip():
            result = await self._messenger.send_message(
                OutgoingMessage(text=response_text, chat_id=message.chat_id)
            )
            self._message_logger.log_outgoing(
                message.chat_id,
                response_text,
                success=result.success,
                message_id=result.message_id,
            )

    async def _create_companion(
        self,
        chat_id: str,
        config: CharacterConfig,
    ) -> None:
        """Create a new companion from the completed onboarding.

        Parameters
        ----------
        chat_id:
            The chat identifier.
        config:
            The character configuration.
        """
        temperature = compute_temperature(config.traits)
        # Capture the current LLM model at creation time.
        # This becomes the companion's "soul" -- the fundamental substrate
        # that processes their memories and personality. Changing the model
        # in .env will only affect NEW companions, not existing ones.
        settings = get_settings()
        record = CharacterBuilder.create_companion_record(
            config, temperature, llm_model=settings.llm_model
        )

        # Add chat_id as the companion ID for easy lookup
        # In a multi-user scenario, you'd generate a UUID and store
        # a mapping, but for single-user self-hosted this works fine
        record["id"] = chat_id

        async with get_session() as session:
            companion = Companion(**record)
            session.add(companion)
            await session.flush()  # Get the companion ID

            # Create initial mood state
            mood_manager = MoodManager(session)
            baseline = mood_manager.compute_baseline(config.traits)
            from mai_companion.personality.mood import resolve_label
            label = resolve_label(baseline)
            await mood_manager._save_mood(
                companion.id,
                baseline,
                label,
                "initial baseline at companion creation",
            )

            # NOTE: We don't save any initial messages here.
            # The user will send the first message, and the companion will respond.
            # This ensures proper message ordering: System → User → Assistant
            # (which is required by chat completion APIs)

            await session.commit()

        logger.info(
            "Created companion: name=%s, language=%s, temp=%.2f",
            config.name,
            config.language,
            temperature,
        )

    async def _get_companion_by_chat(
        self, session, chat_id: str
    ) -> Companion | None:
        """Get the companion for a chat.

        Also sets up the language style in the translation service if
        the companion has one configured.

        Parameters
        ----------
        session:
            The database session.
        chat_id:
            The chat identifier.

        Returns
        -------
        Companion or None
            The companion, or None if not found.
        """
        result = await session.execute(
            select(Companion).where(Companion.id == chat_id)
        )
        companion = result.scalar_one_or_none()

        # Set up language style in translation service
        if companion:
            self._translation_service.set_language_style(
                companion.human_language,
                companion.language_style,
            )

        return companion

    async def _get_recent_messages(
        self, session, companion_id: str, *, limit: int = 30
    ) -> list[Message]:
        """Get recent messages for a companion.

        Parameters
        ----------
        session:
            The database session.
        companion_id:
            The companion's ID.
        limit:
            Maximum number of messages to retrieve.

        Returns
        -------
        list[Message]
            Recent messages, newest first.
        """
        result = await session.execute(
            select(Message)
            .where(Message.companion_id == companion_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
