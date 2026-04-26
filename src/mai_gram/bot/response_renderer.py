"""Telegram response rendering helpers for assistant replies."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from mai_gram.llm.provider import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
)
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.bot.middleware import MessageLogger
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


class ResponseRenderer:
    """Render assistant responses into Telegram-safe message parts."""

    def __init__(self, messenger: Messenger, *, message_logger: MessageLogger) -> None:
        self._messenger = messenger
        self._message_logger = message_logger

    @staticmethod
    def _build_intermediate_display(content: str, reasoning: str, show_reasoning: bool) -> str:
        """Build display text for an intermediate turn before tool calls complete."""
        display = ""
        if show_reasoning and reasoning.strip():
            display = f"💭 Reasoning:\n{reasoning.strip()}"
            if content.strip():
                display += "\n\n───\n\n" + content
        elif content.strip():
            display = content
        return display

    @staticmethod
    def _format_usage_footer(usage: object, cost: float | None, is_byok: bool) -> str:
        """Build a compact token and cost footer string."""
        del is_byok
        if usage is None:
            return ""
        prompt_t = getattr(usage, "prompt_tokens", 0)
        comp_t = getattr(usage, "completion_tokens", 0)
        parts = [f"{prompt_t}/{comp_t} tokens"]
        if cost is not None and cost > 0:
            parts.append(f"${cost:.4f}")
        return " | ".join(parts)

    async def _deliver_error(
        self,
        chat_id: str,
        error_text: str,
        *,
        placeholder_msg_id: str | None,
        keyboard: object = None,
        sent_msg_ids: list[str],
        max_attempts: int = 5,
    ) -> None:
        """Deliver an error message to the user with retry on failure."""
        for attempt in range(1, max_attempts + 1):
            if placeholder_msg_id:
                result = await self._messenger.edit_message(
                    chat_id,
                    placeholder_msg_id,
                    error_text,
                    keyboard=keyboard,
                )
                if result.success:
                    sent_msg_ids.append(placeholder_msg_id)
                    return
            else:
                result = await self._messenger.send_message(
                    OutgoingMessage(text=error_text, chat_id=chat_id, keyboard=keyboard)
                )
                if result.success and result.message_id:
                    sent_msg_ids.append(result.message_id)
                    return
            if attempt < max_attempts:
                delay = 2.0 * attempt
                logger.warning(
                    "Failed to deliver error (attempt %d/%d), retrying in %.0fs",
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        logger.error("Could not deliver error message after %d attempts", max_attempts)

    @staticmethod
    def _user_friendly_error(exc: BaseException) -> str:
        """Map an exception to a user-facing error message with actionable advice."""
        if isinstance(exc, LLMAuthenticationError):
            return (
                "⚠️ API authentication failed.\n\n"
                "The bot's API key is invalid or expired. "
                "Please contact the bot administrator."
            )
        if isinstance(exc, LLMRateLimitError):
            msg = "⏳ Rate limit reached — the AI provider is temporarily overloaded."
            if exc.retry_after is not None:
                msg += f"\n\nPlease wait ~{int(exc.retry_after)}s and try again."
            else:
                msg += "\n\nPlease wait a moment and tap Regenerate."
            return msg
        if isinstance(exc, LLMModelNotFoundError):
            return (
                "❌ The selected model is no longer available.\n\n"
                "Use /reset and /start to choose a different model."
            )
        if isinstance(exc, LLMContextLengthError):
            return (
                "❌ The conversation is too long for this model's context window.\n\n"
                'Use "✂ Cut this & above" on a message to trim history, '
                "or /reset to start fresh."
            )
        if isinstance(exc, LLMProviderError):
            status = f" (HTTP {exc.status_code})" if exc.status_code else ""
            return (
                f"⚠️ AI provider error{status}.\n\n"
                "This is usually temporary. Tap Regenerate to retry."
            )
        if isinstance(exc, LLMError):
            return "⚠️ Something went wrong with the AI provider.\n\nTap Regenerate to retry."
        return (
            "❌ Unexpected error while generating a response.\n\n"
            "Tap Regenerate to retry, or use /reset if the problem persists."
        )

    async def _commit_overflow(
        self,
        *,
        tg_chat_id: str,
        header_html: str,
        reasoning_committed: bool,
        placeholder_msg_id: str | None,
        sent_msg_ids: list[str],
        remaining_content: str,
        current_content: str,
        committed_content_offset: int,
    ) -> tuple[int, str | None]:
        """Commit overflowing streamed content into finalized messages."""
        from mai_gram.core.md_to_telegram import markdown_to_html
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH

        if header_html and not reasoning_committed:
            if placeholder_msg_id:
                await self._edit_part(tg_chat_id, placeholder_msg_id, header_html)
                sent_msg_ids.append(placeholder_msg_id)
            else:
                message_id = await self._send_part(tg_chat_id, header_html)
                if message_id:
                    sent_msg_ids.append(message_id)
            placeholder_msg_id = None

        while len(remaining_content) > 0:
            chunk_text = remaining_content[: SAFE_MAX_LENGTH - 200]
            para_break = chunk_text.rfind("\n\n")
            if para_break > len(chunk_text) // 3:
                chunk_text = chunk_text[:para_break]
            elif (nl := chunk_text.rfind("\n")) > len(chunk_text) // 3:
                chunk_text = chunk_text[:nl]

            chunk_html = markdown_to_html(chunk_text)
            if len(chunk_html) > SAFE_MAX_LENGTH:
                chunk_html = chunk_html[: SAFE_MAX_LENGTH - 10]

            if placeholder_msg_id:
                await self._edit_part(tg_chat_id, placeholder_msg_id, chunk_html)
                sent_msg_ids.append(placeholder_msg_id)
                placeholder_msg_id = None
            else:
                message_id = await self._send_part(tg_chat_id, chunk_html)
                if message_id:
                    sent_msg_ids.append(message_id)

            committed_content_offset += len(chunk_text)
            remaining_content = current_content[committed_content_offset:]

            if len(remaining_content) <= SAFE_MAX_LENGTH - 200:
                break

        new_placeholder: str | None = None
        if remaining_content.strip():
            content_html = markdown_to_html(remaining_content) + " ▍"
            result = await self._messenger.send_message(
                OutgoingMessage(
                    text=content_html,
                    chat_id=tg_chat_id,
                    parse_mode="html",
                )
            )
            if result.success:
                new_placeholder = result.message_id
        return committed_content_offset, new_placeholder

    async def _send_part(
        self,
        chat_id: str,
        text: str,
        *,
        keyboard: object = None,
    ) -> str | None:
        """Send one message part, falling back to plain text if HTML fails."""
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH

        result = await self._messenger.send_message(
            OutgoingMessage(text=text, chat_id=chat_id, parse_mode="html", keyboard=keyboard)
        )
        if result.success:
            return result.message_id

        error = (result.error or "").lower()
        if "too long" in error or "message is too long" in error:
            return await self._send_part_split(chat_id, text, keyboard=keyboard)
        if "parse entities" in error or "can't find end tag" in error:
            plain = re.sub(r"<[^>]+>", "", text)
            if len(plain) > SAFE_MAX_LENGTH:
                return await self._send_part_split(chat_id, text, keyboard=keyboard)
            result = await self._messenger.send_message(
                OutgoingMessage(text=plain, chat_id=chat_id, keyboard=keyboard)
            )
            if result.success:
                return result.message_id
        return None

    async def _send_part_split(
        self,
        chat_id: str,
        text: str,
        *,
        keyboard: object = None,
    ) -> str | None:
        """Emergency split when a rendered part still exceeds the platform limit."""
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH, split_html_safe

        plain = re.sub(r"<[^>]+>", "", text)
        chunks = split_html_safe(plain, max_len=SAFE_MAX_LENGTH)
        last_id: str | None = None
        for index, chunk in enumerate(chunks):
            is_last = index == len(chunks) - 1
            result = await self._messenger.send_message(
                OutgoingMessage(
                    text=chunk,
                    chat_id=chat_id,
                    keyboard=keyboard if is_last else None,
                )
            )
            if result.success and result.message_id:
                last_id = result.message_id
        return last_id

    async def _send_long_message(
        self,
        chat_id: str,
        raw_text: str,
        *,
        header_html: str = "",
        keyboard: object = None,
    ) -> list[str]:
        """Send markdown content as one or more independently rendered HTML parts."""
        from mai_gram.core.md_to_telegram import markdown_to_html
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH, split_html_safe

        sent_ids: list[str] = []
        header_sent = False

        max_first = SAFE_MAX_LENGTH // 2 if header_html else SAFE_MAX_LENGTH
        raw_parts = split_html_safe(raw_text, max_len=max_first)
        if len(raw_parts) > 1:
            rest = split_html_safe("".join(raw_parts[1:]), max_len=SAFE_MAX_LENGTH)
            raw_parts = [raw_parts[0]] + rest

        for index, raw_part in enumerate(raw_parts):
            is_last = index == len(raw_parts) - 1
            html_part = markdown_to_html(raw_part)

            if index == 0 and header_html and not header_sent:
                combined = header_html + "\n\n" + html_part
                if len(combined) <= SAFE_MAX_LENGTH:
                    html_part = combined
                    header_sent = True
                else:
                    header_id = await self._send_part(chat_id, header_html)
                    if header_id:
                        sent_ids.append(header_id)
                    header_sent = True

            message_id = await self._send_part(
                chat_id,
                html_part,
                keyboard=keyboard if is_last else None,
            )
            if message_id:
                sent_ids.append(message_id)
        return sent_ids

    async def _edit_part(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        keyboard: object = None,
    ) -> bool:
        """Edit one message part, falling back to plain text if HTML fails."""
        result = await self._messenger.edit_message(
            chat_id,
            message_id,
            text,
            parse_mode="html",
            keyboard=keyboard,
        )
        if result.success:
            return True

        error = (result.error or "").lower()
        if "parse entities" in error or "can't find end tag" in error:
            plain = re.sub(r"<[^>]+>", "", text)
            result = await self._messenger.edit_message(
                chat_id,
                message_id,
                plain,
                keyboard=keyboard,
            )
            return result.success
        return False

    async def _finalize_placeholder(
        self,
        chat_id: str,
        placeholder_msg_id: str,
        raw_text: str,
        *,
        header_html: str = "",
        keyboard: object = None,
    ) -> list[str]:
        """Edit the placeholder with the first chunk and send the remainder."""
        from mai_gram.core.md_to_telegram import markdown_to_html
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH, split_html_safe

        max_first = SAFE_MAX_LENGTH // 2 if header_html else SAFE_MAX_LENGTH
        raw_parts = split_html_safe(raw_text, max_len=max_first)
        if len(raw_parts) > 1:
            rest = split_html_safe("".join(raw_parts[1:]), max_len=SAFE_MAX_LENGTH)
            raw_parts = [raw_parts[0]] + rest

        first_html = markdown_to_html(raw_parts[0])
        if header_html:
            combined = header_html + "\n\n" + first_html
            if len(combined) <= SAFE_MAX_LENGTH:
                first_html = combined
            else:
                await self._edit_part(chat_id, placeholder_msg_id, header_html)
                initial_extra_ids: list[str] = []
                for index, raw_part in enumerate(raw_parts):
                    is_last = index == len(raw_parts) - 1
                    html_part = markdown_to_html(raw_part)
                    message_id = await self._send_part(
                        chat_id,
                        html_part,
                        keyboard=keyboard if is_last else None,
                    )
                    if message_id:
                        initial_extra_ids.append(message_id)
                return initial_extra_ids

        if len(raw_parts) == 1:
            await self._edit_part(chat_id, placeholder_msg_id, first_html, keyboard=keyboard)
            return []

        await self._edit_part(chat_id, placeholder_msg_id, first_html)

        extra_ids: list[str] = []
        for index, raw_part in enumerate(raw_parts[1:], start=1):
            is_last = index == len(raw_parts) - 1
            html_part = markdown_to_html(raw_part)
            message_id = await self._send_part(
                chat_id,
                html_part,
                keyboard=keyboard if is_last else None,
            )
            if message_id:
                extra_ids.append(message_id)
        return extra_ids

    async def _send_response(
        self,
        chat_id: str,
        *,
        response_text: str | None,
        response_reasoning: str | None = None,
        show_reasoning: bool = False,
        keyboard: object = None,
    ) -> list[str]:
        """Send the final assistant response, splitting it if needed."""
        if not response_text or not response_text.strip():
            return []

        from mai_gram.core.md_to_telegram import format_reasoning_html

        header_html = ""
        if show_reasoning and response_reasoning and response_reasoning.strip():
            header_html = format_reasoning_html(response_reasoning, expandable=True)

        sent_ids = await self._send_long_message(
            chat_id,
            response_text,
            header_html=header_html,
            keyboard=keyboard,
        )
        self._message_logger.log_outgoing(
            chat_id,
            response_text,
            success=bool(sent_ids),
            message_id=sent_ids[-1] if sent_ids else None,
        )
        return sent_ids
