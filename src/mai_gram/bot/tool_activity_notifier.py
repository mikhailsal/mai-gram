"""Tool-call persistence and user-facing display helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.bot.conversation_executor import AssistantTurnRequest
    from mai_gram.llm.provider import ToolCall
    from mai_gram.messenger.base import Messenger


class ToolActivityNotifier:
    """Persist tool activity and optionally surface it to the user."""

    def __init__(self, messenger: Messenger) -> None:
        self._messenger = messenger

    def build_callbacks(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
    ) -> tuple[
        Callable[..., Awaitable[None]],
        Callable[..., Awaitable[None]],
    ]:
        async def _on_tool_call_display(*, content: str, tool_calls: list[ToolCall]) -> None:
            await request.message_store.save_message(
                request.chat.id,
                "assistant",
                content or "",
                tool_calls=tool_calls,
                timezone_name=request.timezone_name,
                show_datetime=request.show_datetime,
            )
            await self._maybe_send_tool_call_display(request, sent_msg_ids, tool_calls)

        async def _on_tool_result_display(
            *,
            tool_call_id: str,
            tool_name: str,
            arguments: str,
            result: object,
            content: str,
            error: str | None,
            server_name: str | None,
        ) -> None:
            del arguments, server_name
            await request.message_store.save_message(
                request.chat.id,
                "tool",
                content,
                tool_call_id=tool_call_id,
                timezone_name=request.timezone_name,
                show_datetime=request.show_datetime,
            )
            await self._maybe_send_tool_result_display(
                request,
                sent_msg_ids,
                tool_name=tool_name,
                result=result,
                error=error,
            )

        return _on_tool_call_display, _on_tool_result_display

    async def _maybe_send_tool_call_display(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        tool_calls: list[ToolCall],
    ) -> None:
        if not request.show_tool_calls:
            return
        lines = self.tool_call_lines(tool_calls)
        if not lines:
            return
        result = await self._messenger.send_message(
            OutgoingMessage(text="\n".join(lines), chat_id=request.telegram_chat_id)
        )
        if result.success and result.message_id:
            sent_msg_ids.append(result.message_id)

    @staticmethod
    def tool_call_lines(tool_calls: list[ToolCall]) -> list[str]:
        if not tool_calls:
            return []
        lines: list[str] = []
        for tool_call in tool_calls:
            name = tool_call.name
            args = tool_call.arguments
            try:
                args_dict = json.loads(args) if isinstance(args, str) else args
                args_str = ", ".join(f"{key}={value!r}" for key, value in args_dict.items())
            except (json.JSONDecodeError, TypeError, AttributeError):
                args_str = str(args)
            lines.append(f"🔧 {name}({args_str})")
        return lines

    async def _maybe_send_tool_result_display(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        *,
        tool_name: str,
        result: object,
        error: str | None,
    ) -> None:
        if not request.show_tool_calls:
            return
        result_text = self.tool_result_text(tool_name=tool_name, result=result, error=error)
        tool_result = await self._messenger.send_message(
            OutgoingMessage(text=result_text, chat_id=request.telegram_chat_id)
        )
        if tool_result.success and tool_result.message_id:
            sent_msg_ids.append(tool_result.message_id)

    @staticmethod
    def tool_result_text(*, tool_name: str, result: object, error: str | None) -> str:
        if error:
            return f"❌ {tool_name}: {error}"
        result_str = str(result) if result is not None else ""
        if len(result_str) > 200:
            result_str = result_str[:200] + "…"
        return f"✅ {tool_name}: {result_str}" if result_str else f"✅ {tool_name}"
