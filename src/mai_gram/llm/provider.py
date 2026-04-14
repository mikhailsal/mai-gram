"""Abstract LLM provider interface.

Defines the contract that all LLM backends (OpenRouter, local models, etc.)
must implement.  Also houses shared data classes for messages, responses,
and token-usage accounting.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


# ---------------------------------------------------------------------------
# Message primitives
# ---------------------------------------------------------------------------

class MessageRole(str, enum.Enum):
    """Roles that a chat message can have."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Tool/function definition exposed to the model."""

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool call emitted by the model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


# ---------------------------------------------------------------------------
# Token usage & response
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class LLMResponse:
    """Complete (non-streaming) response from an LLM."""

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A single token/chunk emitted during streaming."""

    content: str
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM provider errors."""


class LLMAuthenticationError(LLMError):
    """Raised when the API key is invalid or missing."""


class LLMRateLimitError(LLMError):
    """Raised when the provider returns a rate-limit response."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMModelNotFoundError(LLMError):
    """Raised when the requested model identifier is unknown."""


class LLMContextLengthError(LLMError):
    """Raised when the prompt exceeds the model's context window."""


class LLMProviderError(LLMError):
    """Raised for generic / unexpected provider-side errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface that every LLM backend must implement.

    Subclasses only need to override the three abstract methods below.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request and return the full response.

        Parameters
        ----------
        messages:
            The conversation history (system + user + assistant turns).
        model:
            Override the default model for this call.
        temperature:
            Sampling temperature (0.0 = deterministic, 2.0 = very random).
        max_tokens:
            Maximum number of tokens to generate.  ``None`` lets the
            provider decide.
        tools:
            Optional list of tool definitions the model may call.
            When provided, the model can emit ``tool_calls`` in the
            response instead of (or alongside) text content.
        tool_choice:
            Controls how the model selects tools.  Common values:
            ``"auto"`` (model decides), ``"none"`` (no tools),
            ``"required"`` (must call a tool), or a dict specifying
            a particular function
            (``{"type": "function", "function": {"name": "..."}}``)
        """

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat-completion response token-by-token.

        Yields ``StreamChunk`` objects.  The final chunk will carry a
        non-``None`` ``finish_reason``.

        Parameters
        ----------
        messages:
            The conversation history (system + user + assistant turns).
        model:
            Override the default model for this call.
        temperature:
            Sampling temperature (0.0 = deterministic, 2.0 = very random).
        max_tokens:
            Maximum number of tokens to generate.  ``None`` lets the
            provider decide.
        tools:
            Optional list of tool definitions the model may call.
        tool_choice:
            Controls how the model selects tools (``"auto"``,
            ``"none"``, ``"required"``, or a function-specific dict).
        """
        # The yield below is only so that type-checkers see this as an
        # async generator; subclasses will provide the real implementation.
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        """Return an *approximate* token count for the given messages.

        This is used by the memory/prompt builder to stay within the
        context window.  Exact counts are model-specific; a reasonable
        heuristic (e.g. ``len(text) // 4``) is acceptable.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the provider (HTTP clients, etc.)."""
