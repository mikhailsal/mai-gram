"""LLM provider abstraction and OpenRouter client.

Public API:
    - LLMProvider: Abstract base class for all LLM backends
    - OpenRouterProvider: OpenRouter.ai implementation
    - ChatMessage, MessageRole: Message primitives
    - LLMResponse, StreamChunk, TokenUsage: Response types
    - LLMError and subclasses: Typed error hierarchy
"""

from mai_companion.llm.openrouter import OpenRouterProvider
from mai_companion.llm.provider import (
    ChatMessage,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    MessageRole,
    StreamChunk,
    TokenUsage,
)

__all__ = [
    "ChatMessage",
    "LLMAuthenticationError",
    "LLMContextLengthError",
    "LLMError",
    "LLMModelNotFoundError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "MessageRole",
    "OpenRouterProvider",
    "StreamChunk",
    "TokenUsage",
]
