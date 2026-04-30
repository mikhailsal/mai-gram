"""LLM provider abstraction and OpenRouter client."""

from mai_gram.llm.openrouter import OpenRouterProvider
from mai_gram.llm.provider import (
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
    ToolCall,
    ToolDefinition,
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
    "ToolCall",
    "ToolDefinition",
]
