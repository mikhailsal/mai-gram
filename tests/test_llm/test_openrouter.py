"""Tests for the OpenRouter LLM provider.

All tests use mocked HTTP responses – no real API calls are made.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mai_companion.llm.openrouter import OpenRouterProvider
from mai_companion.llm.provider import (
    ChatMessage,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    MessageRole,
    StreamChunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API_KEY = "test-api-key-123"


def _make_chat_response(
    content: str = "Hello!",
    model: str = "openai/gpt-4o",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible chat-completion JSON response."""
    return {
        "id": "gen-abc123",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_stream_lines(
    tokens: list[str],
    model: str = "openai/gpt-4o",
    finish_reason: str = "stop",
) -> str:
    """Build SSE text for a streaming response."""
    lines: list[str] = []
    for token in tokens:
        chunk = {
            "id": "gen-abc123",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")

    # Final chunk with finish_reason
    final = {
        "id": "gen-abc123",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    lines.append(f"data: {json.dumps(final)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


def _make_error_response(message: str, code: int = 400) -> dict[str, Any]:
    """Build an error JSON body."""
    return {"error": {"message": message, "code": code}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """A minimal conversation for testing."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful companion."),
        ChatMessage(role=MessageRole.USER, content="Hi there!"),
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestOpenRouterConstruction:
    """Test provider instantiation and configuration."""

    def test_default_model(self) -> None:
        provider = OpenRouterProvider(api_key=API_KEY)
        assert provider.default_model == "openai/gpt-4o"

    def test_custom_model(self) -> None:
        provider = OpenRouterProvider(api_key=API_KEY, default_model="anthropic/claude-3.5-sonnet")
        assert provider.default_model == "anthropic/claude-3.5-sonnet"

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(LLMAuthenticationError, match="must not be empty"):
            OpenRouterProvider(api_key="")


# ---------------------------------------------------------------------------
# Non-streaming generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    """Test non-streaming chat completion."""

    async def test_basic_response(self, sample_messages: list[ChatMessage]) -> None:
        response_json = _make_chat_response(content="Hello, friend!")

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_json)
        )
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.generate(sample_messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello, friend!"
        assert result.model == "openai/gpt-4o"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.finish_reason == "stop"

        await provider.close()

    async def test_custom_model_override(self, sample_messages: list[ChatMessage]) -> None:
        """Model can be overridden per-call."""
        captured_request: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(200, json=_make_chat_response())

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        await provider.generate(sample_messages, model="meta-llama/llama-3-70b")

        assert captured_request["body"]["model"] == "meta-llama/llama-3-70b"
        await provider.close()

    async def test_temperature_and_max_tokens(self, sample_messages: list[ChatMessage]) -> None:
        """Temperature and max_tokens are forwarded in the request."""
        captured_request: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(200, json=_make_chat_response())

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        await provider.generate(sample_messages, temperature=0.3, max_tokens=512)

        body = captured_request["body"]
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 512
        assert body["stream"] is False
        await provider.close()

    async def test_messages_serialization(self, sample_messages: list[ChatMessage]) -> None:
        """Messages are serialized with role and content."""
        captured_request: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(200, json=_make_chat_response())

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        await provider.generate(sample_messages)

        msgs = captured_request["body"]["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are a helpful companion."}
        assert msgs[1] == {"role": "user", "content": "Hi there!"}
        await provider.close()

    async def test_no_choices_raises(self, sample_messages: list[ChatMessage]) -> None:
        """An empty choices array raises LLMProviderError."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": []})
        )
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(LLMProviderError, match="No choices"):
            await provider.generate(sample_messages)

        await provider.close()

    async def test_error_in_200_body(self, sample_messages: list[ChatMessage]) -> None:
        """Some providers return 200 with an error object in the body."""
        error_body = {"error": {"message": "Model overloaded"}}
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=error_body)
        )
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(LLMProviderError, match="Model overloaded"):
            await provider.generate(sample_messages)

        await provider.close()


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------


class TestHttpErrorMapping:
    """Test that HTTP status codes map to the correct exception types."""

    @pytest.fixture
    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(api_key=API_KEY, max_retries=0)

    async def _make_provider_with_status(
        self,
        status: int,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> OpenRouterProvider:
        resp_headers = headers or {}
        resp_body = body or _make_error_response("test error")

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                status, json=resp_body, headers=resp_headers
            )
        )
        provider = OpenRouterProvider(api_key=API_KEY, max_retries=0)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )
        return provider

    async def test_401_authentication_error(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        provider = await self._make_provider_with_status(401)
        with pytest.raises(LLMAuthenticationError, match="Authentication failed"):
            await provider.generate(sample_messages)
        await provider.close()

    async def test_404_model_not_found(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        provider = await self._make_provider_with_status(404)
        with pytest.raises(LLMModelNotFoundError, match="Model not found"):
            await provider.generate(sample_messages)
        await provider.close()

    async def test_429_rate_limit(self, sample_messages: list[ChatMessage]) -> None:
        provider = await self._make_provider_with_status(
            429, headers={"retry-after": "30"}
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            await provider.generate(sample_messages)
        assert exc_info.value.retry_after == 30.0
        await provider.close()

    async def test_400_context_length(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        body = _make_error_response("This model's context length is exceeded")
        provider = await self._make_provider_with_status(400, body=body)
        with pytest.raises(LLMContextLengthError, match="Context length"):
            await provider.generate(sample_messages)
        await provider.close()

    async def test_500_server_error(self, sample_messages: list[ChatMessage]) -> None:
        provider = await self._make_provider_with_status(500)
        with pytest.raises(LLMProviderError, match="Server error"):
            await provider.generate(sample_messages)
        await provider.close()

    async def test_generic_4xx_error(self, sample_messages: list[ChatMessage]) -> None:
        provider = await self._make_provider_with_status(422)
        with pytest.raises(LLMProviderError, match="Request failed"):
            await provider.generate(sample_messages)
        await provider.close()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test that transient failures are retried."""

    async def test_retries_on_500(self, sample_messages: list[ChatMessage]) -> None:
        """A 500 followed by a 200 succeeds after retry."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    500, json=_make_error_response("internal error")
                )
            return httpx.Response(200, json=_make_chat_response(content="Recovered!"))

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key=API_KEY, max_retries=2)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.generate(sample_messages)
        assert result.content == "Recovered!"
        assert call_count == 2
        await provider.close()

    async def test_no_retry_on_auth_error(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        """401 errors are NOT retried."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(401, json=_make_error_response("unauthorized"))

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key=API_KEY, max_retries=2)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(LLMAuthenticationError):
            await provider.generate(sample_messages)

        # Should have been called only once – no retry
        assert call_count == 1
        await provider.close()

    async def test_retries_on_rate_limit(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        """429 errors ARE retried."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(429, json=_make_error_response("rate limited"))
            return httpx.Response(200, json=_make_chat_response(content="Finally!"))

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key=API_KEY, max_retries=2)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.generate(sample_messages)
        assert result.content == "Finally!"
        assert call_count == 3
        await provider.close()

    async def test_exhausted_retries_raises(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        """When max retries are exhausted, the last error is raised."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                500, json=_make_error_response("always broken")
            )
        )
        provider = OpenRouterProvider(api_key=API_KEY, max_retries=1)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(LLMProviderError, match="Server error"):
            await provider.generate(sample_messages)

        await provider.close()


# ---------------------------------------------------------------------------
# Streaming generate_stream()
# ---------------------------------------------------------------------------


class TestGenerateStream:
    """Test streaming chat completion."""

    async def test_basic_stream(self, sample_messages: list[ChatMessage]) -> None:
        """Tokens arrive one by one via SSE."""
        tokens = ["Hello", ", ", "friend", "!"]
        sse_text = _make_stream_lines(tokens)

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        chunks: list[StreamChunk] = []
        async for chunk in provider.generate_stream(sample_messages):
            chunks.append(chunk)

        # Content chunks + final finish_reason chunk
        contents = [c.content for c in chunks if c.content]
        assert contents == tokens

        # The last meaningful chunk should carry the finish_reason
        finish_chunks = [c for c in chunks if c.finish_reason]
        assert len(finish_chunks) == 1
        assert finish_chunks[0].finish_reason == "stop"

        await provider.close()

    async def test_stream_error_in_body(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        """An error object inside a stream chunk raises LLMProviderError."""
        error_line = 'data: {"error": {"message": "Model overloaded"}}\n\n'
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=error_line.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        provider = OpenRouterProvider(api_key=API_KEY)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(LLMProviderError, match="Model overloaded"):
            async for _chunk in provider.generate_stream(sample_messages):
                pass

        await provider.close()

    async def test_stream_http_error(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        """HTTP errors during streaming are properly mapped."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                401, json=_make_error_response("bad token")
            )
        )
        provider = OpenRouterProvider(api_key=API_KEY, max_retries=0)
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(LLMAuthenticationError):
            async for _chunk in provider.generate_stream(sample_messages):
                pass

        await provider.close()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Test the heuristic token counter."""

    async def test_empty_messages(self) -> None:
        provider = OpenRouterProvider(api_key=API_KEY)
        count = await provider.count_tokens([])
        assert count == 0
        await provider.close()

    async def test_single_message(self) -> None:
        provider = OpenRouterProvider(api_key=API_KEY)
        messages = [ChatMessage(role=MessageRole.USER, content="Hello world")]
        count = await provider.count_tokens(messages)
        # (16 overhead + 11 chars) // 4 = 6
        assert count == (16 + len("Hello world")) // 4
        await provider.close()

    async def test_multiple_messages(self) -> None:
        provider = OpenRouterProvider(api_key=API_KEY)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=MessageRole.USER, content="Hi"),
        ]
        count = await provider.count_tokens(messages)
        expected = (16 + len("You are helpful.") + 16 + len("Hi")) // 4
        assert count == expected
        await provider.close()


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestHeaders:
    """Test that request headers are set correctly."""

    async def test_authorization_header(self, sample_messages: list[ChatMessage]) -> None:
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_make_chat_response())

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(api_key="my-secret-key")
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        await provider.generate(sample_messages)

        assert captured_headers["authorization"] == "Bearer my-secret-key"
        await provider.close()

    async def test_custom_referer_and_title(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_make_chat_response())

        transport = httpx.MockTransport(handler)
        provider = OpenRouterProvider(
            api_key=API_KEY,
            http_referer="https://my-site.com",
            app_title="My App",
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        await provider.generate(sample_messages)

        assert captured_headers["http-referer"] == "https://my-site.com"
        assert captured_headers["x-title"] == "My App"
        await provider.close()


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------


class TestClose:
    """Test resource cleanup."""

    async def test_close_does_not_error(self) -> None:
        provider = OpenRouterProvider(api_key=API_KEY)
        await provider.close()
        # Double close should also be safe
        await provider.close()
