"""OpenRouter API client.

Implements the ``LLMProvider`` interface using the OpenRouter service
(https://openrouter.ai), which exposes an OpenAI-compatible
``/chat/completions`` endpoint and routes requests to many models
(GPT-4o, Claude, Llama, Mistral, …).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from mai_gram.llm.openrouter_support import (
    EMPTY_STREAM_ERROR,
    decode_sse_json,
    parse_inline_stream_error,
    parse_response,
    parse_stream_chunk,
    parse_tool_calls,
    serialize_message,
    serialize_tool_definition,
)
from mai_gram.llm.provider import (
    ChatMessage,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=45.0,
    write=10.0,
    pool=10.0,
)


def _log_stream_chunk(data: Any, chunk: StreamChunk, elapsed_ms: float) -> None:
    if chunk.usage is not None:
        logger.info(
            "Usage data: %s (resolved cost=%.6f, byok=%s)",
            data.get("usage") if isinstance(data, dict) else None,
            chunk.cost or 0,
            chunk.is_byok,
        )
    logger.debug(
        "SSE chunk at %.0fms: content=%d reasoning=%d tc=%s fin=%s",
        elapsed_ms,
        len(chunk.content),
        len(chunk.reasoning or ""),
        bool(chunk.tool_calls_delta),
        chunk.finish_reason,
    )


class OpenRouterProvider(LLMProvider):
    """LLM provider backed by the OpenRouter API.

    Parameters
    ----------
    api_key:
        OpenRouter API key.
    default_model:
        Model identifier used when ``model`` is not passed to individual calls
        (e.g. ``"openai/gpt-4o"``).
    base_url:
        Override the API base URL (useful for tests / proxies).
    http_referer:
        Optional ``HTTP-Referer`` sent with every request (OpenRouter
        recommends this for analytics).
    app_title:
        Optional ``X-Title`` header (displayed on the OpenRouter dashboard).
    timeout:
        Request timeout — either an ``httpx.Timeout`` or a flat number of
        seconds (applied to every phase).
    max_retries:
        How many times to retry on transient 5xx / network errors.
    """

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str = "openai/gpt-4o",
        base_url: str = OPENROUTER_BASE_URL,
        http_referer: str | None = None,
        app_title: str = "mai-gram",
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise LLMAuthenticationError("OpenRouter API key must not be empty")

        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._active_requests = 0

        resolved_timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_title:
            headers["X-Title"] = app_title

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=resolved_timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        """Currently configured default model identifier."""
        return self._default_model

    @property
    def active_requests(self) -> int:
        """Number of LLM requests currently in flight."""
        return self._active_requests

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a non-streaming chat-completion request."""
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            extra_params=extra_params,
        )

        data = await self._post_with_retry("/chat/completions", payload)

        return self._parse_response(data)

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat-completion response as SSE chunks."""
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            extra_params=extra_params,
        )

        async for chunk in self._stream_with_retry("/chat/completions", payload):
            yield chunk

    async def count_tokens(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
    ) -> int:
        """Estimate token count using a simple heuristic.

        A rough rule of thumb for English text is ~4 characters per token.
        We also account for the per-message overhead that the OpenAI
        chat format adds (~4 tokens per message for role metadata).
        """
        total_chars = 0
        for msg in messages:
            # ~4 tokens of overhead per message (role, delimiters)
            total_chars += 16  # 4 tokens x 4 chars/token
            total_chars += len(msg.content)
        return total_chars // 4

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[ToolDefinition] | None,
        tool_choice: str | dict[str, Any] | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the JSON request body.

        ``extra_params`` (from models.toml per-model config) are merged into
        the payload.  Keys like ``provider``, ``reasoning``, ``temperature``,
        ``max_tokens`` and any other OpenRouter-accepted field are supported.
        Explicit call-site values for temperature/max_tokens take precedence
        only when they differ from the defaults.
        """
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": [self._serialize_message(m) for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            payload["include"] = ["usage"]
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = [self._serialize_tool_definition(tool) for tool in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if extra_params:
            for key, value in extra_params.items():
                if key in ("model", "messages", "stream", "tools", "tool_choice"):
                    continue
                payload[key] = value

        return payload

    @staticmethod
    def _serialize_tool_definition(tool: ToolDefinition) -> dict[str, Any]:
        return serialize_tool_definition(tool)

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        return serialize_message(message)

    # -- Non-streaming request with retry ---------------------------------

    async def _post_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST *path* with automatic retry on transient failures."""
        model = payload.get("model", self._default_model)
        msg_count = len(payload.get("messages", []))
        logger.info("LLM request started (model=%s, messages=%d)", model, msg_count)

        last_exc: Exception | None = None
        self._active_requests += 1
        try:
            for attempt in range(1, self._max_retries + 2):  # +2 because range is exclusive
                try:
                    response = await self._client.post(path, json=payload)
                    self._raise_for_status(response)
                    return response.json()  # type: ignore[no-any-return]
                except (LLMRateLimitError, LLMProviderError) as exc:
                    last_exc = exc
                    if attempt <= self._max_retries:
                        logger.warning(
                            "OpenRouter request failed (attempt %d/%d): %s",
                            attempt,
                            self._max_retries + 1,
                            exc,
                        )
                        continue
                    raise
                except httpx.TransportError as exc:
                    last_exc = exc
                    if attempt <= self._max_retries:
                        logger.warning(
                            "Network error (attempt %d/%d): %s",
                            attempt,
                            self._max_retries + 1,
                            exc,
                        )
                        continue
                    raise LLMProviderError(
                        f"Network error after {attempt} attempts: {exc}"
                    ) from exc

            raise LLMProviderError(f"Request failed: {last_exc}") from last_exc  # pragma: no cover
        finally:
            self._active_requests -= 1

    # -- Streaming request with retry --------------------------------------

    async def _stream_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Stream SSE from *path* with automatic retry on transient failures."""
        model = payload.get("model", self._default_model)
        msg_count = len(payload.get("messages", []))
        logger.info("LLM stream started (model=%s, messages=%d)", model, msg_count)

        last_exc: Exception | None = None
        self._active_requests += 1
        try:
            for attempt in range(1, self._max_retries + 2):
                try:
                    async for chunk in self._stream_sse(path, payload):
                        yield chunk
                    return  # successful stream completed
                except (LLMRateLimitError, LLMProviderError) as exc:
                    last_exc = exc
                    if attempt <= self._max_retries:
                        logger.warning(
                            "OpenRouter stream failed (attempt %d/%d): %s",
                            attempt,
                            self._max_retries + 1,
                            exc,
                        )
                        continue
                    raise
                except httpx.TransportError as exc:
                    last_exc = exc
                    if attempt <= self._max_retries:
                        logger.warning(
                            "Network error during stream (attempt %d/%d): %s",
                            attempt,
                            self._max_retries + 1,
                            exc,
                        )
                        continue
                    raise LLMProviderError(
                        f"Network error after {attempt} attempts: {exc}"
                    ) from exc

            raise LLMProviderError(  # pragma: no cover
                f"Stream failed: {last_exc}"
            ) from last_exc
        finally:
            self._active_requests -= 1

    async def _stream_sse(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Low-level SSE streaming."""
        import time as _time

        _t0 = _time.monotonic()
        async with self._client.stream("POST", path, json=payload) as response:
            self._raise_for_status(response)
            logger.debug(
                "SSE stream opened (%.1fms after request)",
                (_time.monotonic() - _t0) * 1000,
            )

            any_data_received = False

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if not line.startswith("data: "):
                    error_msg = parse_inline_stream_error(line)
                    if error_msg is not None:
                        raise LLMProviderError(f"Stream error: {error_msg}")
                    logger.debug("Skipping non-SSE line: %s", line[:120])
                    continue

                data_str = line[len("data: ") :]

                if data_str == "[DONE]":
                    if not any_data_received:
                        raise LLMProviderError(EMPTY_STREAM_ERROR)
                    return

                data = decode_sse_json(data_str)
                if data is None:
                    logger.debug("Skipping non-JSON SSE line: %s", data_str[:100])
                    continue

                any_data_received = True

                chunk = parse_stream_chunk(data)
                if chunk is None:
                    continue

                _elapsed = (_time.monotonic() - _t0) * 1000
                _log_stream_chunk(data, chunk, _elapsed)
                yield chunk

            if not any_data_received:
                raise LLMProviderError(EMPTY_STREAM_ERROR)

    # -- Response parsing --------------------------------------------------

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """Turn a raw JSON response dict into an ``LLMResponse``."""
        return parse_response(data)

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
        return parse_tool_calls(raw_tool_calls)

    # -- HTTP status → typed exception mapping -----------------------------

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Translate HTTP error codes into typed ``LLMError`` subclasses.

        Works for both regular and streaming responses. When the body
        has not been read yet (streaming context), we fall back to the
        status code alone.
        """
        if response.is_success:
            return

        status = response.status_code

        msg = f"HTTP {status}"
        try:
            body = response.json()
            error = body.get("error", {})
            msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        except (ValueError, json.JSONDecodeError):
            try:
                if response.text:
                    msg = response.text[:500]
            except (UnicodeDecodeError, AttributeError):
                logger.debug("Failed to read error response body text")

        if status == 401:
            raise LLMAuthenticationError(f"Authentication failed: {msg}")
        if status == 404:
            raise LLMModelNotFoundError(f"Model not found: {msg}")
        if status == 429:
            retry_after_raw = response.headers.get("retry-after")
            retry_after = float(retry_after_raw) if retry_after_raw else None
            raise LLMRateLimitError(f"Rate limited: {msg}", retry_after=retry_after)
        if status == 400 and "context" in msg.lower():
            raise LLMContextLengthError(f"Context length exceeded: {msg}")
        if status >= 500:
            raise LLMProviderError(f"Server error ({status}): {msg}", status_code=status)

        raise LLMProviderError(f"Request failed ({status}): {msg}", status_code=status)
