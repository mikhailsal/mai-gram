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
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 120.0  # seconds – generous for large generations


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
        Request timeout in seconds.
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
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise LLMAuthenticationError("OpenRouter API key must not be empty")

        self._api_key = api_key
        self._default_model = default_model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

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
            timeout=httpx.Timeout(timeout),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        """Currently configured default model identifier."""
        return self._default_model

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
            total_chars += 16  # 4 tokens × 4 chars/token
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
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.reasoning is not None:
            payload["reasoning"] = message.reasoning
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        return payload

    # -- Non-streaming request with retry ---------------------------------

    async def _post_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST *path* with automatic retry on transient failures."""
        last_exc: Exception | None = None

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
                raise LLMProviderError(f"Network error after {attempt} attempts: {exc}") from exc

        # Should not be reached, but just in case
        raise LLMProviderError(f"Request failed: {last_exc}") from last_exc  # pragma: no cover

    # -- Streaming request with retry --------------------------------------

    async def _stream_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Stream SSE from *path* with automatic retry on transient failures."""
        last_exc: Exception | None = None

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
                raise LLMProviderError(f"Network error after {attempt} attempts: {exc}") from exc

        raise LLMProviderError(f"Stream failed: {last_exc}") from last_exc  # pragma: no cover

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

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                # OpenAI-compatible SSE format: "data: {...}" or "data: [DONE]"
                if not line.startswith("data: "):
                    continue

                data_str = line[len("data: ") :]

                if data_str == "[DONE]":
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON SSE line: %s", data_str[:100])
                    continue

                # Handle inline error objects that some providers send
                if "error" in data:
                    error_msg = data["error"]
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                    raise LLMProviderError(f"Stream error: {error_msg}")

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning", "")
                finish_reason = choices[0].get("finish_reason")
                raw_tool_calls = delta.get("tool_calls")

                tool_calls_delta: list[dict[str, Any]] | None = None
                if isinstance(raw_tool_calls, list) and raw_tool_calls:
                    tool_calls_delta = raw_tool_calls

                usage_obj: TokenUsage | None = None
                cost_val: float | None = None
                is_byok_val = False
                usage_data = data.get("usage")
                if isinstance(usage_data, dict):
                    usage_obj = TokenUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )
                    is_byok_val = bool(usage_data.get("is_byok", False))
                    raw_cost = usage_data.get("cost") or 0.0
                    if is_byok_val:
                        cost_details = usage_data.get("cost_details") or {}
                        inference_cost = (
                            cost_details.get("upstream_inference_cost")
                            or usage_data.get("native_tokens_cost")
                            or usage_data.get("inference_cost")
                            or 0.0
                        )
                        cost_val = float(raw_cost) + float(inference_cost)
                    else:
                        cost_val = float(raw_cost) if raw_cost else None
                    logger.info(
                        "Usage data: %s (resolved cost=%.6f, byok=%s)",
                        usage_data,
                        cost_val or 0,
                        is_byok_val,
                    )

                if content or reasoning or finish_reason or tool_calls_delta or usage_obj:
                    _elapsed = (_time.monotonic() - _t0) * 1000
                    logger.debug(
                        "SSE chunk at %.0fms: content=%d reasoning=%d tc=%s fin=%s",
                        _elapsed,
                        len(content or ""),
                        len(reasoning or ""),
                        bool(tool_calls_delta),
                        finish_reason,
                    )
                    yield StreamChunk(
                        content=content or "",
                        finish_reason=finish_reason,
                        reasoning=reasoning or None,
                        tool_calls_delta=tool_calls_delta,
                        usage=usage_obj,
                        cost=cost_val,
                        is_byok=is_byok_val,
                    )

    # -- Response parsing --------------------------------------------------

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """Turn a raw JSON response dict into an ``LLMResponse``."""
        # Handle error responses that came back with HTTP 200
        if "error" in data:
            error = data["error"]
            msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise LLMProviderError(f"API error: {msg}")

        choices = data.get("choices", [])
        if not choices:
            raise LLMProviderError("No choices in API response")

        message = choices[0].get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or None
        tool_calls = OpenRouterProvider._parse_tool_calls(message.get("tool_calls"))
        finish_reason = choices[0].get("finish_reason", "")

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            model=data.get("model", ""),
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            reasoning=reasoning,
        )

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
        if not isinstance(raw_tool_calls, list):
            return []

        parsed_calls: list[ToolCall] = []
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_data = tool_call.get("function")
            if not isinstance(function_data, dict):
                continue

            tool_call_id = tool_call.get("id")
            name = function_data.get("name")
            arguments = function_data.get("arguments")
            if not isinstance(tool_call_id, str) or not isinstance(name, str):
                continue

            parsed_calls.append(
                ToolCall(
                    id=tool_call_id,
                    name=name,
                    arguments=arguments if isinstance(arguments, str) else "",
                )
            )

        return parsed_calls

    # -- HTTP status → typed exception mapping -----------------------------

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Translate HTTP error codes into typed ``LLMError`` subclasses."""
        if response.is_success:
            return

        status = response.status_code

        # Try to extract an error message from the body
        try:
            body = response.json()
            error = body.get("error", {})
            msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        except Exception:
            msg = response.text[:500] if response.text else f"HTTP {status}"

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

        # Anything else
        raise LLMProviderError(f"Request failed ({status}): {msg}", status_code=status)
