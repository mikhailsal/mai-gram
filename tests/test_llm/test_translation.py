"""Tests for llm/translation.py -- LLM-powered translation service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mai_companion.llm.translation import TranslationService, _parse_numbered_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_provider(response_content: str) -> AsyncMock:
    """Create a mock LLM provider that returns the given content."""
    provider = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    provider.generate.return_value = mock_response
    return provider


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    """Verify language detection via LLM."""

    async def test_detect_language(self) -> None:
        provider = _make_mock_provider("Russian")
        service = TranslationService(llm_provider=provider)
        result = await service.detect_language("русский")
        assert result == "Russian"
        provider.generate.assert_called_once()

    async def test_detect_language_strips_whitespace(self) -> None:
        provider = _make_mock_provider("  Spanish.  ")
        service = TranslationService(llm_provider=provider)
        result = await service.detect_language("espanol")
        assert result == "Spanish"

    async def test_detect_language_uses_low_temperature(self) -> None:
        provider = _make_mock_provider("English")
        service = TranslationService(llm_provider=provider)
        await service.detect_language("english")
        call_kwargs = provider.generate.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.0


# ---------------------------------------------------------------------------
# Single text translation
# ---------------------------------------------------------------------------

class TestTranslate:
    """Verify single text translation."""

    async def test_translate_to_russian(self) -> None:
        provider = _make_mock_provider("Привет, мир!")
        service = TranslationService(llm_provider=provider)
        result = await service.translate("Hello, world!", "Russian")
        assert result == "Привет, мир!"

    async def test_english_passthrough_no_llm_call(self) -> None:
        provider = _make_mock_provider("")
        service = TranslationService(llm_provider=provider)
        result = await service.translate("Hello", "English")
        assert result == "Hello"
        provider.generate.assert_not_called()

    async def test_english_passthrough_case_insensitive(self) -> None:
        provider = _make_mock_provider("")
        service = TranslationService(llm_provider=provider)
        result = await service.translate("Hello", "english")
        assert result == "Hello"
        provider.generate.assert_not_called()

    async def test_caching(self) -> None:
        provider = _make_mock_provider("Hola")
        service = TranslationService(llm_provider=provider)

        # First call
        result1 = await service.translate("Hello", "Spanish")
        assert result1 == "Hola"

        # Second call -- should use cache
        result2 = await service.translate("Hello", "Spanish")
        assert result2 == "Hola"

        # LLM should only have been called once
        assert provider.generate.call_count == 1

    async def test_different_languages_not_cached_together(self) -> None:
        provider = _make_mock_provider("Bonjour")
        service = TranslationService(llm_provider=provider)

        await service.translate("Hello", "French")
        # Change mock response for next call
        new_response = MagicMock()
        new_response.content = "Hola"
        provider.generate.return_value = new_response

        result = await service.translate("Hello", "Spanish")
        assert result == "Hola"
        assert provider.generate.call_count == 2

    async def test_clear_cache(self) -> None:
        provider = _make_mock_provider("Hola")
        service = TranslationService(llm_provider=provider)

        await service.translate("Hello", "Spanish")
        service.clear_cache()

        # After clearing, should call LLM again
        await service.translate("Hello", "Spanish")
        assert provider.generate.call_count == 2


# ---------------------------------------------------------------------------
# Batch translation
# ---------------------------------------------------------------------------

class TestTranslateBatch:
    """Verify batch translation."""

    async def test_batch_translate(self) -> None:
        provider = _make_mock_provider("1. Hola\n2. Mundo")
        service = TranslationService(llm_provider=provider)
        results = await service.translate_batch(
            ["Hello", "World"], "Spanish"
        )
        assert results == ["Hola", "Mundo"]

    async def test_batch_english_passthrough(self) -> None:
        provider = _make_mock_provider("")
        service = TranslationService(llm_provider=provider)
        results = await service.translate_batch(
            ["Hello", "World"], "English"
        )
        assert results == ["Hello", "World"]
        provider.generate.assert_not_called()

    async def test_batch_empty_list(self) -> None:
        provider = _make_mock_provider("")
        service = TranslationService(llm_provider=provider)
        results = await service.translate_batch([], "Spanish")
        assert results == []
        provider.generate.assert_not_called()

    async def test_batch_uses_cache(self) -> None:
        provider = _make_mock_provider("Hola")
        service = TranslationService(llm_provider=provider)

        # Pre-cache one translation
        await service.translate("Hello", "Spanish")

        # Now batch with the cached item + a new one
        new_response = MagicMock()
        new_response.content = "1. Mundo"
        provider.generate.return_value = new_response

        results = await service.translate_batch(
            ["Hello", "World"], "Spanish"
        )
        assert results[0] == "Hola"  # from cache
        assert results[1] == "Mundo"  # from LLM

    async def test_batch_caches_results(self) -> None:
        provider = _make_mock_provider("1. Hola\n2. Mundo")
        service = TranslationService(llm_provider=provider)

        await service.translate_batch(["Hello", "World"], "Spanish")

        # Individual translate should use cache
        result = await service.translate("Hello", "Spanish")
        assert result == "Hola"
        # Only 1 LLM call total (the batch)
        assert provider.generate.call_count == 1


# ---------------------------------------------------------------------------
# _parse_numbered_response
# ---------------------------------------------------------------------------

class TestParseNumberedResponse:
    """Verify numbered response parsing."""

    def test_standard_format(self) -> None:
        result = _parse_numbered_response("1. Hello\n2. World", 2)
        assert result == ["Hello", "World"]

    def test_parenthesis_format(self) -> None:
        result = _parse_numbered_response("1) Hello\n2) World", 2)
        assert result == ["Hello", "World"]

    def test_no_numbers(self) -> None:
        result = _parse_numbered_response("Hello\nWorld", 2)
        assert result == ["Hello", "World"]

    def test_empty_lines_skipped(self) -> None:
        result = _parse_numbered_response("1. Hello\n\n2. World\n", 2)
        assert result == ["Hello", "World"]

    def test_single_item(self) -> None:
        result = _parse_numbered_response("1. Hello", 1)
        assert result == ["Hello"]
