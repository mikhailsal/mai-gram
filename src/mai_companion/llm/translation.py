"""LLM-powered translation service for multilingual onboarding.

Provides language detection and text translation using the existing LLM
provider.  Includes an in-memory cache to avoid repeated LLM calls when
the human navigates back and forth during onboarding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole

logger = logging.getLogger(__name__)

# System prompts used for translation tasks
_DETECT_LANGUAGE_PROMPT = (
    "You are a language identification expert. The human typed the following "
    "text to indicate their preferred language. Respond with ONLY the English "
    "name of the language (e.g., 'Russian', 'Spanish', 'Japanese', 'English'). "
    "Do not add any explanation, punctuation, or extra words."
)

_TRANSLATE_PROMPT = (
    "You are a professional translator. Translate the following text to {language}. "
    "Preserve the original meaning, tone, and nuance precisely. "
    "Do not add explanations or commentary. "
    "For personality-related vocabulary, keep the nuance intact "
    "(e.g., 'blunt' should not become 'rude'). "
    "Respond with ONLY the translated text."
)

_TRANSLATE_BATCH_PROMPT = (
    "You are a professional translator. Translate each of the following numbered "
    "texts to {language}. Preserve meaning, tone, and nuance precisely. "
    "For personality-related vocabulary, keep the nuance intact. "
    "Respond with ONLY the translations, one per line, numbered to match the input. "
    "Format: 1. <translation>\\n2. <translation>\\n..."
)


@dataclass
class TranslationService:
    """LLM-powered translation with caching.

    Parameters
    ----------
    llm_provider:
        The LLM backend to use for translation requests.
    """

    llm_provider: LLMProvider
    _cache: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)

    async def detect_language(self, human_input: str) -> str:
        """Identify the language from free-text input.

        The human may type anything: "Russian", "русский", "Espanol",
        "french", "日本語", etc.  The LLM normalises this to an English
        language name.

        Parameters
        ----------
        human_input:
            Raw text the human typed to indicate their language.

        Returns
        -------
        str
            English name of the detected language (e.g. "Russian").
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_DETECT_LANGUAGE_PROMPT),
            ChatMessage(role=MessageRole.USER, content=human_input.strip()),
        ]
        response = await self.llm_provider.generate(
            messages, temperature=0.0, max_tokens=20
        )
        detected = response.content.strip().strip(".")
        logger.info("Detected language '%s' from input '%s'", detected, human_input)
        return detected

    async def translate(self, text: str, target_language: str) -> str:
        """Translate a single text to the target language.

        If *target_language* is ``"English"``, returns the original text
        without making an LLM call.

        Results are cached so repeated translations of the same text to
        the same language are free.

        Parameters
        ----------
        text:
            The English text to translate.
        target_language:
            The language to translate into (e.g. "Russian", "Spanish").

        Returns
        -------
        str
            The translated text.
        """
        if target_language.lower() == "english":
            return text

        cache_key = (text, target_language)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = _TRANSLATE_PROMPT.format(language=target_language)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt),
            ChatMessage(role=MessageRole.USER, content=text),
        ]
        response = await self.llm_provider.generate(
            messages, temperature=0.1, max_tokens=len(text) * 3
        )
        translated = response.content.strip()
        self._cache[cache_key] = translated
        logger.debug(
            "Translated to %s: '%s' -> '%s'",
            target_language,
            text[:60],
            translated[:60],
        )
        return translated

    async def translate_batch(
        self, texts: list[str], target_language: str
    ) -> list[str]:
        """Translate multiple texts in a single LLM call.

        More efficient than calling :meth:`translate` in a loop because
        all texts are sent in one request.

        Parameters
        ----------
        texts:
            List of English texts to translate.
        target_language:
            The language to translate into.

        Returns
        -------
        list[str]
            Translated texts in the same order as the input.
        """
        if target_language.lower() == "english":
            return list(texts)

        if not texts:
            return []

        # Check cache -- only send untranslated texts to LLM
        results: list[str | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cache_key = (text, target_language)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [r for r in results if r is not None]

        # Build numbered input for uncached texts
        numbered_input = "\n".join(
            f"{idx + 1}. {texts[i]}" for idx, i in enumerate(uncached_indices)
        )

        prompt = _TRANSLATE_BATCH_PROMPT.format(language=target_language)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt),
            ChatMessage(role=MessageRole.USER, content=numbered_input),
        ]
        response = await self.llm_provider.generate(
            messages,
            temperature=0.1,
            max_tokens=sum(len(texts[i]) for i in uncached_indices) * 3,
        )

        # Parse numbered response
        translated_lines = _parse_numbered_response(
            response.content, expected_count=len(uncached_indices)
        )

        for idx, orig_idx in enumerate(uncached_indices):
            if idx < len(translated_lines):
                translation = translated_lines[idx]
            else:
                # Fallback: if parsing failed, return original text
                translation = texts[orig_idx]
                logger.warning(
                    "Batch translation missing line %d, falling back to original",
                    idx + 1,
                )

            self._cache[(texts[orig_idx], target_language)] = translation
            results[orig_idx] = translation

        return [r if r is not None else texts[i] for i, r in enumerate(results)]

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._cache.clear()


def _parse_numbered_response(response: str, expected_count: int) -> list[str]:
    """Parse a numbered list response from the LLM.

    Handles formats like:
        1. Translation one
        2. Translation two

    Also handles cases where the LLM omits numbers or uses different
    formatting.
    """
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    parsed: list[str] = []

    for line in lines:
        # Strip leading number + period/parenthesis: "1. text" -> "text"
        stripped = line
        for i, ch in enumerate(line):
            if ch.isdigit():
                continue
            if ch in ".)" and i > 0:
                stripped = line[i + 1 :].strip()
                break
            # No number prefix -- use the line as-is
            break
        parsed.append(stripped)

    return parsed
