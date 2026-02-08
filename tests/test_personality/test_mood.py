"""Tests for personality/mood.py -- dynamic mood system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_companion.db.models import Companion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
from mai_companion.personality.mood import (
    MoodCoordinates,
    MoodManager,
    MoodSnapshot,
    evaluate_message_sentiment,
    mood_to_behavior_hints,
    mood_to_prompt_section,
    resolve_label,
)

# ---------------------------------------------------------------------------
# MoodCoordinates
# ---------------------------------------------------------------------------

class TestMoodCoordinates:
    """Verify MoodCoordinates dataclass."""

    def test_clamped(self) -> None:
        coords = MoodCoordinates(valence=1.5, arousal=-2.0)
        clamped = coords.clamped()
        assert clamped.valence == 1.0
        assert clamped.arousal == -1.0

    def test_clamped_no_change_when_in_range(self) -> None:
        coords = MoodCoordinates(valence=0.5, arousal=-0.3)
        clamped = coords.clamped()
        assert clamped.valence == 0.5
        assert clamped.arousal == -0.3


# ---------------------------------------------------------------------------
# resolve_label
# ---------------------------------------------------------------------------

class TestResolveLabel:
    """Verify mood label resolution from coordinates."""

    @pytest.mark.parametrize(
        "valence, arousal, expected_labels",
        [
            # Positive + High arousal
            (0.5, 0.5, {"enthusiastic", "excited"}),
            # Positive + Neutral
            (0.5, 0.0, {"pleased", "happy"}),
            # Positive + Low arousal
            (0.5, -0.5, {"content", "serene"}),
            # Neutral + High arousal
            (0.0, 0.5, {"alert", "restless"}),
            # Neutral + Neutral
            (0.0, 0.0, {"neutral"}),
            # Neutral + Low arousal
            (0.0, -0.5, {"relaxed", "drowsy"}),
            # Negative + High arousal
            (-0.5, 0.5, {"frustrated", "irritated"}),
            # Negative + Neutral
            (-0.5, 0.0, {"sad", "gloomy"}),
            # Negative + Low arousal
            (-0.5, -0.5, {"melancholic", "depleted"}),
        ],
    )
    def test_label_in_expected_set(
        self, valence: float, arousal: float, expected_labels: set[str]
    ) -> None:
        label = resolve_label(MoodCoordinates(valence=valence, arousal=arousal))
        assert label in expected_labels, (
            f"Label '{label}' not in {expected_labels} for v={valence}, a={arousal}"
        )

    def test_extreme_positive_excited(self) -> None:
        label = resolve_label(MoodCoordinates(valence=0.9, arousal=0.9))
        assert label == "excited"

    def test_extreme_negative_depleted(self) -> None:
        label = resolve_label(MoodCoordinates(valence=-0.9, arousal=-0.9))
        assert label == "depleted"


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

class TestSentimentAnalysis:
    """Verify rule-based sentiment analysis."""

    def test_positive_message(self) -> None:
        sentiment, intensity = evaluate_message_sentiment(
            "I am so happy today! This is amazing!!!"
        )
        assert sentiment > 0
        assert intensity > 0

    def test_negative_message(self) -> None:
        sentiment, intensity = evaluate_message_sentiment(
            "I feel terrible and exhausted"
        )
        assert sentiment < 0
        assert intensity > 0

    def test_neutral_message(self) -> None:
        sentiment, intensity = evaluate_message_sentiment(
            "The weather is 20 degrees"
        )
        assert sentiment == 0.0
        assert intensity == 0.0

    def test_mixed_message(self) -> None:
        sentiment, intensity = evaluate_message_sentiment(
            "I'm happy but also a bit sad"
        )
        # Should have some intensity but unclear sentiment
        assert intensity > 0

    def test_exclamation_marks_boost_intensity(self) -> None:
        _, intensity_calm = evaluate_message_sentiment("I am happy")
        _, intensity_excited = evaluate_message_sentiment("I am happy!!!")
        assert intensity_excited >= intensity_calm

    def test_intense_markers_boost_intensity(self) -> None:
        # Use a longer message so base intensity is lower, leaving room for boost
        _, intensity_normal = evaluate_message_sentiment(
            "I went to the store and bought some things and felt happy about it"
        )
        _, intensity_very = evaluate_message_sentiment(
            "I went to the store and bought some things and felt extremely happy about it"
        )
        assert intensity_very >= intensity_normal

    def test_empty_message(self) -> None:
        sentiment, intensity = evaluate_message_sentiment("")
        assert sentiment == 0.0
        assert intensity == 0.0


# ---------------------------------------------------------------------------
# Mood prompt generation
# ---------------------------------------------------------------------------

class TestMoodPromptSection:
    """Verify mood-to-prompt conversion."""

    def test_known_label_generates_section(self) -> None:
        mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.5, 0.5),
            label="excited",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        section = mood_to_prompt_section(mood)
        assert "## Current mood" in section
        assert "excited" in section.lower()

    def test_cause_included_when_present(self) -> None:
        mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.5, 0.5),
            label="excited",
            cause="great news from human",
            timestamp=datetime.now(timezone.utc),
        )
        section = mood_to_prompt_section(mood)
        assert "great news from human" in section

    def test_unknown_label_fallback(self) -> None:
        mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.0, 0.0),
            label="some_unknown_mood",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        section = mood_to_prompt_section(mood)
        assert "some_unknown_mood" in section

    def test_all_known_labels_have_templates(self) -> None:
        """Every label that resolve_label can produce should have a template."""
        from mai_companion.personality.mood import _MOOD_PROMPT_TEMPLATES

        known_labels = set()
        # Test a grid of coordinates
        for v in [-0.8, -0.5, 0.0, 0.5, 0.8]:
            for a in [-0.8, -0.5, 0.0, 0.5, 0.8]:
                label = resolve_label(MoodCoordinates(v, a))
                known_labels.add(label)

        for label in known_labels:
            assert label in _MOOD_PROMPT_TEMPLATES, (
                f"Label '{label}' has no prompt template"
            )


# ---------------------------------------------------------------------------
# Behavior hints
# ---------------------------------------------------------------------------

class TestBehaviorHints:
    """Verify mood_to_behavior_hints output."""

    def test_positive_mood_increases_patience(self) -> None:
        mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.8, 0.0),
            label="happy",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        hints = mood_to_behavior_hints(mood)
        assert hints["patience_modifier"] > 0

    def test_negative_mood_decreases_patience(self) -> None:
        mood = MoodSnapshot(
            coordinates=MoodCoordinates(-0.8, 0.0),
            label="sad",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        hints = mood_to_behavior_hints(mood)
        assert hints["patience_modifier"] < 0

    def test_hints_have_expected_keys(self) -> None:
        mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.0, 0.0),
            label="neutral",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        hints = mood_to_behavior_hints(mood)
        assert "patience_modifier" in hints
        assert "humor_modifier" in hints
        assert "verbosity_modifier" in hints
        assert "topic_inclination" in hints


# ---------------------------------------------------------------------------
# MoodManager (with real DB)
# ---------------------------------------------------------------------------

class TestMoodManager:
    """Verify MoodManager with real database sessions."""

    @pytest.fixture
    def default_traits(self) -> dict[str, float]:
        return {
            "warmth": 0.5,
            "humor": 0.5,
            "patience": 0.5,
            "directness": 0.5,
            "laziness": 0.5,
            "mood_volatility": 0.5,
        }

    @pytest.fixture
    async def companion_id(self, session: AsyncSession) -> str:
        companion = Companion(
            name="TestBot",
            personality_traits="{}",
            system_prompt="test",
        )
        session.add(companion)
        await session.flush()
        return companion.id

    async def test_get_current_mood_returns_baseline_when_empty(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        mood = await manager.get_current_mood(companion_id, default_traits)
        assert mood.label
        assert mood.cause == "initial baseline"

    async def test_compute_baseline_warm_patient(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        baseline = manager.compute_baseline({"warmth": 0.9, "patience": 0.9})
        # Warm -> positive valence, patient -> low arousal
        assert baseline.valence > 0
        assert baseline.arousal < 0

    async def test_compute_baseline_cold_impatient(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        baseline = manager.compute_baseline({"warmth": 0.1, "patience": 0.1})
        # Cold -> negative valence, impatient -> positive arousal
        assert baseline.valence < 0
        assert baseline.arousal > 0

    async def test_compute_baseline_medium_is_neutral(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        baseline = manager.compute_baseline({"warmth": 0.5, "patience": 0.5})
        assert baseline.valence == 0.0
        assert baseline.arousal == 0.0

    async def test_apply_reactive_shift_positive(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        mood = await manager.apply_reactive_shift(
            companion_id,
            sentiment_score=0.8,
            intensity=0.7,
            cause="good news",
            traits=default_traits,
        )
        assert mood.coordinates.valence > 0
        assert mood.cause == "good news"

    async def test_apply_reactive_shift_negative(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        mood = await manager.apply_reactive_shift(
            companion_id,
            sentiment_score=-0.8,
            intensity=0.7,
            cause="bad news",
            traits=default_traits,
        )
        assert mood.coordinates.valence < 0

    async def test_apply_reactive_shift_saves_to_db(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        await manager.apply_reactive_shift(
            companion_id,
            sentiment_score=0.5,
            intensity=0.5,
            cause="test",
            traits=default_traits,
        )
        # Should now be retrievable
        mood = await manager.get_current_mood(companion_id, default_traits)
        assert mood.cause == "test"

    async def test_apply_spontaneous_shift(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        mood = await manager.apply_spontaneous_shift(
            companion_id, volatility=0.5, traits=default_traits
        )
        assert mood.cause == "spontaneous shift"
        assert -1.0 <= mood.coordinates.valence <= 1.0
        assert -1.0 <= mood.coordinates.arousal <= 1.0

    async def test_apply_decay_moves_toward_baseline(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        # First, shift mood away from baseline
        await manager.apply_reactive_shift(
            companion_id,
            sentiment_score=0.9,
            intensity=1.0,
            cause="big event",
            traits=default_traits,
        )
        shifted = await manager.get_current_mood(companion_id, default_traits)

        # Apply decay for 10 hours
        decayed = await manager.apply_decay(
            companion_id, traits=default_traits, hours_elapsed=10.0
        )

        baseline = manager.compute_baseline(default_traits)
        # After decay, valence should be closer to baseline than before
        dist_before = abs(shifted.coordinates.valence - baseline.valence)
        dist_after = abs(decayed.coordinates.valence - baseline.valence)
        assert dist_after < dist_before

    async def test_apply_decay_zero_hours_no_change(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        # Shift mood
        await manager.apply_reactive_shift(
            companion_id,
            sentiment_score=0.5,
            intensity=0.5,
            cause="test",
            traits=default_traits,
        )
        before = await manager.get_current_mood(companion_id, default_traits)

        # Decay with 0 hours should barely change
        after = await manager.apply_decay(
            companion_id, traits=default_traits, hours_elapsed=0.0
        )
        assert abs(after.coordinates.valence - before.coordinates.valence) < 0.01

    async def test_get_current_mood_returns_latest(
        self, session: AsyncSession, companion_id: str, default_traits: dict[str, float]
    ) -> None:
        manager = MoodManager(session)
        # Apply two shifts
        await manager.apply_reactive_shift(
            companion_id, 0.5, 0.5, "first", default_traits
        )
        await manager.apply_reactive_shift(
            companion_id, -0.5, 0.5, "second", default_traits
        )
        mood = await manager.get_current_mood(companion_id, default_traits)
        assert mood.cause == "second"


# ---------------------------------------------------------------------------
# LLM mood analysis
# ---------------------------------------------------------------------------

class TestLLMMoodAnalysis:
    """Verify LLM-based mood analysis."""

    async def test_llm_analysis_parses_response(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        mock_provider = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "0.4,0.2"
        mock_provider.generate.return_value = mock_response

        current_mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.0, 0.0),
            label="neutral",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )

        result = await manager.request_llm_mood_analysis(
            ["Hello!", "How are you?"], current_mood, mock_provider
        )
        assert result is not None
        assert abs(result.valence - 0.4) < 0.01
        assert abs(result.arousal - 0.2) < 0.01

    async def test_llm_analysis_returns_none_on_empty_messages(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        mock_provider = AsyncMock()
        current_mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.0, 0.0),
            label="neutral",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        result = await manager.request_llm_mood_analysis(
            [], current_mood, mock_provider
        )
        assert result is None

    async def test_llm_analysis_returns_none_on_bad_response(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        mock_provider = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "I'm not sure about that"
        mock_provider.generate.return_value = mock_response

        current_mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.0, 0.0),
            label="neutral",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        result = await manager.request_llm_mood_analysis(
            ["test"], current_mood, mock_provider
        )
        assert result is None

    async def test_llm_analysis_returns_none_on_exception(
        self, session: AsyncSession
    ) -> None:
        manager = MoodManager(session)
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM error")

        current_mood = MoodSnapshot(
            coordinates=MoodCoordinates(0.0, 0.0),
            label="neutral",
            cause=None,
            timestamp=datetime.now(timezone.utc),
        )
        result = await manager.request_llm_mood_analysis(
            ["test"], current_mood, mock_provider
        )
        assert result is None
