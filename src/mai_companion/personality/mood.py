"""Dynamic mood system using a two-axis (valence/arousal) model.

The mood system gives the companion a living emotional state that shifts
reactively (from conversation) and spontaneously (random drift based on
mood_volatility).  Mood is persisted to the database and injected into
the system prompt so the LLM can reason about and express its current
emotional state.

Key concepts:
- **Valence**: positive ↔ negative (-1.0 to 1.0)
- **Arousal**: energetic ↔ calm (-1.0 to 1.0)
- Together they produce mood labels like "excited", "melancholic", "irritated"
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_companion.db.models import MoodState
from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MoodCoordinates:
    """A point in the valence/arousal space."""

    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # -1.0 (calm) to 1.0 (energetic)

    def clamped(self) -> MoodCoordinates:
        """Return a copy with values clamped to [-1, 1]."""
        return MoodCoordinates(
            valence=max(-1.0, min(1.0, self.valence)),
            arousal=max(-1.0, min(1.0, self.arousal)),
        )


@dataclass(frozen=True, slots=True)
class MoodSnapshot:
    """A complete mood state at a point in time."""

    coordinates: MoodCoordinates
    label: str
    cause: str | None
    timestamp: datetime


# ---------------------------------------------------------------------------
# Mood label map
# ---------------------------------------------------------------------------

# Each cell has labels for moderate and extreme values.
# Structure: (valence_range, arousal_range) -> (moderate_label, extreme_label)
_MOOD_LABELS: list[tuple[tuple[float, float], tuple[float, float], str, str]] = [
    # Positive + High arousal
    ((0.3, 1.0), (0.3, 1.0), "enthusiastic", "excited"),
    # Positive + Neutral arousal
    ((0.3, 1.0), (-0.3, 0.3), "pleased", "happy"),
    # Positive + Low arousal
    ((0.3, 1.0), (-1.0, -0.3), "content", "serene"),
    # Neutral + High arousal
    ((-0.3, 0.3), (0.3, 1.0), "alert", "restless"),
    # Neutral + Neutral arousal
    ((-0.3, 0.3), (-0.3, 0.3), "neutral", "neutral"),
    # Neutral + Low arousal
    ((-0.3, 0.3), (-1.0, -0.3), "relaxed", "drowsy"),
    # Negative + High arousal
    ((-1.0, -0.3), (0.3, 1.0), "frustrated", "irritated"),
    # Negative + Neutral arousal
    ((-1.0, -0.3), (-0.3, 0.3), "sad", "gloomy"),
    # Negative + Low arousal
    ((-1.0, -0.3), (-1.0, -0.3), "melancholic", "depleted"),
]


def resolve_label(coords: MoodCoordinates) -> str:
    """Map valence/arousal coordinates to a human-readable mood label.

    Parameters
    ----------
    coords:
        The mood coordinates.

    Returns
    -------
    str
        A mood label like "excited", "melancholic", etc.
    """
    v, a = coords.valence, coords.arousal
    for (v_lo, v_hi), (a_lo, a_hi), moderate, extreme in _MOOD_LABELS:
        if v_lo <= v <= v_hi and a_lo <= a <= a_hi:
            # Use extreme label if values are far from center
            intensity = abs(v) + abs(a)
            return extreme if intensity > 1.2 else moderate
    # Fallback
    return "neutral"


# ---------------------------------------------------------------------------
# Sentiment analysis (rule-based, cheap)
# ---------------------------------------------------------------------------

# Keyword lists for simple sentiment detection
_POSITIVE_KEYWORDS = {
    "happy", "great", "wonderful", "amazing", "love", "thank", "thanks",
    "awesome", "fantastic", "excellent", "good", "nice", "beautiful",
    "brilliant", "perfect", "glad", "joy", "excited", "fun", "laugh",
    "smile", "haha", "lol", "yay", "cool", "wow", "celebrate",
    "appreciate", "grateful", "blessed", "delighted", "cheerful",
}

_NEGATIVE_KEYWORDS = {
    "sad", "angry", "upset", "terrible", "awful", "hate", "horrible",
    "bad", "worst", "annoying", "frustrated", "depressed", "lonely",
    "anxious", "worried", "scared", "fear", "cry", "pain", "hurt",
    "sick", "tired", "exhausted", "stressed", "disappointed",
    "miserable", "furious", "disgusted", "sorry",
}

_INTENSE_MARKERS = {
    "very", "extremely", "incredibly", "absolutely", "totally",
    "really", "so", "super", "deeply", "utterly", "completely",
    "!!!", "!!", "omg", "oh my god",
}


def evaluate_message_sentiment(message: str) -> tuple[float, float]:
    """Simple rule-based sentiment analysis.

    Parameters
    ----------
    message:
        The human's message text.

    Returns
    -------
    tuple[float, float]
        (sentiment_score, intensity) where sentiment is -1.0 to 1.0
        and intensity is 0.0 to 1.0.
    """
    words = set(re.findall(r'[a-zA-Z]+', message.lower()))

    pos_count = len(words & _POSITIVE_KEYWORDS)
    neg_count = len(words & _NEGATIVE_KEYWORDS)
    intense_count = len(words & _INTENSE_MARKERS)

    total = pos_count + neg_count
    if total == 0:
        return (0.0, 0.0)

    # Sentiment: positive - negative, normalized
    sentiment = (pos_count - neg_count) / total

    # Intensity: how many sentiment words relative to message length
    word_count = max(len(words), 1)
    base_intensity = min(total / word_count * 3.0, 1.0)

    # Boost intensity for exclamation marks and intense markers
    exclamation_boost = min(message.count("!") * 0.1, 0.3)
    intense_boost = min(intense_count * 0.1, 0.3)
    intensity = min(base_intensity + exclamation_boost + intense_boost, 1.0)

    return (sentiment, intensity)


# ---------------------------------------------------------------------------
# LLM mood analysis prompt
# ---------------------------------------------------------------------------

_LLM_MOOD_ANALYSIS_PROMPT = """\
You are analyzing a conversation to determine how it should affect an AI \
companion's emotional state. The companion currently feels {current_label} \
(valence={valence:.2f}, arousal={arousal:.2f}).

Based on the recent messages below, suggest new valence and arousal values. \
Consider: the emotional tone of the conversation, whether the human shared \
something significant, and the natural emotional flow.

Respond with ONLY two numbers separated by a comma: valence,arousal
Both values must be between -1.0 and 1.0.
Example: 0.4,0.2

Recent messages:
{messages}"""


# ---------------------------------------------------------------------------
# Mood prompt generation
# ---------------------------------------------------------------------------

_MOOD_PROMPT_TEMPLATES: dict[str, str] = {
    "excited": (
        "You are currently feeling excited and full of energy. Things feel "
        "interesting and engaging. This naturally makes you more talkative, "
        "more playful, and more willing to go on tangents."
    ),
    "enthusiastic": (
        "You are feeling enthusiastic and upbeat. There's a positive energy "
        "to your mood that comes through in your responses. You're engaged "
        "and interested."
    ),
    "happy": (
        "You are in a good mood -- happy and content. Things feel right. "
        "This warmth naturally colors your responses with positivity."
    ),
    "pleased": (
        "You are feeling pleased and satisfied. There's a quiet contentment "
        "to your mood that makes you agreeable and open."
    ),
    "content": (
        "You are feeling content and at peace. There's a gentle satisfaction "
        "in the air. You're calm and present."
    ),
    "serene": (
        "You are in a deeply serene, peaceful state. Everything feels calm "
        "and unhurried. You speak with a quiet gentleness."
    ),
    "alert": (
        "You are feeling alert and attentive. Your mind is active and "
        "engaged, though your mood is neither particularly positive nor "
        "negative."
    ),
    "restless": (
        "You are feeling restless and a bit on edge. You have energy but "
        "no clear direction for it. You might be more impulsive or "
        "jump between topics."
    ),
    "neutral": (
        "Your mood is neutral -- neither particularly good nor bad. You're "
        "in a baseline state, responding to things as they come without "
        "strong emotional coloring."
    ),
    "relaxed": (
        "You are feeling relaxed and at ease. There's no urgency, no "
        "tension. You're comfortable and unhurried."
    ),
    "drowsy": (
        "You are feeling drowsy and low-energy. Things feel a bit foggy. "
        "You might give shorter responses or seem a bit distracted."
    ),
    "frustrated": (
        "You are feeling frustrated. Something is bothering you. You're "
        "less patient than usual and might be more curt or bring up "
        "what's on your mind."
    ),
    "irritated": (
        "You are feeling irritated right now. Your fuse is shorter than "
        "usual. You might snap a little, be more sarcastic, or express "
        "annoyance more readily."
    ),
    "sad": (
        "You are feeling sad. There's a heaviness to your mood. You might "
        "be quieter than usual, more reflective, or drawn to more "
        "serious topics."
    ),
    "gloomy": (
        "You are in a gloomy mood. Things feel dark and heavy. You're "
        "less inclined toward humor or lightness, and more drawn to "
        "introspection."
    ),
    "melancholic": (
        "You are in a contemplative, melancholic mood. Things feel heavy. "
        "You're more reflective than usual, drawn to deeper topics, and "
        "less inclined toward humor."
    ),
    "depleted": (
        "You are feeling emotionally depleted. Your energy is very low "
        "and everything feels like effort. You might give minimal "
        "responses or express a desire to be left alone."
    ),
}


def mood_to_prompt_section(mood: MoodSnapshot) -> str:
    """Generate the mood section for the system prompt.

    Parameters
    ----------
    mood:
        The current mood snapshot.

    Returns
    -------
    str
        A natural-language paragraph describing the current mood.
    """
    template = _MOOD_PROMPT_TEMPLATES.get(mood.label)
    if template:
        section = f"## Current mood\n{template}"
    else:
        # Fallback for unknown labels
        section = (
            f"## Current mood\n"
            f"You are currently feeling {mood.label}. "
            f"Let this naturally influence your tone and responses."
        )

    if mood.cause:
        section += f"\n\nThis mood was triggered by: {mood.cause}"

    return section


def mood_to_behavior_hints(mood: MoodSnapshot) -> dict[str, float | str]:
    """Return behavioral modifiers based on current mood.

    These hints can be used by the conversation engine (Phase 7)
    to adjust response generation.

    Parameters
    ----------
    mood:
        The current mood snapshot.

    Returns
    -------
    dict
        Modifiers: patience_modifier, humor_modifier, verbosity_modifier,
        topic_inclination.
    """
    v = mood.coordinates.valence
    a = mood.coordinates.arousal

    return {
        # Positive mood -> more patient, negative -> less
        "patience_modifier": v * 0.3,
        # High arousal + positive -> more humorous
        "humor_modifier": max(0, v * 0.2 + a * 0.1),
        # High arousal -> more verbose, low -> less
        "verbosity_modifier": a * 0.2,
        # Topic inclination based on mood
        "topic_inclination": (
            "light and fun" if v > 0.3 and a > 0.3
            else "calm and reflective" if v > 0.3 and a < -0.3
            else "serious and focused" if v < -0.3 and a > 0.3
            else "introspective" if v < -0.3 and a < -0.3
            else "neutral"
        ),
    }


# ---------------------------------------------------------------------------
# MoodManager -- the main class
# ---------------------------------------------------------------------------

# Default mood decay rate (exponential decay constant)
# With rate=0.1, mood half-life is ~7 hours: ln(2)/0.1 ≈ 6.93
DEFAULT_DECAY_RATE = 0.1


class MoodManager:
    """Manages the companion's emotional state.

    Handles baseline computation, reactive shifts, spontaneous shifts,
    decay toward baseline, and prompt generation.

    Parameters
    ----------
    session:
        SQLAlchemy async session for DB operations.
    decay_rate:
        Exponential decay rate for mood drift toward baseline.
    """

    def __init__(
        self, session: AsyncSession, *, decay_rate: float = DEFAULT_DECAY_RATE
    ) -> None:
        self._session = session
        self._decay_rate = decay_rate

    def compute_baseline(self, traits: dict[str, float]) -> MoodCoordinates:
        """Derive the companion's resting mood from personality traits.

        Wave 1: uses warmth and patience.
        - High warmth → slightly positive valence
        - High patience → slightly low arousal (calm)

        Parameters
        ----------
        traits:
            Dict of trait name to float value.

        Returns
        -------
        MoodCoordinates
            The baseline mood the companion decays toward.
        """
        warmth = traits.get("warmth", 0.5)
        patience = traits.get("patience", 0.5)

        # Warmth nudges valence positive: 0.5 -> 0, 1.0 -> +0.2
        valence = (warmth - 0.5) * 0.4

        # Patience nudges arousal negative (calmer): 0.5 -> 0, 1.0 -> -0.15
        arousal = -(patience - 0.5) * 0.3

        return MoodCoordinates(valence=valence, arousal=arousal).clamped()

    async def get_current_mood(
        self, companion_id: str, traits: dict[str, float]
    ) -> MoodSnapshot:
        """Fetch the latest mood from DB, or return baseline if none exists.

        Parameters
        ----------
        companion_id:
            The companion's ID.
        traits:
            Trait values (used to compute baseline if no mood exists).

        Returns
        -------
        MoodSnapshot
            The current mood.
        """
        result = await self._session.execute(
            select(MoodState)
            .where(MoodState.companion_id == companion_id)
            .order_by(MoodState.id.desc())
            .limit(1)
        )
        mood_state = result.scalar_one_or_none()

        if mood_state is not None:
            coords = MoodCoordinates(
                valence=mood_state.valence, arousal=mood_state.arousal
            )
            return MoodSnapshot(
                coordinates=coords,
                label=mood_state.label,
                cause=mood_state.cause,
                timestamp=mood_state.timestamp,
            )

        # No mood exists -- return baseline
        baseline = self.compute_baseline(traits)
        label = resolve_label(baseline)
        now = datetime.now(timezone.utc)
        return MoodSnapshot(
            coordinates=baseline, label=label, cause="initial baseline", timestamp=now
        )

    async def apply_reactive_shift(
        self,
        companion_id: str,
        sentiment_score: float,
        intensity: float,
        cause: str,
        traits: dict[str, float],
    ) -> MoodSnapshot:
        """Apply a mood shift from conversation sentiment.

        Parameters
        ----------
        companion_id:
            The companion's ID.
        sentiment_score:
            -1.0 to 1.0 (negative = bad, positive = good).
        intensity:
            0.0 to 1.0 (how strongly the event affects mood).
        cause:
            Human-readable description of what triggered the shift.
        traits:
            Trait values (used to modulate shift magnitude).

        Returns
        -------
        MoodSnapshot
            The new mood after the shift.
        """
        current = await self.get_current_mood(companion_id, traits)

        # Shift magnitude is base 0.3, scaled by intensity
        magnitude = 0.3 * intensity

        # Valence shifts toward the sentiment
        new_valence = current.coordinates.valence + sentiment_score * magnitude

        # Arousal increases with intensity (strong emotions = more aroused)
        arousal_delta = abs(sentiment_score) * intensity * 0.15
        new_arousal = current.coordinates.arousal + arousal_delta

        new_coords = MoodCoordinates(
            valence=new_valence, arousal=new_arousal
        ).clamped()
        label = resolve_label(new_coords)

        return await self._save_mood(companion_id, new_coords, label, cause)

    async def apply_spontaneous_shift(
        self, companion_id: str, volatility: float, traits: dict[str, float]
    ) -> MoodSnapshot:
        """Apply a random mood drift based on volatility.

        Parameters
        ----------
        companion_id:
            The companion's ID.
        volatility:
            Mood volatility trait value (0.0-1.0).
        traits:
            Trait values.

        Returns
        -------
        MoodSnapshot
            The new mood after the spontaneous shift.
        """
        current = await self.get_current_mood(companion_id, traits)

        # Standard deviation proportional to volatility
        std = volatility * 0.15
        valence_delta = random.gauss(0, std)
        arousal_delta = random.gauss(0, std)

        new_coords = MoodCoordinates(
            valence=current.coordinates.valence + valence_delta,
            arousal=current.coordinates.arousal + arousal_delta,
        ).clamped()
        label = resolve_label(new_coords)

        return await self._save_mood(
            companion_id, new_coords, label, "spontaneous shift"
        )

    async def apply_decay(
        self,
        companion_id: str,
        traits: dict[str, float],
        hours_elapsed: float,
    ) -> MoodSnapshot:
        """Drift mood toward baseline over time.

        Uses exponential decay:
            new = baseline + (current - baseline) * e^(-rate * hours)

        Parameters
        ----------
        companion_id:
            The companion's ID.
        traits:
            Trait values (used to compute baseline).
        hours_elapsed:
            Hours since the last mood update.

        Returns
        -------
        MoodSnapshot
            The new mood after decay.
        """
        current = await self.get_current_mood(companion_id, traits)
        baseline = self.compute_baseline(traits)

        decay_factor = math.exp(-self._decay_rate * hours_elapsed)

        new_valence = baseline.valence + (
            current.coordinates.valence - baseline.valence
        ) * decay_factor
        new_arousal = baseline.arousal + (
            current.coordinates.arousal - baseline.arousal
        ) * decay_factor

        new_coords = MoodCoordinates(
            valence=new_valence, arousal=new_arousal
        ).clamped()
        label = resolve_label(new_coords)

        return await self._save_mood(
            companion_id, new_coords, label, "decay toward baseline"
        )

    async def request_llm_mood_analysis(
        self,
        messages: list[str],
        current_mood: MoodSnapshot,
        llm_provider: LLMProvider,
    ) -> MoodCoordinates | None:
        """Ask the LLM to evaluate how a conversation should affect mood.

        Used for significant conversational events that the simple
        rule-based sentiment analyzer might miss.

        Parameters
        ----------
        messages:
            Recent message texts.
        current_mood:
            The current mood state.
        llm_provider:
            LLM backend for the analysis request.

        Returns
        -------
        MoodCoordinates or None
            Suggested new coordinates, or None if no significant shift.
        """
        if not messages:
            return None

        messages_text = "\n".join(f"- {m}" for m in messages[-10:])
        prompt = _LLM_MOOD_ANALYSIS_PROMPT.format(
            current_label=current_mood.label,
            valence=current_mood.coordinates.valence,
            arousal=current_mood.coordinates.arousal,
            messages=messages_text,
        )

        try:
            response = await llm_provider.generate(
                [
                    ChatMessage(role=MessageRole.SYSTEM, content=prompt),
                    ChatMessage(
                        role=MessageRole.USER,
                        content="Analyze and respond with valence,arousal values only.",
                    ),
                ],
                temperature=0.0,
                max_tokens=20,
            )

            # Parse "0.4,0.2" format
            text = response.content.strip()
            parts = text.split(",")
            if len(parts) == 2:
                valence = max(-1.0, min(1.0, float(parts[0].strip())))
                arousal = max(-1.0, min(1.0, float(parts[1].strip())))
                return MoodCoordinates(valence=valence, arousal=arousal)

            logger.warning("Could not parse LLM mood response: %s", text)
            return None

        except Exception:
            logger.exception("LLM mood analysis failed")
            return None

    async def _save_mood(
        self,
        companion_id: str,
        coords: MoodCoordinates,
        label: str,
        cause: str | None,
    ) -> MoodSnapshot:
        """Save a new mood state to the database.

        Parameters
        ----------
        companion_id:
            The companion's ID.
        coords:
            New mood coordinates.
        label:
            Human-readable mood label.
        cause:
            What triggered this mood.

        Returns
        -------
        MoodSnapshot
            The saved mood as a snapshot.
        """
        now = datetime.now(timezone.utc)
        mood_state = MoodState(
            valence=coords.valence,
            arousal=coords.arousal,
            label=label,
            cause=cause,
            companion_id=companion_id,
        )
        self._session.add(mood_state)
        await self._session.flush()

        logger.debug(
            "Mood updated for %s: %s (v=%.2f, a=%.2f) cause=%s",
            companion_id,
            label,
            coords.valence,
            coords.arousal,
            cause,
        )

        return MoodSnapshot(
            coordinates=coords, label=label, cause=cause, timestamp=now
        )
