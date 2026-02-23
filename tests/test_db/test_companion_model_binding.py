"""Tests for companion LLM model binding.

Verifies that the LLM model is stored per-companion and not changed
by global configuration updates. This protects companion identity --
the model is the companion's "soul".
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mai_companion.db.models import Companion
from mai_companion.personality.character import (
    CharacterBuilder,
    CommunicationStyle,
    Gender,
    Verbosity,
)


class TestCompanionModelField:
    """Tests for the llm_model field on Companion model."""

    async def test_companion_has_default_model(self, session: AsyncSession) -> None:
        """A companion created with just a name gets the default model."""
        companion = Companion(name="Aria")
        session.add(companion)
        await session.flush()

        assert companion.llm_model == "openai/gpt-4o"

    async def test_companion_model_can_be_set(self, session: AsyncSession) -> None:
        """A companion can be created with a specific model."""
        companion = Companion(
            id="test-companion-001",
            name="Luna",
            llm_model="anthropic/claude-3-opus",
        )
        session.add(companion)
        await session.flush()

        result = await session.execute(
            select(Companion).where(Companion.id == "test-companion-001")
        )
        loaded = result.scalar_one()
        assert loaded.llm_model == "anthropic/claude-3-opus"

    async def test_companion_model_persists_after_reload(
        self, session: AsyncSession
    ) -> None:
        """The model is correctly persisted and retrieved from the database."""
        companion = Companion(
            id="persist-test",
            name="TestBot",
            llm_model="google/gemini-pro",
        )
        session.add(companion)
        await session.flush()

        # Clear the session cache and reload
        session.expire(companion)

        result = await session.execute(
            select(Companion).where(Companion.id == "persist-test")
        )
        reloaded = result.scalar_one()
        assert reloaded.llm_model == "google/gemini-pro"


class TestCharacterBuilderModelBinding:
    """Tests for model binding in CharacterBuilder.create_companion_record."""

    def test_create_companion_record_includes_model(self) -> None:
        """create_companion_record includes the llm_model field."""
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        record = CharacterBuilder.create_companion_record(
            config, temperature=0.65, llm_model="anthropic/claude-3-sonnet"
        )

        assert "llm_model" in record
        assert record["llm_model"] == "anthropic/claude-3-sonnet"

    def test_create_companion_record_default_model(self) -> None:
        """create_companion_record uses default model if not specified."""
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)

        assert record["llm_model"] == "openai/gpt-4o"

    def test_create_companion_record_preserves_all_fields(self) -> None:
        """Adding llm_model doesn't break other fields."""
        config = CharacterBuilder.from_preset(
            "Luna",
            "caring_guide",
            language="Russian",
            style=CommunicationStyle.FORMAL,
            verbosity=Verbosity.DETAILED,
        )
        config.gender = Gender.FEMALE
        config.language_style = "pre-revolutionary orthography"

        record = CharacterBuilder.create_companion_record(
            config, temperature=0.75, llm_model="openai/gpt-4-turbo"
        )

        # Verify all existing fields are still present
        assert record["name"] == "Luna"
        assert record["human_language"] == "Russian"
        assert record["temperature"] == 0.75
        assert record["gender"] == "female"
        assert record["language_style"] == "pre-revolutionary orthography"
        assert record["communication_style"] == "formal"
        assert record["verbosity"] == "detailed"
        assert record["relationship_stage"] == "getting_to_know"
        # And the new field
        assert record["llm_model"] == "openai/gpt-4-turbo"


class TestModelIsolation:
    """Tests verifying that companions maintain their own models independently."""

    async def test_different_companions_can_have_different_models(
        self, session: AsyncSession
    ) -> None:
        """Multiple companions can each have their own model."""
        companion1 = Companion(
            id="comp-1",
            name="Alpha",
            llm_model="openai/gpt-4o",
        )
        companion2 = Companion(
            id="comp-2",
            name="Beta",
            llm_model="anthropic/claude-3-opus",
        )
        companion3 = Companion(
            id="comp-3",
            name="Gamma",
            llm_model="google/gemini-pro",
        )

        session.add_all([companion1, companion2, companion3])
        await session.flush()

        # Verify each has its own model
        result = await session.execute(select(Companion).order_by(Companion.id))
        companions = result.scalars().all()

        assert len(companions) == 3
        assert companions[0].llm_model == "openai/gpt-4o"
        assert companions[1].llm_model == "anthropic/claude-3-opus"
        assert companions[2].llm_model == "google/gemini-pro"

    async def test_model_not_affected_by_other_companions(
        self, session: AsyncSession
    ) -> None:
        """Creating a new companion doesn't affect existing companions' models."""
        # Create first companion
        original = Companion(
            id="original",
            name="Original",
            llm_model="anthropic/claude-3-opus",
        )
        session.add(original)
        await session.flush()

        # Create second companion with different model
        new_companion = Companion(
            id="new",
            name="New",
            llm_model="openai/gpt-4-turbo",
        )
        session.add(new_companion)
        await session.flush()

        # Verify original is unchanged
        result = await session.execute(
            select(Companion).where(Companion.id == "original")
        )
        reloaded_original = result.scalar_one()
        assert reloaded_original.llm_model == "anthropic/claude-3-opus"


class TestModelDocumentation:
    """Tests verifying the model field has appropriate documentation."""

    def test_model_field_has_doc(self) -> None:
        """The llm_model field should have documentation explaining its purpose."""
        from mai_companion.db.models import Companion

        # Get the column info
        column = Companion.__table__.columns["llm_model"]
        assert column.doc is not None
        assert "soul" in column.doc.lower()  # References the philosophical concept
