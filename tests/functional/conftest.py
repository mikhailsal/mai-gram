"""Functional test fixtures and helpers."""

from __future__ import annotations

import json
import os
import shutil
try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select

from mai_companion.bot.handler import BotHandler
from mai_companion.bot.middleware import RateLimitConfig
from mai_companion.clock import Clock
from mai_companion.config import Settings
from mai_companion.db import close_db, get_session, init_db, reset_db_state, run_migrations
from mai_companion.db.models import Companion, MoodState
from mai_companion.debug import LLMLoggerProvider
from mai_companion.llm.openrouter import OpenRouterProvider
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore
from mai_companion.messenger.base import IncomingMessage, MessageType
from mai_companion.messenger.console import ConsoleMessenger
from mai_companion.personality.character import (
    CharacterBuilder,
    CharacterConfig,
    CommunicationStyle,
    Gender,
    Verbosity,
)
from mai_companion.personality.mood import MoodManager, resolve_label
from mai_companion.personality.temperature import compute_temperature


@dataclass(frozen=True)
class FunctionalConfig:
    llm_model: str
    summary_threshold: int
    short_term_limit: int
    wiki_context_limit: int
    tool_max_iterations: int
    data_dir: str
    debug_logging: bool


@dataclass
class _ChatRuntime:
    messenger: ConsoleMessenger
    output: StringIO
    handler: BotHandler
    logger_provider: LLMLoggerProvider | None


class FunctionalHarness:
    """Reusable runtime helper for functional tests."""

    def __init__(
        self,
        *,
        config: FunctionalConfig,
        settings: Settings,
        llm_base: OpenRouterProvider,
    ) -> None:
        self.config = config
        self.settings = settings
        self._llm_base = llm_base
        self._chat_runtimes: dict[str, _ChatRuntime] = {}
        self._chat_clocks: dict[str, Clock] = {}
        self._user_by_chat: dict[str, str] = {}

    def _clock_provider(self, chat_id: str) -> Clock:
        return self._chat_clocks.setdefault(chat_id, Clock())

    def _set_target_date(self, chat_id: str, target_date: date | None) -> None:
        if target_date is None:
            return
        self._chat_clocks[chat_id] = Clock.for_target_date(target_date)

    @staticmethod
    def _latest_ai_response(console_output: str) -> str:
        marker = "--- AI Response ---"
        if marker not in console_output:
            return console_output.strip()
        chunk = console_output.rsplit(marker, maxsplit=1)[1]
        lines = chunk.strip().splitlines()
        body: list[str] = []
        for line in lines:
            if line.strip() == "--- Buttons ---":
                break
            body.append(line)
        return "\n".join(body).strip()

    @staticmethod
    def _all_ai_responses(console_output: str) -> list[str]:
        """Extract *every* ``--- AI Response ---`` block from console output.

        Returns a list of response texts (one per block).  Useful for
        verifying multi-message behaviour where intermediate messages
        produce separate blocks.
        """
        marker = "--- AI Response ---"
        if marker not in console_output:
            stripped = console_output.strip()
            return [stripped] if stripped else []

        parts = console_output.split(marker)
        responses: list[str] = []
        for part in parts[1:]:  # skip everything before the first marker
            lines = part.strip().splitlines()
            body: list[str] = []
            for line in lines:
                if line.strip() == "--- Buttons ---":
                    break
                body.append(line)
            text = "\n".join(body).strip()
            if text:
                responses.append(text)
        return responses

    async def _ensure_chat(self, chat_id: str, user_id: str = "functional-user") -> _ChatRuntime:
        existing = self._chat_runtimes.get(chat_id)
        if existing is not None:
            return existing

        output = StringIO()
        messenger = ConsoleMessenger(output=output)
        logger_provider = (
            LLMLoggerProvider(
                self._llm_base,
                chat_id=chat_id,
                base_dir=Path(self.settings.memory_data_dir) / "debug_logs",
                clock=self._clock_provider(chat_id),
            )
            if self.config.debug_logging
            else None
        )
        handler = BotHandler(
            messenger,
            logger_provider or self._llm_base,
            rate_limit_config=RateLimitConfig(
                messages_per_minute=10_000,
                messages_per_hour=100_000,
                cooldown_seconds=0,
            ),
            memory_data_dir=self.settings.memory_data_dir,
            summary_threshold=self.config.summary_threshold,
            wiki_context_limit=self.config.wiki_context_limit,
            short_term_limit=self.config.short_term_limit,
            tool_max_iterations=self.config.tool_max_iterations,
            clock_provider=self._clock_provider,
        )
        runtime = _ChatRuntime(
            messenger=messenger,
            output=output,
            handler=handler,
            logger_provider=logger_provider,
        )
        self._chat_runtimes[chat_id] = runtime
        self._user_by_chat[chat_id] = user_id
        self._chat_clocks.setdefault(chat_id, Clock())
        return runtime

    async def send_start(self, chat_id: str, *, user_id: str = "functional-user") -> str:
        runtime = await self._ensure_chat(chat_id, user_id=user_id)
        incoming = IncomingMessage(
            platform="console",
            chat_id=chat_id,
            user_id=user_id,
            message_id=f"start-{int(datetime.now(timezone.utc).timestamp())}",
            message_type=MessageType.COMMAND,
            command="start",
            text="/start",
            timestamp=datetime.now(timezone.utc),
        )
        await runtime.messenger.dispatch_message(incoming)
        rendered = runtime.output.getvalue()
        runtime.output.seek(0)
        runtime.output.truncate(0)
        return rendered

    async def send_callback(
        self,
        chat_id: str,
        callback_data: str,
        *,
        user_id: str | None = None,
        target_date: date | None = None,
    ) -> str:
        runtime = await self._ensure_chat(chat_id, user_id=user_id or "functional-user")
        self._set_target_date(chat_id, target_date)
        sender_id = user_id or self._user_by_chat.get(chat_id, "functional-user")
        await runtime.messenger.dispatch_callback(
            chat_id=chat_id,
            user_id=sender_id,
            callback_data=callback_data,
        )
        rendered = runtime.output.getvalue()
        runtime.output.seek(0)
        runtime.output.truncate(0)
        return rendered

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        target_date: date | None = None,
    ) -> str:
        runtime = await self._ensure_chat(chat_id, user_id=user_id or "functional-user")
        self._set_target_date(chat_id, target_date)
        sender_id = user_id or self._user_by_chat.get(chat_id, "functional-user")
        await runtime.messenger.dispatch_text(chat_id=chat_id, user_id=sender_id, text=text)
        rendered = runtime.output.getvalue()
        runtime.output.seek(0)
        runtime.output.truncate(0)
        return self._latest_ai_response(rendered)

    async def send_message_multi(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        target_date: date | None = None,
    ) -> list[str]:
        """Send a message and return *all* AI response blocks.

        Unlike :meth:`send_message` which returns only the last response,
        this returns every ``--- AI Response ---`` block — useful for
        testing multi-message flows (e.g. the sleep tool).
        """
        runtime = await self._ensure_chat(chat_id, user_id=user_id or "functional-user")
        self._set_target_date(chat_id, target_date)
        sender_id = user_id or self._user_by_chat.get(chat_id, "functional-user")
        await runtime.messenger.dispatch_text(chat_id=chat_id, user_id=sender_id, text=text)
        rendered = runtime.output.getvalue()
        runtime.output.seek(0)
        runtime.output.truncate(0)
        return self._all_ai_responses(rendered)

    async def create_companion_directly(
        self,
        chat_id: str,
        *,
        name: str = "TestBot",
        language: str = "English",
        traits: dict[str, float] | None = None,
        communication_style: CommunicationStyle = CommunicationStyle.FORMAL,
        verbosity: Verbosity = Verbosity.CONCISE,
        gender: Gender = Gender.NEUTRAL,
        language_style: str | None = None,
    ) -> Companion:
        """Insert a companion directly into the DB, bypassing onboarding.

        This is cheaper than :meth:`complete_onboarding` (zero LLM calls)
        and gives full control over personality traits and system prompt.
        Useful for deterministic tests that need a specific character.
        """
        if traits is None:
            traits = {
                "warmth": 0.3,
                "humor": 0.0,
                "patience": 0.9,
                "directness": 0.9,
                "laziness": 0.0,
                "mood_volatility": 0.1,
            }

        config = CharacterConfig(
            name=name,
            language=language,
            traits=traits,
            gender=gender,
            language_style=language_style,
            communication_style=communication_style,
            verbosity=verbosity,
        )
        temperature = compute_temperature(traits)
        record = CharacterBuilder.create_companion_record(config, temperature)
        record["id"] = chat_id

        async with get_session() as session:
            companion = Companion(**record)
            session.add(companion)
            await session.flush()

            # Create initial mood state
            mood_manager = MoodManager(session)
            baseline = mood_manager.compute_baseline(traits)
            label = resolve_label(baseline)
            await mood_manager._save_mood(
                companion.id,
                baseline,
                label,
                "initial baseline at companion creation",
            )
            await session.commit()

        return companion

    async def complete_onboarding(
        self,
        chat_id: str,
        *,
        language: str = "English",
        companion_name: str = "Mira",
        preset_key: str = "balanced_friend",
        user_id: str = "functional-user",
    ) -> None:
        await self.send_start(chat_id, user_id=user_id)
        await self.send_message(chat_id, language, user_id=user_id)
        await self.send_message(chat_id, companion_name, user_id=user_id)
        await self.send_callback(chat_id, "personality:presets", user_id=user_id)
        await self.send_callback(chat_id, f"preset:{preset_key}", user_id=user_id)
        await self.send_callback(chat_id, "preset_confirm:yes", user_id=user_id)
        await self.send_callback(chat_id, "appearance:skip", user_id=user_id)
        await self.send_callback(chat_id, "confirm:yes", user_id=user_id)

    async def get_companion(self, chat_id: str) -> Companion | None:
        async with get_session() as session:
            result = await session.execute(select(Companion).where(Companion.id == chat_id))
            return result.scalar_one_or_none()

    async def get_wiki_entries(self, companion_id: str) -> list[dict[str, Any]]:
        async with get_session() as session:
            store = WikiStore(session, data_dir=self.settings.memory_data_dir)
            entries = await store.list_entries(companion_id)
        return [
            {
                "key": entry.key,
                "value": entry.value,
                "importance": int(entry.importance),
            }
            for entry in entries
        ]

    def get_summaries(self, companion_id: str) -> list[dict[str, str]]:
        store = SummaryStore(self.settings.memory_data_dir)
        summaries = store.get_all_summaries(companion_id)
        return [
            {
                "summary_type": item.summary_type,
                "period": item.period,
                "content": item.content,
            }
            for item in summaries
        ]

    def get_summary_path(self, companion_id: str, summary_type: str, period: str) -> Path:
        return (
            Path(self.settings.memory_data_dir)
            / companion_id
            / "summaries"
            / summary_type
            / f"{period}.md"
        )

    def get_debug_log(self, chat_id: str, target_date: date) -> list[dict[str, Any]]:
        path = (
            Path(self.settings.memory_data_dir)
            / "debug_logs"
            / chat_id
            / f"{target_date.isoformat()}.jsonl"
        )
        if not path.exists():
            return []
        payloads: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payloads.append(json.loads(line))
        return payloads

    def get_wiki_changelog(self, companion_id: str) -> list[dict[str, Any]]:
        path = Path(self.settings.memory_data_dir) / companion_id / "wiki" / "changelog.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    async def seed_messages(self, chat_id: str, seed_file: str) -> int:
        seed_path = Path(__file__).parent / "seed_data" / seed_file
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_path}")

        async with get_session() as session:
            result = await session.execute(select(Companion).where(Companion.id == chat_id))
            companion = result.scalar_one_or_none()
            if companion is None:
                raise RuntimeError(f"Companion '{chat_id}' must exist before seeding messages")

            store = MessageStore(session)
            count = 0
            offsets: dict[date, int] = {}
            for line in seed_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                role = payload["role"]
                content = payload["content"]
                target = date.fromisoformat(payload["date"])
                second = offsets.get(target, 0)
                offsets[target] = second + 1
                timestamp = datetime.combine(target, time(12, 0), tzinfo=timezone.utc) + timedelta(
                    seconds=second
                )
                await store.save_message(chat_id, role, content, timestamp=timestamp)
                count += 1
        return count

    def get_cost_totals(self) -> dict[str, int | float]:
        calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        estimated_cost = 0.0
        for runtime in self._chat_runtimes.values():
            if runtime.logger_provider is None:
                continue
            stats = runtime.logger_provider.get_session_stats()
            calls += int(stats["llm_calls"])
            prompt_tokens += int(stats["prompt_tokens"])
            completion_tokens += int(stats["completion_tokens"])
            total_tokens += int(stats["total_tokens"])
            estimated_cost += float(stats["session_cost_usd"])
        return {
            "calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
        }


@pytest.fixture(scope="session")
def functional_config() -> FunctionalConfig:
    """Load test-specific functional configuration from TOML."""
    config_path = Path(__file__).parent / "functional_config.toml"
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return FunctionalConfig(
        llm_model=str(payload["llm"]["model"]),
        summary_threshold=int(payload["memory"]["summary_threshold"]),
        short_term_limit=int(payload["memory"]["short_term_limit"]),
        wiki_context_limit=int(payload["memory"]["wiki_context_limit"]),
        tool_max_iterations=int(payload["memory"]["tool_max_iterations"]),
        data_dir=str(payload["testing"]["data_dir"]),
        debug_logging=bool(payload["testing"]["debug_logging"]),
    )


@pytest.fixture(scope="session")
def functional_paths(functional_config: FunctionalConfig) -> dict[str, Path]:
    """Create and return isolated filesystem paths for the functional session."""
    root_dir = Path(functional_config.data_dir).resolve()
    session_dir = root_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    data_dir = session_dir / "data"
    db_path = session_dir / "functional.db"
    session_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root_dir,
        "session": session_dir,
        "data_dir": data_dir,
        "db_path": db_path,
    }


@pytest.fixture(scope="session")
def functional_settings(
    functional_config: FunctionalConfig,
    functional_paths: dict[str, Path],
) -> Settings:
    """Build a Settings object for functional runtime isolation."""
    return Settings(
        telegram_bot_token="functional-token",
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        llm_model=functional_config.llm_model,
        database_url=f"sqlite+aiosqlite:///{functional_paths['db_path']}",
        memory_data_dir=str(functional_paths["data_dir"]),
        summary_threshold=functional_config.summary_threshold,
        wiki_context_limit=functional_config.wiki_context_limit,
        short_term_limit=functional_config.short_term_limit,
        tool_max_iterations=functional_config.tool_max_iterations,
        allowed_users="",
        debug=True,
    )


@pytest.fixture(scope="session")
async def functional_runtime(
    functional_config: FunctionalConfig,
    functional_settings: Settings,
    functional_paths: dict[str, Path],
) -> FunctionalHarness:
    """End-to-end runtime harness with a real OpenRouter provider."""
    if not functional_settings.openrouter_api_key.strip():
        pytest.skip("OPENROUTER_API_KEY is required for functional LLM tests.")

    with patch("mai_companion.bot.handler.get_settings", lambda: functional_settings):
        reset_db_state()
        engine = await init_db(functional_settings.database_url, echo=False)
        await run_migrations(engine)
        llm_base = OpenRouterProvider(
            api_key=functional_settings.openrouter_api_key,
            default_model=functional_settings.llm_model,
        )
        harness = FunctionalHarness(
            config=functional_config,
            settings=functional_settings,
            llm_base=llm_base,
        )
        try:
            yield harness
        finally:
            totals = harness.get_cost_totals()
            print("\n--- Test Session Cost ---")
            print(f"Total LLM calls: {totals['calls']}")
            print(
                "Total tokens: "
                f"{totals['total_tokens']} ({totals['prompt_tokens']} prompt + "
                f"{totals['completion_tokens']} completion)"
            )
            print(f"Estimated cost: ${totals['estimated_cost_usd']:.4f}")
            await llm_base.close()
            await close_db()
            if functional_paths["session"].exists():
                shutil.rmtree(functional_paths["session"])

