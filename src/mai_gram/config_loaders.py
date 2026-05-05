"""File-backed config loaders used by the Settings facade."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib

    TOMLDecodeError = tomllib.TOMLDecodeError
else:
    import tomli as tomllib

    TOMLDecodeError = tomllib.TOMLDecodeError

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Per-prompt display and tool/MCP settings loaded from a companion TOML file."""

    show_reasoning: bool = True
    show_tool_calls: bool = True
    send_datetime: bool | None = None
    tools_enabled: list[str] | None = None
    tools_disabled: list[str] | None = None
    mcp_servers_enabled: list[str] | None = None
    mcp_servers_disabled: list[str] | None = None


@dataclass
class BotConfig:
    """Per-bot configuration loaded from ``config/bots.toml``."""

    token: str
    allowed_users: list[int] | None = None
    allowed_models: list[str] | None = None
    allowed_prompts: list[str] | None = None
    allowed_templates: list[str] | None = None


_MODEL_META_KEYS = frozenset({"enabled", "title", "id", "max_context_tokens"})


class ModelsConfigLoader:
    """Load and mtime-cache the shared models TOML file.

    Model sections live under ``[models."<key>"]``.  Each section may contain:

    * ``enabled`` (bool, default ``True``) -- whether the model is offered to
      users during chat setup.
    * ``title`` (str, optional) -- human-friendly label shown in selection UI.
    * ``id`` (str, optional) -- real OpenRouter model identifier when the
      section key is an alias (allows the same model with different params).
    * Any other keys are treated as OpenRouter request-body overrides.
    """

    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._cache: dict[str, Any] = {}
        self._mtime: float = 0.0

    def refresh(self) -> dict[str, Any]:
        config_path = self._config_path
        if not config_path.exists():
            self._cache = {}
            self._mtime = 0.0
            return {}
        try:
            mtime = config_path.stat().st_mtime
        except OSError:
            self._cache = {}
            self._mtime = 0.0
            return {}

        if mtime != self._mtime:
            with config_path.open("rb") as f:
                self._cache = tomllib.load(f)
            self._mtime = mtime
            logger.info("Config reloaded: %s (mtime=%.3f)", config_path, mtime)
        return self._cache

    def _iter_model_sections(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(key, section_dict)`` for every model sub-table."""
        models = self.refresh().get("models", {})
        return [(k, v) for k, v in models.items() if isinstance(v, dict)]

    def get_enabled_models(self, default_model: str) -> list[str]:
        """Return keys of models that are enabled (``enabled`` defaults to ``True``)."""
        sections = self._iter_model_sections()
        if not sections:
            return [default_model]
        return [key for key, sec in sections if sec.get("enabled", True)]

    # Backward-compatible alias used by Settings.get_allowed_models
    def get_allowed_models(self, default_model: str) -> list[str]:
        return self.get_enabled_models(default_model)

    def get_default_model(self, default_model: str) -> str:
        data = self.refresh()
        result: str = data.get("models", {}).get("default", default_model)
        return result

    def get_model_title(self, model_key: str) -> str | None:
        """Return the display title for *model_key*, or ``None`` if not set."""
        data = self.refresh()
        section = data.get("models", {}).get(model_key)
        if isinstance(section, dict):
            title = section.get("title")
            return str(title) if title is not None else None
        return None

    def get_model_id(self, model_key: str) -> str:
        """Resolve the real OpenRouter model ID for *model_key*.

        If the section defines ``id``, that value is used; otherwise the
        section key itself is assumed to be the real model identifier.
        """
        data = self.refresh()
        section = data.get("models", {}).get(model_key)
        if isinstance(section, dict):
            real_id = section.get("id")
            if real_id:
                return str(real_id)
        return model_key

    def get_max_context_tokens(self, model_key: str) -> int:
        """Resolve the context-truncation budget for *model_key*.

        Resolution order:
        1. Per-model ``max_context_tokens`` in ``[models."<key>"]``
        2. Global ``max_context_tokens`` in ``[models]``
        3. ``0`` (disabled -- no truncation)

        A value of ``0`` disables automatic context truncation entirely.
        """
        data = self.refresh()
        models = data.get("models", {})

        section = models.get(model_key)
        if isinstance(section, dict):
            per_model = section.get("max_context_tokens")
            if per_model is not None:
                return int(per_model)

        global_val = models.get("max_context_tokens")
        if global_val is not None:
            return int(global_val)

        return 0

    def get_model_params(self, model_key: str) -> dict[str, Any]:
        """Return API-level overrides for *model_key* (meta-keys stripped)."""
        data = self.refresh()
        section = data.get("models", {}).get(model_key, {})
        if not isinstance(section, dict):
            return {}
        return {k: v for k, v in section.items() if k not in _MODEL_META_KEYS}

    def get_tool_filter(self) -> tuple[list[str] | None, list[str] | None]:
        data = self.refresh()
        tools_section = data.get("tools", {})
        return tools_section.get("enabled"), tools_section.get("disabled")

    def get_external_mcp_config(self) -> dict[str, dict[str, Any]]:
        import json as _json

        data = self.refresh()
        mcp_section = data.get("mcp", {})
        mcp_json_path = mcp_section.get("mcp_config_path", "")
        whitelist = set(mcp_section.get("external_servers", []))

        if not mcp_json_path or not whitelist:
            return {}

        json_path = Path(mcp_json_path).expanduser()
        if not json_path.exists():
            return {}

        with json_path.open(encoding="utf-8") as f:
            mcp_data = _json.load(f)

        servers_raw = mcp_data.get("mcpServers", {})
        result: dict[str, dict[str, Any]] = {}
        for name, config in servers_raw.items():
            if name in whitelist:
                result[name] = config
        return result


class BotsConfigLoader:
    """Load and mtime-cache ``bots.toml``."""

    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._cache: list[BotConfig] | None = None
        self._mtime: float = 0.0

    def get_bot_configs(self) -> list[BotConfig]:
        config_path = self._config_path
        if not config_path.exists():
            return []

        try:
            mtime = config_path.stat().st_mtime
        except OSError:
            return []

        if mtime != self._mtime or self._cache is None:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
            self._mtime = mtime

            configs: list[BotConfig] = []
            for entry in data.get("bots", []):
                token = entry.get("token", "").strip()
                if not token:
                    logger.warning("Skipping bot entry with empty token in bots.toml")
                    continue
                configs.append(
                    BotConfig(
                        token=token,
                        allowed_users=entry.get("allowed_users"),
                        allowed_models=entry.get("allowed_models"),
                        allowed_prompts=entry.get("allowed_prompts"),
                        allowed_templates=entry.get("allowed_templates"),
                    )
                )
            self._cache = configs
            logger.info(
                "Loaded %d bot(s) from %s (mtime=%.3f)",
                len(configs),
                config_path,
                mtime,
            )

        return self._cache

    def get_bot_config_by_token(self, token: str) -> BotConfig | None:
        for bot_config in self.get_bot_configs():
            if bot_config.token == token:
                return bot_config
        return None


class PromptConfigLoader:
    """Load prompt text files and companion TOML config files."""

    def __init__(self, prompts_dir: str) -> None:
        self._prompts_dir = Path(prompts_dir)

    def get_available_prompts(self) -> dict[str, str]:
        prompts_path = self._prompts_dir
        if not prompts_path.exists():
            return {}
        result: dict[str, str] = {}
        for prompt_file in sorted(prompts_path.iterdir()):
            if prompt_file.is_file() and prompt_file.suffix in (".txt", ".md"):
                result[prompt_file.stem] = prompt_file.read_text(encoding="utf-8").strip()
        return result

    def get_prompt_config(self, prompt_name: str) -> PromptConfig:
        config_path = self._prompts_dir / f"{prompt_name}.toml"
        if not config_path.exists():
            return PromptConfig()
        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)

            tools = data.get("tools", {})
            mcp = data.get("mcp_servers", {})
            return PromptConfig(
                show_reasoning=data.get("show_reasoning", True),
                show_tool_calls=data.get("show_tool_calls", True),
                send_datetime=data.get("send_datetime"),
                tools_enabled=tools.get("enabled"),
                tools_disabled=tools.get("disabled"),
                mcp_servers_enabled=mcp.get("enabled"),
                mcp_servers_disabled=mcp.get("disabled"),
            )
        except (TOMLDecodeError, ValueError, TypeError, AttributeError):
            logger.warning("Failed to parse prompt config: %s", config_path)
            return PromptConfig()
