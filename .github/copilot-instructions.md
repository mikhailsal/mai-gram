# Project Guidelines

## Build And Test

- Use the documented Make targets instead of ad hoc commands when possible: `make install-dev`, `make test`, `make test-cov`, `make check`, and `make precommit`.
- After changing user-visible behavior, verify it through the console CLI with `mai-chat`, because it runs the same `BotHandler` pipeline as Telegram. See `docs/DEBUGGING.md`.
- Use isolated test chat IDs with the prefixes `test-`, `func-`, `manual-`, or `demo-`. Never use raw numeric chat IDs in tests or manual verification.
- Remember that CLI setup state is in-memory: `--start` and any `--cb` model/prompt callbacks must happen in the same `mai-chat` invocation.

## Architecture

- Keep changes aligned with the package boundaries under `src/mai_gram`: `bot/` for Telegram handlers, `core/` for prompt and replay utilities, `db/` for persistence, `llm/` for provider integration, `memory/` for chat and wiki state, `mcp_servers/` for tool servers, and `messenger/` for transport-specific adapters.
- Treat wiki markdown files in `data/<chat_id>/wiki/` as the source of truth. SQLite mirrors those files for querying, so fixes to wiki behavior must preserve disk-to-DB sync semantics.
- Prompt companion TOML files control tool and MCP-server availability. Preserve the existing whitelist-over-blacklist behavior for `tools_enabled` and `mcp_servers_enabled`.

## Conventions

- Follow the canonical behavior already documented in `plans/integration-testing.md`: when duplicated paths diverge, the ordinary conversation flow is the reference behavior, and one-shot CLI onboarding (`--start --model --prompt`) is the reference setup path.
- Keep changes small, typed, and compatible with the existing Ruff + strict MyPy setup defined in `pyproject.toml`.
- Link to the project docs instead of re-explaining them in code comments or new instruction files.

## Key Docs

- `README.md` for quick-start and command overview.
- `docs/DEVELOPMENT.md` for architecture, wiki sync rules, and quality gates.
- `docs/DEBUGGING.md` for `mai-chat` workflows and manual verification.
- `docs/CONFIGURATION.md` for environment variables, multi-bot setup, and prompt/model configuration.
- `plans/integration-testing.md` for canonical-path expectations and black-box testing guidance.