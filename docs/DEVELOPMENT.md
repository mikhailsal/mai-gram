# Development

## Setup

```bash
pip install -e ".[dev]"
```

This also installs the git pre-commit hook automatically.

## Project Structure

```
src/mai_gram/
├── bot/          # Telegram message handling and setup flow
├── core/         # Prompt builder, markdown converter
├── db/           # SQLAlchemy models, migrations, database
├── debug/        # LLM call logger and cost tracker
├── llm/          # LLM provider abstraction (OpenRouter)
├── mcp_servers/  # MCP tool servers (wiki, messages, external)
├── memory/       # Message store, wiki store, summarizer (inactive)
├── messenger/    # Messenger abstraction (Telegram, console)
├── config.py     # Settings via pydantic-settings
├── main.py       # Application entry point
└── console_runner.py  # CLI interface
```

## Testing

```bash
pytest                     # Run all tests
pytest -x                  # Stop on first failure
pytest --cov=mai_gram      # With coverage
make test-cov              # Coverage with 90% enforcement
```

## Linting & Formatting

```bash
ruff check .               # Lint
ruff format .              # Format
mypy src/mai_gram          # Type check
make check                 # All three at once
make fix                   # Auto-fix lint + reformat
```

## Pre-commit Hook

A git pre-commit hook enforces quality gates before every commit:

1. **Lint** — `ruff check` passes with zero errors
2. **Format** — `ruff format --check` confirms consistent formatting
3. **Type check** — `mypy` passes with zero errors (strict mode)
4. **Tests + coverage** — all tests pass with ≥ 90% code coverage

The hook is installed automatically by `make install-dev`. To install manually:

```bash
make install-hooks
```

To skip the hook for a single commit (use sparingly):

```bash
git commit --no-verify
```

To run the same checks manually without committing:

```bash
make precommit
```

## Auto-Reload

```bash
python -m mai_gram.main --reload
```

Watches `src/` for Python file changes and auto-restarts.
