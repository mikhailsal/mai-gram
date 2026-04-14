# Development

## Setup

```bash
pip install -e ".[dev]"
```

## Project Structure

```
src/mai_gram/
├── bot/          # Telegram message handling and setup flow
├── core/         # Prompt builder
├── db/           # SQLAlchemy models, migrations, database
├── debug/        # LLM call logger and cost tracker
├── llm/          # LLM provider abstraction (OpenRouter)
├── mcp_servers/  # MCP tool servers (wiki, messages)
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
```

## Linting & Formatting

```bash
ruff check .               # Lint
ruff format .              # Format
mypy src/mai_gram          # Type check
```

## Auto-Reload

```bash
python -m mai_gram.main --reload
```

Watches `src/` for Python file changes and auto-restarts.
