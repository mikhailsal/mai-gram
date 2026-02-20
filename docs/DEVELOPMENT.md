# Development Guide

> **Contributing to mAI Companion — setup, testing, and guidelines.**

This guide is for developers who want to contribute to mAI Companion or extend it for their own needs.

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/mai-companion.git
cd mai-companion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Development Dependencies

The `[dev]` extra includes:

| Package | Purpose |
|---------|---------|
| `pytest` | Test framework |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reporting |
| `ruff` | Linting and formatting |
| `mypy` | Type checking |

---

## Project Structure

```
mai-companion/
├── src/mai_companion/          # Main source code
│   ├── bot/                    # Telegram bot handling
│   ├── core/                   # Core engine (prompt building)
│   ├── db/                     # Database models and migrations
│   ├── llm/                    # LLM provider abstraction
│   ├── memory/                 # Memory system
│   ├── messenger/              # Messenger abstraction
│   └── personality/            # Personality and mood
│
├── tests/                      # Test suite
│   ├── functional/             # End-to-end functional tests
│   ├── test_bot/               # Bot unit tests
│   ├── test_mcp/               # MCP server & bridge tests
│   ├── test_memory/            # Memory unit tests
│   └── ...
│
├── docs/                       # Documentation
├── data/                       # Runtime data (gitignored)
└── pyproject.toml              # Project configuration
```

---

## Running the Application

### Telegram Mode

```bash
# With environment variables
TELEGRAM_BOT_TOKEN=xxx OPENROUTER_API_KEY=xxx python -m mai_companion.main

# Or with .env file
python -m mai_companion.main
```

### Console Mode

For testing without Telegram:

```bash
mai-chat
```

This provides an interactive console interface.

### Console Replay Mode

Replay conversation logs for testing:

```bash
mai-chat --replay path/to/messages.jsonl
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mai_companion --cov-report=html

# Run specific test file
pytest tests/test_memory/test_manager.py

# Run specific test
pytest tests/test_memory/test_manager.py::test_save_message

# Run with verbose output
pytest -v
```

### Test Categories

| Directory | Type | Description |
|-----------|------|-------------|
| `tests/test_*` | Unit | Fast, isolated component tests |
| `tests/functional/` | Functional | End-to-end scenario tests |

### Functional Tests

Functional tests simulate real conversations:

```bash
# Run functional tests
pytest tests/functional/

# Run specific functional test
pytest tests/functional/test_onboarding.py
```

Functional tests use:
- Mock LLM responses
- Simulated time (Clock abstraction)
- Isolated test databases

### Writing Tests

```python
import pytest
from mai_companion.memory.manager import MemoryManager

@pytest.fixture
async def memory_manager(session):
    # Setup fixture
    return MemoryManager(...)

async def test_save_message(memory_manager):
    # Arrange
    companion_id = "test-companion"
    
    # Act
    message = await memory_manager.save_message(
        companion_id, "user", "Hello!"
    )
    
    # Assert
    assert message.content == "Hello!"
    assert message.role == "user"
```

---

## Code Quality

### Linting

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Formatting

```bash
# Check formatting
ruff format --check .

# Apply formatting
ruff format .
```

### Type Checking

```bash
# Run mypy
mypy src/mai_companion
```

### Pre-commit Checks

Before committing, run:

```bash
ruff check .
ruff format .
mypy src/mai_companion
pytest
```

---

## Code Style

### General Guidelines

- Follow PEP 8
- Use type hints everywhere
- Write docstrings for public functions
- Keep functions focused and small
- Prefer explicit over implicit

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `MemoryManager` |
| Functions | snake_case | `save_message` |
| Constants | UPPER_SNAKE | `DEFAULT_LIMIT` |
| Private | _prefix | `_internal_method` |

### Docstrings

Use Google-style docstrings:

```python
async def save_message(
    self,
    companion_id: str,
    role: str,
    content: str,
) -> Message:
    """Save a message to the database.
    
    Parameters
    ----------
    companion_id:
        The companion's unique identifier.
    role:
        Message role ('user' or 'assistant').
    content:
        The message text content.
    
    Returns
    -------
    Message
        The saved message object.
    
    Raises
    ------
    ValueError
        If role is invalid.
    """
```

---

## Architecture Patterns

### Dependency Injection

Components receive dependencies through constructors:

```python
class MemoryManager:
    def __init__(
        self,
        message_store: MessageStore,
        summary_store: SummaryStore,
        wiki_store: WikiStore,
    ) -> None:
        self._message_store = message_store
        self._summary_store = summary_store
        self._wiki_store = wiki_store
```

### Protocol-Based Abstractions

Use protocols for interfaces:

```python
from typing import Protocol

class Messenger(Protocol):
    async def send_message(self, message: OutgoingMessage) -> SendResult: ...
    async def edit_message(self, chat_id: str, msg_id: str, text: str) -> bool: ...
```

### Async Throughout

All I/O operations are async:

```python
async def get_messages(self, companion_id: str) -> list[Message]:
    result = await self._session.execute(
        select(Message).where(Message.companion_id == companion_id)
    )
    return list(result.scalars().all())
```

---

## Adding New Features

### Adding a New Trait

1. Add to `TraitName` enum in `personality/traits.py`:

```python
class TraitName(str, enum.Enum):
    # ... existing traits
    NEW_TRAIT = "new_trait"
```

2. Add definition to `TRAIT_DEFINITIONS`:

```python
TRAIT_DEFINITIONS[TraitName.NEW_TRAIT] = TraitDefinition(
    name=TraitName.NEW_TRAIT,
    display_name="New Trait",
    description="What this trait controls",
    low_label="Low end behavior",
    high_label="High end behavior",
)
```

3. Add behavioral instructions:

```python
TRAIT_BEHAVIORAL_INSTRUCTIONS[(TraitName.NEW_TRAIT, TraitLevel.VERY_LOW)] = "..."
# ... for all levels
```

4. Update presets if needed
5. Add tests

### Adding a New Messenger

1. Implement the `Messenger` protocol in `messenger/`:

```python
class NewMessenger:
    async def send_message(self, message: OutgoingMessage) -> SendResult:
        # Implementation
        ...
    
    async def edit_message(self, chat_id: str, msg_id: str, text: str) -> bool:
        # Implementation
        ...
```

2. Add configuration options
3. Update the main entry point
4. Add tests

---

## Database Migrations

### Adding a New Model

1. Define the model in `db/models.py`:

```python
class NewModel(Base):
    __tablename__ = "new_models"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    companion_id: Mapped[str] = mapped_column(ForeignKey("companions.id"))
    # ... fields
```

2. Add migration in `db/migrations.py` if needed
3. Update tests

### Schema Changes

For existing tables, add migration logic:

```python
async def migrate_v2(session: AsyncSession) -> None:
    """Add new_column to existing table."""
    await session.execute(
        text("ALTER TABLE messages ADD COLUMN new_column TEXT")
    )
```

---

## Terminology Compliance

When writing code and documentation, follow the project terminology:

| Use | Don't Use |
|-----|-----------|
| AI | bot, assistant, agent |
| human | user, owner, client |
| companion | — |

See [TERMINOLOGY.md](TERMINOLOGY.md) for the complete glossary.

---

## Debugging

For a comprehensive guide on inspecting conversations, viewing wiki entries, replaying tool calls, and troubleshooting — see **[DEBUGGING.md](DEBUGGING.md)**.

### Quick Reference

```bash
# Enable verbose application logging
DEBUG=true python -m mai_companion.main

# Inspect Telegram conversation history
mai-chat -c <chat-id> --history

# Check wiki entries
mai-chat -c <chat-id> --wiki

# Replay conversation with tool events
mai-chat -c <chat-id> --replay --date 2026-02-19

# Inspect the full LLM prompt
mai-chat -c <chat-id> --show-prompt

# Send a message with debug logging
mai-chat -c <chat-id> --debug "Hello"
```

### Logging in Code

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed info for debugging")
logger.info("General information")
logger.warning("Something unexpected")
logger.error("Something went wrong")
```

---

## Performance Considerations

### Token Budgeting

Be mindful of context window limits:
- Track token usage in prompt building
- Truncate oldest content first
- Log warnings when approaching limits

### Database Queries

- Use async queries
- Add indexes for frequent queries
- Batch operations when possible

### Memory Usage

- Stream large responses
- Clean up temporary data
- Use generators for large datasets

---

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG (if exists)
3. Run full test suite
4. Create git tag
5. Build and push Docker image

---

## Getting Help

- Check existing issues on GitHub
- Read the documentation thoroughly
- Ask questions in discussions

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical overview
- [TERMINOLOGY.md](TERMINOLOGY.md) — Project terminology
- [CONFIGURATION.md](CONFIGURATION.md) — All settings
