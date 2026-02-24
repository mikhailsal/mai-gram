# Architecture

> **Technical overview of how mAI Companion works.**

This document explains the system's components, data flow, and design decisions.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Your Server                                  │
│                                                                           │
│  ┌────────────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Messenger Layer      │  │    Core      │  │    Personality       │  │
│  │  ┌──────────────────┐  │  │   Engine     │  │    System            │  │
│  │  │ @bot_1 (Telegram)│◄─┼─►│              │◄►│  (Traits + Mood)    │  │
│  │  ├──────────────────┤  │  │  BotHandler  │  └──────────────────────┘  │
│  │  │ @bot_2 (Telegram)│◄─┤  │  (per bot)   │                            │
│  │  ├──────────────────┤  │  └──────┬───────┘                            │
│  │  │ @bot_3 (Telegram)│◄─┘        │                                     │
│  │  └──────────────────┘           ▼                                     │
│  └────────────────────────┘  ┌──────────────┐  ┌──────────────────────┐  │
│                              │   Prompt     │  │    LLM Provider      │  │
│  ┌──────────────┐            │   Builder    │◄►│   (OpenRouter)       │  │
│  │   Memory     │◄──────────►│              │  │   (shared)           │  │
│  │   System     │            └──────────────┘  └──────────────────────┘  │
│  └──────────────┘                                                        │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                        Storage Layer                              │    │
│  │   SQLite (messages, companions, moods)  │  Files (wiki, summaries)│    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

The system supports **multiple Telegram bots** running simultaneously. Each bot has its own `TelegramMessenger` and `BotHandler` instance, while the LLM provider and database are shared. This allows a single human to have independent companions in different bots.

---

## Component Overview

| Component | Location | Purpose |
|-----------|----------|---------|
| **Messenger** | `src/mai_companion/messenger/` | Telegram integration, message handling |
| **Bot** | `src/mai_companion/bot/` | Request handling, onboarding flow |
| **Core** | `src/mai_companion/core/` | Prompt building, conversation engine |
| **Personality** | `src/mai_companion/personality/` | Traits, mood, character config |
| **Memory** | `src/mai_companion/memory/` | Messages, summaries, wiki, forgetting |
| **MCP Servers** | `src/mai_companion/mcp_servers/` | Tool servers (wiki, messages, sleep), bridge, manager |
| **LLM** | `src/mai_companion/llm/` | OpenRouter client, translation |
| **Database** | `src/mai_companion/db/` | SQLAlchemy models, migrations |

---

## Messenger Layer

The messenger layer abstracts communication channels. Currently supports:

- **Telegram** (`telegram.py`) — Primary interface via python-telegram-bot (supports multiple bot instances)
- **Console** (`console.py`) — For testing and development

### Multi-Bot Architecture

Each Telegram bot token creates a separate `TelegramMessenger` + `BotHandler` pair:

```
                    ┌─ TelegramMessenger(@bot_1) ─ BotHandler ─┐
                    │                                            │
Human ──Telegram──►├─ TelegramMessenger(@bot_2) ─ BotHandler ─┼──► Shared LLM
                    │                                            │
                    └─ TelegramMessenger(@bot_3) ─ BotHandler ─┘
```

Each messenger resolves its `bot_id` (username) from the Telegram API on startup. This `bot_id` is injected into every `IncomingMessage`, enabling the handler to construct a **composite companion ID**: `user_id@bot_username`.

### Message Flow

```
Human Message → Telegram API → Messenger(@bot_id) → Handler → Core Engine
                                                                    ↓
Human ← Telegram API ← Messenger ← Handler ← AI Response
```

### Abstraction

```python
class Messenger(Protocol):
    async def send_message(self, message: OutgoingMessage) -> SendResult: ...
    async def edit_message(self, chat_id: str, message_id: str, text: str) -> bool: ...
```

The `IncomingMessage` dataclass includes a `bot_id` field that identifies which bot received the message:

```python
@dataclass(frozen=True, slots=True)
class IncomingMessage:
    platform: str
    chat_id: str
    user_id: str
    bot_id: str = ""   # Bot username (e.g., "my_companion_bot")
    message_id: str
    # ...
```

This allows future support for WhatsApp, Signal, or other platforms.

---

## Personality System

The personality system gives each AI a unique character.

### Traits

13 traits organized in three implementation waves:

```
Wave 1 (Current):
├── Warmth         (0.0-1.0)  How caring and affectionate
├── Humor          (0.0-1.0)  How playful and witty
├── Patience       (0.0-1.0)  How thorough and unhurried
├── Directness     (0.0-1.0)  How blunt and frank
├── Laziness       (0.0-1.0)  How much effort is avoided
└── Mood Volatility(0.0-1.0)  How dramatically mood shifts

Wave 2 (Planned):
├── Assertiveness
├── Curiosity
├── Emotional Depth
├── Independence
└── Helpfulness

Wave 3 (Planned):
├── Proactiveness
└── Special Speech
```

### Trait → Behavior Mapping

Each trait level maps to specific behavioral instructions injected into the system prompt:

```python
TRAIT_BEHAVIORAL_INSTRUCTIONS = {
    (TraitName.WARMTH, TraitLevel.VERY_HIGH): (
        "You are deeply nurturing and affectionate. You radiate warmth in "
        "every interaction..."
    ),
    # ... more mappings
}
```

### Mood System

Two-axis emotional model:

```
                    Energetic (+1.0)
                         │
         Irritated       │       Excited
                         │
    Negative ───────────┼─────────── Positive
    (-1.0)               │            (+1.0)
                         │
         Melancholic     │       Serene
                         │
                    Calm (-1.0)
```

Mood shifts through:
- **Reactive shifts** — Response to conversation sentiment
- **Spontaneous shifts** — Random drift based on volatility trait
- **Decay** — Gradual return to baseline over time

➡️ See [PERSONALITY.md](PERSONALITY.md) for complete details.

---

## Memory System

Memory is organized in layers that mirror human cognition.

### Memory Layers

```
┌─────────────────────────────────────────────────────────┐
│                    Context Window                        │
│  ┌─────────────────────────────────────────────────────┐│
│  │  System Prompt (personality + mood + relationship)  ││
│  ├─────────────────────────────────────────────────────┤│
│  │  Wiki Entries (top 20 by importance)                ││
│  ├─────────────────────────────────────────────────────┤│
│  │  Memory Summaries (monthly → weekly → daily)        ││
│  ├─────────────────────────────────────────────────────┤│
│  │  Short-Term Messages (last ~30 messages)            ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Short-Term Memory

Recent messages stored in SQLite with full content:

```python
class Message(Base):
    id: int
    companion_id: str
    role: str           # "user" or "assistant"
    content: str
    timestamp: datetime
    is_proactive: bool
```

### Daily Summaries

When a day has enough messages (default: 20+), they're summarized:

```
data/<companion-id>/summaries/daily/2024-01-15.md
```

Summary content:
```markdown
Discussed work stress and upcoming vacation plans. Human mentioned 
feeling overwhelmed with the project deadline. Shared excitement 
about the trip to Japan next month.
```

### Wiki Knowledge Base

Structured facts with importance scores:

```
data/<companion-id>/wiki/
├── 0900_human-name.md        # Importance: 900
├── 0850_human-birthday.md    # Importance: 850
├── 0500_favorite-food.md     # Importance: 500
└── 0300_mentioned-coworker.md # Importance: 300
```

File naming: `{importance:04d}_{sanitized-key}.md`

### Forgetting Engine

Natural memory decay:
- Low-importance entries lose importance over time
- Entries reaching importance ≤ 0 are deleted
- High-importance entries (like name) are protected

➡️ See [MEMORY.md](MEMORY.md) for complete details.

---

## Prompt Builder

The prompt builder assembles context for each LLM request.

### Context Assembly

```python
async def build_context(companion, mood, clock) -> list[ChatMessage]:
    # 1. Build system prompt
    system_prompt = (
        base_personality_prompt
        + mood_section
        + relationship_section
        + current_time
        + wiki_entries
        + memory_summaries
    )
    
    # 2. Add conversation history
    messages = await get_short_term_messages(limit=30)
    
    # 3. Token budgeting
    while token_count > max_tokens:
        truncate_oldest_summaries()
    
    return [system_message, *conversation_messages]
```

### Token Budgeting

Context is capped at ~120K tokens. When exceeded:
1. Oldest monthly summaries removed first
2. Then weekly summaries
3. Then daily summaries
4. Short-term messages are preserved

---

## MCP Tool System

The companion has access to tools via an MCP (Model Context Protocol) compatible
server architecture. Each tool server exposes a `list_tools` / `call_tool`
interface and is registered with the central `MCPManager`.

### Available Tool Servers

| Server | Tools | Purpose |
|--------|-------|---------|
| **messages** | `search_messages` | Full-text search over conversation history |
| **wiki** | `wiki_create`, `wiki_edit`, `wiki_read`, `wiki_search` | Persistent knowledge base CRUD |
| **sleep** | `sleep` | Pause between messages for multi-message delivery |

### Multi-Message Delivery (Sleep Tool)

The `sleep` tool enables a natural texting experience. When the companion
wants to send several short messages instead of one long one, it:

1. Writes the first part of its reply as assistant content
2. Calls the `sleep` tool (with an optional duration, default 1s, max 5s)
3. The bridge's `on_intermediate_content` callback fires, sending the text
   immediately as a separate message
4. After the pause, the LLM continues and can produce the next message

This creates the illusion of a real person typing multiple messages.

### Agentic Loop

The `run_with_tools` bridge function implements a loop:

```
LLM generates → has tool calls? → execute tools → feed results back → repeat
                  ↓ no
              return final response
```

When the LLM produces text alongside tool calls, the text is delivered to
the human via the `on_intermediate_content` callback before the tools run.

---

## LLM Provider

Abstracted interface for language model access.

### OpenRouter Integration

```python
class OpenRouterProvider(LLMProvider):
    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse: ...
    
    async def count_tokens(self, messages: list[ChatMessage]) -> int: ...
```

### Temperature Derivation

Temperature is computed from personality traits:

```python
def compute_temperature(traits: dict[str, float]) -> float:
    # Higher warmth, humor → higher temperature (more creative)
    # Higher directness, patience → lower temperature (more consistent)
    base = 0.7
    warmth_factor = (traits.get("warmth", 0.5) - 0.5) * 0.2
    humor_factor = (traits.get("humor", 0.5) - 0.5) * 0.15
    directness_factor = -(traits.get("directness", 0.5) - 0.5) * 0.1
    return clamp(base + warmth_factor + humor_factor + directness_factor, 0.3, 1.0)
```

---

## Database Schema

SQLite with SQLAlchemy async.

### Companion Identification

Companions are identified by a **composite ID** in the format `user_id@bot_username` (e.g., `186215217@my_companion_bot`). This ensures that the same human can have independent companions in different bots, each with its own personality, memory, and conversation history.

For companions created via the console runner (without a Telegram bot), the ID is simply the `chat_id`.

### Core Tables

```sql
-- Companions (AI entities)
-- id format: "user_id@bot_username" (e.g., "186215217@my_companion_bot")
CREATE TABLE companions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    language TEXT DEFAULT 'English',
    system_prompt TEXT,
    traits JSON,
    relationship_stage TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Messages (conversation history)
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    companion_id TEXT REFERENCES companions(id),
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP,
    is_proactive BOOLEAN DEFAULT FALSE
);

-- Mood States (emotional history)
CREATE TABLE mood_states (
    id INTEGER PRIMARY KEY,
    companion_id TEXT REFERENCES companions(id),
    valence REAL,      -- -1.0 to 1.0
    arousal REAL,      -- -1.0 to 1.0
    label TEXT,        -- 'excited', 'melancholic', etc.
    cause TEXT,
    timestamp TIMESTAMP
);

-- Knowledge Entries (wiki)
CREATE TABLE knowledge_entries (
    id INTEGER PRIMARY KEY,
    companion_id TEXT REFERENCES companions(id),
    category TEXT DEFAULT 'wiki',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    importance REAL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## Data Flow Example

### Human sends "I'm feeling stressed about work"

```
1. Telegram receives message
   └─► Messenger extracts text, user_id, chat_id, bot_id
   └─► Companion ID derived: user_id@bot_id

2. Handler processes message
   └─► Saves to message store
   └─► Retrieves companion config (by composite ID)
   └─► Gets current mood

3. Mood system analyzes sentiment
   └─► Detects negative sentiment
   └─► Shifts mood toward concern

4. Prompt builder assembles context
   └─► System prompt (personality + new mood)
   └─► Wiki entries (knows human's job, etc.)
   └─► Recent summaries
   └─► Last 30 messages

5. LLM generates response
   └─► Temperature from personality
   └─► Response reflects mood shift

6. Response saved and sent
   └─► Message stored in DB
   └─► Sent via Telegram
   └─► Summary triggered if threshold met
```

---

## File Structure

```
mai-companion/
├── src/mai_companion/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── config.py            # Settings (pydantic-settings)
│   ├── clock.py             # Time abstraction (for testing)
│   ├── console_runner.py    # CLI interface
│   │
│   ├── bot/
│   │   ├── handler.py       # Message handling
│   │   ├── middleware.py    # Access control
│   │   └── onboarding.py    # Character creation flow
│   │
│   ├── core/
│   │   └── prompt_builder.py # Context assembly
│   │
│   ├── personality/
│   │   ├── character.py     # CharacterConfig, CharacterBuilder
│   │   ├── traits.py        # Trait definitions, presets
│   │   ├── mood.py          # Mood system
│   │   └── temperature.py   # LLM temperature derivation
│   │
│   ├── memory/
│   │   ├── manager.py       # Memory orchestration
│   │   ├── messages.py      # Message store
│   │   ├── summaries.py     # Summary store
│   │   ├── summarizer.py    # Summary generation
│   │   ├── knowledge_base.py # Wiki store
│   │   └── forgetting.py    # Memory decay
│   │
│   ├── llm/
│   │   ├── provider.py      # LLM abstraction
│   │   ├── openrouter.py    # OpenRouter client
│   │   └── translation.py   # Multi-language support
│   │
│   ├── mcp_servers/
│   │   ├── bridge.py        # MCP ↔ OpenAI bridge, agentic loop
│   │   ├── manager.py       # Server registry and tool routing
│   │   ├── messages_server.py # Message search tool
│   │   ├── wiki_server.py   # Wiki CRUD tools
│   │   └── sleep_server.py  # Sleep tool (multi-message delivery)
│   │
│   ├── messenger/
│   │   ├── base.py          # Messenger protocol
│   │   ├── telegram.py      # Telegram implementation
│   │   └── console.py       # Console implementation
│   │
│   └── db/
│       ├── database.py      # Session management
│       ├── models.py        # SQLAlchemy models
│       └── migrations.py    # Schema migrations
│
├── tests/                   # Test suite
├── data/                    # Runtime data (gitignored)
├── docs/                    # Documentation
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Design Decisions

### Why SQLite?

- Simple deployment (single file)
- No separate database server needed
- Sufficient for single-human use
- Easy backups (copy one file)

### Why File-Based Wiki?

- Human-readable and editable
- Easy to backup and version
- Importance visible in filename
- Can be browsed without tools

### Why OpenRouter?

- Access to multiple models (GPT-4, Claude, Llama)
- Single API for model switching
- Pay-as-you-go pricing
- Human controls their own API key

### Why Multiple Bots Instead of One?

- One companion per chat keeps conversation history clean
- No confusion between different AI personalities in the same thread
- Each bot has its own context, memory, and personality
- Feels like messaging different friends, not switching modes
- Composite ID (`user_id@bot_username`) ensures full isolation

### Why Async?

- Telegram API is I/O bound
- LLM calls have high latency
- Database operations can block
- Multiple bots run concurrently in the same event loop
- Async enables responsive handling

---

## See Also

- [MEMORY.md](MEMORY.md) — Deep dive into memory system
- [PERSONALITY.md](PERSONALITY.md) — Deep dive into personality system
- [CONFIGURATION.md](CONFIGURATION.md) — All settings
- [DEVELOPMENT.md](DEVELOPMENT.md) — Contributing guide
