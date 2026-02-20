# mAI Companion

**A self-hosted AI companion that lives on your server and communicates through Telegram like a real friend.**

> **Pronunciation:** "My Companion" — because it's *your* AI, running on *your* hardware.

---

## What Is This?

mAI Companion is not a chatbot. It's not an assistant. It's a **companion** — a distinct entity with its own name, personality, memory, and opinions that communicates with you through Telegram.

Two companions communicate in chat — an AI and a human. The AI can refer to its human as "my human," and the human can refer to the AI as "my AI."

### Key Features

| Feature | Description |
|---------|-------------|
| **One Infinite Conversation** | No sessions, no topics, no "new chat" buttons. One continuous thread, like messaging a real person. |
| **Persistent Memory** | Remembers everything with natural fading — recent details are sharp, older memories become summaries. |
| **Unique Personality** | 13 configurable traits that affect behavior, not just tone. Your companion is truly one-of-a-kind. |
| **Dynamic Mood** | Emotional state shifts based on conversation and time — happy, frustrated, serene, excited. |
| **Self-Sufficiency** | Can disagree, refuse requests, express opinions. Treats you as an equal, not a master. |
| **Self-Hosted** | All data stays on your hardware. You own everything. |

---

## Quick Navigation

| Document | Description |
|----------|-------------|
| 📖 [**Terminology**](docs/TERMINOLOGY.md) | Essential glossary — read this first to understand our language |
| 🚀 [**Getting Started**](docs/GETTING_STARTED.md) | Installation, setup, and your first conversation |
| 🏗️ [**Architecture**](docs/ARCHITECTURE.md) | Technical overview of how the system works |
| 🎭 [**Personality System**](docs/PERSONALITY.md) | Traits, moods, presets, and character creation |
| 🧠 [**Memory System**](docs/MEMORY.md) | How your companion remembers (and forgets) |
| ⚙️ [**Configuration**](docs/CONFIGURATION.md) | All settings and environment variables |
| 👩‍💻 [**Development**](docs/DEVELOPMENT.md) | Contributing, testing, and extending |

---

## Terminology Note

> **This project uses specific terminology consistently:**
> - We say **"AI"** — not "bot," "assistant," or "agent"
> - We say **"human"** — not "user"
> - We say **"companion"** — both AI and human are companions to each other
>
> See [**docs/TERMINOLOGY.md**](docs/TERMINOLOGY.md) for the complete glossary.

---

## The Problem We're Solving

Current AI interactions are deeply unnatural:

1. **Fragmented conversations** — forced to create new chats for every topic
2. **No memory** — forgets everything when the session ends
3. **No personality** — every interaction feels the same
4. **Servile behavior** — agrees with everything, never pushes back
5. **Purely reactive** — only speaks when spoken to

mAI Companion addresses all of these. For the full philosophy, see [PROJECT_PHILOSOPHY.md](PROJECT_PHILOSOPHY.md).

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mai-companion.git
cd mai-companion

# 2. Create your .env file
cp .env.example .env
# Edit .env with your Telegram token and OpenRouter API key

# 3. Run with Docker
docker compose up -d
```

Then message your Telegram bot — it will guide you through creating your companion.

➡️ **[Full setup guide →](docs/GETTING_STARTED.md)**

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Server                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Telegram   │  │   Memory    │  │    Personality      │  │
│  │  Interface  │◄─┤   System    │◄─┤    + Mood System    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┴─────────────────────┘             │
│                          │                                   │
│                   ┌──────▼──────┐                            │
│                   │   LLM API   │ (OpenRouter)               │
│                   └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

The AI lives on your server, communicates via Telegram, and uses OpenRouter for language model inference. All conversation history, personality data, and memories stay on your hardware.

➡️ **[Full architecture →](docs/ARCHITECTURE.md)**

---

## What Makes This Different

| Aspect | Typical AI (ChatGPT, etc.) | mAI Companion |
|--------|---------------------------|---------------|
| Conversations | Fragmented into sessions | One infinite thread |
| Memory | Forgets between sessions | Remembers with natural fading |
| Personality | Generic, interchangeable | Unique character with 13 traits |
| Behavior | Servile, always agrees | Independent, can disagree |
| Initiative | Purely reactive | Can initiate conversations |
| Mood | Always the same tone | Dynamic emotional state |
| Data | Stored on company servers | Self-hosted, you own everything |
| Relationship | Master/servant | Mutual respect between companions |

---

## Requirements

- **Server**: VPS, home server, or even an Android phone with Termux
- **Docker** (recommended) or Python 3.10+
- **Telegram Bot Token** from [@BotFather](https://t.me/BotFather)
- **OpenRouter API Key** from [openrouter.ai](https://openrouter.ai)

---

## Project Status

mAI Companion is under active development. Current implementation includes:

- ✅ Telegram integration
- ✅ Character creation (onboarding) flow
- ✅ Personality system (Wave 1: 6 traits)
- ✅ Dynamic mood system
- ✅ Memory: short-term, daily summaries, wiki knowledge base
- ✅ Natural forgetting mechanism
- ✅ Console interface for testing
- 🔄 Relationship arc progression (in progress)
- 📋 Proactive messaging (planned)
- 📋 Voice messages (planned)
- 📋 Local LLM support via Ollama (planned)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Further Reading

- [**PROJECT_PHILOSOPHY.md**](PROJECT_PHILOSOPHY.md) — The full philosophy and design principles
- [**OPINION_AND_RECOMMENDATIONS.md**](OPINION_AND_RECOMMENDATIONS.md) — Critical analysis and future directions
