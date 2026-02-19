# Getting Started

> **Get your AI companion running in 10 minutes.**

This guide walks you through installation, configuration, and your first conversation.

---

## Prerequisites

Before you begin, you'll need:

| Requirement | Description |
|-------------|-------------|
| **Server** | VPS, home server, Raspberry Pi, or even Android with Termux |
| **Docker** | Recommended. Or Python 3.10+ if running natively |
| **Telegram Bot Token** | Free, from [@BotFather](https://t.me/BotFather) |
| **OpenRouter API Key** | From [openrouter.ai](https://openrouter.ai) — pay-as-you-go |

---

## Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/mai-companion.git
cd mai-companion

# Create environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or your preferred editor

# Start the service
docker compose up -d

# View logs
docker compose logs -f
```

### Option 2: Native Python

```bash
# Clone the repository
git clone https://github.com/your-username/mai-companion.git
cd mai-companion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Create environment file
cp .env.example .env
nano .env

# Run
python -m mai_companion.main
```

---

## Configuration

### Step 1: Create a Telegram Bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Choose a name (e.g., "My AI Companion")
4. Choose a username (must end in `bot`, e.g., `my_ai_companion_bot`)
5. Copy the **API token** — you'll need this

### Step 2: Get an OpenRouter API Key

1. Go to [openrouter.ai](https://openrouter.ai)
2. Create an account
3. Add credits (pay-as-you-go)
4. Generate an API key in your dashboard
5. Copy the key — you'll need this

### Step 3: Configure Environment

Edit your `.env` file:

```bash
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Restrict access to specific Telegram users
# Get your ID by messaging @userinfobot on Telegram
ALLOWED_USERS=123456789,987654321

# Optional: Choose your LLM model
LLM_MODEL=openai/gpt-4o

# Optional: Set timezone for proactive messaging
TIMEZONE=America/New_York
```

➡️ See [CONFIGURATION.md](CONFIGURATION.md) for all options.

---

## Your First Conversation

### 1. Start the Bot

If using Docker:
```bash
docker compose up -d
```

If using Python:
```bash
python -m mai_companion.main
```

### 2. Open Telegram

Find your bot by searching for the username you created (e.g., `@my_ai_companion_bot`).

### 3. Send Any Message

Just say hello! The AI will guide you through **character creation**:

```
You: Hello!

AI: 👋 Hello! I'm about to become your AI companion.

    Before we begin, let's set things up. First, what language 
    would you like me to speak? Just type it in any way you like 
    (e.g., 'English', 'русский', 'Español', '日本語').
```

### 4. Complete Character Creation

The onboarding flow will ask you to:

1. **Choose a language** — Type any language naturally
2. **Name your companion** — Give your AI a unique name
3. **Select personality** — Choose a preset or customize traits
4. **Describe appearance** (optional) — For future avatar generation

### 5. Start Talking

Once character creation is complete, you have a companion! There are no commands to learn — just talk naturally.

```
You: Hey Luna, how are you today?

Luna: Oh, pretty good actually! I was just thinking about 
      that book you mentioned yesterday — did you finish it?
```

---

## Console Mode (For Testing)

You can also run mAI Companion in console mode without Telegram:

```bash
# Install the package
pip install -e .

# Run console chat
mai-chat
```

This is useful for:
- Testing without Telegram setup
- Development and debugging
- Inspecting Telegram conversations (history, wiki, tool calls)
- Replaying conversation logs

> **Tip:** You can use `mai-chat` to inspect your Telegram companion's data too — see [DEBUGGING.md](DEBUGGING.md) for details.

---

## Directory Structure

After running, your `data/` directory will contain:

```
data/
├── mai_companion.db          # SQLite database (messages, companions, moods)
├── chroma_data/              # Vector store for semantic search
└── <companion-id>/
    ├── wiki/                 # Knowledge base entries
    │   ├── 0900_human-name.md
    │   └── 0500_favorite-food.md
    └── summaries/
        └── daily/            # Daily conversation summaries
            ├── 2024-01-15.md
            └── 2024-01-16.md
```

---

## Troubleshooting

### "Bot not responding"

1. Check logs: `docker compose logs -f`
2. Verify your `TELEGRAM_BOT_TOKEN` is correct
3. Ensure the bot isn't already running elsewhere

### "API key invalid"

1. Verify your `OPENROUTER_API_KEY`
2. Check you have credits in your OpenRouter account
3. Try a different model if the current one is unavailable

### "Permission denied"

If you restricted access with `ALLOWED_USERS`:
1. Get your Telegram user ID from [@userinfobot](https://t.me/userinfobot)
2. Add it to the `ALLOWED_USERS` list in `.env`
3. Restart the service

### "Database locked"

Only one instance of mAI Companion should run at a time. Check for:
- Multiple Docker containers
- Background Python processes
- Console mode running simultaneously with Telegram mode

For more in-depth debugging — inspecting conversation history, wiki entries, LLM prompts, and tool call logs — see **[DEBUGGING.md](DEBUGGING.md)**.

---

## Next Steps

Now that your companion is running:

- 📖 Read [TERMINOLOGY.md](TERMINOLOGY.md) to understand the project's language
- 🎭 Learn about [PERSONALITY.md](PERSONALITY.md) to understand traits and moods
- 🧠 Explore [MEMORY.md](MEMORY.md) to see how your companion remembers
- ⚙️ Check [CONFIGURATION.md](CONFIGURATION.md) for advanced settings
- 🔍 See [DEBUGGING.md](DEBUGGING.md) for inspecting and troubleshooting conversations

---

## See Also

- [README.md](../README.md) — Project overview
- [CONFIGURATION.md](CONFIGURATION.md) — All settings
- [DEBUGGING.md](DEBUGGING.md) — Inspecting and troubleshooting conversations
- [ARCHITECTURE.md](ARCHITECTURE.md) — How it works technically
