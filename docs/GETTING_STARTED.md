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

> **Want multiple companions?** You can create up to 3 Telegram bots, each acting as a separate "window" for a different companion. The same human can have a different AI personality in each bot. See [Multi-Bot Setup](#multi-bot-setup-optional) below.

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

# Optional: Additional bots for multiple companions (see Multi-Bot Setup)
TELEGRAM_BOT_TOKEN_2=
TELEGRAM_BOT_TOKEN_3=

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

## Running the Application

Once installation and configuration are complete, you can run mAI Companion in several ways.

### Option 1: Docker (Recommended for Production)

```bash
# Start in background (detached mode)
docker compose up -d

# View logs
docker compose logs -f

# Stop the service
docker compose down

# Restart the service
docker compose restart
```

### Option 2: Native Python

```bash
# Run in foreground (logs visible, Ctrl+C to stop)
python -m mai_companion.main

# Run with auto-reload (recommended for development)
python -m mai_companion.main --reload

# Run in background (Linux/macOS)
python -m mai_companion.main &

# Run in background with nohup (persists after terminal closes)
nohup python -m mai_companion.main > mai_companion.log 2>&1 &

# Check if running
ps aux | grep mai_companion

# Stop background process
pkill -f "python -m mai_companion.main"
```

### Development Mode (Auto-Reload)

When developing, use `--reload` to automatically restart the bot when you change Python files:

```bash
python -m mai_companion.main --reload
```

This watches the `src/` directory for `.py` file changes. When a change is detected, the bot gracefully shuts down and restarts with the new code. You'll see output like:

```
🔄 Auto-reload enabled — watching /path/to/src for changes
   The bot will restart automatically when you save a .py file.

2024-01-15 10:30:00 - INFO - mAI Companion starting up...
...
🔄 Detected changes in: src/mai_companion/bot/handler.py
   Restarting bot process...
```

> **Note:** `--reload` is for development only. For production, run without `--reload` or use systemd (see below).

### Startup Output

When the application starts successfully, you'll see:

```
2024-01-15 10:30:00 - INFO - mAI Companion starting up...
2024-01-15 10:30:00 - INFO - Database ready
2024-01-15 10:30:00 - INFO - LLM provider initialized (model: openai/gpt-4o)
2024-01-15 10:30:00 - INFO - Access control enabled: 1 user(s) allowed
2024-01-15 10:30:01 - INFO - Bot 1/1 started: @my_ai_companion_bot
2024-01-15 10:30:01 - INFO - mAI Companion is running with 1 bot(s)! Press Ctrl+C to stop.
```

With multiple bots configured, you'll see each one start:

```
2024-01-15 10:30:01 - INFO - Bot 1/3 started: @my_ai_companion_bot
2024-01-15 10:30:01 - INFO - Bot 2/3 started: @my_second_bot
2024-01-15 10:30:01 - INFO - Bot 3/3 started: @my_third_bot
2024-01-15 10:30:01 - INFO - mAI Companion is running with 3 bot(s)! Press Ctrl+C to stop.
```

### Running as a System Service (Linux)

For production deployments, you can create a systemd service:

```bash
# Create service file
sudo nano /etc/systemd/system/mai-companion.service
```

```ini
[Unit]
Description=mAI Companion Telegram Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/mai-companion
Environment="PATH=/path/to/mai-companion/venv/bin"
ExecStart=/path/to/mai-companion/venv/bin/python -m mai_companion.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable mai-companion
sudo systemctl start mai-companion

# Check status
sudo systemctl status mai-companion

# View logs
sudo journalctl -u mai-companion -f

# Stop the service
sudo systemctl stop mai-companion
```

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

# Run console chat (test mode by default)
mai-chat -c my-test-companion --start

# For real conversations via CLI (not testing)
mai-chat -c my-companion --real "Hello!"
```

This is useful for:
- Testing without Telegram setup
- Development and debugging
- Inspecting Telegram conversations (history, wiki, tool calls)
- Replaying conversation logs

> **Important:** By default, `mai-chat` runs in **test mode** — the AI is informed this is a test/development scenario, not a real conversation. This aligns with our philosophy of transparency. Use `--real` when you genuinely want to have a conversation via the CLI.

> **Tip:** You can use `mai-chat` to inspect your Telegram companion's data too — see [DEBUGGING.md](DEBUGGING.md) for details.

---

## Multi-Bot Setup (Optional)

mAI Companion supports running **multiple Telegram bots simultaneously**. Each bot acts as a separate "window" for a different companion. The same human can create a unique AI personality in each bot — different name, traits, language, and memories.

### Why Multiple Bots?

Having multiple companions in a single chat would create confusion — mixed histories, overlapping contexts, and an unnatural experience. Separate bots keep each companion's conversation clean and independent, just like messaging different friends.

### How It Works

1. Create additional bots via [@BotFather](https://t.me/BotFather) (up to 3 total)
2. Add their tokens to your `.env`:

```bash
TELEGRAM_BOT_TOKEN=your_primary_bot_token
TELEGRAM_BOT_TOKEN_2=your_second_bot_token
TELEGRAM_BOT_TOKEN_3=your_third_bot_token
```

3. Restart the application — all bots start automatically

Each companion is identified by a composite ID: `user_id@bot_username` (e.g., `186215217@my_ai_companion_bot`). This ensures that the same human's companions in different bots are fully independent — separate personality, memory, wiki, and conversation history.

### Example

```
Bot 1 (@my_companion_bot)     → "Luna" — warm, creative, speaks English
Bot 2 (@my_second_bot)        → "Кай" — direct, analytical, speaks Russian
Bot 3 (@my_third_bot)         → "Hana" — gentle, curious, speaks Japanese
```

All three run from a single mAI Companion instance on your server.

---

## Directory Structure

After running, your `data/` directory will contain:

```
data/
├── mai_companion.db                    # SQLite database (messages, companions, moods)
├── chroma_data/                        # Vector store for semantic search
└── <user_id>@<bot_username>/           # One directory per companion
    ├── wiki/                           # Knowledge base entries
    │   ├── 0900_human-name.md
    │   └── 0500_favorite-food.md
    └── summaries/
        ├── daily/                      # Daily conversation summaries
        │   ├── 2024-01-15.md
        │   ├── 2024-01-16.md
        │   └── .versions/              # Previous versions (from re-consolidation)
        ├── weekly/                     # Weekly rollups
        └── monthly/                    # Monthly rollups
```

> **Note:** Companion directories use the format `<user_id>@<bot_username>` (e.g., `186215217@my_ai_companion_bot`). For companions created via the console runner (without a Telegram bot), the directory is simply the `chat_id`.

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
