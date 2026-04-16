# Getting Started

## Prerequisites

- Python 3.10+
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- An OpenRouter API key (from [openrouter.ai](https://openrouter.ai))

## Installation

```bash
git clone https://github.com/mikhailsal/mai-gram.git
cd mai-gram
pip install -e ".[dev]"
```

## Configuration

1. Copy `.env.example` to `.env`
2. Set `TELEGRAM_BOT_TOKEN` and `OPENROUTER_API_KEY`
3. Optionally configure `ALLOWED_USERS` for access control

## Running

```bash
python -m mai_gram.main
```

## First Chat

1. Open your Telegram bot
2. Send `/start`
3. Select a model from the keyboard
4. Select a system prompt (or type your own)
5. Start chatting!

## Console Mode

For debugging without Telegram:

```bash
# Create a test chat (model + prompt in one command)
mai-chat -c test-demo --start --model google/gemma-4-31b-it:free --prompt default

# Send messages
mai-chat -c test-demo "Hello!"

# Inspect state
mai-chat -c test-demo --history
mai-chat -c test-demo --wiki
mai-chat -c test-demo --show-prompt
```

See [docs/DEBUGGING.md](DEBUGGING.md) for the full reference.
