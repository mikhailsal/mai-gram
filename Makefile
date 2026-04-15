# mai-gram Makefile
# ========================
# Convenience commands for development, testing, and running the application.
#
# Usage: make <target>
# Run `make help` to see all available targets.

.PHONY: help install install-dev run run-dev run-reload \
        chat chat-start chat-list chat-history chat-prompt chat-import \
        test test-v test-cov test-cov-html test-unit test-fast \
        lint lint-fix format format-check typecheck check fix \
        precommit install-hooks \
        docker-build docker-up docker-down docker-logs docker-restart docker-shell \
        clean clean-pyc clean-test clean-all

# Default target
.DEFAULT_GOAL := help

# Colors for help output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# Python interpreter
PYTHON := python3

# Default test chat ID
CHAT ?= test-makefile

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)mai-gram$(RESET) — Telegram-LLM bridge via OpenRouter"
	@echo ""
	@echo "$(YELLOW)Usage:$(RESET) make $(BLUE)<target>$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^[a-zA-Z_-]+:.*##/ { printf "  $(BLUE)%-18s$(RESET) %s\n", $$1, $$2 } \
		/^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""

##@ Installation

install: ## Install the package (production dependencies only)
	$(PYTHON) -m pip install -e .

install-dev: ## Install the package with development dependencies + git hooks
	$(PYTHON) -m pip install -e ".[dev]"
	@$(MAKE) install-hooks

##@ Running the Bot

run: ## Run the Telegram bot (production mode)
	$(PYTHON) -m mai_gram.main

run-dev: run-reload ## Alias for run-reload

run-reload: ## Run with auto-reload (restarts on code changes)
	$(PYTHON) -m mai_gram.main --reload

##@ Console Chat (mai-chat CLI)

chat-start: ## Start a new chat setup (CHAT=test-mychat)
	mai-chat -c $(CHAT) --start

chat-setup: ## Full setup: select model + prompt (CHAT=test-mychat MODEL=openai/gpt-4o-mini PROMPT=default)
	mai-chat -c $(CHAT) --start --cb "model:$(or $(MODEL),openai/gpt-4o-mini)" --cb "prompt:$(or $(PROMPT),default)"

chat: ## Send a message (CHAT=test-mychat MSG="Hello!")
	mai-chat -c $(CHAT) "$(MSG)"

chat-list: ## List all chats with message counts
	mai-chat --list

chat-history: ## Show conversation history (CHAT=test-mychat)
	mai-chat -c $(CHAT) --history

chat-wiki: ## Show wiki entries (CHAT=test-mychat)
	mai-chat -c $(CHAT) --wiki

chat-prompt: ## Show the assembled LLM prompt (CHAT=test-mychat)
	mai-chat -c $(CHAT) --show-prompt

chat-import: ## Import dialogue from JSON file (CHAT=test-mychat FILE=conversation.json)
	mai-chat -c $(CHAT) --import-json $(FILE)

chat-debug: ## Send a message with debug logging (CHAT=test-mychat MSG="Hello!")
	mai-chat -c $(CHAT) --debug "$(MSG)"

##@ Testing

test: ## Run all tests
	pytest

test-v: ## Run all tests with verbose output
	pytest -v

test-cov: ## Run tests with coverage report (enforces 90% minimum)
	pytest --cov=mai_gram --cov-report=term-missing --cov-config=pyproject.toml --cov-fail-under=90

test-cov-html: ## Run tests with HTML coverage report
	pytest --cov=mai_gram --cov-report=html --cov-config=pyproject.toml
	@echo "Coverage report generated in htmlcov/index.html"

test-unit: ## Run unit tests only
	pytest tests/

test-fast: ## Run tests excluding slow markers
	pytest -m "not slow"

##@ Code Quality

lint: ## Check code with ruff linter
	ruff check .

lint-fix: ## Auto-fix linting issues
	ruff check --fix .

format: ## Format code with ruff
	ruff format .

format-check: ## Check code formatting (without changes)
	ruff format --check .

typecheck: ## Run mypy type checker
	mypy src/mai_gram

check: lint format-check typecheck ## Run all code quality checks (lint + format + typecheck)

fix: lint-fix format ## Auto-fix linting issues and format code

##@ Pre-commit

precommit: check test-cov ## Run all pre-commit checks (lint, format, typecheck, tests + 90% coverage)

install-hooks: ## Install git pre-commit hook (enforces quality on every commit)
	@cp scripts/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed. It will run lint, format, typecheck, and tests before each commit."
	@echo "Skip once with: git commit --no-verify"

##@ Docker

docker-build: ## Build Docker image
	docker compose build

docker-up: ## Start the application in Docker (detached)
	docker compose up -d

docker-down: ## Stop and remove Docker containers
	docker compose down

docker-logs: ## View Docker container logs (follow mode)
	docker compose logs -f

docker-restart: ## Restart Docker containers
	docker compose restart

docker-shell: ## Open a shell in the running container
	docker compose exec mai-gram /bin/bash

##@ Cleanup

clean: clean-pyc ## Clean up generated files

clean-pyc: ## Remove Python bytecode and cache files
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-test: ## Remove test artifacts
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

clean-all: clean clean-test ## Remove all generated files (bytecode, test artifacts)
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
