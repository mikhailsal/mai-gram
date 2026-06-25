# mai-gram Makefile
# ========================
# Convenience commands for development, testing, and running the application.
#
# Usage: make <target>
# Run `make help` to see all available targets.

.PHONY: help install install-dev run run-dev run-reload \
        chat chat-start chat-list chat-history chat-prompt chat-import \
	test test-v test-cov test-cov-html test-unit test-integration test-fast \
	test-functional-local test-functional-live test-functional-live-serial test-local \
	lint lint-fix format format-check typecheck size-check check fix \
        precommit install-hooks \
        docker-build docker-up docker-down docker-logs docker-restart docker-shell \
        deploy \
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

test: ## Run all tests (unit + integration + local functional; live skipped without API key)
	pytest
	pytest tests/test_integration -n0

test-v: ## Run all tests with verbose output
	pytest -v
	pytest tests/test_integration -n0 -v

test-cov: ## Run tests with coverage report (enforces minimum from pyproject.toml)
	pytest --cov=mai_gram --cov-report=term-missing --cov-config=pyproject.toml
	pytest tests/test_integration -n0 --cov=mai_gram --cov-config=pyproject.toml --cov-append --cov-report=term-missing

test-cov-html: ## Run tests with HTML coverage report
	pytest --cov=mai_gram --cov-report=html --cov-config=pyproject.toml
	pytest tests/test_integration -n0 --cov=mai_gram --cov-config=pyproject.toml --cov-append --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-unit: ## Run unit tests only (no functional, no integration)
	pytest -m "not functional_local and not functional_live and not functional and not integration"

test-integration: ## Run in-process integration tests (mock providers, no API key, serial)
	pytest tests/test_integration -n0

test-functional-local: ## Run local functional tests (CLI subprocess, no API key)
	pytest -m functional_local

test-functional-live: ## Run live functional tests with OPENROUTER_API_KEY loaded from env or .env
	@set -e; \
	set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	if [ -z "$$OPENROUTER_API_KEY" ]; then \
		echo "OPENROUTER_API_KEY is required for live functional tests"; \
		exit 1; \
	fi; \
	pytest -m "functional_live or functional" --run-functional

test-functional-live-serial: ## Run live functional tests serially (no parallelism)
	@set -e; \
	set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	if [ -z "$$OPENROUTER_API_KEY" ]; then \
		echo "OPENROUTER_API_KEY is required for live functional tests"; \
		exit 1; \
	fi; \
	pytest -n 0 -m "functional_live or functional" --run-functional

test-local: ## Run all local tests (unit + integration + local functional)
	pytest -m "not functional_live and not functional"
	pytest tests/test_integration -n0

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

size-check: ## Enforce Python file/function size limits
	$(PYTHON) scripts/check_code_limits.py --enforce

check: lint format-check typecheck size-check ## Run all code quality checks (lint + format + typecheck + size audit)

fix: lint-fix format ## Auto-fix linting issues and format code

##@ Pre-commit

precommit: check test-cov test-functional-live ## Run all pre-commit checks (local tests first via coverage, then live)

install-hooks: ## Install git pre-commit hook (enforces quality on every commit)
	@cp scripts/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed. It will run lint, format, typecheck, size enforcement, coverage, and live functional tests before each commit."
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

##@ Deployment

deploy: ## Deploy working tree to the remote server (configure deploy.env first)
	./scripts/deploy.sh

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
