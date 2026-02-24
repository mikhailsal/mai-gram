# mAI Companion Makefile
# ========================
# Convenience commands for development, testing, and running the application.
#
# Usage: make <target>
# Run `make help` to see all available targets.

.PHONY: help install install-dev run run-dev run-reload console \
        test test-cov test-functional test-unit \
        lint format typecheck check \
        docker-build docker-up docker-down docker-logs docker-restart \
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

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)mAI Companion$(RESET) — Development Commands"
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

install-dev: ## Install the package with development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

##@ Running the Application

run: ## Run the Telegram bot (production mode)
	$(PYTHON) -m mai_companion.main

run-dev: run-reload ## Alias for run-reload

run-reload: ## Run with auto-reload (restarts on code changes)
	$(PYTHON) -m mai_companion.main --reload

console: ## Run in console mode (interactive, for testing)
	mai-chat

console-test: ## Run console with a test companion
	mai-chat -c test-makefile --start

##@ Testing

test: ## Run all tests
	pytest

test-v: ## Run all tests with verbose output
	pytest -v

test-cov: ## Run tests with coverage report
	pytest --cov=mai_companion --cov-report=term-missing

test-cov-html: ## Run tests with HTML coverage report
	pytest --cov=mai_companion --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-functional: ## Run functional tests only
	pytest tests/functional/

test-unit: ## Run unit tests only (exclude functional)
	pytest --ignore=tests/functional/

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
	mypy src/mai_companion

check: lint format-check typecheck ## Run all code quality checks (lint + format + typecheck)

fix: lint-fix format ## Auto-fix linting issues and format code

##@ Pre-commit

precommit: check test ## Run all checks before committing (lint, format, typecheck, tests)

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
	docker compose exec mai-companion /bin/bash

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
