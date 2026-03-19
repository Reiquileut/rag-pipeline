.PHONY: help up down logs test lint fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start all services (docker-compose)
	docker compose up -d --build

down: ## Stop all services
	docker compose down

logs: ## Follow application logs
	docker compose logs -f api

test: ## Run tests
	pytest -v --tb=short

lint: ## Run linter (ruff)
	ruff check app/ tests/

fmt: ## Format code (ruff)
	ruff format app/ tests/
	ruff check --fix app/ tests/

clean: ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage
