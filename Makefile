.DEFAULT_GOAL := help

.PHONY: help install format lint test coverage clean ci

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install for development
	pip install -e ".[dev]"

format:  ## Auto-format code
	ruff format .
	ruff check --fix .

lint:  ## Run linters
	ruff check .

test:  ## Run tests
	pytest tests/

coverage:  ## Run tests with coverage
	pytest --cov=finetunecheck --cov-report=term-missing tests/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .ruff_cache/ .mypy_cache/ .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

ci: lint test  ## Run CI checks (lint + test)
