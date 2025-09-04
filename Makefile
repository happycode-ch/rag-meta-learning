.PHONY: help install test lint format type-check clean run-tests coverage dev setup

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make install     - Install project dependencies"
	@echo "  make test        - Run all tests with coverage"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code with black and isort"
	@echo "  make type-check  - Run type checking with mypy"
	@echo "  make coverage    - Generate test coverage report"
	@echo "  make clean       - Remove build artifacts and cache files"
	@echo "  make dev         - Run development server"
	@echo "  make pre-commit  - Install pre-commit hooks"

setup:
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pre-commit install

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	pylint src/

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

type-check:
	mypy src/ --strict --ignore-missing-imports

coverage:
	pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

dev:
	python src/main.py --dev

pre-commit:
	pre-commit install
	pre-commit run --all-files