.PHONY: help install install-dev run run-dev test test-cov lint format clean docker-build docker-run docker-stop docker-logs

# Default target
help:
	@echo "Music Genre Classifier - Available commands:"
	@echo ""
	@echo "  Development:"
	@echo "    make install       Install production dependencies"
	@echo "    make install-dev   Install development dependencies"
	@echo "    make run           Run production server"
	@echo "    make run-dev       Run development server with hot reload"
	@echo ""
	@echo "  Testing & Quality:"
	@echo "    make test          Run tests"
	@echo "    make test-cov      Run tests with coverage report"
	@echo "    make lint          Run linters (flake8, mypy)"
	@echo "    make format        Format code (black, isort)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build  Build Docker image"
	@echo "    make docker-run    Run Docker container"
	@echo "    make docker-dev    Run Docker in development mode"
	@echo "    make docker-stop   Stop Docker containers"
	@echo "    make docker-logs   View Docker logs"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         Remove cache and temp files"
	@echo "    make clean-all     Remove all generated files including models"

# Variables
PYTHON := python3
PIP := pip
UVICORN := uvicorn
DOCKER_COMPOSE := docker-compose
IMAGE_NAME := music-genre-classifier
PORT := 8000

# ============== Development ==============

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install || true

run:
	$(UVICORN) app:app --host 0.0.0.0 --port $(PORT)

run-dev:
	$(UVICORN) app:app --reload --host 0.0.0.0 --port $(PORT)

# ============== Testing & Quality ==============

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	mypy . --ignore-missing-imports || true

format:
	black .
	isort .

# ============== Docker ==============

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	$(DOCKER_COMPOSE) up -d

docker-dev:
	$(DOCKER_COMPOSE) --profile dev up app-dev

docker-stop:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-shell:
	docker exec -it $(IMAGE_NAME) /bin/bash

# ============== Cleanup ==============

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf temp/* 2>/dev/null || true

clean-all: clean
	rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true
	rm -rf .eggs/ 2>/dev/null || true

# ============== Data & Training ==============

preprocess:
	$(PYTHON) preprocess_data.py

train:
	$(PYTHON) train_nn.py

predict:
	$(PYTHON) predict.py

# ============== Environment ==============

env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
	else \
		echo ".env file already exists"; \
	fi

check-env:
	@echo "Checking environment..."
	@$(PYTHON) --version
	@$(PIP) --version
	@echo "All checks passed!"
