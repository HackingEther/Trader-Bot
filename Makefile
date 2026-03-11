.PHONY: help install dev up down migrate seed test lint format run worker beat shell

help:
	@echo "Available commands:"
	@echo "  install    Install dependencies"
	@echo "  dev        Install with dev dependencies"
	@echo "  up         Start all services via docker-compose"
	@echo "  down       Stop all services"
	@echo "  migrate    Run Alembic migrations"
	@echo "  seed       Seed database with sample data"
	@echo "  test       Run test suite"
	@echo "  lint       Run linter"
	@echo "  format     Auto-format code"
	@echo "  run        Start FastAPI server locally"
	@echo "  worker     Start Celery worker locally"
	@echo "  beat       Start Celery beat locally"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

up:
	docker-compose up -d

down:
	docker-compose down

migrate:
	alembic upgrade head

downgrade:
	alembic downgrade -1

seed:
	python scripts/seed_symbols.py
	python scripts/seed_sample_bars.py

test:
	pytest --cov=trader --cov-report=term-missing -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run:
	uvicorn trader.main:app --reload --port 8000

worker:
	celery -A trader.celery_app:celery worker --loglevel=info

beat:
	celery -A trader.celery_app:celery beat --loglevel=info

shell:
	python -c "from trader.config import get_settings; print(get_settings())"
