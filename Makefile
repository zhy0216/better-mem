.PHONY: up down migrate run worker test lint typecheck

up:
	docker-compose up -d

down:
	docker-compose down

migrate:
	python -m src.migrations.migrate

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	python -m arq src.worker.settings.WorkerSettings

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	pyright src/

install:
	uv pip install -e ".[dev]"
