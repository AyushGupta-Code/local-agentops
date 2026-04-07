.PHONY: install run test lint format

install:
	pip install -e .[dev]

run:
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

test:
	pytest

lint:
	ruff check app tests
	mypy app

format:
	ruff check --fix app tests
