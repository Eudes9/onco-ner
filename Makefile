# Makefile

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/

format:
	python -m ruff format src/