.PHONY: install-dev format lint test check

RUFF_TARGETS = glowclip tests app.py

install-dev:
	python -m pip install -e '.[dev]'

format:
	ruff format $(RUFF_TARGETS)
	ruff check $(RUFF_TARGETS) --fix

lint:
	ruff check $(RUFF_TARGETS)
	ruff format --check $(RUFF_TARGETS)

test:
	pytest

check: lint test
