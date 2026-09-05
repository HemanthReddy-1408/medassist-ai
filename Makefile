.PHONY: help install test test-live lint fmt demo corpus clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install with dev extras
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:  ## Run the offline suite (no network, no API key)
	.venv/bin/python -m pytest tests/ -q

test-live:  ## Include tests that hit real providers and public APIs
	.venv/bin/python -m pytest tests/ -q -m "live"

lint:  ## Ruff
	.venv/bin/ruff check medassist tests

fmt:  ## Ruff autofix
	.venv/bin/ruff check --fix medassist tests

demo:  ## Live: scrape, index, retrieve, and print the retrieval trace
	.venv/bin/python -m medassist.demo

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .cache

demo-gate:  ## Live: retrieve, generate claims, and run the release gate
	.venv/bin/python -m medassist.demo_gate
