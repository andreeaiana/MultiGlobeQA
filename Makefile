.PHONY: install install-all lint typecheck test clean

install:            ## scoring + tier-1/2 inference
	uv sync --extra models --extra dev

install-all:        ## adds Spark, KG construction and the translation pipelines
	uv sync --extra models --extra kg --extra generate --extra kg-build \
	        --extra pipelines --extra dev

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

typecheck:
	uv run mypy benchmark_eval benchmark_models question_generator

test:
	uv run pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info
