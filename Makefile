SHELL := /bin/bash

.PHONY: setup run-pipeline run-baseline-a run-baseline-b analyze-detections convert-video test lint format

setup:
	uv sync --extra dev --extra viz

run-pipeline:
	uv run polaris match --strategy full-pipeline $(ARGS)

run-baseline-a:
	uv run polaris match --strategy baseline-a $(ARGS)

run-baseline-b:
	uv run polaris match --strategy baseline-b $(ARGS)

analyze-detections:
	uv run polaris analyze-detections $(ARGS)

convert-video:
	uv run polaris convert-video $(ARGS)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests experiments scripts

format:
	uv run ruff format src tests experiments scripts
