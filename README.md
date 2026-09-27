# POLARIS

POLARIS is a research codebase for cross-temporal landmark matching (for example, autumn vs winter rover imagery).

![Cross-season matching example](docs/pair_005_comparison.jpg)

The Python side of this repository is now organized around one reusable package (`src/polaris`) with one CLI (`polaris`).
Legacy script paths are kept as thin wrappers under `scripts/compat/` for compatibility.

## What is in scope

- Reusable Python logic: `src/polaris/`
- Thin experiment entry points: `experiments/*.py`
- Operational helper scripts: `scripts/`
- Swift app (separate deliverable): `app/` (not part of Python packaging)

## Quick start (uv)

1. Install uv: <https://docs.astral.sh/uv/>
2. Sync dependencies:

```bash
make setup
# equivalent: uv sync --extra dev --extra viz
```

3. Run help:

```bash
uv run polaris --help
```

## Main commands

### Match across seasons

```bash
# Full pipeline
uv run polaris match --strategy full-pipeline \
  --detections data/detections.json \
  --autumn-idx 0 \
  --winter-depth /path/to/winter/depth \
  --autumn-depth /path/to/autumn/depth

# Baseline A
uv run polaris match --strategy baseline-a \
  --detections data/detections.json \
  --autumn-idx 0

# Baseline B
uv run polaris match --strategy baseline-b \
  --detections data/detections.json \
  --autumn-idx 0
```

### Analyze OWL-ViT detection thresholds

```bash
uv run polaris analyze-detections \
  --image-dir /path/to/rgb \
  --query-set recommended
```

### Convert image sequence to video

```bash
uv run polaris convert-video \
  --input-dir /path/to/rgb \
  --output-path output.mp4 \
  --fps 30
```

## Experiments

Use the orchestrator for a fixed test set:

```bash
bash experiments/run_all_experiments.sh
```

Results are logged to:

- `artifacts/runs/latest/baseline_a_results.csv`
- `artifacts/runs/latest/baseline_b_results.csv`
- `artifacts/runs/latest/full_pipeline_results.csv`

Legacy non-essential scripts and historical outputs are archived under:

- `artifacts/archive/experiments_legacy/`

## Reproducibility and data docs

- Repro steps: `docs/project/REPRODUCE.md`
- Data contract (expected folders/files): `docs/DATA.md`

## Development

```bash
make test
make lint
```

## Notes on upstream code and licenses

This repository contains upstream FastVLM/LLaVA-derived components under `llava/`, `model_export/`, and `app/`.
Keep license and attribution files intact:

- `LICENSE`
- `docs/legal/LICENSE_MODEL`
- `docs/legal/ACKNOWLEDGEMENTS`
- `CITATION.cff`

Community/governance docs:

- `.github/CODE_OF_CONDUCT.md`
- `.github/CONTRIBUTING.md`
