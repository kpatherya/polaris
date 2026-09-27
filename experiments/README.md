# Experiments

This directory contains only thin entry scripts and the orchestrator for comparing three strategies:

- `baseline-a`: visual embedding + semantic label overlap
- `baseline-b`: geometric keypoint matching
- `full-pipeline`: visual + semantic + depth + geometric fusion

All three scripts delegate to shared code in `src/polaris/matching/runner.py`.

Legacy slide/report scripts and historical generated outputs were moved to:

- `artifacts/archive/experiments_legacy/`

## Run one experiment

```bash
uv run python experiments/baseline_a_autumnwinter_match.py \
  --detections data/detections.json \
  --autumn-idx 0

uv run python experiments/baseline_b_autumnwinter_match.py \
  --detections data/detections.json \
  --autumn-idx 0

uv run python experiments/full_pipeline_autumnwinter_match.py \
  --detections data/detections.json \
  --autumn-idx 0 \
  --winter-depth /path/to/winter/depth \
  --autumn-depth /path/to/autumn/depth
```

## Run the full sweep

```bash
bash experiments/run_all_experiments.sh
```

## Outputs

Generated outputs are written under:

- `artifacts/runs/latest/`

These files are treated as artifacts and should not be committed as source.
