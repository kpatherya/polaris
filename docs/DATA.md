# Data Inputs for POLARIS

This repository does not ship large datasets or generated experiment artifacts.

## Required inputs

### 1) Detection cache for experiment matching

- Recommended path: `data/detections.json`
- Optional legacy snapshot path: `artifacts/archive/experiments_legacy/outputs/detections.json`
- Structure:
  - top-level keys: `autumn`, `winter`
  - each is a list of frames
  - frame fields:
    - `frame_id` (string)
    - `image_path` (absolute or repo-relative path)
    - `detections` (list)
  - each detection field:
    - `label` (string)
    - `bbox` (`[x1, y1, x2, y2]`)
    - optional score/metadata fields

### 2) RGB frame directories (for detection analysis and video conversion)

- Example: `/path/to/rover/.../rgb`
- Expected files: `.png`

### 3) Optional depth directories (for full-pipeline strategy)

- Winter depth path: user-provided with `--winter-depth`
- Autumn depth path: user-provided with `--autumn-depth`
- File matching: same basename as RGB frame, or `.png/.tiff/.tif` variants

## Generated outputs (not source)

- `artifacts/runs/*`
- `pipeline_results/*`
- `data/*`

Historical one-off artifacts and presentation material are archived under:

- `artifacts/archive/experiments_legacy/*`

These should be treated as reproducible outputs and kept out of version control.

## How to regenerate

- Run experiments: `bash experiments/run_all_experiments.sh`
- Run detection analysis: `uv run polaris analyze-detections ...`
