# Reproduce POLARIS Runs (uv)

## 1) Fresh clone

```bash
git clone <repo-url>
cd polaris
```

## 2) Install dependencies

```bash
make setup
```

## 3) Verify environment quickly

```bash
make test
```

## 4) Run one smoke match

```bash
uv run polaris match --strategy baseline-b \
  --detections data/detections.json \
  --autumn-idx 0 \
  --top-k 3
```

## 5) Run all configured experiments

```bash
bash experiments/run_all_experiments.sh
```

## 6) Optional utilities

```bash
uv run polaris analyze-detections --image-dir /path/to/rgb
uv run polaris convert-video --input-dir /path/to/rgb --output-path output.mp4
```

## Assumptions

- You have local access to RGB/depth directories referenced by your detection JSON.
- You have compatible hardware/software for model inference (CPU/MPS/CUDA).
- This document does not pin dataset locations because they are environment-specific.
