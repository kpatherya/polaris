# Contributing to Polaris

Thanks for improving Polaris. This repository is a research pipeline for
cross-temporal landmark matching, so reproducibility and clear assumptions are
critical.

## Environment Assumptions

- Python 3.10+
- PyTorch + Transformers with either `mps`, `cuda`, or `cpu`
- FastVLM checkpoint available locally
- Seasonal RGB/depth datasets available locally or mounted storage

Install project dependencies from `pyproject.toml` using your preferred tooling.

## Reproducible Run Checklist

When opening a pull request that changes behavior, include:

- command line used to run pipeline or experiment scripts
- dataset layout and frame-pair selection settings
- model checkpoint path (use placeholders in docs, not local absolute paths)
- output directory and artifact names used for validation

## Contribution Flow

1. Open an issue for substantial feature work.
2. Keep pull requests focused to one theme (pipeline logic, analysis, docs, or tooling).
3. Include minimal validation evidence (script syntax check, dry run, or sample output).
4. Update README examples if flags, defaults, or outputs change.

## Documentation Rules

- Do not commit examples with machine-specific paths like `/Users/...` or `/Volumes/...`.
- Use portable placeholders such as `/path/to/winter/rgb`.
- Document limitations or hardware assumptions directly in PR descriptions.
