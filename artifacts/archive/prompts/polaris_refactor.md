# Copilot Refactor Prompt — `polaris` repo (uv migration + reusability pass)

```
You are refactoring the repository "polaris" (github.com/kpatherya/polaris), a
research codebase for a cross-temporal object detection/matching pipeline
(rover imagery -> detection -> matching across time). The repo currently
contains:

- Root-level standalone scripts (adaptive_threshold_config.py,
  analyze_detections.py, cross_temporal_pipeline.py) that duplicate logic
  also present in scripts/cross_temporal_matching.py.
- experiments/ containing three large, near-duplicate pipeline scripts
  (baseline_a_autumnwinter_match.py, baseline_b_autumnwinter_match.py,
  full_pipeline_autumnwinter_match.py) plus generated CSVs and a ~4MB
  detections.json committed to git.
- A separate Swift/Xcode app (app/FastVLM...) unrelated to the Python
  pipeline's runtime dependencies.
- model_export/, llava/, docs/, artifacts/, scripts/ as loosely organized
  top-level directories.
- A pyproject.toml with no enforced lockfile discipline.

Your job: turn this into a reproducible, uv-managed Python project with a
single source of truth for pipeline logic, minimal essential scripts, and a
clear boundary between "library code," "CLI entry points," and "research
artifacts." Do not touch app/ (Swift/Xcode) beyond isolating it — it is a
separate deliverable and should not affect Python packaging.
```

## Stage 1 — Inventory and dependency graph (no changes yet)

```
Before changing anything, do a read-only audit:
1. List every .py file at the repo root, in scripts/, and in experiments/.
   For each, identify: (a) what it imports from other local files, (b) what
   external packages it imports, (c) whether it's a CLI entry point, a
   library module, or a one-off/throwaway script.
2. Diff the three files baseline_a_autumnwinter_match.py,
   baseline_b_autumnwinter_match.py, and full_pipeline_autumnwinter_match.py
   line-by-line and summarize what is actually shared logic vs. what differs
   (e.g. thresholds, matching strategy, config flags).
3. Identify every file that is generated output rather than source (result
   CSVs, detections.json, anything under artifacts/) versus files that are
   genuinely inputs or code.
4. Produce this as a markdown table: file path | role (lib/cli/script/
   generated-output/duplicate) | keep/merge/delete | reason.
Do not modify files in this stage. Output only the audit table and a short
narrative of the duplication pattern you found.
```

## Stage 2 — Target layout proposal

```
Based on the Stage 1 audit, propose a new src-layout structure for the
Python portion of this repo, following this shape unless you have a
specific reason to deviate:

polaris/
  pyproject.toml          # uv-managed, single source of dependency truth
  uv.lock
  src/polaris/
    __init__.py
    detection/            # detection-related logic (from analyze_detections.py etc.)
    matching/             # cross-temporal matching logic (unify baseline_a/b/full_pipeline)
    config/               # adaptive_threshold_config.py becomes a config module
    io/                    # video/rover conversion, model download helpers
    cli.py                # single CLI entrypoint(s), e.g. `polaris run-pipeline`
  scripts/                # ONLY bare shell/setup scripts that can't be Python (model downloads, env setup)
  experiments/            # thin, small scripts that import from src/polaris and just set parameters
  docs/
  data/                    # gitignored; document expected inputs
  app/                     # untouched Swift project, isolated

For each duplicated experiment script (baseline_a, baseline_b, full_pipeline),
design ONE parameterized matching function/class in src/polaris/matching/
that takes the differing knobs (threshold, strategy name, etc.) as arguments
or a config object, so the three experiments become three short scripts that
each just call the shared function with different parameters.

Show me this plan as a directory tree plus a 1-2 sentence justification per
module before writing any code.
```

## Stage 3 — uv migration

```
Migrate dependency management to uv:
1. Regenerate pyproject.toml under [project] with an accurate dependency list
   inferred from actual imports found in Stage 1 (do not carry over unused
   or speculative dependencies from any current requirements/setup files).
2. Split dependencies into core (needed to run the pipeline) and optional
   groups (e.g. [project.optional-dependencies] viz, dev) — anything only
   used by one-off plotting/figure scripts (create_example_figure.py,
   create_slide_figure.py) goes in a `viz` extra, not core.
3. Generate a uv.lock via `uv lock` semantics (produce the pyproject.toml
   correctly enough that `uv sync` would work; note any package I need to
   verify manually because you can't run uv yourself).
4. Add a minimal `uv run` based Makefile or justfile with targets like
   `setup`, `run-pipeline`, `test`, `lint` so I never have to remember raw
   commands.
5. Remove any legacy requirements.txt / setup.py / conda env files once
   the pyproject.toml supersedes them, and tell me explicitly what you
   removed and why.
```

## Stage 4 — Collapse duplication into reusable modules

```
Now implement the Stage 2 plan for the code (not yet for experiments/):
1. Move logic from the root-level scripts (adaptive_threshold_config.py,
   analyze_detections.py, cross_temporal_pipeline.py) and scripts/
   cross_temporal_matching.py into the appropriate src/polaris/ submodules,
   resolving the duplication you found between the root script and the
   scripts/ version — keep the more complete/correct implementation and
   delete the redundant one, explaining your choice.
2. Extract the shared matching logic from baseline_a/baseline_b/full_pipeline
   into src/polaris/matching/ as described in Stage 2. Each function must
   have a docstring, type hints, and no hardcoded file paths — paths and
   thresholds come in as parameters or a config dataclass.
3. Build a single CLI (src/polaris/cli.py) using argparse or click that
   exposes subcommands equivalent to what these scripts did
   (e.g. `polaris match --strategy baseline-a`, `polaris match --strategy full-pipeline`).
4. Rewrite experiments/baseline_a_autumnwinter_match.py,
   baseline_b_autumnwinter_match.py, and full_pipeline_autumnwinter_match.py
   as thin scripts (<30 lines each) that just call the shared CLI/library
   function with their specific parameters, for backward-compatible
   reproducibility of the original experiment.
5. Do not silently drop any functionality — if something in the old scripts
   has no clear new home, flag it to me instead of deleting it.
```

## Stage 5 — Strip to bare-essential scripts

```
Apply a "does this need to exist as a script" test to every remaining .py
and .sh file at the repo root and in scripts/:
- If it's a one-time/manual operation (e.g. downloading a model, converting
  a video format), keep it as a small shell or Python script under scripts/,
  but make it idempotent and parameterized via CLI args or env vars instead
  of hardcoded paths.
- If it's reusable pipeline logic, it must live under src/polaris/ and be
  called via the CLI, not run standalone.
- If it's a demo/plotting helper only used once for a slide deck, move it
  under experiments/ or docs/assets/ and mark it clearly as non-essential,
  or propose deleting it if I confirm it's no longer needed.
List every file you intend to delete or relocate with a one-line reason,
and wait for my go-ahead before deleting anything.
```

## Stage 6 — Data and artifact hygiene

```
1. Add data/, artifacts/, experiments/*.csv, experiments/detections.json,
   and any other generated output to .gitignore.
2. Do NOT rewrite git history to purge these from past commits unless I
   explicitly ask — just stop tracking them going forward and tell me the
   `git rm --cached` commands I'd need to run.
3. Add a short docs/DATA.md explaining what inputs each pipeline stage
   expects (paths, formats) and where to obtain or generate them, since the
   actual data files won't ship in the repo anymore.
```

## Stage 7 — Reproducibility verification

```
Produce a REPRODUCE.md that lists the exact commands, in order, to go from
a fresh clone to a working pipeline run using uv (uv sync, then the CLI
commands from Stage 4). Also add a minimal smoke test (tests/test_cli.py or
similar) that imports each src/polaris submodule and calls it with tiny
synthetic input, so `uv run pytest` gives me a fast reproducibility check
without needing the real dataset. Tell me any assumption you had to make
because you couldn't execute the code yourself.
```