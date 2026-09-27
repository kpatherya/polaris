"""Thin wrapper for Baseline B experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polaris.matching.runner import run_match_cli

if __name__ == "__main__":
    raise SystemExit(run_match_cli(default_strategy="baseline-b"))
