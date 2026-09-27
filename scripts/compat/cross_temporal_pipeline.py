"""Legacy compatibility shim for the old monolithic pipeline script.

The original implementation has been archived at:
  artifacts/archive/legacy_cross_temporal_pipeline.py

Use the unified CLI instead:
  uv run polaris match --strategy full-pipeline ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from polaris.matching.primitives import DepthValidator, FastVLMAnalyzer, KeypointMatcher
from polaris.matching.runner import run_match_cli


def main(argv: list[str] | None = None) -> int:
    """Delegate to the modern shared matching runner."""
    print(
        "[deprecated] cross_temporal_pipeline.py now delegates to polaris match. "
        "For full functionality use: uv run polaris match --strategy full-pipeline ..."
    )
    return run_match_cli(argv=argv, default_strategy="full-pipeline")


__all__ = ["DepthValidator", "FastVLMAnalyzer", "KeypointMatcher", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
