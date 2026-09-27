"""Unified command-line interface for the POLARIS research pipeline."""

from __future__ import annotations

import sys

from polaris.detection.analysis import main as analyze_detections_main
from polaris.io.rover_video import main as convert_video_main
from polaris.matching.runner import run_match_cli


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher with subcommands for matching and utilities."""
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in {"-h", "--help"}:
        print("POLARIS cross-temporal toolkit")
        print()
        print("Usage:")
        print("  polaris match ...")
        print("  polaris analyze-detections ...")
        print("  polaris convert-video ...")
        return 0

    command, rest = args[0], args[1:]

    if command == "match":
        return run_match_cli(rest)
    if command == "analyze-detections":
        return analyze_detections_main(rest)
    if command == "convert-video":
        return convert_video_main(rest)

    print(f"Unknown command: {command}")
    print("Run 'polaris --help' for available commands.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
