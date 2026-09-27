"""Compatibility entry point for detection analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from polaris.detection.analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
