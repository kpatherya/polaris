"""Compatibility entry point for frame-sequence video conversion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polaris.io.rover_video import main

if __name__ == "__main__":
    raise SystemExit(main())
