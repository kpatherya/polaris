"""Compatibility shim for threshold configuration.

The source of truth now lives in src/polaris/config/adaptive_thresholds.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from polaris.config.adaptive_thresholds import *
