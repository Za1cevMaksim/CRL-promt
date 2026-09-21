#!/usr/bin/env python3
"""Run the model-capacity and data-size sweep."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "main.py"), "capacity", *sys.argv[1:]]))

