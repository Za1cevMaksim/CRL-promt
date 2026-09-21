#!/usr/bin/env python3
"""Run the emotion classification experiment group."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "main.py"), "classification", *sys.argv[1:]]))

