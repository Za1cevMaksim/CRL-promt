#!/usr/bin/env python3
"""Run the response generation experiment group."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "main.py"), "generation", *sys.argv[1:]]))

