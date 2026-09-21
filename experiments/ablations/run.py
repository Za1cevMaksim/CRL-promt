#!/usr/bin/env python3
"""Run the auxiliary hyperparameter and objective ablations."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "main.py"), "ablations", *sys.argv[1:]]))

