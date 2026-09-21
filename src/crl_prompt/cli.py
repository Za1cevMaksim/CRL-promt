"""Installed console entry point."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Execute the repository-level command-line interface."""
    repository_main = Path(__file__).resolve().parents[2] / "main.py"
    runpy.run_path(str(repository_main), run_name="__main__")

