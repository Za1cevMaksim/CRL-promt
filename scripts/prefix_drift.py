#!/usr/bin/env python3
"""Measure relative movement between a shared prefix start and a trained prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def flatten_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate all floating-point tensors in deterministic key order."""
    tensors = [state[key].float().reshape(-1) for key in sorted(state)]
    if not tensors:
        raise ValueError("The checkpoint does not contain tensors")
    return torch.cat(tensors)


def prefix_drift(start: dict[str, torch.Tensor], trained: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return relative L2 drift and cosine similarity between checkpoints."""
    start_vector = flatten_state(start)
    trained_vector = flatten_state(trained)
    if start_vector.shape != trained_vector.shape:
        raise ValueError("Prefix checkpoints have different shapes")
    difference = trained_vector - start_vector
    return {
        "start_norm": float(torch.linalg.vector_norm(start_vector)),
        "trained_norm": float(torch.linalg.vector_norm(trained_vector)),
        "difference_norm": float(torch.linalg.vector_norm(difference)),
        "relative_drift": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(start_vector).clamp(min=1e-12)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                start_vector.unsqueeze(0), trained_vector.unsqueeze(0)
            )
        ),
    }


def main() -> None:
    """Read two prefix checkpoints and write drift diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    start = torch.load(arguments.start, map_location="cpu", weights_only=True)
    trained = torch.load(arguments.trained, map_location="cpu", weights_only=True)
    result = prefix_drift(start, trained)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

