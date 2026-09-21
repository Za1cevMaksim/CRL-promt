"""Runtime, serialization and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import set_seed


def configure_runtime(seed: int) -> None:
    """Initialize random number generators and CUDA allocation settings."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def save_json(payload: Any, path: str | Path) -> None:
    """Write a JSON file atomically enough for resumable local runs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clear_device_cache() -> None:
    """Release Python and CUDA caches between large model runs."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

