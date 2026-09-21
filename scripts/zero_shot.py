#!/usr/bin/env python3
"""Evaluate a base Qwen2.5 model without a trainable prefix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import AutoModelForCausalLM

from crl_prompt.config import load_config
from crl_prompt.data import CausalLMCollator
from crl_prompt.evaluation import evaluate_model
from crl_prompt.experiments import load_tokenizer, prepare_data
from crl_prompt.utils import save_json


def main() -> None:
    """Run clean source-only zero-shot evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    tokenizer = load_tokenizer(config)
    bundle = prepare_data(config, tokenizer)
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    collator = CausalLMCollator(tokenizer, config.legacy_training_padding)
    result = evaluate_model(
        model,
        bundle.test,
        bundle.test_references,
        bundle.emotion_labels,
        tokenizer,
        config,
        collator,
    )
    save_json(result, arguments.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

