#!/usr/bin/env python3
"""Evaluate one saved prefix on a larger source-only test subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from crl_prompt.config import load_config
from crl_prompt.data import CausalLMCollator, load_dialogue_reference_sets
from crl_prompt.evaluation import evaluate_model
from crl_prompt.experiments import load_tokenizer, prepare_data
from crl_prompt.metrics import bertscore_f1, multireference_rouge_l
from crl_prompt.modeling import load_prefix_model, restore_prefix
from crl_prompt.utils import save_json


def main() -> None:
    """Load a configuration and prefix checkpoint and evaluate them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--multi-reference", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config, {"test_size": arguments.test_size})
    tokenizer = load_tokenizer(config)
    bundle = prepare_data(config, tokenizer)
    model = load_prefix_model(config)
    restore_prefix(
        model,
        torch.load(arguments.checkpoint, map_location="cpu", weights_only=True),
    )
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
    if arguments.bertscore and config.task == "response_generation":
        result["metrics"]["bertscore_f1"] = bertscore_f1(
            result["predictions"], bundle.test_references
        )
    if arguments.multi_reference and config.task == "response_generation":
        reference_sets = load_dialogue_reference_sets(config, "test")[: config.test_size]
        score, per_example = multireference_rouge_l(
            result["predictions"], reference_sets
        )
        result["metrics"]["multireference_rougeL"] = score
        result["per_example_multireference_rougeL"] = per_example
    save_json(result, arguments.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
