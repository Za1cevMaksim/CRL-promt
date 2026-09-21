"""Clean source-only generation and checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .config import ExperimentConfig
from .data import CausalLMCollator, strip_target_tokens
from .metrics import classification_metrics, generation_metrics


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    collator: CausalLMCollator,
) -> list[str]:
    """Generate from source-only inputs without exposing reference targets."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=config.evaluation_batch_size,
        collate_fn=collator,
    )
    predictions: list[str] = []
    old_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    try:
        for batch in loader:
            input_ids, attention_mask = strip_target_tokens(batch, tokenizer.pad_token_id)
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            predictions.extend(text.strip() or " " for text in decoded)
    finally:
        model.config.use_cache = old_cache
    return predictions


@torch.no_grad()
def verbalizer_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    references: Sequence[str],
    labels: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    collator: CausalLMCollator,
    batch_size: int = 16,
) -> tuple[float, list[int]]:
    """Score unique first label tokens at the next-token position."""
    first_tokens = {
        label: tokenizer(" " + label, add_special_tokens=False)["input_ids"][0]
        for label in labels
    }
    if len(set(first_tokens.values())) != len(first_tokens):
        raise ValueError("The first-token verbalizer contains collisions")
    ordered_labels = sorted(first_tokens)
    token_ids = torch.tensor(
        [first_tokens[label] for label in ordered_labels], device=model.device
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct: list[int] = []
    reference_index = 0
    model.eval()
    for batch in loader:
        input_ids, attention_mask = strip_target_tokens(batch, tokenizer.pad_token_id)
        logits = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        ).logits[:, -1, :]
        predicted_indices = logits[:, token_ids].argmax(dim=-1).tolist()
        for predicted_index in predicted_indices:
            correct.append(
                int(ordered_labels[predicted_index] == references[reference_index])
            )
            reference_index += 1
    return float(np.mean(correct)), correct


def evaluate_model(
    model: torch.nn.Module,
    dataset: Dataset,
    references: Sequence[str],
    labels: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    collator: CausalLMCollator,
) -> dict[str, Any]:
    """Evaluate one trained adapter and retain per-example values."""
    predictions = generate_predictions(model, dataset, tokenizer, config, collator)
    if config.task == "emotion_classification":
        metrics, correct = classification_metrics(predictions, references, labels)
        verbalizer, verbalizer_correct = verbalizer_accuracy(
            model, dataset, references, labels, tokenizer, collator
        )
        metrics["verbalizer_accuracy"] = verbalizer
        return {
            "metrics": metrics,
            "predictions": predictions,
            "correct": correct,
            "verbalizer_correct": verbalizer_correct,
        }
    metrics, per_example_rouge = generation_metrics(predictions, references)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "per_example_rougeL": per_example_rouge,
    }

