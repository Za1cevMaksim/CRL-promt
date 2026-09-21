"""Evaluation metrics for emotion labels and generated responses."""

from __future__ import annotations

import re
from collections.abc import Sequence

import evaluate
import numpy as np


def normalize_label(text: str) -> str:
    """Normalize generated label text for vocabulary matching."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def match_emotion(text: str, labels: Sequence[str]) -> str | None:
    """Map generated text to one of the known emotion labels."""
    normalized = normalize_label(text)
    if not normalized:
        return None
    label_map = {normalize_label(label): label for label in labels}
    first_word = normalized.split()[0]
    if first_word in label_map:
        return label_map[first_word]
    if normalized in label_map:
        return label_map[normalized]
    for normalized_label, label in label_map.items():
        if normalized.startswith(normalized_label):
            return label
    for normalized_label, label in label_map.items():
        if normalized_label in normalized:
            return label
    return None


def distinct_n(texts: Sequence[str]) -> tuple[float, float]:
    """Return corpus-level Distinct-1 and Distinct-2."""
    unigrams: set[str] = set()
    bigrams: set[tuple[str, str]] = set()
    token_count = 0
    pair_count = 0
    for text in texts:
        words = text.split()
        unigrams.update(words)
        bigrams.update(zip(words, words[1:]))
        token_count += len(words)
        pair_count += max(0, len(words) - 1)
    return len(unigrams) / max(1, token_count), len(bigrams) / max(1, pair_count)


def classification_metrics(
    predictions: Sequence[str], references: Sequence[str], labels: Sequence[str]
) -> tuple[dict[str, float], list[int]]:
    """Compute exact vocabulary accuracy for generated emotion labels."""
    predicted_labels = [match_emotion(prediction, labels) for prediction in predictions]
    correct = [int(prediction == reference) for prediction, reference in zip(predicted_labels, references)]
    return {"accuracy": float(np.mean(correct))}, correct


def generation_metrics(
    predictions: Sequence[str], references: Sequence[str]
) -> tuple[dict[str, float], list[float]]:
    """Compute ROUGE and diversity metrics for generated responses."""
    rouge = evaluate.load("rouge")
    aggregate = rouge.compute(
        predictions=list(predictions),
        references=list(references),
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )
    per_example = rouge.compute(
        predictions=list(predictions),
        references=list(references),
        rouge_types=["rougeL"],
        use_aggregator=False,
    )["rougeL"]
    distinct1, distinct2 = distinct_n(predictions)
    metrics = {key: float(value) for key, value in aggregate.items()}
    metrics.update({"distinct1": distinct1, "distinct2": distinct2})
    return metrics, [float(value) for value in per_example]


def bertscore_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute rescaled English BERTScore F1."""
    from bert_score import score

    _, _, f1 = score(
        list(predictions),
        list(references),
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return float(f1.mean().item())


def multireference_rouge_l(
    predictions: Sequence[str], reference_sets: Sequence[Sequence[str]]
) -> tuple[float, list[float]]:
    """Compute the maximum ROUGE-L over all references of each dialogue."""
    if len(predictions) != len(reference_sets):
        raise ValueError("Predictions and reference sets must have equal length")
    rouge = evaluate.load("rouge")
    scores: list[float] = []
    for prediction, references in zip(predictions, reference_sets):
        repeated_predictions = [prediction] * len(references)
        values = rouge.compute(
            predictions=repeated_predictions,
            references=list(references),
            rouge_types=["rougeL"],
            use_aggregator=False,
        )["rougeL"]
        scores.append(float(max(values)))
    return float(np.mean(scores)), scores
