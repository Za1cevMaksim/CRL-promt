"""EmpatheticDialogues preparation for classification and generation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as torch_f
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import PreTrainedTokenizerBase

from .config import ExperimentConfig

EMOTION_CONFUSION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"afraid", "terrified", "anxious", "apprehensive"}),
    frozenset({"sad", "disappointed", "devastated", "lonely"}),
    frozenset({"angry", "furious", "annoyed", "disgusted"}),
    frozenset({"surprised", "anticipating", "excited", "joyful"}),
    frozenset({"guilty", "ashamed", "embarrassed"}),
    frozenset({"hopeful", "faithful", "trusting", "prepared", "confident", "proud"}),
    frozenset({"caring", "grateful", "impressed"}),
    frozenset({"sentimental", "nostalgic", "content"}),
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class DatasetBundle:
    """Tokenized splits and string references used during training and evaluation."""

    train: Dataset
    validation: Dataset
    test: Dataset
    reward: Dataset
    validation_references: list[str]
    test_references: list[str]
    reward_references: list[str]
    emotion_labels: list[str]


def _parquet_split(config: ExperimentConfig, split: str) -> Dataset:
    """Load the maintained parquet conversion of EmpatheticDialogues."""
    files = list_repo_files(
        config.dataset_repo,
        repo_type="dataset",
        revision=config.dataset_revision,
    )
    shards = sorted(
        name
        for name in files
        if name.startswith(f"default/{split}/") and name.endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(f"No parquet shards found for split={split}")
    paths = [
        hf_hub_download(
            config.dataset_repo,
            filename=name,
            repo_type="dataset",
            revision=config.dataset_revision,
        )
        for name in shards
    ]
    return load_dataset("parquet", data_files=paths, split="train")


def load_emotion_episodes(config: ExperimentConfig, split: str) -> list[dict[str, str]]:
    """Return one situation and emotion label per conversation."""
    episodes: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in _parquet_split(config, split):
        conversation_id = str(row["conv_id"])
        if conversation_id in seen:
            continue
        seen.add(conversation_id)
        situation = str(row["prompt"]).strip()
        emotion = str(row["context"]).strip()
        if situation and emotion:
            episodes.append({"situation": situation, "emotion": emotion})
    return episodes


def load_dialogue_examples(config: ExperimentConfig, split: str) -> list[dict[str, str]]:
    """Return situation, emotion and the first reply by the second speaker."""
    conversations: dict[str, list[dict[str, Any]]] = {}
    for row in _parquet_split(config, split):
        conversations.setdefault(str(row["conv_id"]), []).append(dict(row))

    examples: list[dict[str, str]] = []
    for conversation_id in sorted(conversations):
        rows = sorted(conversations[conversation_id], key=lambda row: row["utterance_idx"])
        first_speaker = rows[0]["speaker_idx"]
        reply = next((row for row in rows if row["speaker_idx"] != first_speaker), None)
        if reply is None:
            continue
        response = str(reply["utterance"]).replace("_comma_", ",").strip()
        situation = str(rows[0]["prompt"]).strip()
        emotion = str(rows[0]["context"]).strip()
        if response and situation and emotion:
            examples.append(
                {"situation": situation, "emotion": emotion, "response": response}
            )
    return examples


def load_dialogue_reference_sets(
    config: ExperimentConfig, split: str
) -> list[list[str]]:
    """Return every reply by the second speaker for each retained conversation."""
    conversations: dict[str, list[dict[str, Any]]] = {}
    for row in _parquet_split(config, split):
        conversations.setdefault(str(row["conv_id"]), []).append(dict(row))
    reference_sets: list[list[str]] = []
    for conversation_id in sorted(conversations):
        rows = sorted(conversations[conversation_id], key=lambda row: row["utterance_idx"])
        first_speaker = rows[0]["speaker_idx"]
        replies = [
            str(row["utterance"]).replace("_comma_", ",").strip()
            for row in rows
            if row["speaker_idx"] != first_speaker and str(row["utterance"]).strip()
        ]
        if replies:
            reference_sets.append(replies)
    return reference_sets


def make_incoherent_response(text: str, rng: random.Random) -> str | None:
    """Construct a surface-corrupted response by permuting sentences or words."""
    sentences = [part for part in SENTENCE_BOUNDARY.split(text.strip()) if part]
    if len(sentences) == 2:
        return f"{sentences[1]} {sentences[0]}".strip()
    if len(sentences) > 2:
        for _ in range(3):
            rng.shuffle(sentences)
            candidate = " ".join(sentences).strip()
            if candidate != text.strip():
                return candidate
    words = text.split()
    if len(words) >= 2:
        for _ in range(3):
            rng.shuffle(words)
            candidate = " ".join(words)
            if candidate != text.strip():
                return candidate
    return None


def attach_dialogue_negatives(
    rows: list[dict[str, str]], split: str, negative_count: int = 4
) -> int:
    """Attach off-topic and surface-corrupted responses in place."""
    if len(rows) < 2:
        raise ValueError("At least two dialogue examples are required")
    off_topic_rng = random.Random(f"offtopic-{split}")
    incoherent_rng = random.Random(f"incoherent-{split}")
    permutation = list(range(len(rows)))
    off_topic_rng.shuffle(permutation)
    fallbacks = 0

    for index, row in enumerate(rows):
        used = {index}
        candidates: list[int] = []
        offset = 0
        while len(candidates) < negative_count and offset < len(rows) * 2:
            candidate = permutation[(index + offset) % len(rows)]
            offset += 1
            if candidate not in used:
                used.add(candidate)
                candidates.append(candidate)
        while len(candidates) < negative_count:
            candidates.append(candidates[-1])
        for negative_index, candidate in enumerate(candidates, start=1):
            key = "negative_offtopic" if negative_index == 1 else f"negative_offtopic_{negative_index}"
            row[key] = rows[candidate]["response"]

        corrupted = make_incoherent_response(row["response"], incoherent_rng)
        if corrupted is None or corrupted.strip() == row["response"].strip():
            corrupted = row["negative_offtopic"]
            fallbacks += 1
        row["negative_incoherent"] = corrupted
    return fallbacks


def _wrong_label(label: str, labels: Sequence[str], rng: random.Random) -> str:
    """Sample a hard negative label from a semantically close group."""
    for group in EMOTION_CONFUSION_GROUPS:
        if label in group:
            candidates = sorted(group.intersection(labels) - {label})
            if candidates:
                return rng.choice(candidates)
    return rng.choice([candidate for candidate in labels if candidate != label])


def _encode_target(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    maximum: int,
    add_eos: bool,
) -> list[int]:
    """Encode a target and optionally reserve its last token for EOS."""
    target_limit = max(1, maximum - 1) if add_eos else maximum
    prefix = " "
    ids = tokenizer(
        prefix + text,
        add_special_tokens=False,
        truncation=True,
        max_length=target_limit,
    )["input_ids"]
    return ids + [tokenizer.eos_token_id] if add_eos else ids


def prepare_classification_data(
    config: ExperimentConfig, tokenizer: PreTrainedTokenizerBase
) -> DatasetBundle:
    """Create tokenized emotion classification splits and hard negatives."""
    train_all = load_emotion_episodes(config, "train")
    validation_all = load_emotion_episodes(config, "validation")
    test_all = load_emotion_episodes(config, "test")
    labels = sorted({row["emotion"] for row in train_all})
    if len(labels) != 32:
        raise ValueError(f"Expected 32 emotion labels, found {len(labels)}")
    if config.reward_size + config.train_size > len(train_all):
        raise ValueError("Reward and training subsets exceed the training split")

    reward_rows = train_all[: config.reward_size]
    train_rows = train_all[config.reward_size : config.reward_size + config.train_size]
    validation_rows = validation_all[: config.validation_size]
    test_rows = test_all[: config.test_size]
    negative_rng = random.Random(1234)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        output = {key: [] for key in (
            "input_ids", "attention_mask", "labels",
            "negative_input_ids", "negative_attention_mask", "negative_labels",
        )}
        for situation, emotion in zip(batch["situation"], batch["emotion"]):
            source = config.prompt_template.format(situation=situation)
            source_ids = tokenizer(
                source,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_source_length,
            )["input_ids"] or [tokenizer.pad_token_id]
            target_ids = _encode_target(
                tokenizer, emotion, config.max_target_length, add_eos=False
            )
            source_ids = source_ids[: max(1, config.total_length - len(target_ids))]
            full_ids = source_ids + target_ids
            output["input_ids"].append(full_ids)
            output["attention_mask"].append([1] * len(full_ids))
            output["labels"].append([-100] * len(source_ids) + target_ids)

            negative = _wrong_label(emotion, labels, negative_rng)
            negative_ids = _encode_target(
                tokenizer, negative, config.max_target_length, add_eos=False
            )
            negative_full = source_ids + negative_ids
            output["negative_input_ids"].append(negative_full)
            output["negative_attention_mask"].append([1] * len(negative_full))
            output["negative_labels"].append([-100] * len(source_ids) + negative_ids)
        return output

    return _tokenize_bundle(
        train_rows,
        validation_rows,
        test_rows,
        reward_rows,
        tokenize,
        reference_column="emotion",
        labels=labels,
    )


def prepare_generation_data(
    config: ExperimentConfig, tokenizer: PreTrainedTokenizerBase
) -> DatasetBundle:
    """Create tokenized response generation splits and dialogue negatives."""
    train_all = load_dialogue_examples(config, "train")
    validation_all = load_dialogue_examples(config, "validation")
    test_all = load_dialogue_examples(config, "test")
    if config.reward_size + config.train_size > len(train_all):
        raise ValueError("Reward and training subsets exceed the training split")

    reward_rows = [dict(row) for row in train_all[: config.reward_size]]
    train_rows = [
        dict(row)
        for row in train_all[config.reward_size : config.reward_size + config.train_size]
    ]
    validation_rows = [dict(row) for row in validation_all[: config.validation_size]]
    test_rows = [dict(row) for row in test_all[: config.test_size]]
    for split_name, rows in (
        ("reward", reward_rows),
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        attach_dialogue_negatives(rows, split_name, negative_count=4)

    negative_columns = {
        "negative_offtopic": "negative_offtopic",
        "negative_offtopic_2": "negative_offtopic_2",
        "negative_offtopic_3": "negative_offtopic_3",
        "negative_offtopic_4": "negative_offtopic_4",
        "negative_incoherent": "negative_incoherent",
    }

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        prefixes = [""] + [column for column in negative_columns if column in batch]
        output: dict[str, list[list[int]]] = {}
        for prefix in prefixes:
            for suffix in ("input_ids", "attention_mask", "labels"):
                key = suffix if not prefix else f"{prefix}_{suffix}"
                output[key] = []

        for index, (emotion, situation, response) in enumerate(
            zip(batch["emotion"], batch["situation"], batch["response"])
        ):
            source = config.prompt_template.format(emotion=emotion, situation=situation)
            source_ids = tokenizer(
                source,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_source_length,
            )["input_ids"] or [tokenizer.pad_token_id]
            target_ids = _encode_target(
                tokenizer, response, config.max_target_length, add_eos=True
            )
            source_ids = source_ids[: max(1, config.total_length - len(target_ids))]
            full_ids = source_ids + target_ids
            output["input_ids"].append(full_ids)
            output["attention_mask"].append([1] * len(full_ids))
            output["labels"].append([-100] * len(source_ids) + target_ids)

            for column in prefixes[1:]:
                negative_ids = _encode_target(
                    tokenizer, batch[column][index], config.max_target_length, add_eos=True
                )
                negative_ids = negative_ids[: max(1, config.total_length - len(source_ids))]
                full_negative = source_ids + negative_ids
                output[f"{column}_input_ids"].append(full_negative)
                output[f"{column}_attention_mask"].append([1] * len(full_negative))
                output[f"{column}_labels"].append(
                    [-100] * len(source_ids) + negative_ids
                )
        return output

    return _tokenize_bundle(
        train_rows,
        validation_rows,
        test_rows,
        reward_rows,
        tokenize,
        reference_column="response",
        labels=[],
    )


def _tokenize_bundle(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    reward_rows: list[dict[str, str]],
    tokenize: Any,
    reference_column: str,
    labels: list[str],
) -> DatasetBundle:
    """Map one tokenizer over all splits and attach reward indices."""
    raw_splits = [Dataset.from_list(rows) for rows in (
        train_rows, validation_rows, test_rows, reward_rows
    )]
    tokenized = [
        raw.map(tokenize, batched=True, remove_columns=raw.column_names)
        for raw in raw_splits
    ]
    tokenized[3] = tokenized[3].add_column("reference_index", list(range(len(tokenized[3]))))
    for split in tokenized:
        split.set_format("torch")
    return DatasetBundle(
        train=tokenized[0],
        validation=tokenized[1],
        test=tokenized[2],
        reward=tokenized[3],
        validation_references=[row[reference_column] for row in validation_rows],
        test_references=[row[reference_column] for row in test_rows],
        reward_references=[row[reference_column] for row in reward_rows],
        emotion_labels=labels,
    )


class CausalLMCollator:
    """Left-pad positive and negative causal language-model sequences."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, legacy_padding: bool = True
    ) -> None:
        self.tokenizer = tokenizer
        self.legacy_padding = legacy_padding

    @staticmethod
    def _tensor(value: Any) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        return tensor.squeeze(0) if tensor.ndim > 1 else tensor

    @staticmethod
    def _pad(sequences: Sequence[torch.Tensor], value: int) -> torch.Tensor:
        maximum = max(sequence.size(0) for sequence in sequences)
        return torch.stack(
            [
                sequence
                if sequence.size(0) == maximum
                else torch_f.pad(sequence, (maximum - sequence.size(0), 0), value=value)
                for sequence in sequences
            ]
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of tokenized examples."""
        input_padding = 0 if self.legacy_padding else int(self.tokenizer.pad_token_id)
        output: dict[str, torch.Tensor] = {
            "input_ids": self._pad(
                [self._tensor(item["input_ids"]) for item in features], input_padding
            ),
            "attention_mask": self._pad(
                [self._tensor(item["attention_mask"]) for item in features], 0
            ),
            "labels": self._pad(
                [self._tensor(item["labels"]) for item in features], -100
            ),
        }
        prefixes = sorted(
            key[: -len("_input_ids")]
            for key in features[0]
            if key.endswith("_input_ids") and key != "input_ids"
        )
        for prefix in prefixes:
            output[f"{prefix}_input_ids"] = self._pad(
                [self._tensor(item[f"{prefix}_input_ids"]) for item in features],
                int(self.tokenizer.pad_token_id),
            )
            output[f"{prefix}_attention_mask"] = self._pad(
                [self._tensor(item[f"{prefix}_attention_mask"]) for item in features], 0
            )
            output[f"{prefix}_labels"] = self._pad(
                [self._tensor(item[f"{prefix}_labels"]) for item in features], -100
            )
        if "reference_index" in features[0]:
            output["reference_index"] = torch.tensor(
                [int(item["reference_index"]) for item in features], dtype=torch.long
            )
        return output


def strip_target_tokens(
    batch: dict[str, torch.Tensor], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove teacher-forced targets before generation and left-pad sources."""
    labels = batch["labels"]
    target_lengths = (labels != -100).sum(dim=1)
    sequence_length = batch["input_ids"].size(1)
    sources = [
        batch["input_ids"][index, : sequence_length - int(target_lengths[index])]
        for index in range(batch["input_ids"].size(0))
    ]
    masks = [
        batch["attention_mask"][index, : sequence_length - int(target_lengths[index])]
        for index in range(batch["attention_mask"].size(0))
    ]
    maximum = max(source.size(0) for source in sources)
    source_batch = torch.stack(
        [
            source
            if source.size(0) == maximum
            else torch_f.pad(source, (maximum - source.size(0), 0), value=pad_token_id)
            for source in sources
        ]
    )
    mask_batch = torch.stack(
        [
            mask
            if mask.size(0) == maximum
            else torch_f.pad(mask, (maximum - mask.size(0), 0), value=0)
            for mask in masks
        ]
    )
    return source_batch, mask_batch
