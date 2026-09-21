"""Typed configuration objects and YAML loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

TaskKind = Literal["emotion_classification", "response_generation"]
ContrastMode = Literal[
    "none",
    "mask",
    "wrong_label",
    "offtopic",
    "offtopic4",
    "offtopic_grad",
    "incoherent",
]
RLMode = Literal["none", "reinforce", "grpo"]


@dataclass(slots=True)
class ExperimentConfig:
    """Complete configuration for one training run."""

    name: str
    task: TaskKind
    model_name: str = "Qwen/Qwen2.5-3B"
    dataset_repo: str = "empathetic_dialogues"
    dataset_revision: str = "refs/convert/parquet"
    output_dir: str = "outputs/run"
    seed: int = 42

    train_size: int = 6000
    validation_size: int = 200
    test_size: int = 500
    reward_size: int = 600
    max_source_length: int = 128
    max_target_length: int = 48
    max_new_tokens: int = 48

    num_virtual_tokens: int = 20
    prefix_projection: bool = True
    initialization_candidates: int = 4
    initialization_probe_steps: int = 200

    epochs: int = 4
    batch_size: int = 8
    evaluation_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    warmup_steps: int = 200
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True

    contrast_mode: ContrastMode = "none"
    contrast_weight: float = 0.0
    contrast_temperature: float = 0.5
    contrast_dropout: float = 0.1
    contrast_negatives: int = 1
    negative_branch_gradient: bool = False

    rl_mode: RLMode = "none"
    rl_weight: float = 2e-5
    automatic_rl_weight: bool = True
    rl_target_fraction: float = 0.3
    perturbation_std: float = 0.02
    rl_interval: int = 100
    rl_batches: int = 8
    reward_baseline_decay: float = 0.9
    reward_sampling: bool = False
    reward_temperature: float = 0.7
    reward_top_p: float = 0.9

    divergence_ce_threshold: float = 8.0
    divergence_guard_start_step: int = 100
    legacy_training_padding: bool = True
    trust_remote_code: bool = True

    grpo_group_size: int = 4
    grpo_batch_size: int = 2
    grpo_interval: int = 30
    grpo_max_new_tokens: int = 48

    @property
    def prompt_template(self) -> str:
        """Return the input template used by the selected task."""
        if self.task == "emotion_classification":
            return "Situation: {situation}\nEmotion:"
        return "Emotion: {emotion}\nSituation: {situation}\nResponse:"

    @property
    def total_length(self) -> int:
        """Maximum source plus target sequence length."""
        return self.max_source_length + self.max_target_length

    def with_updates(self, **updates: Any) -> "ExperimentConfig":
        """Return a copy with selected fields replaced."""
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to JSON compatible values."""
        return asdict(self)

    def validate(self) -> None:
        """Reject internally inconsistent configurations early."""
        if self.train_size <= 0 or self.validation_size <= 0 or self.test_size <= 0:
            raise ValueError("Dataset sizes must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Training schedule must be positive")
        if self.contrast_mode == "none" and self.contrast_weight != 0:
            raise ValueError("contrast_weight requires a contrast mode")
        if self.rl_mode == "none" and self.automatic_rl_weight:
            self.automatic_rl_weight = False
        if self.task == "emotion_classification" and self.max_new_tokens > 8:
            raise ValueError("Emotion labels require at most eight generated tokens")
        if self.contrast_mode == "offtopic4" and self.contrast_negatives != 4:
            raise ValueError("offtopic4 requires contrast_negatives=4")


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Load one experiment configuration from a YAML file."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    if overrides:
        raw.update(overrides)
    known = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {unknown}")
    config = ExperimentConfig(**raw)
    config.validate()
    return config

