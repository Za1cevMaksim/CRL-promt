"""Generation rewards used by REINFORCE and GRPO experiments."""

from __future__ import annotations

from collections.abc import Sequence

import evaluate
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from .config import ExperimentConfig
from .metrics import match_emotion


@torch.no_grad()
def generation_reward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    references: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    emotion_labels: Sequence[str],
) -> tuple[float, list[str]]:
    """Generate continuations and compute task-level reward."""
    was_training = model.training
    model.eval()
    old_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    generation_arguments = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.reward_sampling,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.reward_sampling:
        generation_arguments.update(
            temperature=config.reward_temperature,
            top_p=config.reward_top_p,
        )
    try:
        generated = model.generate(**generation_arguments)
    finally:
        model.config.use_cache = old_cache
        if was_training:
            model.train()
    continuation = generated[:, input_ids.shape[1] :]
    texts = [text.strip() or " " for text in tokenizer.batch_decode(
        continuation, skip_special_tokens=True
    )]
    if config.task == "emotion_classification":
        reward = np.mean(
            [match_emotion(text, emotion_labels) == reference for text, reference in zip(texts, references)]
        )
        return float(reward), texts
    rouge = evaluate.load("rouge")
    score = rouge.compute(
        predictions=texts,
        references=list(references),
        rouge_types=["rougeL"],
    )["rougeL"]
    return float(score), texts

