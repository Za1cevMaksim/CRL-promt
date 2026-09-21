"""Qwen loading and prefix-tuning initialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
from peft.peft_model import PeftModelForCausalLM
from transformers import AutoModelForCausalLM

from .config import ExperimentConfig

PrefixState: TypeAlias = dict[str, torch.Tensor]


def load_prefix_model(config: ExperimentConfig) -> PeftModelForCausalLM:
    """Load a frozen causal LM and attach a trainable prefix encoder."""
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    base_model.config.use_cache = True
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config.num_virtual_tokens,
        prefix_projection=config.prefix_projection,
        inference_mode=False,
    )
    model = get_peft_model(base_model, prefix_config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.enable_input_require_grads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def prefix_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable prefix encoder parameters."""
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "prompt_encoder" in name
    ]
    if not parameters:
        raise RuntimeError("No trainable prompt_encoder parameters were found")
    return parameters


def snapshot_prefix(model: torch.nn.Module) -> PrefixState:
    """Copy trainable prefix parameters to CPU memory."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in prefix_parameters(model)
    }


def restore_prefix(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Restore a prefix snapshot without modifying frozen model weights."""
    missing: list[str] = []
    with torch.no_grad():
        for name, parameter in prefix_parameters(model):
            if name not in state:
                missing.append(name)
                continue
            parameter.data.copy_(state[name].to(parameter.device, dtype=parameter.dtype))
    if missing:
        raise KeyError(f"Prefix state is missing parameters: {missing}")

