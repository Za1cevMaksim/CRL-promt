"""Trainer implementation for CE, contrastive and REINFORCE objectives."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
import evaluate
from datasets import Dataset
from torch.distributions import Normal
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback

from .config import ExperimentConfig
from .data import CausalLMCollator, strip_target_tokens
from .evaluation import generate_predictions
from .metrics import classification_metrics, generation_metrics
from .modeling import prefix_parameters, restore_prefix, snapshot_prefix
from .reward import generation_reward
from .utils import save_json


class CRLTrainer(Trainer):
    """Optimize a prefix with CE and optional CRL-Prompt auxiliary losses."""

    def __init__(
        self,
        *args: Any,
        experiment_config: ExperimentConfig,
        reward_dataset: Dataset,
        reward_references: list[str],
        emotion_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        **kwargs: Any,
    ) -> None:
        self.experiment_config = experiment_config
        self.reward_dataset = reward_dataset
        self.reward_references = reward_references
        self.emotion_labels = emotion_labels
        self.experiment_tokenizer = tokenizer
        self.reward_history: list[dict[str, float | int]] = []
        self._reward_baseline = 0.0
        self._ce_ema: float | None = None
        self._last_rl_step = -1
        self._last_log_step = -1
        self._rl_weight_calibrated = False
        self._divergence_reported = False
        super().__init__(*args, **kwargs)
        self.prefix_parameter_list = prefix_parameters(self.model)
        embedding_parameters = [
            (name, parameter)
            for name, parameter in self.prefix_parameter_list
            if "embedding" in name
        ]
        self.mask_parameter_list = embedding_parameters or self.prefix_parameter_list
        self._reward_loader = DataLoader(
            reward_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
        )
        self._reward_iterator = iter(self._reward_loader)

    @staticmethod
    def _model_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Discard auxiliary columns before calling the language model."""
        return {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }

    @staticmethod
    def _pool_hidden(hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Average final-layer states over supervised target tokens."""
        if hidden.size(1) != labels.size(1):
            hidden = hidden[:, -labels.size(1) :, :]
        target_mask = (labels != -100).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(min=1)
        return torch_f.normalize(pooled.float(), dim=-1)

    def _next_reward_batch(self) -> dict[str, torch.Tensor]:
        """Cycle over the fixed held-out reward split."""
        try:
            return next(self._reward_iterator)
        except StopIteration:
            self._reward_iterator = iter(self._reward_loader)
            return next(self._reward_iterator)

    def _negative_prefixes(self) -> list[str]:
        """Map each contrast mode to tokenized negative columns."""
        mode = self.experiment_config.contrast_mode
        return {
            "wrong_label": ["negative"],
            "offtopic": ["negative_offtopic"],
            "offtopic_grad": ["negative_offtopic"],
            "offtopic4": [
                "negative_offtopic",
                "negative_offtopic_2",
                "negative_offtopic_3",
                "negative_offtopic_4",
            ],
            "incoherent": ["negative_incoherent"],
        }.get(mode, [])

    def _contrastive_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the repulsion form of the CRL-Prompt contrastive loss."""
        config = self.experiment_config
        negative_states: list[torch.Tensor] = []
        prefixes = self._negative_prefixes()
        if prefixes:
            for prefix in prefixes:
                forward_inputs = {
                    "input_ids": inputs[f"{prefix}_input_ids"],
                    "attention_mask": inputs[f"{prefix}_attention_mask"],
                }
                if config.negative_branch_gradient:
                    output = model(**forward_inputs, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        output = model(**forward_inputs, output_hidden_states=True)
                negative_states.append(
                    self._pool_hidden(
                        output.hidden_states[-1], inputs[f"{prefix}_labels"]
                    )
                )
        else:
            clean_inputs = {
                key: value
                for key, value in self._model_inputs(inputs).items()
                if key != "labels"
            }
            original = {
                name: parameter.data.clone()
                for name, parameter in self.mask_parameter_list
            }
            try:
                for _ in range(config.contrast_negatives):
                    for name, parameter in self.mask_parameter_list:
                        mask = (
                            torch.rand_like(parameter.float()) > config.contrast_dropout
                        ).to(parameter.dtype)
                        parameter.data.copy_((original[name] * mask).to(parameter.dtype))
                    with torch.no_grad():
                        output = model(**clean_inputs, output_hidden_states=True)
                    negative_states.append(
                        self._pool_hidden(output.hidden_states[-1], inputs["labels"])
                    )
                    for name, parameter in self.mask_parameter_list:
                        parameter.data.copy_(original[name])
            finally:
                for name, parameter in self.mask_parameter_list:
                    parameter.data.copy_(original[name])

        stacked = torch.stack(negative_states, dim=1)
        similarities = torch.einsum("bd,bkd->bk", positive, stacked)
        temperature = config.contrast_temperature
        return torch.log1p(
            torch.exp((similarities - 1.0) / temperature).sum(dim=1)
        ).mean()

    def _reinforce_loss(self, model: torch.nn.Module) -> tuple[torch.Tensor, float]:
        """Estimate a parameter-space REINFORCE update from one perturbation."""
        config = self.experiment_config
        log_probabilities: list[torch.Tensor] = []
        noisy_values: dict[str, torch.Tensor] = {}
        for name, parameter in self.prefix_parameter_list:
            parameter_fp32 = parameter.float()
            noise = torch.randn_like(parameter_fp32) * config.perturbation_std
            noisy = (parameter_fp32.detach() + noise).detach()
            log_probabilities.append(
                Normal(parameter_fp32, config.perturbation_std).log_prob(noisy).sum()
            )
            noisy_values[name] = noisy

        original = snapshot_prefix(model)
        rewards: list[float] = []
        try:
            with torch.no_grad():
                for name, parameter in self.prefix_parameter_list:
                    parameter.data.copy_(noisy_values[name].to(parameter.dtype))
            for _ in range(config.rl_batches):
                batch = self._next_reward_batch()
                reference_indices = batch.pop("reference_index")
                source_ids, source_mask = strip_target_tokens(
                    batch, self.experiment_tokenizer.pad_token_id
                )
                references = [
                    self.reward_references[index]
                    for index in reference_indices.tolist()
                ]
                reward, _ = generation_reward(
                    model,
                    source_ids.to(model.device),
                    source_mask.to(model.device),
                    references,
                    self.experiment_tokenizer,
                    config,
                    self.emotion_labels,
                )
                rewards.append(reward)
        finally:
            restore_prefix(model, original)

        reward_value = float(np.mean(rewards))
        advantage = reward_value - self._reward_baseline
        decay = config.reward_baseline_decay
        self._reward_baseline = decay * self._reward_baseline + (1.0 - decay) * reward_value
        raw_loss = -advantage * torch.stack(log_probabilities).sum()

        if config.automatic_rl_weight and not self._rl_weight_calibrated:
            magnitude = abs(float(raw_loss.detach()))
            if magnitude >= 1e-6 and self._ce_ema is not None:
                config.rl_weight = float(
                    np.clip(config.rl_target_fraction * self._ce_ema / magnitude, 1e-8, 1.0)
                )
                self._rl_weight_calibrated = True
        self.reward_history.append(
            {"step": int(self.state.global_step), "reward": reward_value}
        )
        return config.rl_weight * raw_loss, reward_value

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Combine CE, contrastive and parameter-space RL objectives."""
        config = self.experiment_config
        use_contrast = config.contrast_mode != "none" and config.contrast_weight != 0
        model_inputs = self._model_inputs(inputs)
        outputs = model(**model_inputs, output_hidden_states=use_contrast)
        ce_loss = outputs.loss
        ce_value = float(ce_loss.detach())
        self._ce_ema = ce_value if self._ce_ema is None else 0.95 * self._ce_ema + 0.05 * ce_value

        if (
            self.state.global_step >= config.divergence_guard_start_step
            and self._ce_ema > config.divergence_ce_threshold
            and not self._divergence_reported
        ):
            self._divergence_reported = True
            self.control.should_training_stop = True

        contrastive_loss = ce_loss.new_zeros(())
        if use_contrast:
            positive = self._pool_hidden(outputs.hidden_states[-1], model_inputs["labels"])
            contrastive_loss = self._contrastive_loss(model, inputs, positive)

        reinforcement_loss = ce_loss.new_zeros(())
        reward = 0.0
        if (
            config.rl_mode == "reinforce"
            and self.state.global_step % config.rl_interval == 0
            and self._last_rl_step != self.state.global_step
        ):
            self._last_rl_step = self.state.global_step
            reinforcement_loss, reward = self._reinforce_loss(model)

        if self.state.global_step % 10 == 0 and self._last_log_step != self.state.global_step:
            self._last_log_step = self.state.global_step
            print(
                f"step={self.state.global_step} ce_ema={self._ce_ema:.4f} "
                f"contrast={float(contrastive_loss.detach()):.4f} "
                f"rl={float(reinforcement_loss.detach()):.4f} reward={reward:.4f}"
            )

        loss = ce_loss + config.contrast_weight * contrastive_loss + reinforcement_loss
        if not getattr(self, "model_accepts_loss_kwargs", False):
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


class GRPOTrainer(CRLTrainer):
    """Replace periodic CE steps with group-relative token policy updates."""

    def _grpo_loss(self, model: torch.nn.Module) -> tuple[torch.Tensor, float]:
        """Sample response groups and optimize length-normalized token log-probability."""
        config = self.experiment_config
        batch = self._next_reward_batch()
        reference_indices = batch.pop("reference_index")
        source_ids, source_mask = strip_target_tokens(
            batch, self.experiment_tokenizer.pad_token_id
        )
        source_ids = source_ids[: config.grpo_batch_size].to(model.device)
        source_mask = source_mask[: config.grpo_batch_size].to(model.device)
        references = [
            self.reward_references[index]
            for index in reference_indices.tolist()[: config.grpo_batch_size]
        ]
        batch_size = source_ids.size(0)
        group_size = config.grpo_group_size
        expanded_ids = source_ids.repeat_interleave(group_size, dim=0)
        expanded_mask = source_mask.repeat_interleave(group_size, dim=0)

        model.eval()
        old_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=expanded_ids,
                    attention_mask=expanded_mask,
                    max_new_tokens=config.grpo_max_new_tokens,
                    do_sample=True,
                    temperature=config.reward_temperature,
                    top_p=config.reward_top_p,
                    pad_token_id=self.experiment_tokenizer.pad_token_id,
                    eos_token_id=self.experiment_tokenizer.eos_token_id,
                )
        finally:
            model.config.use_cache = old_cache
            model.train()

        new_tokens = generated[:, expanded_ids.size(1) :]
        texts = self.experiment_tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        repeated_references = [
            references[index // group_size] for index in range(batch_size * group_size)
        ]
        rouge = evaluate.load("rouge")
        rewards = np.asarray(
            rouge.compute(
                predictions=texts,
                references=repeated_references,
                rouge_types=["rougeL"],
                use_aggregator=False,
            )["rougeL"],
            dtype=np.float32,
        )
        advantages = torch.zeros(batch_size * group_size, device=model.device)
        group_standard_deviations: list[float] = []
        for group_index in range(batch_size):
            group = rewards[group_index * group_size : (group_index + 1) * group_size]
            group_standard_deviations.append(float(group.std()))
            centered = group - group.mean()
            advantages[group_index * group_size : (group_index + 1) * group_size] = (
                torch.from_numpy(centered).to(model.device)
            )

        eos_id = self.experiment_tokenizer.eos_token_id
        has_eos = (new_tokens == eos_id).any(dim=1)
        eos_positions = (new_tokens == eos_id).float().argmax(dim=1)
        lengths = torch.where(
            has_eos,
            eos_positions + 1,
            torch.full_like(eos_positions, new_tokens.size(1)),
        )
        full_ids = torch.cat([expanded_ids, new_tokens], dim=1)
        full_mask = torch.cat([expanded_mask, torch.ones_like(new_tokens)], dim=1)

        token_log_probabilities: list[torch.Tensor] = []
        chunk_size = 8
        for start in range(0, full_ids.size(0), chunk_size):
            ids = full_ids[start : start + chunk_size]
            mask = full_mask[start : start + chunk_size]
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
            negative_log_probability = torch_f.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="none",
            ).view(ids.size(0), -1)
            token_log_probabilities.append(-negative_log_probability)

        token_log_probability = torch.cat(token_log_probabilities, dim=0)
        generated_length = new_tokens.size(1)
        positions = torch.arange(generated_length, device=model.device).unsqueeze(0)
        generated_mask = positions < lengths.unsqueeze(1)
        full_generated_mask = torch.zeros_like(token_log_probability, dtype=torch.bool)
        full_generated_mask[:, -generated_length:] = generated_mask
        sequence_log_probability = (
            token_log_probability * full_generated_mask
        ).sum(dim=1) / full_generated_mask.sum(dim=1).clamp(min=1)
        raw_loss = -(advantages.detach() * sequence_log_probability).mean()

        if config.automatic_rl_weight and not self._rl_weight_calibrated:
            magnitude = abs(float(raw_loss.detach()))
            if magnitude >= 1e-6 and self._ce_ema is not None:
                config.rl_weight = float(
                    np.clip(config.rl_target_fraction * self._ce_ema / magnitude, 1e-8, 1.0)
                )
                self._rl_weight_calibrated = True
        reward = float(rewards.mean())
        self.reward_history.append(
            {
                "step": int(self.state.global_step),
                "reward": reward,
                "group_std": float(np.mean(group_standard_deviations)),
            }
        )
        return config.rl_weight * raw_loss, reward

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Use GRPO on scheduled events and CE on all remaining steps."""
        config = self.experiment_config
        scheduled = (
            model.training
            and self.state.global_step % config.grpo_interval == 0
            and self._last_rl_step != self.state.global_step
        )
        if scheduled:
            self._last_rl_step = self.state.global_step
            loss, reward = self._grpo_loss(model)
            if self.state.global_step % 10 == 0:
                print(
                    f"step={self.state.global_step} grpo={float(loss.detach()):.4f} "
                    f"reward={reward:.4f}"
                )
            return (loss, None) if return_outputs else loss
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )


class BestGenerationMetricCallback(TrainerCallback):
    """Save the prefix from the best source-only validation epoch."""

    def __init__(
        self,
        config: ExperimentConfig,
        dataset: Dataset,
        references: list[str],
        emotion_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        collator: CausalLMCollator,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.references = references
        self.emotion_labels = emotion_labels
        self.tokenizer = tokenizer
        self.collator = collator
        self.best = -math.inf
        self.best_epoch: float | None = None
        self.history: list[dict[str, float]] = []

    @torch.no_grad()
    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: torch.nn.Module | None = None,
        **_: Any,
    ) -> None:
        """Evaluate one epoch and persist the best trainable prefix."""
        if model is None:
            return
        was_training = model.training
        predictions = generate_predictions(
            model,
            self.dataset,
            self.tokenizer,
            self.config,
            self.collator,
        )
        if self.config.task == "emotion_classification":
            metrics, _ = classification_metrics(
                predictions, self.references, self.emotion_labels
            )
            value = metrics["accuracy"]
            metric_name = "accuracy"
        else:
            metrics, _ = generation_metrics(predictions, self.references)
            value = metrics["rougeL"]
            metric_name = "rougeL"
        if was_training:
            model.train()

        epoch = float(state.epoch or 0.0)
        self.history.append({"epoch": epoch, "value": value})
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if value > self.best:
            self.best = value
            self.best_epoch = epoch
            torch.save(snapshot_prefix(model), output_dir / "best_prefix.pt")
        save_json(
            {
                "metric": metric_name,
                "history": self.history,
                "best": self.best,
                "best_epoch": self.best_epoch,
            },
            output_dir / "validation_history.json",
        )
