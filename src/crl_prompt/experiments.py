"""Resume-safe training and paired experiment orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrainingArguments
from transformers import AutoModelForCausalLM

from .config import ExperimentConfig, load_config
from .data import (
    CausalLMCollator,
    DatasetBundle,
    prepare_classification_data,
    prepare_generation_data,
)
from .evaluation import evaluate_model
from .metrics import bertscore_f1, multireference_rouge_l
from .modeling import PrefixState, load_prefix_model, restore_prefix, snapshot_prefix
from .trainer import BestGenerationMetricCallback, CRLTrainer, GRPOTrainer
from .utils import clear_device_cache, configure_runtime, load_json, save_json


def load_tokenizer(config: ExperimentConfig) -> PreTrainedTokenizerBase:
    """Load a tokenizer with left padding and an explicit pad token."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def prepare_data(
    config: ExperimentConfig, tokenizer: PreTrainedTokenizerBase
) -> DatasetBundle:
    """Dispatch to the selected task-specific preprocessing pipeline."""
    if config.task == "emotion_classification":
        return prepare_classification_data(config, tokenizer)
    return prepare_generation_data(config, tokenizer)


def _validation_ce(
    model: torch.nn.Module,
    dataset: Dataset,
    collator: CausalLMCollator,
    batch_size: int,
) -> float:
    """Compute token-weighted validation cross-entropy."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    token_count = 0
    with torch.no_grad():
        for batch in loader:
            model_batch = {
                key: value.to(model.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            output = model(**model_batch)
            supervised = int((model_batch["labels"] != -100).sum())
            total_loss += float(output.loss) * supervised
            token_count += supervised
    if was_training:
        model.train()
    return total_loss / max(1, token_count)


def select_common_start(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    collator: CausalLMCollator,
    config: ExperimentConfig,
) -> PrefixState:
    """Probe several prefix initializations and return the lowest-CE candidate."""
    if config.initialization_candidates <= 0:
        return snapshot_prefix(model)
    prompt_encoder = getattr(model, "prompt_encoder", None)
    if prompt_encoder is None:
        raise RuntimeError("The PEFT model has no prompt_encoder")

    candidates: list[tuple[float, PrefixState]] = []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for candidate_index in range(config.initialization_candidates):
        candidate_seed = config.seed * 1000 + candidate_index
        configure_runtime(candidate_seed)
        for module in prompt_encoder.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        for parameter in trainable:
            parameter.data = parameter.data.float()

        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        loader = DataLoader(
            bundle.train,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=torch.Generator().manual_seed(candidate_seed),
        )
        model.train()
        for step, batch in enumerate(loader):
            if step >= config.initialization_probe_steps:
                break
            model_batch = {
                key: value.to(model.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(**model_batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
        validation_ce = _validation_ce(
            model, bundle.validation, collator, config.batch_size
        )
        candidates.append((validation_ce, snapshot_prefix(model)))

    candidates.sort(key=lambda item: item[0])
    best_state = candidates[0][1]
    restore_prefix(model, best_state)
    return best_state


def _training_arguments(config: ExperimentConfig) -> TrainingArguments:
    """Translate the project configuration to Transformers arguments."""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.evaluation_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config.seed,
        prediction_loss_only=True,
        max_grad_norm=config.max_grad_norm,
    )


def run_experiment(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    tokenizer: PreTrainedTokenizerBase,
    initial_state: PrefixState,
    force: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one configuration, resuming from results when possible."""
    output_dir = Path(config.output_dir)
    result_path = output_dir / "results.json"
    if result_path.exists() and not force:
        return load_json(result_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(config.seed)
    model = load_prefix_model(config)
    restore_prefix(model, initial_state)
    collator = CausalLMCollator(tokenizer, config.legacy_training_padding)
    validation_callback = BestGenerationMetricCallback(
        config,
        bundle.validation,
        bundle.validation_references,
        bundle.emotion_labels,
        tokenizer,
        collator,
    )
    trainer_class = GRPOTrainer if config.rl_mode == "grpo" else CRLTrainer
    trainer = trainer_class(
        model=model,
        args=_training_arguments(config),
        experiment_config=config,
        reward_dataset=bundle.reward,
        reward_references=bundle.reward_references,
        emotion_labels=bundle.emotion_labels,
        tokenizer=tokenizer,
        train_dataset=bundle.train,
        eval_dataset=bundle.validation,
        data_collator=collator,
        callbacks=[validation_callback],
    )

    started = time.time()
    trainer.train()
    elapsed_minutes = (time.time() - started) / 60.0
    best_path = output_dir / "best_prefix.pt"
    if best_path.exists():
        restore_prefix(model, torch.load(best_path, map_location="cpu", weights_only=True))
    evaluation = evaluate_model(
        model,
        bundle.test,
        bundle.test_references,
        bundle.emotion_labels,
        tokenizer,
        config,
        collator,
    )
    result = {
        "experiment": config.name,
        "config": config.to_dict(),
        "test": evaluation,
        "validation": {
            "best": validation_callback.best,
            "best_epoch": validation_callback.best_epoch,
            "history": validation_callback.history,
        },
        "reward_history": trainer.reward_history,
        "training_time_minutes": elapsed_minutes,
    }
    save_json(result, result_path)
    del trainer, model
    clear_device_cache()
    return result


def run_paired_suite(
    config_paths: Sequence[str | Path],
    seeds: Sequence[int],
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run configurations from a shared probed prefix for every seed."""
    return run_paired_configs(
        [load_config(path) for path in config_paths], seeds, output_root, force
    )


def run_paired_configs(
    base_configs: Sequence[ExperimentConfig],
    seeds: Sequence[int],
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run already constructed configurations from one common start per seed."""
    if not base_configs:
        raise ValueError("At least one configuration is required")
    signature = {
        (config.task, config.model_name, config.train_size) for config in base_configs
    }
    if len(signature) != 1:
        raise ValueError("A paired suite must share task, model and train size")

    output_root = Path(output_root)
    summary_rows: list[dict[str, Any]] = []
    for seed in seeds:
        reference = base_configs[0].with_updates(seed=seed)
        tokenizer = load_tokenizer(reference)
        bundle = prepare_data(reference, tokenizer)
        seed_root = output_root / f"seed_{seed}"
        shared_path = seed_root / "shared_start.pt"
        if shared_path.exists() and not force:
            shared_state = torch.load(shared_path, map_location="cpu", weights_only=True)
        else:
            configure_runtime(seed)
            probe_model = load_prefix_model(reference)
            collator = CausalLMCollator(tokenizer, reference.legacy_training_padding)
            shared_state = select_common_start(probe_model, bundle, collator, reference)
            seed_root.mkdir(parents=True, exist_ok=True)
            torch.save(shared_state, shared_path)
            del probe_model
            clear_device_cache()

        row: dict[str, Any] = {"seed": seed}
        for base_config in base_configs:
            run_config = base_config.with_updates(
                seed=seed,
                output_dir=str(seed_root / base_config.name),
            )
            result = run_experiment(
                run_config,
                bundle,
                tokenizer,
                shared_state,
                force=force,
            )
            row[base_config.name] = result["test"]["metrics"]
        summary_rows.append(row)
        save_json({"rows": summary_rows}, output_root / "summary.json")
    return {"rows": summary_rows}


def run_zero_shot_evaluation(
    config: ExperimentConfig, output_path: str | Path, force: bool = False
) -> dict[str, Any]:
    """Evaluate a base model without attaching a trainable prefix."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return load_json(output_path)
    configure_runtime(config.seed)
    tokenizer = load_tokenizer(config)
    bundle = prepare_data(config, tokenizer)
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    collator = CausalLMCollator(tokenizer, config.legacy_training_padding)
    evaluation = evaluate_model(
        model,
        bundle.test,
        bundle.test_references,
        bundle.emotion_labels,
        tokenizer,
        config,
        collator,
    )
    result = {"config": config.to_dict(), "test": evaluation}
    save_json(result, output_path)
    del model
    clear_device_cache()
    return result


def reevaluate_output_tree(
    output_root: str | Path,
    destination_root: str | Path,
    force: bool = False,
) -> None:
    """Re-evaluate every saved best prefix on the larger article test subsets."""
    output_root = Path(output_root)
    destination_root = Path(destination_root)
    for checkpoint in sorted(output_root.rglob("best_prefix.pt")):
        result_path = checkpoint.parent / "results.json"
        if not result_path.exists():
            continue
        previous = load_json(result_path)
        config = ExperimentConfig(**previous["config"])
        test_size = 2542 if config.task == "emotion_classification" else 1000
        config = config.with_updates(test_size=test_size)
        relative = checkpoint.parent.relative_to(output_root)
        destination = destination_root / relative / "results.json"
        if destination.exists() and not force:
            continue
        tokenizer = load_tokenizer(config)
        bundle = prepare_data(config, tokenizer)
        model = load_prefix_model(config)
        restore_prefix(
            model,
            torch.load(checkpoint, map_location="cpu", weights_only=True),
        )
        collator = CausalLMCollator(tokenizer, config.legacy_training_padding)
        evaluation = evaluate_model(
            model,
            bundle.test,
            bundle.test_references,
            bundle.emotion_labels,
            tokenizer,
            config,
            collator,
        )
        if config.task == "response_generation":
            evaluation["metrics"]["bertscore_f1"] = bertscore_f1(
                evaluation["predictions"], bundle.test_references
            )
            from .data import load_dialogue_reference_sets

            reference_sets = load_dialogue_reference_sets(config, "test")[:test_size]
            multireference, values = multireference_rouge_l(
                evaluation["predictions"], reference_sets
            )
            evaluation["metrics"]["multireference_rougeL"] = multireference
            evaluation["per_example_multireference_rougeL"] = values
        save_json({"config": config.to_dict(), "test": evaluation}, destination)
        del model
        clear_device_cache()
