#!/usr/bin/env python3
"""Command-line entry point for reproducing the experiments in the paper."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if TYPE_CHECKING:
    from crl_prompt.config import ExperimentConfig

CONFIG_DIR = PROJECT_ROOT / "configs"

CLASSIFICATION_CONFIGS = [
    CONFIG_DIR / "classification_mle.yaml",
    CONFIG_DIR / "classification_mask.yaml",
    CONFIG_DIR / "classification_wrong_label.yaml",
    CONFIG_DIR / "classification_reinforce.yaml",
    CONFIG_DIR / "classification_full.yaml",
]
GENERATION_CONFIGS = [
    CONFIG_DIR / "generation_mle.yaml",
    CONFIG_DIR / "generation_mask.yaml",
    CONFIG_DIR / "generation_offtopic.yaml",
    CONFIG_DIR / "generation_incoherent.yaml",
    CONFIG_DIR / "generation_reinforce.yaml",
]


def _seed_values(raw: Sequence[int] | None, default: Sequence[int]) -> list[int]:
    """Use explicit seeds when provided and a protocol default otherwise."""
    return list(raw) if raw else list(default)


def run_classification(args: argparse.Namespace) -> None:
    """Run the 3B emotion classification comparison."""
    from crl_prompt.experiments import run_paired_suite

    run_paired_suite(
        CLASSIFICATION_CONFIGS,
        _seed_values(args.seeds, range(42, 49)),
        PROJECT_ROOT / "outputs" / "classification",
        force=args.force,
    )


def run_generation(args: argparse.Namespace) -> None:
    """Run the 3B response generation comparison."""
    from crl_prompt.experiments import run_paired_suite

    run_paired_suite(
        GENERATION_CONFIGS,
        _seed_values(args.seeds, range(42, 49)),
        PROJECT_ROOT / "outputs" / "generation",
        force=args.force,
    )


def run_grpo(args: argparse.Namespace) -> None:
    """Run the GRPO comparison from the same starts as the generation baseline."""
    from crl_prompt.experiments import run_paired_suite

    run_paired_suite(
        [CONFIG_DIR / "generation_mle.yaml", CONFIG_DIR / "generation_grpo.yaml"],
        _seed_values(args.seeds, (42, 43, 44)),
        PROJECT_ROOT / "outputs" / "grpo",
        force=args.force,
    )


def _capacity_variants(
    model_name: str, train_size: int
) -> list[ExperimentConfig]:
    """Create the MLE, mask and REINFORCE configurations for one sweep point."""
    from crl_prompt.config import load_config

    configs = [
        load_config(CONFIG_DIR / "classification_mle.yaml"),
        load_config(CONFIG_DIR / "classification_mask.yaml"),
        load_config(CONFIG_DIR / "classification_reinforce.yaml"),
    ]
    model_short = model_name.rsplit("-", maxsplit=1)[-1].lower()
    return [
        config.with_updates(
            model_name=model_name,
            train_size=train_size,
            name=f"{config.name}_{model_short}_{train_size}",
            initialization_candidates=0,
            warmup_steps=min(
                200,
                max(
                    10,
                    math.ceil(train_size / config.batch_size) * config.epochs // 10,
                ),
            ),
        )
        for config in configs
    ]


def run_capacity(args: argparse.Namespace) -> None:
    """Run the capacity and training-set-size sweep used in the paper."""
    from crl_prompt.experiments import run_paired_configs

    protocol = [
        ("Qwen/Qwen2.5-0.5B", 250, [42, 43, 44, 45, 46, 47, 48, 52, 53, 54]),
        ("Qwen/Qwen2.5-0.5B", 1000, [42, 43, 44]),
        ("Qwen/Qwen2.5-0.5B", 6000, [42, 43, 44, 45, 46, 47, 48, 52, 53, 54]),
        ("Qwen/Qwen2.5-1.5B", 6000, [42, 43, 44]),
        ("Qwen/Qwen2.5-3B", 250, [42, 43, 44]),
        ("Qwen/Qwen2.5-3B", 500, [42, 43, 44]),
        ("Qwen/Qwen2.5-3B", 1000, [42, 43, 44]),
        ("Qwen/Qwen2.5-3B", 6000, [42, 43, 44]),
    ]
    for model_name, train_size, default_seeds in protocol:
        seeds = _seed_values(args.seeds, default_seeds)
        model_short = model_name.rsplit("-", maxsplit=1)[-1].lower()
        run_paired_configs(
            _capacity_variants(model_name, train_size),
            seeds,
            PROJECT_ROOT / "outputs" / "capacity" / f"{model_short}_{train_size}",
            force=args.force,
        )


def run_ablations(args: argparse.Namespace) -> None:
    """Run auxiliary objective and hyperparameter checks from the journal."""
    from crl_prompt.experiments import run_paired_suite

    seeds = _seed_values(args.seeds, (42, 43, 44))
    run_paired_suite(
        [
            CONFIG_DIR / "classification_mle.yaml",
            CONFIG_DIR / "classification_article_values.yaml",
        ],
        seeds,
        PROJECT_ROOT / "outputs" / "ablations" / "classification",
        force=args.force,
    )


def run_zero_shot(args: argparse.Namespace) -> None:
    """Evaluate the zero-shot model-size ladder and generation baseline."""
    from crl_prompt.experiments import run_zero_shot_evaluation

    base_classification = load_config_lazy(CONFIG_DIR / "classification_mle.yaml")
    zero_shot_root = PROJECT_ROOT / "outputs" / "zero_shot"
    for model_name in (
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
    ):
        config = base_classification.with_updates(model_name=model_name)
        short_name = model_name.rsplit("-", maxsplit=1)[-1].lower()
        run_zero_shot_evaluation(
            config, zero_shot_root / f"classification_{short_name}.json", args.force
        )
    generation = load_config_lazy(CONFIG_DIR / "generation_mle.yaml")
    run_zero_shot_evaluation(
        generation, zero_shot_root / "generation_3b.json", args.force
    )


def run_large_evaluation(args: argparse.Namespace) -> None:
    """Evaluate all available trained prefixes on the article test sizes."""
    from crl_prompt.experiments import reevaluate_output_tree

    reevaluate_output_tree(
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "outputs" / "large_evaluation",
        force=args.force,
    )


def load_config_lazy(path: Path) -> "ExperimentConfig":
    """Load a YAML configuration without importing training dependencies."""
    from crl_prompt.config import load_config

    return load_config(path)
    run_paired_suite(
        [
            CONFIG_DIR / "generation_mle.yaml",
            CONFIG_DIR / "generation_offtopic4.yaml",
            CONFIG_DIR / "generation_offtopic_gradient.yaml",
            CONFIG_DIR / "generation_reinforce_sampled.yaml",
        ],
        seeds,
        PROJECT_ROOT / "outputs" / "ablations" / "generation",
        force=args.force,
    )


def run_all(args: argparse.Namespace) -> None:
    """Execute every training group and regenerate the paper figures."""
    run_classification(args)
    run_generation(args)
    run_grpo(args)
    run_capacity(args)
    run_ablations(args)
    run_zero_shot(args)
    run_large_evaluation(args)
    from crl_prompt.reporting import build_figures

    build_figures(PROJECT_ROOT)


def show_summary(_: argparse.Namespace) -> None:
    """Print the paper tables without importing the training stack."""
    from crl_prompt.reporting import print_paper_results

    print_paper_results(PROJECT_ROOT)


def make_figures(_: argparse.Namespace) -> None:
    """Rebuild figures from saved result files."""
    from crl_prompt.reporting import build_figures

    build_figures(PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """Create the project command-line interface."""
    parser = argparse.ArgumentParser(
        description="Reproduce CRL-Prompt dialogue experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("classification", run_classification),
        ("generation", run_generation),
        ("grpo", run_grpo),
        ("capacity", run_capacity),
        ("ablations", run_ablations),
        ("zero-shot", run_zero_shot),
        ("large-eval", run_large_evaluation),
        ("all", run_all),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--seeds", nargs="+", type=int)
        subparser.add_argument("--force", action="store_true")
        subparser.set_defaults(handler=handler)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.set_defaults(handler=show_summary)
    figures_parser = subparsers.add_parser("figures")
    figures_parser.set_defaults(handler=make_figures)
    return parser


def main() -> None:
    """Parse CLI arguments and dispatch the selected command."""
    arguments = build_parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
