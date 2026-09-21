"""Paper result summaries and reproducible figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any



def _load_results(project_root: Path) -> dict[str, Any]:
    """Read the result values transcribed from the final article."""
    path = project_root / "results" / "paper_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def print_paper_results(project_root: Path) -> None:
    """Print the main article tables in a compact terminal format."""
    results = _load_results(project_root)
    print("Emotion classification, Qwen2.5-3B")
    print(f"zero-shot: {results['classification_3b']['zero_shot']:.3f}")
    for row in results["classification_3b"]["rows"]:
        print(
            f"{row['configuration']:<30} "
            f"test500={row['test_500_mean']:.3f}±{row['test_500_std']:.3f} "
            f"full={row['full_test_mean']:.3f}±{row['full_test_std']:.3f}"
        )
    print("\nResponse generation, Qwen2.5-3B")
    print(f"zero-shot ROUGE-L: {results['generation_3b']['zero_shot_rougeL']:.3f}")
    for row in results["generation_3b"]["rows"]:
        print(
            f"{row['configuration']:<30} "
            f"R-L(500)={row['rougeL_500_mean']:.3f}±{row['rougeL_500_std']:.3f}"
        )


def _save_figure(figure: Any, output_base: Path) -> None:
    """Save a figure as both PNG and PDF."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def _capacity_figure(results: dict[str, Any], output_dir: Path) -> None:
    """Plot paired differences to MLE across model sizes."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = results["capacity_full_train"]
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for offset, mean_key, std_key, label, color in (
        (-0.08, "mask_delta_mean", "mask_delta_std", "Contrast mask", "#4472C4"),
        (0.08, "rl_delta_mean", "rl_delta_std", "CE + REINFORCE", "#ED7D31"),
    ):
        axis.errorbar(
            positions + offset,
            [row[mean_key] for row in rows],
            yerr=[row[std_key] for row in rows],
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(positions, [row["model"] for row in rows])
    axis.set_xlabel("Qwen2.5 model size")
    axis.set_ylabel("Accuracy difference to MLE")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, output_dir / "capacity_effect")


def _data_figure(results: dict[str, Any], output_dir: Path) -> None:
    """Plot classification accuracy against training-set size."""
    import matplotlib.pyplot as plt

    rows = results["data_sweep"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, model in zip(axes, ("0.5B", "3B")):
        subset = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: row["examples"],
        )
        for mean_key, std_key, label, color in (
            ("mle_mean", "mle_std", "MLE", "#555555"),
            ("mask_mean", "mask_std", "Contrast mask", "#4472C4"),
            ("rl_mean", "rl_std", "CE + REINFORCE", "#ED7D31"),
        ):
            axis.errorbar(
                [row["examples"] for row in subset],
                [row[mean_key] for row in subset],
                yerr=[row[std_key] for row in subset],
                marker="o",
                capsize=3,
                label=label,
                color=color,
            )
        axis.set_xscale("log")
        axis.set_title(f"Qwen2.5-{model}")
        axis.set_xlabel("Training examples")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Emotion classification accuracy")
    axes[1].legend(loc="lower right")
    _save_figure(figure, output_dir / "data_efficiency")


def _grpo_figure(project_root: Path, output_dir: Path) -> None:
    """Plot held-out GRPO reward histories from the completed runs."""
    import matplotlib.pyplot as plt

    path = project_root / "results" / "grpo_reward_history.json"
    histories = json.loads(path.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for seed, history in sorted(histories.items()):
        if not history:
            continue
        axis.plot(
            [entry["step"] for entry in history],
            [entry["reward"] for entry in history],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"run {seed}",
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Held-out ROUGE-L reward")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_dir / "grpo_reward")


def build_figures(project_root: Path) -> None:
    """Regenerate the three figures used to summarize the experiments."""
    results = _load_results(project_root)
    output_dir = project_root / "figures" / "generated"
    _capacity_figure(results, output_dir)
    _data_figure(results, output_dir)
    _grpo_figure(project_root, output_dir)
    print(f"Figures written to {output_dir}")
