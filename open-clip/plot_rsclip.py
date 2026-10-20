"""Plot RS-CLIP curriculum losses, source exposure, and pseudo confidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict[str, object]:
    path = path.resolve()
    if path.is_dir():
        path /= "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("strategy") != "rsclip":
        raise ValueError(f"Not an RS-CLIP metrics file: {path}")
    return document


def _save(figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _add_round_boundaries(axes, units: list[int], rounds: list[int]) -> None:
    """Mark transitions between curriculum rounds on one or more axes."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for index in range(1, len(rounds)):
        if rounds[index] != rounds[index - 1]:
            for axis in axes:
                axis.axvline(units[index] - 0.5, color="grey", alpha=0.35)


def plot_training(document: dict[str, object], output_dir: Path) -> Path:
    records = document.get("epochs", [])
    if not records:
        raise ValueError("RS-CLIP metrics contain no completed training units")
    units = [record["epoch"] for record in records]
    train_loss = [record["train"]["loss"] for record in records]
    validation_loss = [record["validation"]["selection_loss"] for record in records]
    rounds = [record["round"] for record in records]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(units, train_loss, label="train")
    axis.plot(units, validation_loss, "--", label="validation")
    _add_round_boundaries(axis, units, rounds)
    axis.set(title="RS-CLIP loss by training unit", xlabel="Training unit", ylabel="Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save(figure, output_dir, "loss.png")


def plot_source_exposure(document: dict[str, object], output_dir: Path) -> Path:
    records = document.get("epochs", [])
    if not records:
        raise ValueError("RS-CLIP metrics contain no completed training units")
    units = [record["epoch"] for record in records]
    rounds = [record["round"] for record in records]
    caption_images = [record["train"].get("caption_images", 0) for record in records]
    pseudo_images = [record["train"].get("pseudo_images", 0) for record in records]
    caption_texts = [record["train"].get("caption_texts", 0) for record in records]
    pseudo_texts = [record["train"].get("pseudo_texts", 0) for record in records]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for axis, caption_counts, pseudo_counts, ylabel in (
        (axes[0], caption_images, pseudo_images, "Processed image records"),
        (axes[1], caption_texts, pseudo_texts, "Processed text records"),
    ):
        axis.bar(units, caption_counts, label="real-caption supervision")
        axis.bar(
            units,
            pseudo_counts,
            bottom=caption_counts,
            label="pseudo supervision",
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    _add_round_boundaries(list(axes), units, rounds)
    axes[0].set_title("RS-CLIP optimizer exposure by source")
    axes[1].set_xlabel("Training unit")
    return _save(figure, output_dir, "source_exposure.png")


def plot_selection(document: dict[str, object], output_dir: Path) -> Path:
    rounds = document.get("rounds", [])
    if not rounds:
        raise ValueError("RS-CLIP metrics contain no completed rounds")
    indices = [record["round"] for record in rounds]
    means = [record["selection"]["mean_confidence"] for record in rounds]
    minimums = [record["selection"]["minimum_confidence"] for record in rounds]
    maximums = [record["selection"]["maximum_confidence"] for record in rounds]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(indices, means, marker="o", label="mean")
    axis.fill_between(indices, minimums, maximums, alpha=0.2, label="min–max")
    axis.set(
        title="Pseudo-label confidence by curriculum round",
        xlabel="Round",
        ylabel="Confidence",
    )
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save(figure, output_dir, "pseudo_confidence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    document = load_metrics(args.run)
    run_path = args.run.resolve()
    output_dir = args.output_dir or (
        run_path if run_path.is_dir() else run_path.parent
    ) / "plots"
    for path in (
        plot_training(document, output_dir),
        plot_source_exposure(document, output_dir),
        plot_selection(document, output_dir),
    ):
        print(path)


if __name__ == "__main__":
    main()
