"""Evaluate a manually downloaded OpenCLIP checkpoint on the RSICD test split."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import open_clip
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from dataloader import RSICDZeroShotDataset
from prompt_utils import build_text_features as build_prompt_text_features
from prompt_utils import display_class_name


PROMPT_TEMPLATES = (
    "An overhead view of a {}",
    "A top-down satellite shot of a {}",
    "A high-resolution image showing a {}",
    "A geospatial view depicting a {}",
    "An image taken from above showing a {}",
)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "zeroshot_config.yaml"
DEFAULT_OUTPUT_DIR = Path("results/open_clip_zeroshot")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    pretrained_tag: str
    checkpoint_path: Path


@dataclass(frozen=True)
class DatasetConfig:
    image_dir: Path
    json_path: Path
    class_dir: Path
    dataset_split: str


def load_config_values(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Zero-shot config does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"Zero-shot config must contain a mapping: {config_path}")
    return values


def require_config_strings(values: dict[str, object], *keys: str) -> None:
    missing_keys = [key for key in keys if not isinstance(values.get(key), str)]
    if missing_keys:
        raise ValueError(
            f"Zero-shot config is missing string values for: {', '.join(missing_keys)}"
        )


def resolve_config_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_model_config(config_path: Path) -> ModelConfig:
    """Load a local OpenCLIP checkpoint selection from YAML."""
    values = load_config_values(config_path)
    require_config_strings(values, "model_name", "pretrained_tag", "checkpoint_path")
    return ModelConfig(
        model_name=values["model_name"],
        pretrained_tag=values["pretrained_tag"],
        checkpoint_path=resolve_config_path(config_path, values["checkpoint_path"]),
    )


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load local RSICD resource paths from YAML."""
    values = load_config_values(config_path)
    require_config_strings(values, "image_dir", "json_path", "class_dir", "dataset_split")
    return DatasetConfig(
        image_dir=resolve_config_path(config_path, values["image_dir"]),
        json_path=resolve_config_path(config_path, values["json_path"]),
        class_dir=resolve_config_path(config_path, values["class_dir"]),
        dataset_split=values["dataset_split"],
    )


def apply_model_overrides(args: argparse.Namespace, model_config: ModelConfig) -> ModelConfig:
    """Apply optional command-line overrides without enabling remote downloads."""
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()

    return replace(
        model_config,
        model_name=args.model_name or model_config.model_name,
        pretrained_tag=args.pretrained or model_config.pretrained_tag,
        checkpoint_path=checkpoint_path or model_config.checkpoint_path,
    )


def build_text_features(
    class_names: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Encode and ensemble the configured prompts for every RSICD class."""
    return build_prompt_text_features(
        class_names, PROMPT_TEMPLATES, model, tokenizer, device
    )


def list_pretrained() -> None:
    """Print model/checkpoint pairs supported by the installed OpenCLIP version."""
    for model_name, pretrained in open_clip.list_pretrained():
        print(f"{model_name}\t{pretrained}")


def require_checkpoint(model_config: ModelConfig) -> Path:
    """Return the configured local checkpoint without allowing remote downloads."""
    if not model_config.checkpoint_path.is_file():
        raise FileNotFoundError(
            "OpenCLIP checkpoint is missing: "
            f"{model_config.checkpoint_path}. Download it manually and update "
            "checkpoint_path in the zero-shot config or pass --checkpoint-path."
        )
    return model_config.checkpoint_path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device_name}. Use auto, cpu, or cuda.")
    return device


def calculate_metrics(ground_truths: list[str], predictions: list[str]) -> dict[str, float]:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        ground_truths,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(ground_truths, predictions)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1_score),
    }


def evaluate(
    data_loader: DataLoader,
    class_names: list[str],
    model: torch.nn.Module,
    text_features: torch.Tensor,
    device: torch.device,
) -> tuple[list[str], list[dict[str, object]]]:
    """Classify each test image and retain ordered per-image predictions."""
    ground_truths: list[str] = []
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for image_batch, label_batch, filename_batch in data_loader:
            image_features = functional.normalize(model.encode_image(image_batch.to(device)), dim=-1)
            similarities = image_features @ text_features.T
            predicted_indices = similarities.argmax(dim=-1).tolist()

            for label, filename, predicted_index, scores in zip(
                label_batch, filename_batch, predicted_indices, similarities
            ):
                prediction = class_names[predicted_index]
                ground_truths.append(label)
                predictions.append(
                    {
                        "filename": filename,
                        "ground_truth": label,
                        "prediction": prediction,
                        "similarity": float(scores[predicted_index].item()),
                    }
                )

    return ground_truths, predictions


def write_results(
    output_dir: Path,
    metrics: dict[str, object],
    predictions: list[dict[str, object]],
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as predictions_file:
        json.dump(predictions, predictions_file, indent=2)
        predictions_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML model/checkpoint configuration (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--model-name", help="Override the config model_name.")
    parser.add_argument(
        "--pretrained",
        help="Override the config pretrained_tag; weights remain local.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Override the config local checkpoint_path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N test images; omit to evaluate the full test split.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for metrics.json and predictions.json (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--list-pretrained",
        action="store_true",
        help="List installed-version OpenCLIP model/checkpoint pairs and exit.",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.list_pretrained:
        list_pretrained()
        return

    dataset_config = load_dataset_config(args.config)
    model_config = apply_model_overrides(args, load_model_config(args.config))
    checkpoint_path = require_checkpoint(model_config)
    pretrained_config = open_clip.get_pretrained_cfg(
        model_config.model_name, model_config.pretrained_tag
    )
    if pretrained_config is None:
        raise RuntimeError(
            f"Installed OpenCLIP does not provide preprocessing configuration for "
            f"{model_config.model_name} / {model_config.pretrained_tag}."
        )

    device = resolve_device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config.model_name,
        pretrained=str(checkpoint_path),
        device=device,
        image_mean=pretrained_config["mean"],
        image_std=pretrained_config["std"],
        image_interpolation=pretrained_config["interpolation"],
        image_resize_mode=pretrained_config["resize_mode"],
    )
    tokenizer = open_clip.get_tokenizer(model_config.model_name)
    model.eval()

    dataset = RSICDZeroShotDataset(
        image_dir=dataset_config.image_dir,
        json_path=dataset_config.json_path,
        class_dir=dataset_config.class_dir,
        transform=preprocess,
        split=dataset_config.dataset_split,
    )
    if args.max_samples is None:
        evaluation_dataset = dataset
    else:
        evaluation_dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if not evaluation_dataset:
        raise ValueError("No RSICD test samples selected for evaluation.")

    text_features = build_text_features(dataset.class_names, model, tokenizer, device)
    data_loader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False)
    ground_truths, predictions = evaluate(
        data_loader, dataset.class_names, model, text_features, device
    )
    predicted_labels = [record["prediction"] for record in predictions]
    metric_values = calculate_metrics(ground_truths, predicted_labels)
    metrics: dict[str, object] = {
        "model_name": model_config.model_name,
        "pretrained_tag": model_config.pretrained_tag,
        "checkpoint_path": str(checkpoint_path),
        "image_dir": str(dataset_config.image_dir),
        "json_path": str(dataset_config.json_path),
        "class_dir": str(dataset_config.class_dir),
        "device": str(device),
        "num_samples": len(predictions),
        "num_classes": len(dataset.class_names),
        **metric_values,
    }
    write_results(args.output_dir, metrics, predictions)

    print(f"Model: {model_config.model_name} / {model_config.pretrained_tag}")
    print(f"Evaluated samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Results: {args.output_dir}")
    if len(predictions) == 1:
        prediction = predictions[0]
        print(f"Image: {prediction['filename']}")
        print(f"Ground truth: {prediction['ground_truth']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Similarity: {prediction['similarity']:.6f}")


if __name__ == "__main__":
    main()
