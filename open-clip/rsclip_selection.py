"""Class-balanced pseudo-label selection for RS-CLIP."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from prompt_utils import build_text_features, expand_class_prompts


def load_class_names(class_dir: str | Path) -> list[str]:
    """Load only class-file stems; ground-truth file contents stay untouched."""
    class_dir = Path(class_dir)
    if not class_dir.is_dir():
        raise FileNotFoundError(f"RSICD class directory does not exist: {class_dir}")
    class_names = [path.stem for path in sorted(class_dir.iterdir()) if path.is_file()]
    if not class_names:
        raise ValueError(f"No class-list files found in: {class_dir}")
    return class_names


@torch.no_grad()
def score_unlabeled_images(
    model: torch.nn.Module,
    data_loader: DataLoader,
    class_names: Sequence[str],
    prompt_templates: Sequence[str],
    tokenizer,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """Return filenames and per-image class probabilities in loader order."""
    model.eval()
    text_features = build_text_features(
        class_names, prompt_templates, model, tokenizer, device
    )
    filenames: list[str] = []
    probability_batches: list[torch.Tensor] = []
    for images, batch_filenames in data_loader:
        images = images.to(device)
        image_features = functional.normalize(model.encode_image(images), dim=-1)
        logits = model.logit_scale.exp() * image_features @ text_features.T
        probabilities = functional.softmax(logits, dim=1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite probabilities during pseudo-label scoring")
        filenames.extend(batch_filenames)
        probability_batches.append(probabilities.cpu())
    if not probability_batches:
        raise ValueError("No unlabeled images were scored")
    return filenames, torch.cat(probability_batches)


def select_top_k_per_class(
    filenames: Sequence[str],
    probabilities: torch.Tensor,
    class_names: Sequence[str],
    prompt_templates: Sequence[str],
    top_k: int,
) -> list[dict[str, object]]:
    """Select top-K independently per class with deterministic filename ties."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [images, classes]")
    if probabilities.shape != (len(filenames), len(class_names)):
        raise ValueError("Probability shape does not match filenames and classes")
    if top_k > len(filenames):
        raise ValueError(
            f"top_k ({top_k}) exceeds the number of unlabeled images ({len(filenames)})"
        )

    records: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        ranked = sorted(
            range(len(filenames)),
            key=lambda index: (
                -float(probabilities[index, class_index].item()), filenames[index]
            ),
        )[:top_k]
        prompts = expand_class_prompts([class_name], prompt_templates)
        for rank, image_index in enumerate(ranked, start=1):
            records.append(
                {
                    "filename": filenames[image_index],
                    "class_name": class_name,
                    "confidence": float(probabilities[image_index, class_index].item()),
                    "rank_within_class": rank,
                    "prompts": prompts,
                }
            )
    return records


def selection_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    filenames = [str(record["filename"]) for record in records]
    class_counts = Counter(str(record["class_name"]) for record in records)
    confidences = [float(record["confidence"]) for record in records]
    unique_images = len(set(filenames))
    return {
        "pairs": len(records),
        "unique_images": unique_images,
        "overlapping_pairs": len(records) - unique_images,
        "class_counts": dict(sorted(class_counts.items())),
        "minimum_confidence": min(confidences),
        "maximum_confidence": max(confidences),
        "mean_confidence": sum(confidences) / len(confidences),
    }


def write_selection_manifest(
    path: Path,
    round_number: int,
    top_k: int,
    records: Sequence[dict[str, object]],
) -> str:
    """Write deterministic selection evidence and return its SHA-256 digest."""
    document = {
        "round": round_number,
        "top_k_per_class": top_k,
        "summary": selection_summary(records),
        "records": list(records),
    }
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
