"""Shared prompt construction and encoding helpers for OpenCLIP workflows."""

import re
from collections.abc import Sequence

import torch
import torch.nn.functional as functional


def display_class_name(class_name: str) -> str:
    """Convert RSICD class filenames into readable prompt text."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", class_name)
    return spaced.replace("_", " ").lower()


def expand_class_prompts(
    class_names: Sequence[str], prompt_templates: Sequence[str]
) -> list[str]:
    """Expand templates in class-major order."""
    if not class_names:
        raise ValueError("At least one class name is required")
    if not prompt_templates:
        raise ValueError("At least one prompt template is required")
    invalid = [template for template in prompt_templates if template.count("{}") != 1]
    if invalid:
        raise ValueError("Every prompt template must contain exactly one '{}' placeholder")
    return [
        template.format(display_class_name(class_name))
        for class_name in class_names
        for template in prompt_templates
    ]


def build_text_features(
    class_names: Sequence[str],
    prompt_templates: Sequence[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Encode, average, and normalize prompt features for every class."""
    prompts = expand_class_prompts(class_names, prompt_templates)
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = functional.normalize(model.encode_text(text_tokens), dim=-1)
        text_features = text_features.reshape(
            len(class_names), len(prompt_templates), -1
        ).mean(dim=1)
    return functional.normalize(text_features, dim=-1)
