"""Configuration schema and validation for RS-CLIP curriculum training."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from dataloader import CAPTION_POLICIES
from main_finetune import TEXT_TO_IMAGE_WEIGHTINGS, resolve_config_path


BUDGET_MODES = frozenset({"epochs", "steps"})


@dataclass(frozen=True)
class CurriculumRound:
    top_k_per_class: int
    budget: int


@dataclass(frozen=True)
class RSCLIPConfig:
    model_name: str
    pretrained_tag: str
    checkpoint_path: Path
    image_dir: Path
    json_path: Path
    class_dir: Path
    captioned_split: str
    unlabeled_split: str
    supervised_train_fraction: float
    caption_policy: str
    text_to_image_weighting: str
    retain_caption_supervision: bool
    pseudo_prompt_templates: tuple[str, ...]
    budget_mode: str
    curriculum: tuple[CurriculumRound, ...]
    seed: int
    batch_size: int
    num_workers: int
    learning_rate: float
    warmup_fraction: float
    output_dir: Path
    optimizer: str = "adafactor"
    weight_decay: float = 0.0


def _require_strings(values: dict[str, object], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if not isinstance(values.get(key), str)]
    if missing:
        raise ValueError(
            "RS-CLIP config is missing string values for: " + ", ".join(missing)
        )


def load_rsclip_config(config_path: Path) -> RSCLIPConfig:
    """Load and strictly validate a local RS-CLIP curriculum configuration."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"RS-CLIP config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"RS-CLIP config must contain a mapping: {config_path}")

    _require_strings(
        values,
        (
            "model_name",
            "pretrained_tag",
            "checkpoint_path",
            "image_dir",
            "json_path",
            "class_dir",
            "captioned_split",
            "unlabeled_split",
            "caption_policy",
            "text_to_image_weighting",
            "budget_mode",
            "output_dir",
        ),
    )
    retain_captions = values.get("retain_caption_supervision")
    if not isinstance(retain_captions, bool):
        raise ValueError("retain_caption_supervision must be true or false")

    raw_templates = values.get("pseudo_prompt_templates")
    if not isinstance(raw_templates, list) or not raw_templates or not all(
        isinstance(template, str) for template in raw_templates
    ):
        raise ValueError("pseudo_prompt_templates must be a non-empty list of strings")
    if any(template.count("{}") != 1 for template in raw_templates):
        raise ValueError(
            "Every pseudo prompt template must contain exactly one '{}' placeholder"
        )

    raw_rounds = values.get("curriculum")
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise ValueError("curriculum must be a non-empty list of rounds")
    rounds: list[CurriculumRound] = []
    try:
        for raw_round in raw_rounds:
            if not isinstance(raw_round, dict):
                raise TypeError
            curriculum_round = CurriculumRound(
                top_k_per_class=int(raw_round["top_k_per_class"]),
                budget=int(raw_round["budget"]),
            )
            if curriculum_round.top_k_per_class <= 0 or curriculum_round.budget <= 0:
                raise ValueError
            rounds.append(curriculum_round)
        seed = int(values["seed"])
        train_fraction = float(values["supervised_train_fraction"])
        batch_size = int(values["batch_size"])
        num_workers = int(values["num_workers"])
        learning_rate = float(values["learning_rate"])
        warmup_fraction = float(values["warmup_fraction"])
        optimizer = str(values.get("optimizer", "adafactor")).lower()
        weight_decay = float(values.get("weight_decay", 0.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "RS-CLIP config has invalid numeric or curriculum settings"
        ) from error

    top_ks = [item.top_k_per_class for item in rounds]
    if top_ks != sorted(top_ks):
        raise ValueError("curriculum top_k_per_class values must be nondecreasing")
    if values["budget_mode"] not in BUDGET_MODES:
        raise ValueError("budget_mode must be 'epochs' or 'steps'")
    if values["captioned_split"] == values["unlabeled_split"]:
        raise ValueError("captioned_split and unlabeled_split must be distinct")
    if "test" in {values["captioned_split"], values["unlabeled_split"]}:
        raise ValueError("The RSICD test split cannot be used for RS-CLIP training")
    if values["caption_policy"] not in CAPTION_POLICIES:
        raise ValueError(f"Unsupported caption_policy: {values['caption_policy']!r}")
    if values["text_to_image_weighting"] not in TEXT_TO_IMAGE_WEIGHTINGS:
        raise ValueError(
            "Unsupported text_to_image_weighting: "
            f"{values['text_to_image_weighting']!r}"
        )
    if not 0 < train_fraction < 1:
        raise ValueError(
            "supervised_train_fraction must be strictly between zero and one"
        )
    if batch_size <= 0 or num_workers < 0 or learning_rate <= 0:
        raise ValueError(
            "batch_size and learning_rate must be positive; "
            "num_workers cannot be negative"
        )
    if optimizer not in {"adafactor", "adamw"}:
        raise ValueError("optimizer must be either 'adafactor' or 'adamw'")
    if weight_decay < 0 or not 0 <= warmup_fraction < 1:
        raise ValueError("Invalid weight_decay or warmup_fraction")

    return RSCLIPConfig(
        model_name=values["model_name"],
        pretrained_tag=values["pretrained_tag"],
        checkpoint_path=resolve_config_path(config_path, values["checkpoint_path"]),
        image_dir=resolve_config_path(config_path, values["image_dir"]),
        json_path=resolve_config_path(config_path, values["json_path"]),
        class_dir=resolve_config_path(config_path, values["class_dir"]),
        captioned_split=values["captioned_split"],
        unlabeled_split=values["unlabeled_split"],
        supervised_train_fraction=train_fraction,
        caption_policy=values["caption_policy"],
        text_to_image_weighting=values["text_to_image_weighting"],
        retain_caption_supervision=retain_captions,
        pseudo_prompt_templates=tuple(raw_templates),
        budget_mode=values["budget_mode"],
        curriculum=tuple(rounds),
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        warmup_fraction=warmup_fraction,
        output_dir=resolve_config_path(config_path, values["output_dir"]),
        optimizer=optimizer,
        weight_decay=weight_decay,
    )
