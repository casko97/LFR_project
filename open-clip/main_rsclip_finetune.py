"""RS-CLIP curriculum fine-tuning with optional real-caption retention."""

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import random
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from dataloader import (
    RSICDImageOnlyDataset,
    RSICDPseudoPairDataset,
    SourceTaggedDataset,
    build_supervised_train_val_datasets,
    collate_supervised_captions,
    set_dataset_epoch,
)
from main_finetune import (
    TeeStream,
    create_optimizer_and_scheduler,
    hash_filename_order,
    load_local_openclip_model,
    mean_epoch_statistics,
    move_batch_to_device,
    prepare_output_dir,
    resolve_device,
    set_seed,
    supervised_contrastive_loss,
    validate_one_epoch,
    write_metrics,
)
from rsclip_config import RSCLIPConfig, load_rsclip_config
from rsclip_selection import (
    load_class_names,
    score_unlabeled_images,
    select_top_k_per_class,
    selection_summary,
    write_selection_manifest,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).with_name("configs") / "rsclip" / "rsclip_pilot_mixed.yaml"
)


def _build_round_training_dataset(
    caption_dataset: Dataset,
    pseudo_dataset: Dataset,
    retain_caption_supervision: bool,
) -> Dataset:
    """Build one shuffled pool while keeping pseudo-only mode caption-free."""
    if retain_caption_supervision:
        return ConcatDataset(
            [SourceTaggedDataset(caption_dataset, "caption"), pseudo_dataset]
        )
    return pseudo_dataset


def _resolve_round_budget(
    budget_mode: str,
    configured_budget: int,
    batches_per_pass: int,
    max_steps_per_round: int | None,
) -> tuple[int, int, int]:
    """Return effective budget, optimizer steps, and number of training units."""
    if configured_budget <= 0 or batches_per_pass <= 0:
        raise ValueError("Round budget and batches per pass must be positive")
    if budget_mode == "epochs":
        return (
            configured_budget,
            configured_budget * batches_per_pass,
            configured_budget,
        )
    if budget_mode == "steps":
        effective_budget = configured_budget
        if max_steps_per_round is not None:
            if max_steps_per_round <= 0:
                raise ValueError("max_steps_per_round must be positive")
            effective_budget = min(effective_budget, max_steps_per_round)
        return effective_budget, effective_budget, 1
    raise ValueError(f"Unsupported budget mode: {budget_mode!r}")


def _source_counts(batch: dict[str, object]) -> tuple[dict[str, int], dict[str, int]]:
    sources = batch["sources"]
    mapping = batch["caption_to_image"]
    image_counts = {"caption": 0, "pseudo": 0}
    caption_counts = {"caption": 0, "pseudo": 0}
    for source in sources:
        image_counts[source] = image_counts.get(source, 0) + 1
    for image_index in mapping.tolist():
        source = sources[image_index]
        caption_counts[source] = caption_counts.get(source, 0) + 1
    return image_counts, caption_counts


def _train_batches(
    model: torch.nn.Module,
    batches: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    text_to_image_weighting: str,
    step_limit: int,
) -> dict[str, object]:
    model.train()
    batch_statistics: list[dict[str, float]] = []
    processed_filenames: list[str] = []
    image_counts = {"caption": 0, "pseudo": 0}
    caption_counts = {"caption": 0, "pseudo": 0}
    for batch in itertools.islice(batches, step_limit):
        processed_filenames.extend(batch["filenames"])
        batch_images, batch_captions = _source_counts(batch)
        for source in image_counts:
            image_counts[source] += batch_images.get(source, 0)
            caption_counts[source] += batch_captions.get(source, 0)
        optimizer.zero_grad(set_to_none=True)
        loss, statistics = supervised_contrastive_loss(
            model,
            *move_batch_to_device(batch, device),
            text_to_image_weighting=text_to_image_weighting,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite RS-CLIP training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, 4.6052)
        scheduler.step()
        batch_statistics.append({"loss": float(loss.detach().item()), **statistics})
    metrics: dict[str, object] = mean_epoch_statistics(batch_statistics)
    metrics["objective_loss"] = metrics["loss"]
    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    metrics["batches"] = len(batch_statistics)
    metrics["optimizer_steps"] = len(batch_statistics)
    metrics["images"] = len(processed_filenames)
    metrics["captions"] = sum(caption_counts.values())
    metrics["captions_per_image"] = metrics["captions"] / metrics["images"]
    metrics["caption_images"] = image_counts["caption"]
    metrics["pseudo_images"] = image_counts["pseudo"]
    metrics["caption_texts"] = caption_counts["caption"]
    metrics["pseudo_texts"] = caption_counts["pseudo"]
    metrics["image_order_sha256"] = hash_filename_order(processed_filenames)
    return metrics


def _cycled_batches(
    data_loader: DataLoader,
    dataset: Dataset,
    max_batches_per_pass: int | None,
) -> Iterator[dict[str, object]]:
    pass_index = 0
    while True:
        set_dataset_epoch(dataset, pass_index)
        yielded = 0
        for batch in data_loader:
            if max_batches_per_pass is not None and yielded >= max_batches_per_pass:
                break
            yielded += 1
            yield batch
        if yielded == 0:
            raise ValueError("No RS-CLIP training batches selected")
        pass_index += 1


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: RSCLIPConfig,
    round_index: int,
    unit_index: int,
    total_steps: int,
    manifest_sha256: str,
    best_validation_loss: float,
    train_metrics: dict[str, object],
    validation_metrics: dict[str, object],
) -> dict[str, object]:
    state: dict[str, object] = {
        "round_index": round_index,
        "unit_index": unit_index,
        "budget_mode": config.budget_mode,
        "retain_caption_supervision": config.retain_caption_supervision,
        "manifest_sha256": manifest_sha256,
        "round_total_steps": total_steps,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def verify_last_checkpoint(path: Path, config: RSCLIPConfig) -> int:
    """Restore the final model, optimizer, and scheduler into fresh objects."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "round_index",
        "round_total_steps",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError("RS-CLIP checkpoint is missing: " + ", ".join(missing))
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        model = open_clip.create_model(config.model_name, pretrained=None, device="cpu")
    finally:
        logging.disable(previous_disable_level)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, int(state["round_total_steps"])
    )
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state["round_index"])


def _manifest_hash(records: list[dict[str, object]]) -> str:
    payload = json.dumps(records, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_rsclip(args: argparse.Namespace, config: RSCLIPConfig) -> None:
    device = resolve_device(args.device)
    set_seed(config.seed)
    model, tokenizer, train_transform, validation_transform = load_local_openclip_model(
        config, device
    )
    caption_train_dataset, validation_dataset = build_supervised_train_val_datasets(
        image_dir=config.image_dir,
        json_path=config.json_path,
        train_transform=train_transform,
        validation_transform=validation_transform,
        tokenizer=tokenizer,
        train_fraction=config.supervised_train_fraction,
        seed=config.seed,
        source_split=config.captioned_split,
        train_caption_policy=config.caption_policy,
    )
    unlabeled_dataset = RSICDImageOnlyDataset(
        config.image_dir,
        config.json_path,
        validation_transform,
        config.unlabeled_split,
        max_samples=args.max_unlabeled_images,
    )
    class_names = load_class_names(config.class_dir)
    common_loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "drop_last": False,
    }
    selection_loader = DataLoader(
        unlabeled_dataset, shuffle=False, **common_loader_options
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=collate_supervised_captions,
        **common_loader_options,
    )

    configured_rounds = list(config.curriculum)
    if args.max_rounds is not None:
        configured_rounds = configured_rounds[: args.max_rounds]
    metrics: dict[str, object] = {
        "strategy": "rsclip",
        "model_name": config.model_name,
        "pretrained_tag": config.pretrained_tag,
        "checkpoint_path": str(config.checkpoint_path),
        "image_dir": str(config.image_dir),
        "json_path": str(config.json_path),
        "class_dir": str(config.class_dir),
        "device": str(device),
        "seed": config.seed,
        "captioned_split": config.captioned_split,
        "unlabeled_split": config.unlabeled_split,
        "test_split_used": False,
        "retain_caption_supervision": config.retain_caption_supervision,
        "caption_policy": config.caption_policy,
        "text_to_image_weighting": config.text_to_image_weighting,
        "validation_caption_policy": "all_unique",
        "validation_text_to_image_weighting": "per_caption",
        "pseudo_prompt_templates": list(config.pseudo_prompt_templates),
        "budget_mode": config.budget_mode,
        "caption_train_images_available": len(caption_train_dataset),
        "validation_images": len(validation_dataset),
        "unlabeled_images": len(unlabeled_dataset),
        "num_classes": len(class_names),
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "curriculum": [asdict(item) for item in configured_rounds],
        "rounds": [],
        "epochs": [],
    }
    write_metrics(config.output_dir, metrics)
    print(f"Model: {config.model_name} / {config.pretrained_tag}")
    print(f"Device: {device}")
    print(f"Caption retention: {config.retain_caption_supervision}")
    print(f"Budget mode: {config.budget_mode}")
    print(
        "Caption train/validation/unlabeled images: "
        f"{len(caption_train_dataset)}/{len(validation_dataset)}/{len(unlabeled_dataset)}"
    )

    best_validation_loss = float("inf")
    global_event = 0
    last_round_number = 0
    for round_index, curriculum_round in enumerate(configured_rounds):
        round_number = round_index + 1
        last_round_number = round_number
        print(
            f"Round {round_number}/{len(configured_rounds)}: "
            f"selecting top-{curriculum_round.top_k_per_class} per class"
        )
        filenames, probabilities = score_unlabeled_images(
            model,
            selection_loader,
            class_names,
            config.pseudo_prompt_templates,
            tokenizer,
            device,
        )
        pseudo_records = select_top_k_per_class(
            filenames,
            probabilities,
            class_names,
            config.pseudo_prompt_templates,
            curriculum_round.top_k_per_class,
        )
        round_dir = config.output_dir / f"round_{round_number:02d}"
        manifest_path = round_dir / "pseudo_labels.json"
        manifest_sha256 = write_selection_manifest(
            manifest_path,
            round_number,
            curriculum_round.top_k_per_class,
            pseudo_records,
        )
        pseudo_dataset = RSICDPseudoPairDataset(
            config.image_dir,
            pseudo_records,
            train_transform,
            tokenizer,
            config.pseudo_prompt_templates,
        )
        training_dataset = _build_round_training_dataset(
            caption_train_dataset,
            pseudo_dataset,
            config.retain_caption_supervision,
        )
        shuffle_generator = torch.Generator().manual_seed(config.seed + round_index)
        train_loader = DataLoader(
            training_dataset,
            shuffle=True,
            generator=shuffle_generator,
            collate_fn=collate_supervised_captions,
            **common_loader_options,
        )
        batches_per_pass = min(
            len(train_loader), args.max_train_batches or len(train_loader)
        )
        round_budget, round_total_steps, training_unit_count = _resolve_round_budget(
            config.budget_mode,
            curriculum_round.budget,
            batches_per_pass,
            args.max_steps_per_round,
        )
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config, round_total_steps
        )
        round_record: dict[str, object] = {
            "round": round_number,
            "top_k_per_class": curriculum_round.top_k_per_class,
            "configured_budget": curriculum_round.budget,
            "effective_budget": round_budget,
            "budget_mode": config.budget_mode,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "selection": selection_summary(pseudo_records),
            "caption_records_in_pool": (
                len(caption_train_dataset) if config.retain_caption_supervision else 0
            ),
            "pseudo_records_in_pool": len(pseudo_dataset),
            "history": [],
        }

        training_units = range(training_unit_count)
        for unit_index in training_units:
            if config.budget_mode == "epochs":
                set_dataset_epoch(training_dataset, unit_index)
                train_metrics = _train_batches(
                    model,
                    iter(train_loader),
                    optimizer,
                    scheduler,
                    device,
                    config.text_to_image_weighting,
                    batches_per_pass,
                )
            else:
                train_metrics = _train_batches(
                    model,
                    _cycled_batches(
                        train_loader, training_dataset, args.max_train_batches
                    ),
                    optimizer,
                    scheduler,
                    device,
                    config.text_to_image_weighting,
                    round_total_steps,
                )
            validation_metrics = validate_one_epoch(
                model, validation_loader, device, args.max_validation_batches
            )
            validation_loss = float(validation_metrics["selection_loss"])
            is_best = validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
            state = _checkpoint_state(
                model,
                optimizer,
                scheduler,
                config,
                round_index,
                unit_index,
                round_total_steps,
                manifest_sha256,
                best_validation_loss,
                train_metrics,
                validation_metrics,
            )
            torch.save(state, config.output_dir / "ckp_last.pth")
            torch.save(state, round_dir / "ckp_last.pth")
            if is_best:
                torch.save(state, config.output_dir / "ckp_best.pth")
            global_event += 1
            event = {
                "epoch": global_event,
                "round": round_number,
                "round_unit": unit_index + 1,
                "train": train_metrics,
                "validation": validation_metrics,
                "is_best": is_best,
            }
            round_record["history"].append(event)
            metrics["epochs"].append(event)
            metrics["best_validation_loss"] = best_validation_loss
            write_metrics(config.output_dir, metrics)
            print(
                f"Round {round_number}, unit {unit_index + 1}: "
                f"train_loss={train_metrics['loss']:.6f} "
                f"validation_loss={validation_loss:.6f} "
                f"caption_images={train_metrics['caption_images']} "
                f"pseudo_images={train_metrics['pseudo_images']}"
            )
        metrics["rounds"].append(round_record)
        write_metrics(config.output_dir, metrics)

    if last_round_number == 0:
        raise ValueError("No curriculum rounds selected")
    restored_round_index = verify_last_checkpoint(
        config.output_dir / "ckp_last.pth", config
    )
    metrics["checkpoint_load_round"] = restored_round_index + 1
    metrics["checkpoint_restore_verified"] = True
    write_metrics(config.output_dir, metrics)
    print(f"Artifacts: {config.output_dir}")
    print(f"Checkpoint restore verified for round {restored_round_index + 1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--max-unlabeled-images", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-steps-per-round", type=int)
    args = parser.parse_args()
    for name in (
        "batch_size",
        "max_rounds",
        "max_unlabeled_images",
        "max_train_batches",
        "max_validation_batches",
        "max_steps_per_round",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_rsclip_config(config_path)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir.resolve())
    prepare_output_dir(config.output_dir, config_path)
    log_path = config.output_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Writing terminal output to: {log_path}")
            run_rsclip(args, config)


if __name__ == "__main__":
    main()
