"""Unit tests for RS-CLIP curriculum construction and optimization."""

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from main_rsclip_finetune import (  # noqa: E402
    _build_round_training_dataset,
    _resolve_round_budget,
    _source_counts,
    _train_batches,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_projection = torch.nn.Linear(1, 2, bias=False)
        self.text_projection = torch.nn.Embedding(4, 2)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_projection(images[:, 0, 0, 0].unsqueeze(1))

    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        return self.text_projection(texts[:, 0])


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, source: str | None = None) -> None:
        self.source = source

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int):
        sample = torch.ones(1, 1, 1), torch.tensor([[1]]), "image.jpg"
        if self.source is None:
            return sample
        return (*sample, self.source)


class RSCLIPTrainingTests(unittest.TestCase):
    def make_batch(self, sources: list[str]) -> dict[str, object]:
        return {
            "images": torch.tensor([[[[1.0]]], [[[2.0]]]]),
            "texts": torch.tensor([[0], [1], [2]]),
            "caption_to_image": torch.tensor([0, 0, 1]),
            "filenames": ["caption.jpg", "pseudo.jpg"],
            "sources": sources,
        }

    def test_training_pool_toggle_includes_or_excludes_real_captions(self) -> None:
        caption_dataset = PairDataset()
        pseudo_dataset = PairDataset("pseudo")

        mixed = _build_round_training_dataset(
            caption_dataset, pseudo_dataset, retain_caption_supervision=True
        )
        pseudo_only = _build_round_training_dataset(
            caption_dataset, pseudo_dataset, retain_caption_supervision=False
        )

        self.assertEqual(len(mixed), 2)
        self.assertEqual(
            [mixed[index][3] for index in range(2)], ["caption", "pseudo"]
        )
        self.assertIs(pseudo_only, pseudo_dataset)
        self.assertEqual(pseudo_only[0][3], "pseudo")

    def test_epoch_and_step_budget_accounting_is_exact(self) -> None:
        self.assertEqual(
            _resolve_round_budget("epochs", 2, 7, max_steps_per_round=None),
            (2, 14, 2),
        )
        self.assertEqual(
            _resolve_round_budget("steps", 300, 7, max_steps_per_round=None),
            (300, 300, 1),
        )
        self.assertEqual(
            _resolve_round_budget("steps", 300, 7, max_steps_per_round=3),
            (3, 3, 1),
        )

    def test_source_counts_map_every_text_to_its_image_source(self) -> None:
        images, texts = _source_counts(self.make_batch(["caption", "pseudo"]))
        self.assertEqual(images, {"caption": 1, "pseudo": 1})
        self.assertEqual(texts, {"caption": 2, "pseudo": 1})

    def test_mixed_training_reports_sources_steps_and_finite_update(self) -> None:
        model = TinyModel()
        initial_parameters = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = CountingScheduler()
        batch = self.make_batch(["caption", "pseudo"])

        metrics = _train_batches(
            model,
            iter([batch, batch]),
            optimizer,
            scheduler,
            torch.device("cpu"),
            "per_caption",
            step_limit=1,
        )

        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(scheduler.steps, 1)
        self.assertEqual(metrics["caption_images"], 1)
        self.assertEqual(metrics["pseudo_images"], 1)
        self.assertEqual(metrics["caption_texts"], 2)
        self.assertEqual(metrics["pseudo_texts"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(initial_parameters, model.parameters())
            )
        )

    def test_pseudo_only_training_reports_zero_caption_exposure(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = CountingScheduler()
        batch = self.make_batch(["pseudo", "pseudo"])

        metrics = _train_batches(
            model,
            iter([batch]),
            optimizer,
            scheduler,
            torch.device("cpu"),
            "per_caption",
            step_limit=1,
        )

        self.assertEqual(metrics["caption_images"], 0)
        self.assertEqual(metrics["caption_texts"], 0)
        self.assertEqual(metrics["pseudo_images"], 2)
        self.assertEqual(metrics["pseudo_texts"], 3)


if __name__ == "__main__":
    unittest.main()
