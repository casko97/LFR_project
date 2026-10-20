"""Focused tests for the RS-CLIP dataset adapters in the shared loader."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from dataloader import (  # noqa: E402
    RSICDImageOnlyDataset,
    RSICDPseudoPairDataset,
    SourceTaggedDataset,
    collate_supervised_captions,
)


def tensor_transform(_image):
    return torch.zeros(3, 2, 2)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> torch.Tensor:
        self.calls.append(list(texts))
        return torch.arange(len(texts) * 3).reshape(len(texts), 3)


class TinyCaptionDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.epoch = -1

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int):
        return torch.ones(1, 1, 1), torch.tensor([[1, 2]]), "caption.jpg"

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class RSCLIPDatasetTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        Image.new("RGB", (3, 3), color="red").save(path)

    def test_image_only_split_ignores_caption_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_image(image_dir / "train.jpg")
            self._write_image(image_dir / "test.jpg")
            metadata = {
                "images": [
                    {
                        "filename": "train.jpg",
                        "split": "train",
                        "sentences": [{"tokens": []}, {"invalid": object.__name__}],
                    },
                    {
                        "filename": "test.jpg",
                        "split": "test",
                        "sentences": [{"tokens": ["must", "not", "load"]}],
                    },
                ]
            }
            json_path = root / "dataset.json"
            json_path.write_text(json.dumps(metadata), encoding="utf-8")

            dataset = RSICDImageOnlyDataset(
                image_dir, json_path, tensor_transform, "train"
            )

            self.assertEqual(dataset.filenames, ["train.jpg"])
            image, filename = dataset[0]
            self.assertEqual(filename, "train.jpg")
            self.assertEqual(tuple(image.shape), (3, 2, 2))

    def test_pseudo_dataset_expands_templates_and_keeps_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_dir = Path(temporary_directory)
            self._write_image(image_dir / "same.jpg")
            tokenizer = RecordingTokenizer()
            records = [
                {"filename": "same.jpg", "class_name": "BareLand"},
                {"filename": "same.jpg", "class_name": "Desert"},
            ]
            dataset = RSICDPseudoPairDataset(
                image_dir,
                records,
                tensor_transform,
                tokenizer,
                ["an aerial {}", "a satellite {}"],
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                tokenizer.calls[0], ["an aerial bare land", "a satellite bare land"]
            )
            self.assertEqual(dataset[0][3], "pseudo")
            self.assertEqual(dataset[1][2], "same.jpg")

    def test_source_collation_and_epoch_forwarding(self) -> None:
        base = TinyCaptionDataset()
        tagged = SourceTaggedDataset(base, "caption")
        tagged.set_epoch(4)
        batch = collate_supervised_captions(
            [
                tagged[0],
                (
                    torch.zeros(1, 1, 1),
                    torch.tensor([[3, 4], [5, 6]]),
                    "pseudo.jpg",
                    "pseudo",
                ),
            ]
        )

        self.assertEqual(base.epoch, 4)
        self.assertEqual(batch["sources"], ["caption", "pseudo"])
        torch.testing.assert_close(
            batch["caption_to_image"], torch.tensor([0, 1, 1])
        )


if __name__ == "__main__":
    unittest.main()
