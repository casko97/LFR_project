"""Focused tests for RS-CLIP pseudo-label scoring and selection."""

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from prompt_utils import build_text_features  # noqa: E402
from rsclip_selection import (  # noqa: E402
    load_class_names,
    score_unlabeled_images,
    select_top_k_per_class,
    selection_summary,
    write_selection_manifest,
)


class IndexTokenizer:
    def __call__(self, texts: list[str]) -> torch.Tensor:
        lookup = {
            "first class a": 0,
            "second class a": 1,
            "first class b": 2,
            "second class b": 3,
        }
        return torch.tensor([[lookup[text]] for text in texts])


class FeatureModel(torch.nn.Module):
    def __init__(self, logit_scale: float = 1.0) -> None:
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(logit_scale)))
        self.register_buffer(
            "text_features",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.0, 1.0],
                    [0.2, 0.8],
                ]
            ),
        )

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_features[tokens[:, 0]]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :, 0, 0]


class ImageDataset(Dataset):
    def __init__(self) -> None:
        self.features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index].reshape(2, 1, 1), f"{index}.jpg"


class RSCLIPScoringTests(unittest.TestCase):
    def test_prompt_features_are_averaged_then_normalized(self) -> None:
        model = FeatureModel()
        actual = build_text_features(
            ["ClassA", "ClassB"],
            ["first {}", "second {}"],
            model,
            IndexTokenizer(),
            torch.device("cpu"),
        )
        normalized_rows = functional.normalize(model.text_features, dim=-1)
        expected = functional.normalize(
            normalized_rows.reshape(2, 2, 2).mean(dim=1), dim=-1
        )
        torch.testing.assert_close(actual, expected)

    def test_batched_scoring_uses_logit_scale_and_class_softmax(self) -> None:
        model = FeatureModel(logit_scale=2.0)
        filenames, actual = score_unlabeled_images(
            model,
            DataLoader(ImageDataset(), batch_size=2, shuffle=False),
            ["ClassA", "ClassB"],
            ["first {}", "second {}"],
            IndexTokenizer(),
            torch.device("cpu"),
        )
        text_features = build_text_features(
            ["ClassA", "ClassB"],
            ["first {}", "second {}"],
            model,
            IndexTokenizer(),
            torch.device("cpu"),
        )
        image_features = functional.normalize(ImageDataset().features, dim=-1)
        expected = functional.softmax(2.0 * image_features @ text_features.T, dim=1)

        self.assertEqual(filenames, ["0.jpg", "1.jpg", "2.jpg"])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(3))


class RSCLIPSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filenames = ["b.jpg", "a.jpg", "c.jpg"]
        self.probabilities = torch.tensor(
            [[0.9, 0.9], [0.9, 0.2], [0.1, 0.8]]
        )
        self.class_names = ["ClassA", "ClassB"]
        self.templates = ["a {}", "overhead {}"]

    def _select(self, top_k: int) -> list[dict[str, object]]:
        return select_top_k_per_class(
            self.filenames,
            self.probabilities,
            self.class_names,
            self.templates,
            top_k,
        )

    def test_selection_is_balanced_deterministic_and_keeps_overlap(self) -> None:
        records = self._select(top_k=2)

        self.assertEqual(
            [(record["class_name"], record["filename"]) for record in records],
            [
                ("ClassA", "a.jpg"),
                ("ClassA", "b.jpg"),
                ("ClassB", "b.jpg"),
                ("ClassB", "c.jpg"),
            ],
        )
        self.assertEqual(
            records[0]["prompts"], ["a class a", "overhead class a"]
        )
        summary = selection_summary(records)
        self.assertEqual(summary["class_counts"], {"ClassA": 2, "ClassB": 2})
        self.assertEqual(summary["overlapping_pairs"], 1)

    def test_increasing_k_retains_each_classes_previous_selection(self) -> None:
        smaller = self._select(top_k=1)
        larger = self._select(top_k=2)
        for class_name in self.class_names:
            smaller_names = {
                record["filename"]
                for record in smaller
                if record["class_name"] == class_name
            }
            larger_names = {
                record["filename"]
                for record in larger
                if record["class_name"] == class_name
            }
            self.assertLessEqual(smaller_names, larger_names)

    def test_class_vocabulary_uses_stems_without_parsing_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            class_dir = Path(temporary_directory)
            (class_dir / "BareLand.txt").write_bytes(b"\xff\xfe")
            (class_dir / "Port.txt").write_text("unused.jpg\n", encoding="utf-8")

            self.assertEqual(load_class_names(class_dir), ["BareLand", "Port"])

    def test_manifest_is_deterministic_and_hash_matches_written_bytes(self) -> None:
        records = self._select(top_k=2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "round_01" / "pseudo_labels.json"
            first_digest = write_selection_manifest(path, 1, 2, records)
            first_payload = path.read_bytes()
            second_digest = write_selection_manifest(path, 1, 2, records)

            self.assertEqual(first_digest, second_digest)
            self.assertEqual(first_digest, hashlib.sha256(first_payload).hexdigest())
            document = json.loads(first_payload)
            self.assertEqual(document["round"], 1)
            self.assertEqual(document["top_k_per_class"], 2)
            self.assertEqual(document["records"], records)

    def test_invalid_shapes_and_excessive_k_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            select_top_k_per_class(
                self.filenames,
                torch.ones(3),
                self.class_names,
                self.templates,
                1,
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self._select(top_k=4)


if __name__ == "__main__":
    unittest.main()
