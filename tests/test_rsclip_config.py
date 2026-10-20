"""Tests for RS-CLIP configuration validation and pilot parity."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
CONFIG_DIR = OPEN_CLIP_DIR / "configs" / "rsclip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from rsclip_config import load_rsclip_config  # noqa: E402


class RSCLIPConfigTests(unittest.TestCase):
    def _values(self) -> dict:
        return yaml.safe_load(
            (CONFIG_DIR / "rsclip_pilot_mixed.yaml").read_text(encoding="utf-8")
        )

    def _load_values(self, values: dict):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
            return load_rsclip_config(config_path)

    def test_pilot_uses_required_splits_schedule_and_enabled_toggle(self) -> None:
        config = load_rsclip_config(CONFIG_DIR / "rsclip_pilot_mixed.yaml")

        self.assertEqual(config.captioned_split, "val")
        self.assertEqual(config.unlabeled_split, "train")
        self.assertEqual(config.supervised_train_fraction, 0.8)
        self.assertTrue(config.retain_caption_supervision)
        self.assertEqual(config.budget_mode, "epochs")
        self.assertEqual(
            [(item.top_k_per_class, item.budget) for item in config.curriculum],
            [(5, 2), (10, 2), (20, 2)],
        )

    def test_pilot_configs_differ_only_by_toggle_and_output(self) -> None:
        mixed = yaml.safe_load(
            (CONFIG_DIR / "rsclip_pilot_mixed.yaml").read_text(encoding="utf-8")
        )
        pseudo_only = yaml.safe_load(
            (CONFIG_DIR / "rsclip_pilot_pseudo_only.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(mixed.pop("retain_caption_supervision"))
        self.assertFalse(pseudo_only.pop("retain_caption_supervision"))
        self.assertNotEqual(mixed.pop("output_dir"), pseudo_only.pop("output_dir"))
        self.assertEqual(mixed, pseudo_only)

    def test_relative_paths_are_resolved_from_config_location(self) -> None:
        config = self._load_values(self._values())

        self.assertTrue(config.checkpoint_path.is_absolute())
        self.assertTrue(config.image_dir.is_absolute())
        self.assertTrue(config.json_path.is_absolute())
        self.assertTrue(config.class_dir.is_absolute())
        self.assertTrue(config.output_dir.is_absolute())

    def test_retain_caption_supervision_requires_a_yaml_boolean(self) -> None:
        for invalid_value in ("false", 0, None):
            values = self._values()
            values["retain_caption_supervision"] = invalid_value
            with self.subTest(value=invalid_value), self.assertRaisesRegex(
                ValueError, "true or false"
            ):
                self._load_values(values)

    def test_test_or_overlapping_source_splits_are_rejected(self) -> None:
        cases = [
            ("test", "train", "test split"),
            ("val", "test", "test split"),
            ("val", "val", "must be distinct"),
        ]
        for captioned, unlabeled, message in cases:
            values = self._values()
            values["captioned_split"] = captioned
            values["unlabeled_split"] = unlabeled
            with self.subTest(
                captioned=captioned, unlabeled=unlabeled
            ), self.assertRaisesRegex(ValueError, message):
                self._load_values(values)

    def test_prompt_template_validation(self) -> None:
        for templates in ([], ["no placeholder"], ["{} and {}"], "a {}"):
            values = self._values()
            values["pseudo_prompt_templates"] = templates
            with self.subTest(templates=templates), self.assertRaisesRegex(
                ValueError, "prompt"
            ):
                self._load_values(values)

    def test_curriculum_and_budget_validation(self) -> None:
        cases = [
            ([], "non-empty"),
            ([{"top_k_per_class": 0, "budget": 1}], "invalid numeric"),
            ([{"top_k_per_class": 5, "budget": 0}], "invalid numeric"),
            (
                [
                    {"top_k_per_class": 10, "budget": 1},
                    {"top_k_per_class": 5, "budget": 1},
                ],
                "nondecreasing",
            ),
        ]
        for curriculum, message in cases:
            values = self._values()
            values["curriculum"] = curriculum
            with self.subTest(curriculum=curriculum), self.assertRaisesRegex(
                ValueError, message
            ):
                self._load_values(values)

        values = self._values()
        values["budget_mode"] = "minutes"
        with self.assertRaisesRegex(ValueError, "epochs.*steps"):
            self._load_values(values)

    def test_step_budget_mode_is_supported_without_changing_rounds(self) -> None:
        values = copy.deepcopy(self._values())
        values["budget_mode"] = "steps"
        values["curriculum"] = [
            {"top_k_per_class": 5, "budget": 300},
            {"top_k_per_class": 10, "budget": 300},
        ]

        config = self._load_values(values)

        self.assertEqual(config.budget_mode, "steps")
        self.assertEqual([item.budget for item in config.curriculum], [300, 300])


if __name__ == "__main__":
    unittest.main()
