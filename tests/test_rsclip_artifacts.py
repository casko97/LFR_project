"""Tests for RS-CLIP plots and zero-shot comparison configuration."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
CONFIG_DIR = OPEN_CLIP_DIR / "configs" / "rsclip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

from main_zeroshot_compare import load_comparison_dataset, load_models  # noqa: E402
from plot_rsclip import (  # noqa: E402
    load_metrics,
    plot_selection,
    plot_source_exposure,
    plot_training,
)


def example_metrics() -> dict[str, object]:
    epochs = []
    for unit, round_number in enumerate((1, 1, 2, 2), start=1):
        epochs.append(
            {
                "epoch": unit,
                "round": round_number,
                "train": {
                    "loss": 1.0 / unit,
                    "caption_images": unit,
                    "pseudo_images": unit + 1,
                    "caption_texts": unit * 2,
                    "pseudo_texts": unit + 2,
                },
                "validation": {"selection_loss": 1.1 / unit},
            }
        )
    return {
        "strategy": "rsclip",
        "retain_caption_supervision": True,
        "epochs": epochs,
        "rounds": [
            {
                "round": 1,
                "selection": {
                    "minimum_confidence": 0.2,
                    "mean_confidence": 0.4,
                    "maximum_confidence": 0.7,
                },
            },
            {
                "round": 2,
                "selection": {
                    "minimum_confidence": 0.3,
                    "mean_confidence": 0.5,
                    "maximum_confidence": 0.8,
                },
            },
        ],
    }


class RSCLIPArtifactTests(unittest.TestCase):
    def test_plotting_writes_all_expected_nonempty_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            document = example_metrics()

            paths = [
                plot_training(document, output_dir),
                plot_source_exposure(document, output_dir),
                plot_selection(document, output_dir),
            ]

            self.assertEqual(
                {path.name for path in paths},
                {"loss.png", "source_exposure.png", "pseudo_confidence.png"},
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_metrics_loader_accepts_a_run_directory_and_rejects_other_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            metrics_path = run_dir / "metrics.json"
            document = example_metrics()
            metrics_path.write_text(json.dumps(document), encoding="utf-8")

            self.assertEqual(load_metrics(run_dir), document)
            document["strategy"] = "supervised"
            metrics_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Not an RS-CLIP"):
                load_metrics(metrics_path)

    def test_zero_shot_comparison_uses_only_test_and_root_best_checkpoints(self) -> None:
        config_path = CONFIG_DIR / "zeroshot_compare_config.yaml"
        dataset = load_comparison_dataset(config_path)
        models = load_models(config_path)

        self.assertEqual(dataset.dataset_split, "test")
        self.assertEqual(
            [model.label for model in models],
            ["non_finetuned_baseline", "rsclip_mixed", "rsclip_pseudo_only"],
        )
        self.assertEqual(
            [model.checkpoint_type for model in models],
            ["openclip", "finetune", "finetune"],
        )
        self.assertTrue(models[1].checkpoint_path.as_posix().endswith("ckp_best.pth"))
        self.assertTrue(models[2].checkpoint_path.as_posix().endswith("ckp_best.pth"))


if __name__ == "__main__":
    unittest.main()
