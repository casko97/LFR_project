# RS-CLIP curriculum experiments

This directory configures an RS-CLIP-style class-balanced pseudo-label
curriculum for RSICD. It follows the pseudo-only curriculum from the
[RS-CLIP paper](https://doi.org/10.1016/j.jag.2023.103497) and
[reference implementation](https://github.com/lx709/RS-CLIP), with an optional
repository-specific extension that retains real-caption supervision.

## Data isolation

- `dataset_rsicd.json` `train` records are loaded strictly as unlabeled images;
  their captions are never read by the RS-CLIP runner.
- `val` records are deterministically divided by whole image into 80% caption
  training and 20% held-out validation using the configured seed.
- The held-out subset always uses all unique captions and the common per-caption validation objective.

## Caption-retention modes

The pilot configurations have identical seeds, schedules, prompts, and
optimization settings apart from their toggle and output directory:

- `rsclip_pilot_mixed.yaml` sets `retain_caption_supervision: true`. Every
  round trains from one shuffled pool containing real-caption records and
  pseudo image–prompt pairs.
- `rsclip_pilot_pseudo_only.yaml` sets
  `retain_caption_supervision: false`. No captioned records participate in an
  optimizer update; this is the mode corresponding to the original RS-CLIP
  pseudo-only strategy.

Both modes regenerate predictions from the current model at the beginning of
every round, independently select the top K images for each class, retain
cross-class overlaps as separate pairs, and carry model weights into the next
round. Optimizer and scheduler state are reset per round.

## Running

Run the complete mixed and pseudo-only pilots from the repository root:

```bash
python open-clip/main_rsclip_finetune.py \
  --config open-clip/configs/rsclip/rsclip_pilot_mixed.yaml \
  --device cuda

python open-clip/main_rsclip_finetune.py \
  --config open-clip/configs/rsclip/rsclip_pilot_pseudo_only.yaml \
  --device cuda
```

For a bounded smoke run:

```bash
python open-clip/main_rsclip_finetune.py \
  --config open-clip/configs/rsclip/rsclip_pilot_mixed.yaml \
  --device cuda --batch-size 2 --max-rounds 1 \
  --max-unlabeled-images 64 --max-train-batches 1 \
  --max-validation-batches 1 --output-dir results/rsclip_smoke
```

* The supplied pilots use K=`5 → 10 → 20` and two epochs per round. 
* For an exact optimizer-update budget, set `budget_mode: steps` and specify each round's
`budget` 
* `--max-steps-per-round` provides a bounded override for smoke tests, `--max-train-batches` limits batches per epoch or per cycled pass, while`--max-rounds` and `--max-unlabeled-images` bound selection work.

## Artifacts and plots

Each round writes `round_NN/pseudo_labels.json` and
`round_NN/ckp_last.pth`. The run root contains:

- `ckp_best.pth`, selected globally by held-out real-caption validation loss;
- `ckp_last.pth`, the final completed training unit;
- `metrics.json`, including the toggle, curriculum, selection statistics,
  image/text exposure by source, and validation history;
- `config.yaml`, the copied input configuration;
- `train.log`, the captured terminal output.

Checkpoints include model, optimizer, scheduler and RNG state, round/unit
identity, pseudo-manifest identity, source-aware training metrics, and
validation metrics. They are loadable by the existing comparison evaluator as
`checkpoint_type: finetune`.

Generate loss, source-exposure, and pseudo-confidence plots with:

```bash
python open-clip/plot_rsclip.py results/rsclip_pilot_mixed
python open-clip/plot_rsclip.py results/rsclip_pilot_pseudo_only
```

The source plot separately reports real-caption and pseudo image/text exposure;
the loss and exposure plots mark curriculum-round boundaries.

## Zero-shot comparison

After both complete pilots have produced `ckp_best.pth`, compare them with the
same non-fine-tuned OpenCLIP baseline on the untouched RSICD `test` split:

```bash
python open-clip/main_zeroshot_compare.py \
  --models-config open-clip/configs/rsclip/zeroshot_compare_config.yaml \
  --device cuda \
  --output-dir evaluations/rsclip_comparison
```

The comparison config intentionally points to the root-level best checkpoints,
so no changes to the existing evaluator are required.
