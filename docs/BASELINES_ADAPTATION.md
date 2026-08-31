# Baselines adapted from `baselines_original.ipynb`

## Status

This is a static code adaptation of the original notebook retained at the
repository root as `baselines_original.ipynb` (SHA-256
`1394b92658325fb8a135eb1fc94db0527d6167bbdcbe620bcbe4ba7147da1fd7`).
The user requested that it not be run. No baseline training, evaluation, package
installation, or pretrained-weight download was performed during adaptation.

The file rename also updates its stored Google Drive folder strings from the old
notebook name to `baselines_original`; no executable logic or notebook outputs
were otherwise changed.

The notebook's stored outputs are archival notebook state, not results verified by
this repository. Some cells have null or out-of-order execution counters, and two
stored ResNet printouts disagree despite referencing the same variable. They are
therefore intentionally not copied into the project result tables.

## Implemented baselines

| CLI name | Notebook model | Backbone/initialization | Epochs | Learning rate |
|---|---|---|---:|---:|
| `resnet18` | ResNet baseline | torchvision ResNet-18 default ImageNet weights; full fine-tune | 10 | 1e-4 |
| `openclip_linear` | CLIP baseline | frozen OpenCLIP ViT-B/32, `laion2b_s34b_b79k`; linear head | 20 | 1e-3 |
| `npr` | NPR baseline | notebook's two-stage bottleneck CNN over nearest-neighbor reconstruction residuals | 25 | 2e-4 |
| `vib` | VIB baseline | frozen OpenCLIP ViT-L/14, `openai`; 256-dimensional stochastic bottleneck | 10 | 1e-4 |

All use binary labels `real=0`, `ai=1`, `BCEWithLogitsLoss`, Adam, batch size
32, and threshold 0.5. NPR retains its StepLR schedule (`step_size=10`,
`gamma=0.9`). VIB retains `beta=1e-4`, dropout 0.5, and the notebook's
`softplus(raw_std - 5)` parameterization.

## Optional dependencies—not installed this round

The main GLowCLIP installation is unchanged. The notebook baselines additionally
need:

- `torchvision` for ResNet-18 and image transformations;
- `open_clip_torch` for the OpenCLIP and VIB baselines.

They are recorded in `requirements-baselines.txt` but were deliberately not
installed. When the user decides to run the reproduction, they may install them
manually in an appropriate environment:

```bash
pip install -r requirements-baselines.txt
```

This command was documented only; it was not executed. The ResNet and OpenCLIP
factories may also download pretrained weights on first use. Cache or provision
those weights before enabling offline mode.

The original notebook imported `scikit-learn`, `matplotlib`, and `pytz`. The
adaptation uses repository-native metrics, writes JSON/CSV rather than interactive
plots, and uses standard UTC timestamps, so those three packages are not required.

## Expected data layout

Pass either the `images` directory or its parent:

```text
images/
├── train/
│   └── pair_id/{real.jpg,ai.jpg}
├── validation/
│   └── pair_id/{real.jpg,ai.jpg}
└── test/
    └── pair_id/{real.jpg,ai.jpg}
```

JPG/JPEG, PNG, WebP, BMP, and TIFF files are accepted. Every pair must contain
exactly one real and one AI member. The runner rejects incomplete pairs, duplicate
role files, missing splits, and pair IDs shared across splits before training.

## Running later

Run one baseline:

```bash
python -m glowclip.baselines.runner \
  --config configs/baselines.yaml \
  --data-root /path/to/two_dataset_combined/images \
  --baseline resnet18 \
  --output-dir outputs/baselines
```

Repeat `--baseline` to select multiple models. Omit it to run all four. Optional
CLI overrides are available for device, worker count, and seed.

Each baseline writes:

```text
outputs/baselines/<baseline>/
├── best.pt
├── history.json
├── test_metrics.json
└── test_predictions.csv
```

`run_summary.json` records the source-notebook hash, data counts, device, seed,
selected epoch, validation AUC, and test metrics.

## Correctness and portability fixes

The adaptation preserves architectures and headline hyperparameters while fixing
the following notebook problems:

1. Google Drive mounting, Colab shell commands, hard-coded `/content` paths, and
   in-notebook installation cells were replaced by normal CLI arguments.
2. The CLIP notebook assigned `clip_test_data` from `val_dir`; the repository uses
   `test_dir` and audits split isolation.
3. Loader functions ignored their `num_workers` arguments and always used four;
   the configured/CLI value is now honored, including the zero-worker case.
4. The notebook silently accepted partial pairs; the repository fails loudly.
5. No global random seed was set. The adaptation records and applies seed 42.
6. Checkpoint cells for CLIP, NPR, and VIB all saved `model.state_dict()` from the
   ResNet variable, labeled every model `ResNet18`, and referenced undefined
   `best_val_auc`. Each model now saves its own validation-selected state and
   metadata.
7. Frozen OpenCLIP weights are omitted from checkpoints and reconstructed from the
   recorded backbone/tag, preventing unnecessary multi-gigabyte duplication.
8. VIB sampled a random latent during validation and test. Training remains
   stochastic, but evaluation uses posterior mean `mu` for deterministic metrics.
9. Notebook test reporting depended on mutable global state and contained stale,
   inconsistent cell outputs. The runner evaluates the untouched test split once,
   after validation selection, and writes machine-readable predictions.
10. GPU-name printing assumed CUDA was available; device resolution now supports
    CPU and fails clearly when unavailable CUDA is explicitly requested.

## Deliberately preserved notebook choices

- OpenCLIP linear-probe training uses the notebook's evaluation transform rather
  than the unused `clip_train_transform` returned by OpenCLIP.
- VIB uses the single OpenCLIP preprocessing transform for every split, as in the
  notebook.
- ResNet and NPR preprocessing, optimizers, NPR architecture, schedules, and fixed
  threshold remain as written.

These choices make the code traceable to the supplied baseline rather than silently
turning it into a different experiment.
