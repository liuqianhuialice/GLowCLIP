# Complete GLowCLIP handoff report

## 1. Handoff summary

This is the canonical consolidated report for the GLowCLIP Real/AIGC classifier
trained on `glow_dataset`. It combines the dataset audit and
rearrangement record, model description, training protocol, validation history,
held-out test results, robustness results, limitations, repository verification,
and deployment instructions.

**Run date:** 2026-08-31 UTC  
**Task:** binary image classification (`0 = Real`, `1 = AIGC`)  
**Fine-tune start:** fresh seed-42 LoRA/head initialization, with no resume or old
fine-tune checkpoint  
**Backbone:** cached public `openai/clip-vit-base-patch16`  
**Selected checkpoint:** epoch 5, chosen on validation only  
**Frozen operating threshold:** 0.4474602937698364  
**Held-out test policy:** one fixed-threshold test evaluation after model and
threshold selection

The selected model achieved **0.993439 ROC-AUC**, **0.992898 transform-family ×
level macro ROC-AUC**, and **96.2668% accuracy** on 3,268 held-out images. The
dedicated same-pair robustness suite retained **0.983072 ROC-AUC at severity 5**.

The complete handoff package has this layout:

```text
GLowCLIP-handoff/
├── README_FIRST.md
├── COMPLETE_HANDOFF_REPORT.md
├── SHA256SUMS
├── checkpoint/
│   └── final_handoff_checkpoint_glow_dataset.pt
└── github-repo/
    ├── README.md
    ├── MODEL_CARD.md
    ├── baselines_original.ipynb
    ├── configs/
    ├── docs/
    ├── glowclip/
    ├── results/
    ├── tests/
    └── data/
        └── glow_dataset/   # rearranged dataset
```

The pretrained CLIP backbone is not embedded in the fine-tune checkpoint or
package. Transformers loads it from an existing Hugging Face cache or downloads it
once when network access is allowed.

## 2. Executive results

| Result | Value |
|---|---:|
| Test images / matched pairs | 3,268 / 1,634 |
| Test ROC-AUC | **0.993439** |
| Test ROC-AUC 95% pair-bootstrap CI | [0.991503, 0.995216] |
| Family × level macro ROC-AUC | **0.992898** |
| Macro-AUC 95% pair-bootstrap CI | [0.990795, 0.994806] |
| Accuracy | **0.962668** |
| AIGC sensitivity | **0.960832** |
| Real-image recall (TNR) | **0.964504** |
| Average precision | 0.993963 |
| Severity-5 same-pair ROC-AUC | **0.983072** |

At the frozen threshold, the confusion counts were TP=1,570, TN=1,576, FP=58,
and FN=64. The AIGC image scored above its matched real image in 1,623 of 1,634
pairs (99.33%).

Real-image recall (also called the true-negative rate, TNR) is the fraction of
real images correctly classified as real: TN / (TN + FP).

The headline results do not describe every domain equally well. DeepGuardDB test
ROC-AUC was 0.9157 with 82.02% accuracy. The small 30-pair IMAGEN group achieved
0.8450 ROC-AUC. Its level-1 subgroup contains only 20 images and produced the
minimum eligible generator × level AUC of 0.8100.

## 3. Original dataset

The source archive was:

```text
glow_dataset.zip
```

Its size was 2,292,819,469 bytes. ZIP integrity testing passed before extraction.
The extracted source tree was 2,324,233,916 apparent bytes across 62,869 files:

```text
glow_dataset/
├── README.md
├── dataset_summary.json
├── audit/
├── scripts/
├── final_dataset/
│   ├── images/{train,validation,test}/u*/{real,ai}.{jpg,png}
│   ├── images_clean/{train,validation,test}/u*/{real,ai}.{jpg,png}
│   ├── manifests/{train,validation,test,all_images,pairs}.csv
│   └── clean_summary.json
└── robustness_eval/
    ├── images/{clean,t1,t3,t5}/u*/{real,ai}.jpg
    ├── manifests/{clean,t1,t3,t5,all_conditions}.csv
    ├── eval_summary.json
    └── severity.json
```

### 3.1 Dataset composition

| Split | Pairs | Images | Real | AIGC |
|---|---:|---:|---:|---:|
| Train | 13,081 | 26,162 | 13,081 | 13,081 |
| Validation | 1,636 | 3,272 | 1,636 | 1,636 |
| Test | 1,634 | 3,268 | 1,634 | 1,634 |
| **Total** | **16,351** | **32,702** | **16,351** | **16,351** |

| Source | Pairs |
|---|---:|
| Defactify | 8,670 |
| DeepGuardDB | 894 |
| SID_Set | 6,787 |

| Generator | Pairs |
|---|---:|
| DALLE3 | 2,032 |
| FLUX | 6,787 |
| IMAGEN | 298 |
| Midjourney | 1,734 |
| SD | 298 |
| SD21 | 1,734 |
| SD3 | 1,734 |
| SDXL | 1,734 |

There are 9,564 `coco_ladder` pairs and 6,787 `sid_ladder` pairs. Assigned
transform levels contain 5,451 level-1 pairs, 5,451 level-3 pairs, and 5,449
level-5 pairs. Levels from the two families are tracked separately because their
operations and effective severity are not assumed to be equivalent.

## 4. How the original dataset was rearranged

The repository command used for normalization was:

```bash
python -m glowclip.normalize_dataset \
  --source-root /path/to/glow_dataset \
  --output-root data/glow_dataset \
  --link-mode hardlink \
  --verify-hashes
```

Normalization was non-destructive. The source package was not edited. In the
working environment, image files were hard-linked so the clean layout did not
temporarily duplicate image bytes. The delivered tarball stores the actual image
contents and does not depend on those original hard links.

### 4.1 Path mapping

| Original path | Rearranged path | Purpose |
|---|---|---|
| `final_dataset/images/<split>/<pair>/<role>.<ext>` | `images/<split>/<pair>/<role>.<ext>` | Train/validation/test image root |
| `final_dataset/manifests/train.csv` | `manifests/train.csv` | Training rows |
| `final_dataset/manifests/validation.csv` | `manifests/validation.csv` | Model and threshold selection |
| `final_dataset/manifests/test.csv` | `manifests/test.csv` | Held-out labeled test |
| `final_dataset/manifests/all_images.csv` | `manifests/all_images.csv` | Full assigned-level image index |
| `final_dataset/manifests/pairs.csv` | `manifests/pairs.csv` | One row per real/AIGC pair |
| `robustness_eval/images/<condition>/<pair>/<role>.jpg` | `robustness_eval/images/<condition>/<pair>/<role>.jpg` | Evaluation-only repeated conditions |
| `robustness_eval/manifests/*.csv` | `robustness_eval/manifests/*.csv` | Normalized robustness manifests |

No pixels were resized, re-encoded, renamed, or relabeled during rearrangement.
Only directory placement and manifest compatibility fields/paths changed.

### 4.2 Main-manifest field mapping

The original main manifest already contained source provenance, labels,
generator, degradation metadata, dimensions, format, path, and SHA-256. The
normalizer preserved every original column and added:

| Added field | Value |
|---|---|
| `dataset_pair_id` | Copy of the source pair identifier |
| `source` | Copy of `source_dataset` |
| `source_path` | Copy of `source_relative_path` |

It rewrote only `output_path`:

```text
before: final_dataset/images/train/u000001/real.jpg
after:  train/u000001/real.jpg
```

For `pairs.csv`, it similarly added `dataset_pair_id` and `source`, then removed
the `final_dataset/images/` prefix from `real_path` and `ai_path`.

### 4.3 Robustness-manifest field mapping

The original robustness rows did not use the loader's `split` and
`transform_family` fields. The normalizer preserved the original columns and
added:

| Added field | Value |
|---|---|
| `dataset_pair_id` | Copy of the source pair identifier |
| `source` | Copy of `source_dataset` |
| `source_path` | Empty because no separate original path is supplied |
| `split` | Copy of `condition`: `clean`, `t1`, `t3`, or `t5` |
| `transform_family` | Constant `glow_robustness` |

It stripped the packaging prefix from `output_path`:

```text
before: robustness_eval/images/clean/u000002/real.jpg
after:  clean/u000002/real.jpg
```

### 4.4 Resulting runnable structure

```text
data/glow_dataset/
├── images/
│   ├── train/uXXXXXX/{real,ai}.{jpg,png}
│   ├── validation/uXXXXXX/{real,ai}.{jpg,png}
│   └── test/uXXXXXX/{real,ai}.{jpg,png}
├── manifests/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── all_images.csv
│   └── pairs.csv
├── normalization_summary.json
└── robustness_eval/
    ├── images/
    │   ├── clean/uXXXXXX/{real,ai}.jpg
    │   ├── t1/uXXXXXX/{real,ai}.jpg
    │   ├── t3/uXXXXXX/{real,ai}.jpg
    │   └── t5/uXXXXXX/{real,ai}.jpg
    ├── manifests/
    │   ├── clean.csv
    │   ├── t1.csv
    │   ├── t3.csv
    │   ├── t5.csv
    │   └── all_conditions.csv
    └── normalization_summary.json
```

The rearranged tree contains 40,362 files and 1,605,993,348 apparent bytes:
32,702 main images, 7,648 robustness images, ten CSV manifests, and two
normalization summaries. It contains no symlinks.

### 4.5 What was deliberately excluded

- `final_dataset/images_clean` was not used for training or included in the
  rearranged tree. It covers only 11,238 of 16,351 pairs; 5,113 FLUX pairs are
  unavailable, so substituting it would make the training population incomplete.
- `audit/` and `scripts/` describe how the source delivery was constructed but are
  not runtime dataset inputs.
- The original package README and summaries were consolidated into this report and
  the normalized `normalization_summary.json` files.
- `robustness_eval` was retained, but isolated under its own root and never exposed
  to training or validation checkpoint selection.

### 4.6 Rearrangement and leakage audit

The normalizer:

- verified SHA-256 and successful 224×224 decoding for all 32,702 assigned-level
  images and all 7,648 robustness images;
- required exactly one real image with label 0 and one AIGC image with label 1 per
  main pair;
- required generator, transform family, transform level, and operations to agree
  within each pair;
- confirmed zero pair-ID overlap between train, validation, and test;
- confirmed zero source-pair, caption-group, and byte-hash overlap across splits;
- confirmed that the main and robustness image trees exactly matched their
  manifests;
- confirmed that all 956 robustness pairs appear under all four conditions;
- independently checked that the source and rearranged sample images had the same
  inode in hard-link mode.

The source audit identified four very high visual-descriptor cross-source real
image candidates. All four were train-to-train rather than cross-split, so they
were retained without creating validation or test leakage. These were similarity
candidates, not byte-hash overlaps.

## 5. Model

GLowCLIP is a binary forensic image classifier based on
`openai/clip-vit-base-patch16`:

- global branch: final CLS token projected to 256 dimensions;
- local branch: final 14×14 patch grid, pointwise projection, depthwise
  convolution, and mean/std aggregation;
- fusion: input-dependent 256-channel global/local gate;
- LoRA: rank 8, alpha 16, dropout 0.05 on Q/V projections in vision layers 8–11;
- training views: reference plus independently sampled online compound degradation;
- loss: fused classification, auxiliary branch classification, feature
  consistency, and prediction consistency.

Parameter counts:

| Stage | Trainable parameters |
|---|---:|
| Head-only | 990,083 |
| Joint head + LoRA | 1,088,387 |
| Entire model | 86,887,811 |

The handoff checkpoint is 13,123,697 bytes and contains trained adapters/heads,
optimizer/run state, model configuration, threshold, and validation metrics. It
loads the public CLIP backbone separately.

Checkpoint SHA-256:

```text
28f28d7c65a6a96e546fa46e899b3766dd4c058e0302a8a8a49bf24f8f121271
```

## 6. Training protocol

- Seed: 42
- Epochs: one head-only epoch followed by four joint LoRA epochs
- Batch size: 16
- Gradient accumulation: 2
- Effective batch size: 32
- Optimizer: AdamW
- Head learning rate: 5e-4
- LoRA learning rate: 1e-4
- Weight decay: 0.01
- Precision: BF16
- Hardware: one NVIDIA GeForce RTX 5090
- Software: Python 3.12.14, PyTorch 2.7.1+cu128, Transformers 5.16.1
- Approximate five-epoch wall time: 738 seconds (12.3 minutes)
- Network policy: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; cached backbone
  only
- Resume policy: no resume and no previous fine-tune checkpoint

“Fresh fine-tune” means newly initialized LoRA adapters and heads trained from
epoch 1. It does not mean that the public CLIP backbone was initialized randomly.

Checkpoint selection used the unweighted mean of the six transform-family × level
validation AUCs. Each epoch fitted one Youden threshold on validation. The test set
did not influence either choice.

## 7. Training and validation history

| Epoch | Stage | Train loss | Val ROC-AUC | Val robust AUC | Val worst-group AUC | Val accuracy | Threshold | Mean gate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Heads | 0.3878 | 0.9794 | 0.9807 | 0.8403 | 0.9279 | 0.4637 | 0.456 |
| 2 | Joint | 0.2759 | 0.9894 | 0.9884 | **0.8819** | 0.9514 | 0.7050 | 0.397 |
| 3 | Joint | 0.1904 | 0.9915 | 0.9910 | 0.8438 | 0.9563 | 0.6011 | 0.344 |
| 4 | Joint | 0.1484 | 0.9920 | 0.9913623 | 0.8611 | 0.9581 | 0.3919 | 0.330 |
| **5** | **Joint** | **0.1196** | **0.9921** | **0.9914147** | **0.8750** | **0.9590** | **0.4475** | **0.338** |

Epoch 5 exceeded epoch 4 on the predeclared robust score by 0.0000524 and was
selected. Epoch 2 had the best validation worst-group AUC, showing the tradeoff
between aggregate family/level behavior and the noisiest small subgroup. The
fusion gate stayed far from its 0.05/0.95 collapse boundaries.

## 8. Held-out test results

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| ROC-AUC | **0.993439** | [0.991503, 0.995216] |
| Transform-family × level macro ROC-AUC | **0.992898** | [0.990795, 0.994806] |
| Accuracy | **0.962668** | [0.955630, 0.969094] |
| Balanced accuracy | **0.962668** | — |
| AIGC sensitivity/recall | **0.960832** | [0.951025, 0.970012] |
| Real-image recall (TNR) | **0.964504** | [0.955324, 0.973072] |
| Precision | 0.964373 | — |
| F1 | 0.962600 | — |
| Average precision | 0.993963 | — |
| Brier score | 0.028810 | — |
| Binary log loss | 0.105447 | — |
| 10-bin expected calibration error | 0.013938 | — |

Bootstrap intervals used 2,000 deterministic percentile replicates sampled at the
`dataset_pair_id` level, preserving each real/AIGC pair.

### 8.1 Confusion matrix at threshold 0.447460

| Actual / predicted | Real | AIGC |
|---|---:|---:|
| Real | **1,576** | 58 |
| AIGC | 64 | **1,570** |

Mean AIGC probability was 0.0452 for real images and 0.9498 for AIGC images. The
mean matched score margin (`AIGC − Real`) was 0.9046 and the median was 0.9919.

### 8.2 By transform family and assigned level

| Family / level | Images | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | Errors |
|---|---:|---:|---:|---:|---:|---:|
| COCO / 1 | 700 | 0.9895 | 0.9486 | 0.9400 | 0.9571 | 36 |
| COCO / 3 | 604 | 0.9837 | 0.9404 | 0.9272 | 0.9536 | 36 |
| COCO / 5 | 608 | 0.9876 | 0.9474 | 0.9441 | 0.9507 | 32 |
| SID / 1 | 476 | 0.9998 | 0.9895 | 0.9958 | 0.9832 | 5 |
| SID / 3 | 436 | 0.9999 | 0.9931 | 1.0000 | 0.9862 | 3 |
| SID / 5 | 444 | 0.9970 | 0.9775 | 0.9865 | 0.9685 | 10 |

### 8.3 By source dataset

| Source | Pairs | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | Errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepGuardDB | 89 | 0.9157 | 0.8202 | 0.7079 | 0.9326 | 32 |
| Defactify | 867 | 0.9918 | 0.9585 | 0.9608 | 0.9562 | 72 |
| SID_Set | 678 | 0.9990 | 0.9867 | 0.9941 | 0.9794 | 18 |

### 8.4 By generator-associated pair group

Each row contains generated images from that generator and their matched real
images.

| Generator | Pairs | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | Errors |
|---|---:|---:|---:|---:|---:|---:|
| DALLE3 | 203 | 0.9865 | 0.9360 | 0.9507 | 0.9212 | 26 |
| FLUX | 678 | 0.9990 | 0.9867 | 0.9941 | 0.9794 | 18 |
| IMAGEN | 30 | 0.8450 | 0.7333 | 0.6000 | 0.8667 | 16 |
| Midjourney | 174 | 0.9974 | 0.9741 | 0.9598 | 0.9885 | 9 |
| SD | 30 | 0.9633 | 0.8833 | 0.8333 | 0.9333 | 7 |
| SD21 | 172 | 0.9864 | 0.9535 | 0.9244 | 0.9826 | 16 |
| SD3 | 173 | 0.9898 | 0.9480 | 0.9364 | 0.9595 | 18 |
| SDXL | 174 | 0.9951 | 0.9655 | 0.9885 | 0.9425 | 12 |

The three lowest generator × level AUCs were IMAGEN/level 1 (0.8100, 20 images),
IMAGEN/level 3 (0.8642, 18 images), and IMAGEN/level 5 (0.8678, 22 images). The
18-image group is shown for transparency but excluded from the configured
worst-group metric because it is below the 20-image minimum.

## 9. Dedicated same-pair robustness evaluation

The separate robustness tree contains 956 identical test pairs from Defactify and
DeepGuardDB under clean, t1, t3, and t5 conditions. SID_Set has no robustness
tree. The frozen epoch-5 checkpoint and threshold were used without refitting.

| Condition | Images | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 1,912 | 0.9890 | 0.9493 | 0.9372 | 0.9613 | 37 | 60 |
| t1 | 1,912 | 0.9895 | 0.9508 | 0.9372 | 0.9644 | 34 | 60 |
| t3 | 1,912 | 0.9878 | 0.9477 | 0.9362 | 0.9592 | 39 | 61 |
| t5 | 1,912 | 0.9831 | 0.9357 | 0.9320 | 0.9393 | 58 | 65 |

Severity 5 reduced ROC-AUC by 0.00596 and accuracy by 0.01360 relative to clean.
The small t1 improvement is compatible with benign recompression. Across all four
conditions, Defactify ROC-AUC was 0.9923 and DeepGuardDB ROC-AUC was 0.9189.

## 10. Using the package

### 10.1 Install the repository

After extracting the tarball:

```bash
cd GLowCLIP-handoff/github-repo
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e .
```

Install a PyTorch build appropriate for the target GPU. RTX 50-series/Blackwell
GPUs require CUDA 12.8 or newer kernels.

### 10.2 Predict arbitrary unlabeled and unpaired images

The inference path scores each image independently. It does not require labels,
tags, manifests, class directories, matched pairs, or special filenames. A user's
test data may be a flat folder:

```text
my_test_images/
├── IMG_0001.jpg
├── holiday-photo.png
├── upload_83.webp
└── anything.jpeg
```

From `github-repo/`:

```bash
python -m glowclip.predict /path/to/my_test_images/ \
  --checkpoint ../checkpoint/final_handoff_checkpoint_glow_dataset.pt \
  --output predictions.json
```

Directory scanning is recursive. Supported formats are JPG/JPEG, PNG, WebP, BMP,
and TIFF, case-insensitively. Non-image files are ignored. EXIF orientation, RGB
conversion, aspect-ratio-preserving letterboxing, 224×224 resizing, and CLIP
normalization happen automatically.

Each result contains:

```json
{
  "path": "/path/to/my_test_images/IMG_0001.jpg",
  "fake_probability": 0.996623,
  "prediction": "AIGC",
  "gate_mean": 0.393444
}
```

`fake_probability` is the estimated AIGC probability. The checkpoint's threshold
is used unless `--threshold` is supplied. An unlabeled folder can produce
predictions but cannot produce accuracy/AUC without ground-truth labels.

This exact scenario was verified with the real checkpoint on a flat directory of
arbitrarily named images. No pairing or manifest code was invoked. A dedicated
regression test covers flat unlabeled input, mixed extensions, ignored non-images,
and path deduplication.

### 10.3 Re-run the held-out labeled evaluation

```bash
python -m glowclip.evaluate \
  --checkpoint ../checkpoint/final_handoff_checkpoint_glow_dataset.pt \
  --manifest data/glow_dataset/manifests/test.csv \
  --image-root data/glow_dataset/images \
  --output-dir outputs/glow_dataset/test_best_robust
```

Do not pass `--fit-threshold` for final test reporting.

### 10.4 Re-run the same-pair robustness evaluation

```bash
python -m glowclip.evaluate \
  --checkpoint ../checkpoint/final_handoff_checkpoint_glow_dataset.pt \
  --manifest data/glow_dataset/robustness_eval/manifests/all_conditions.csv \
  --image-root data/glow_dataset/robustness_eval/images \
  --output-dir outputs/glow_dataset/robustness_same_pairs
```

### 10.5 Re-run training from epoch 1

The dataset is already located at the paths used by the supplied configuration:

```bash
python -m glowclip.train --config configs/glow_dataset.yaml
```

When the backbone is already cached and network access must remain disabled:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m glowclip.train --config configs/glow_dataset.yaml
```

Do not pass `--resume` when reproducing the fresh fine-tune.

## 11. Repository verification

The repository was checked in both its working tree and a clean packaging tree:

- Ruff lint: passed
- Ruff formatting verification: passed
- Pytest: **12 passed**
- Python wheel construction: passed
- Offline two-batch training smoke test using the cached backbone: passed
- Complete five-epoch fine-tune: passed
- Held-out evaluation using the selected checkpoint: passed
- Same-pair robustness evaluation: passed
- Checkpoint reload and prediction: passed
- Flat unlabeled/unpaired directory prediction with arbitrary filenames: passed
- Clean packaged-repository prediction with the real checkpoint: passed

The GitHub repository intentionally ignores `data/`, `outputs/`, checkpoints,
model caches, and Python/build caches. The complete tarball injects the normalized
dataset under the ignored `github-repo/data/` directory solely to make this handoff
runnable after extraction.

## 12. Limitations and responsible use

- All eight evaluated generator labels occur in training; this is not an
  unseen-generator benchmark.
- Generator and source are confounded. FLUX comes entirely from SID_Set, while
  IMAGEN and SD occur only in DeepGuardDB.
- IMAGEN and SD have only 30 test pairs each. Their per-level groups contain 18–22
  images, so their estimates are unstable.
- DeepGuardDB is materially weaker than the other two sources.
- The same-pair robustness suite excludes SID_Set.
- The stored threshold is calibrated for this validation distribution. New
  generators, camera sources, screenshots, crops, or recompression pipelines can
  shift calibration and error rates.
- The model returns a probability-like forensic signal, not proof of provenance.
  It should not be the sole basis for legal, punitive, moderation, or authorship
  decisions.

## 13. Included machine-readable records

Inside `github-repo/results/`:

- `test_metrics.json`: complete overall and subgroup test metrics
- `robustness_same_pairs_metrics.json`: complete robustness metrics
- `bootstrap_summary.json`: pair-cluster confidence intervals
- `test_analysis.json`: calibration, average precision, and matched-pair summaries
- `validation_history.csv`: all five epochs
- `run_manifest.json`: checkpoint, data, environment, and test-policy provenance

Inside the package root, `SHA256SUMS` covers every included file other than the
checksum list itself. An adjacent checksum file distributed next to the tarball
verifies the compressed archive as a whole.
