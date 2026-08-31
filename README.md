<p align="center">
  <img src="assets/glowclip-banner.svg" width="100%" alt="GLowCLIP — Global–Local Weighted Feature Fusion for Robust AI-Generated Image Detection">
</p>

<h1 align="center">GLowCLIP</h1>

<p align="center">
  <strong>Global–Local Weighted Feature Fusion for Robust AI-Generated Image Detection</strong>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Backbone" src="https://img.shields.io/badge/Backbone-CLIP_ViT--B%2F16-8B5CF6">
  <img alt="Task" src="https://img.shields.io/badge/Task-Real_vs_AIGC-0EA5E9">
  <img alt="Robustness" src="https://img.shields.io/badge/Training-Degradation_aware-EC4899">
  <a href="https://huggingface.co/spaces/xhan0219/GLowCLIP"><img alt="Hugging Face Space" src="https://img.shields.io/badge/Live_Demo-Hugging_Face-FFD21E"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#dataset-construction">Dataset</a> ·
  <a href="#results-at-a-glance">Results</a> ·
  <a href="#how-it-works">Architecture</a> ·
  <a href="https://huggingface.co/spaces/xhan0219/GLowCLIP">Live demo</a> ·
  <a href="#training-and-evaluation">Train & evaluate</a> ·
  <a href="#documentation">Documentation</a>
</p>

GLowCLIP is a robust binary forensic classifier for distinguishing real images
from AI-generated images (`0 = Real`, `1 = AIGC`). It combines CLIP's global CLS
representation with local patch-token evidence, then learns how much to trust each
feature channel for every input image.

The name describes the model directly: **G**lobal + **Lo**cal + **W**eighted +
**CLIP**.

> [!IMPORTANT]
> **Your deployment images do not need labels, matched real/AI pairs, special
> filenames, or class folders.** Point `glowclip-predict` at any image or directory;
> every image is scored independently.

## Why GLowCLIP?

| Capability | What the repository provides |
|---|---|
| Global + local evidence | A CLS-token branch captures semantic/structural cues while a spatial patch branch captures local forensic traces. |
| Input-adaptive fusion | A learned 256-channel gate weights global and local evidence separately for every image. |
| Efficient fine-tuning | Rank-8 LoRA updates only the Q/V projections in the last four CLIP vision layers; the remaining backbone stays frozen. |
| Robustness training | Reference and online compound-degraded views share labels and are coupled with feature- and prediction-consistency losses. |
| Real-world inference | Recursive, unlabeled, unpaired folder inference supports JPEG, PNG, WebP, BMP, and TIFF. |
| Reproducible evaluation | Fixed-threshold held-out testing, subgroup metrics, pair-bootstrap confidence intervals, and machine-readable results are included. |

## Dataset construction

`glow_dataset` was built by consolidating three complementary source datasets into
a unified, balanced real-versus-AIGC benchmark. It contains **16,351 matched
real–AI pairs / 32,702 images**, with exactly 16,351 real and 16,351 AI-generated
images.

| Source | Matched pairs | Images |
|---|---:|---:|
| Defactify | 8,670 | 17,340 |
| SID_Set | 6,787 | 13,574 |
| DeepGuardDB | 894 | 1,788 |
| **Total** | **16,351** | **32,702** |

The AI side covers eight generator families: FLUX (6,787 pairs), DALL·E 3
(2,032), Midjourney, Stable Diffusion 2.1, Stable Diffusion 3, and SDXL (1,734
each), plus Imagen and Stable Diffusion (298 each).
The pairs are divided at pair level into 13,081 training, 1,636 validation, and
1,634 test pairs, so the two members of a pair always remain in the same split.
All 32,702 files are verified with SHA-256 hashes, and cross-source duplicate
auditing is included in the data-preparation pipeline.

### Construction pipeline

1. **Source integration:** collect complementary real and AI images from
   Defactify, SID_Set, and DeepGuardDB and map them into one common schema.
2. **Pairing and semantic quality control:** organize images into real–AI pairs
   using native caption relationships or CLIP image similarity. Defactify pairs
   satisfy a CLIP similarity threshold of at least 0.75, while SID pairs satisfy
   a threshold of at least 0.80.
3. **Semantic matching:** favor pairs with related subjects and broad semantic
   content, with the goal of reducing obvious topic-level class differences.
4. **Pair-aware splitting:** assign the complete pair to train, validation, or
   test before model training, preventing the two sides of a pair from crossing
   split boundaries.
5. **Degradation construction:** create controlled modified views for
   degradation-aware training and robustness evaluation while preserving the
   original label.
6. **Integrity and duplicate audit:** verify every delivered file with SHA-256
   and inspect cross-source duplicate candidates before evaluation.

### Dataset quality and advantages

- **Substantial real-image coverage.** The dataset contains a large and diverse
  collection of genuine photographs rather than relying on a small real-image
  reference pool.
- **Semantically aligned, harder examples.** AI images are paired with real images
  through native caption relationships or CLIP semantic similarity rather than
  random class-level concatenation. This reduces obviously unrelated content; it
  does not assume that every AI image is photorealistic or visually
  indistinguishable from a photograph.
- **Reduced semantic shortcuts.** Real and AI images are paired and selected to
  reduce easy differences in subject matter and broad semantic content. The
  design aims to make topic-only classification less effective, without claiming
  that composition, viewpoint, lighting, color, or style are fully matched.
- **Multi-generator coverage.** Eight generator families provide broader visual
  diversity than a benchmark built around a single synthesis model.
- **Deployment-oriented robustness.** Clean images and controlled modified-image
  evaluation support measurement under compression and other real-world image
  degradation.

### Dataset novelty

The dataset contribution is not only its size. Its key novelty is the combination
of strict real–AI pairing, pair-level splitting, semantically aligned hard pairs,
multi-source and multi-generator coverage, integrity verification, duplicate
auditing, and degradation-aware evaluation in one reproducible benchmark. This
design specifically reduces dataset shortcuts and makes the benchmark better
aligned with difficult real-world detection.

## Results at a glance

The selected epoch-5 checkpoint was fine-tuned with fresh LoRA adapters and heads
on `glow_dataset`. The public pretrained CLIP backbone was loaded from
cache; it was not retrained from random initialization. Training used random seed
42 on a single NVIDIA GeForce RTX 5090 with 32 GB of GPU memory.

### Held-out modified-image test split

The main test split contains the dataset's assigned degradation levels rather than
the optional clean-image tree: 3,268 images / 1,634 matched pairs, with the
validation-selected threshold frozen before test evaluation.

| Model | Accuracy | AUROC | AP | F1-score |
|---|---:|---:|---:|---:|
| **GLowCLIP (ours)** | **96.27%** | **0.9934** | **0.9940** | **96.26%** |
| ResNet+LP | 87.34% | 0.9561 | 0.9555 | 88.09% |
| CLIP+LP | 89.60% | 0.9614 | 0.9632 | 89.59% |
| NPR(CVPR 2024) | 78.82% | 0.8836 | 0.8850 | 80.61% |
| VIB(CVPR 2025) | 90.59% | 0.9693 | 0.9703 | 90.76% |

GLowCLIP additionally achieved **0.9929** transform-family × level macro
ROC-AUC and **96.08%** AIGC recall.

Against the strongest reported baseline in the table, VIB, GLowCLIP improves
accuracy by **5.68 percentage points**, AUROC by **0.0241**, AP by **0.0237**, and
F1-score by **5.50 percentage points**. Compared with the CLIP linear-probe
baseline, it improves both accuracy and F1-score by **6.67 percentage points** and
AUROC by **0.0320**. These gains show the value of learning global–local fusion and
degradation consistency beyond using a frozen CLIP representation alone.

See the [full training report](docs/TRAINING_REPORT.md) for confidence intervals,
subgroup results, and limitations.

## How it works

### Method construction

GLowCLIP is built around five components:

1. A pretrained CLIP ViT-B/16 backbone extracts a shared visual token sequence.
2. A global branch projects the CLS token into a 256-dimensional representation
   for semantic and structural evidence.
3. A local branch processes the 14 × 14 patch-token grid to preserve spatially
   localized forensic traces.
4. A learned 256-channel gate decides, independently for each input, how much
   global and local evidence to retain in the fused representation.
5. Degradation-aware training couples each reference image with an online
   compound-degraded view using classification, feature-consistency, and
   prediction-consistency objectives.

Rank-8 LoRA adapters update only the Q/V projections in the last four CLIP vision
layers. The rest of the pretrained backbone remains frozen, concentrating
fine-tuning capacity on the task-specific evidence and fusion heads.

### Method advantages and novelty

- **Adaptive global–local reasoning.** Unlike a fixed average or concatenation,
  the channel-wise gate changes the fusion decision for every image and feature
  channel.
- **Fine-grained forensic sensitivity.** The local branch retains patch-level
  traces that can disappear when classification relies only on a global token.
- **Semantic context without losing local evidence.** The global branch helps
  interpret structure and content while the local branch focuses on spatial
  artifacts.
- **Robustness learned explicitly.** Feature and prediction consistency encourage
  the same decision after realistic image degradation instead of treating
  robustness as an evaluation-only property.
- **Parameter-efficient adaptation.** Targeted LoRA updates avoid full-backbone
  fine-tuning while still adapting the pretrained representation.

The key design contribution of GLowCLIP is the integration of per-image,
per-channel global–local gating with degradation-consistent LoRA adaptation.
Together, these components address both clean-image discrimination and stability
after image transformation in a single end-to-end detector.

```mermaid
flowchart LR
    I[Input image] --> C[CLIP ViT-B/16]
    C --> G[Global branch<br/>CLS token]
    C --> L[Local branch<br/>14 × 14 patch grid]
    G --> W{Channel-wise<br/>fusion gate}
    L --> W
    W --> F[Fused 256-D feature]
    F --> P[Real / AIGC score]

    style I fill:#0f172a,stroke:#38bdf8,color:#fff
    style C fill:#172554,stroke:#60a5fa,color:#fff
    style G fill:#164e63,stroke:#22d3ee,color:#fff
    style L fill:#4c1d95,stroke:#c084fc,color:#fff
    style W fill:#581c87,stroke:#e879f9,color:#fff
    style F fill:#3b0764,stroke:#f0abfc,color:#fff
    style P fill:#701a75,stroke:#f5d0fe,color:#fff
```

During training, each reference image is paired with an online compound-degraded
view. Both pass through the same network. The objective combines fused
classification, auxiliary global/local classification, cosine feature
consistency, and symmetric prediction consistency:

```text
L = L_fused + λ_aux L_aux + λ_feature L_feature + λ_prediction L_prediction
```

At inference time, only one forward pass per input image is required.

## Live demo

Try GLowCLIP in the public
[Hugging Face Space](https://huggingface.co/spaces/xhan0219/GLowCLIP). Upload a
JPEG, PNG, WebP, BMP, or TIFF image to view its AIGC score, predicted label,
decision threshold, and mean global–local fusion-gate value.

## Quick start

### 1. Install

Python 3.10 or later is required. Install the PyTorch build appropriate for your
hardware, then install GLowCLIP:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e .
```

For an RTX 50-series / Blackwell GPU, use a CUDA 12.8-or-newer PyTorch build. This
is one known-good setup:

```bash
pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The public `openai/clip-vit-base-patch16` backbone is fetched from Hugging Face on
first use unless it is already cached. To use an existing local cache without any
download, prefix commands with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 2. Add the checkpoint

Download the trained [`ckpt_GLOWCLIP.pt`](https://drive.google.com/file/d/1pFgzd-avyml6hsxmdwEwpfBs4AUYSult/view?usp=sharing)
and place it anywhere convenient, for example:

```text
checkpoints/ckpt_GLOWCLIP.pt
```

The checkpoint is 13,123,697 bytes and has SHA-256 checksum
`28f28d7c65a6a96e546fa46e899b3766dd4c058e0302a8a8a49bf24f8f121271`.
It stores the model configuration and validation-selected decision threshold
`0.4474602938`.

Checkpoints produced before the GLowCLIP rename remain compatible: their tensor
keys and model configuration did not change.

### 3. Predict your own images

A flat folder is enough:

```text
my_test_images/
├── IMG_0001.jpg
├── holiday-photo.png
├── upload_83.webp
└── any-name-is-fine.jpeg
```

```bash
glowclip-predict my_test_images/ \
  --checkpoint checkpoints/ckpt_GLOWCLIP.pt \
  --output predictions.json
```

You can mix individual files and directories. Directory traversal is recursive:

```bash
glowclip-predict image.jpg first_directory/ second_directory/ \
  --checkpoint checkpoints/ckpt_GLOWCLIP.pt \
  --output predictions.json
```

Each result includes the source path, AI-generated score, predicted label,
and diagnostic mean fusion-gate value:

```json
{
  "threshold": 0.4474602938,
  "predictions": [
    {
      "path": "/data/upload_83.webp",
      "fake_score": 0.982741,
      "prediction": "AIGC",
      "gate_mean": 0.331204
    }
  ]
}
```

EXIF orientation correction, RGB conversion, aspect-ratio-preserving
letterboxing, resizing, and CLIP normalization are automatic.

## Training and evaluation

### Supported training layout

Training and labeled evaluation use manifests and matched pairs so split leakage
can be audited:

```text
data/
├── images/
│   ├── train/pair_*/{real,ai}.*
│   ├── validation/pair_*/{real,ai}.*
│   └── test/pair_*/{real,ai}.*
└── manifests/{train,validation,test}.csv
```

Unlabeled inference does **not** require this layout; it is only for training and
metric-producing evaluation.

### Prepare the original archives

```bash
glowclip-prepare \
  --images-zip images.zip \
  --manifests-zip manifests.zip \
  --output-dir data \
  --verify-hashes
```

### Normalize `glow_dataset`

The normalizer creates a clean, non-destructive layout and can use hard links to
avoid duplicating image bytes:

```bash
glowclip-normalize-dataset \
  --source-root /path/to/glow_dataset \
  --output-root data/glow_dataset \
  --link-mode hardlink \
  --verify-hashes
```

The main training tree contains assigned transformed images. Optional clean and
evaluation-only trees are excluded from training.

### Fine-tune

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
glowclip-train --config configs/glow_dataset.yaml
```

The default schedule is one head-only epoch followed by four joint LoRA epochs in
BF16. Override YAML values from the command line without modifying a config:

```bash
glowclip-train \
  --config configs/default.yaml \
  --set data.batch_size=8 \
  --set train.gradient_accumulation=4
```

Resume an interrupted run with `--resume outputs/glowclip/last.pt`.

### Evaluate labeled images

```bash
glowclip-evaluate \
  --checkpoint checkpoints/ckpt_GLOWCLIP.pt \
  --manifest data/glow_dataset/manifests/test.csv \
  --image-root data/glow_dataset/images \
  --output-dir outputs/test
```

The checkpoint's validation-selected threshold is used by default. Do not pass
`--fit-threshold` when producing final test metrics.

## Repository map

```text
GLowCLIP/
├── assets/glowclip-banner.svg
├── configs/
│   ├── default.yaml
│   ├── glow_dataset.yaml
│   └── baselines.yaml
├── docs/
│   ├── GLOWCLIP_Method_Guide.md
│   ├── TRAINING_REPORT.md
│   ├── COMPLETE_HANDOFF_REPORT.md
│   └── BASELINES_ADAPTATION.md
├── glowclip/
│   ├── model.py
│   ├── degradations.py
│   ├── normalize_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── baselines/
├── results/                 # compact machine-readable result summaries
├── tests/
├── baselines_original.ipynb
├── MODEL_CARD.md
├── Makefile
└── pyproject.toml
```

The source repository intentionally excludes datasets, model caches, prediction
rows, and trained checkpoints.

## Adapted notebook baselines

The four models from `baselines_original.ipynb` are available as maintainable
modules: ImageNet-pretrained ResNet-18, frozen OpenCLIP ViT-B/32 + linear probe,
NPR, and a VIB head over frozen OpenCLIP ViT-L/14.

The reported baseline results are shown beside GLowCLIP in
[Results at a glance](#results-at-a-glance).

Their optional dependencies are listed separately and are **not** installed with
GLowCLIP:

```bash
pip install -r requirements-baselines.txt
glowclip-baseline-reproduction \
  --data-root /path/to/images \
  --baseline resnet18 \
  --output-dir outputs/baselines
```

See the [baseline adaptation notes](docs/BASELINES_ADAPTATION.md) for implementation
details.

## Development

Install development dependencies and run the complete local check:

```bash
pip install -e '.[dev]'
make check
```

The unit tests use a tiny dummy vision backbone. They do not download CLIP weights
or require a dataset.

## Documentation

| Document | Contents |
|---|---|
| [Method guide](docs/GLOWCLIP_Method_Guide.md) | Architecture, fusion mechanism, LoRA placement, degradation-aware loss, and implementation notes |
| [Model card](MODEL_CARD.md) | Intended use, training data summary, evaluation, limitations, and checkpoint identity |
| [Complete handoff report](docs/COMPLETE_HANDOFF_REPORT.md) | Consolidated data, training, testing, robustness, and delivery record |
| [Training report](docs/TRAINING_REPORT.md) | Main experiment, confidence intervals, subgroup analysis, and robustness results |

## Responsible use and limitations

GLowCLIP produces a forensic score, not proof of provenance.
Performance can shift with unseen generators, screenshots, recompression, editing,
camera pipelines, and source-domain changes. Generator and source are also partly
confounded in the training corpus. Do not use a single model score as the sole
basis for punitive, legal, authorship, or moderation decisions.

For the full scope of validated and unvalidated behavior, read the
[model card](MODEL_CARD.md).
