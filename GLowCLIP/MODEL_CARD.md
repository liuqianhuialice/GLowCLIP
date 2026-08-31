# Model card: GLowCLIP Real/AIGC detector

## Model description

GLowCLIP is a binary forensic image classifier built on
`openai/clip-vit-base-patch16`. It uses the final CLS token for global evidence and
the final 14×14 patch-token grid for local evidence. A lightweight spatial patch
head aggregates patch mean and standard deviation, and a learned channel-wise gate
fuses the two 256-dimensional branches.

The CLIP backbone is frozen except for rank-8 LoRA residuals on query and value
projections in vision encoder layers 8–11. Joint training uses reference and
online compound-degraded views with classification, auxiliary branch,
feature-consistency, and prediction-consistency losses.

## Intended use

- Research and evaluation of real-versus-AIGC image classification
- Controlled batch scoring where the operating threshold is validated for the
  deployment domain
- Reproduction and extension of global/local forensic feature fusion

The output is a probabilistic signal, not proof of image provenance. It should not
be the sole basis for punitive, legal, moderation, or authorship decisions.

## Training data summary

The selected experiment used `glow_dataset`, containing 16,351 matched
real/AIGC pairs:

- 13,081 train pairs
- 1,636 validation pairs
- 1,634 test pairs
- sources: Defactify, DeepGuardDB, and SID_Set
- AIGC generators: DALLE3, FLUX, IMAGEN, Midjourney, SD, SD21, SD3, and SDXL
- transform families: `coco_ladder` and `sid_ladder`, each with levels 1, 3, and 5

All 32,702 assigned-level images passed SHA-256 and decode checks. Pair IDs,
source-pair IDs, caption groups, and byte hashes had zero overlap across splits.
The dataset itself is not distributed with this repository.

Fine-tuning started with newly initialized LoRA adapters and classification/fusion
heads at seed 42; no earlier fine-tune checkpoint was loaded. The public pretrained
CLIP backbone was loaded from an existing local cache.

## Evaluation results

The validation-selected epoch-5 checkpoint and its fixed threshold of
0.4474602938 achieved:

| Metric | Held-out result |
|---|---:|
| ROC-AUC | 0.993439 |
| ROC-AUC 95% pair-bootstrap CI | [0.991503, 0.995216] |
| Transform-family × level macro ROC-AUC | 0.992898 |
| Macro-AUC 95% pair-bootstrap CI | [0.990795, 0.994806] |
| Average precision | 0.993963 |
| Accuracy | 0.962668 |
| AIGC recall | 0.960832 |
| Brier score | 0.028810 |

The confusion counts were TP=1,570, TN=1,576, FP=58, and FN=64 on 3,268
images. Full subgroup and robustness results are in
[the training report](docs/TRAINING_REPORT.md).

## Limitations

- All eight evaluated generator labels occur in training; this is not an
  unseen-generator benchmark.
- Source and generator are confounded. FLUX comes from SID_Set, while IMAGEN and
  SD occur only in DeepGuardDB.
- DeepGuardDB was materially weaker (0.9157 ROC-AUC, 82.02% accuracy), and the
  30-pair IMAGEN test group achieved 0.8450 ROC-AUC.
- Per-level IMAGEN/SD results are based on only 18–22 images and are unstable.
- The dedicated same-pair robustness suite covers Defactify and DeepGuardDB but
  not SID_Set.
- Calibration and the stored threshold can shift with new generators, image
  sources, recompression, screenshots, or capture devices.
- Dataset-specific correlations may not represent intrinsic properties of all
  AI-generated imagery.

## Checkpoint information

The selected checkpoint is intentionally distributed outside the source archive:

```text
final_handoff_checkpoint_glow_dataset.pt
```

It is 13,123,697 bytes, stores the trained adapters/heads and run state, and loads
the public `openai/clip-vit-base-patch16` backbone separately. Expected SHA-256:

```text
28f28d7c65a6a96e546fa46e899b3766dd4c058e0302a8a8a49bf24f8f121271
```

## Environment

The five-epoch run completed in approximately 12.3 minutes on one NVIDIA GeForce
RTX 5090 using BF16, PyTorch 2.7.1+cu128, and Transformers 5.16.1. Hardware,
software versions, and data-loader behavior can affect throughput and exact
reproducibility.
