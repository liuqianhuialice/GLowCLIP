# GLowCLIP training and test report: glow_dataset

> The source repository deliberately excludes the dataset, raw prediction rows,
> pretrained Hugging Face cache, and trained weights. The selected fine-tune
> checkpoint is delivered separately.

**Run date:** 2026-08-31 UTC  
**Task:** binary image classification (`0 = Real`, `1 = AIGC`)  
**Fine-tune policy:** fresh seed-42 LoRA/head initialization; no previous
fine-tune checkpoint or resume state  
**Selected model:** validation-best robust checkpoint from epoch 5  
**Test policy:** checkpoint and threshold selected on validation, followed by one
fixed-threshold evaluation of the held-out test split

## Executive summary

The selected GLowCLIP model achieved **0.9934 ROC-AUC** and **96.27% accuracy**
on 3,268 held-out images (1,634 matched real/AIGC pairs). Its transform-family ×
level macro ROC-AUC was **0.9929**. Pair-cluster bootstrap 95% intervals were
**[0.9915, 0.9952]** for ROC-AUC and **[0.9556, 0.9691]** for accuracy.

At the frozen validation-selected threshold of 0.44746, the test confusion counts
were TP=1,570, TN=1,576, FP=58, and FN=64. The AIGC member scored above its matched
real member in 1,623 of 1,634 pairs (**99.33%**).

The package's separate same-pair robustness suite showed a modest decline from
clean to severity 5: ROC-AUC changed from **0.9890 to 0.9831**, and accuracy from
**94.93% to 93.57%**. This suite covers Defactify and DeepGuardDB, not SID_Set.

Aggregate results are strong, but performance is not uniform. DeepGuardDB test
ROC-AUC was 0.9157 and accuracy was 82.02%. The small IMAGEN group (30 pairs) had
0.8450 ROC-AUC; its 20-image level-1 subgroup produced the reported minimum
eligible generator × level AUC of 0.8100. These subgroup estimates are far less
stable than the full-test result.

## Dataset rearrangement and integrity

The original package was normalized non-destructively before training. Its
manifests used a source-specific pair identifier and paths rooted at
`final_dataset/images/`; the repository normalizer added loader-compatible fields,
rewrote paths relative to a clean image root, and hard-linked images into:

```text
data/glow_dataset/
├── images/{train,validation,test}/u*/{real,ai}.*
├── manifests/{train,validation,test,all_images,pairs}.csv
└── robustness_eval/
    ├── images/{clean,t1,t3,t5}/u*/{real,ai}.jpg
    └── manifests/{clean,t1,t3,t5,all_conditions}.csv
```

Hard links avoided duplicating image bytes. The source package was left unchanged.
The training manifests include only each pair's assigned transform level; the
clean tree and same-pair robustness tree were kept outside training and model
selection.

### Main split

| Split | Pairs | Images | Real | AIGC |
|---|---:|---:|---:|---:|
| Train | 13,081 | 26,162 | 13,081 | 13,081 |
| Validation | 1,636 | 3,272 | 1,636 | 1,636 |
| Test | 1,634 | 3,268 | 1,634 | 1,634 |
| **Total** | **16,351** | **32,702** | **16,351** | **16,351** |

Pair sources were Defactify (8,670), DeepGuardDB (894), and SID_Set (6,787).
The AIGC generators were DALLE3, FLUX, IMAGEN, Midjourney, SD, SD21, SD3, and
SDXL. The COCO-derived and SID-derived packages use different degradation ladders,
so levels with the same number are not assumed to have identical severity across
families.

The normalization audit:

- verified the SHA-256 and 224×224 decode of all 32,702 assigned-level images;
- required exactly one real (`0`) and one AIGC (`1`) image per pair;
- checked agreement of generator, transform family, level, and operations within
  every pair;
- found zero pair-ID, source-pair, caption-group, or byte-hash overlap across
  train, validation, and test;
- confirmed that the image tree and manifests contain exactly the same files;
- retained four high-similarity cross-source real-image candidates because all
  four were train-to-train and therefore introduced no validation/test leakage.

The optional clean tree covers only 11,238 of 16,351 pairs; 5,113 FLUX pairs are
unavailable. It was therefore not substituted for the complete assigned-level
training set. The separate robustness suite contains the same 956 test pairs under
clean, t1, t3, and t5 conditions (7,648 images total); all of its hashes and image
dimensions were also verified.

## Model and optimization

- Backbone: cached public `openai/clip-vit-base-patch16`
- Features: global CLS branch plus local 14×14 patch-token branch
- Fusion: learned 256-channel global/local gate
- LoRA: rank 8, alpha 16, dropout 0.05, Q/V projections in vision layers 8–11
- Parameters: 86,887,811 total; 990,083 trainable in the head-only stage and
  1,088,387 trainable in the joint stage
- Training schedule: one head-only epoch followed by four joint LoRA epochs
- Optimizer: AdamW; head LR 5e-4; LoRA LR 1e-4; weight decay 0.01
- Effective batch: 32 images (batch 16, gradient accumulation 2)
- Loss: fused classification, auxiliary branch classification, feature
  consistency, and prediction consistency over reference/online-degraded views
- Precision/hardware: BF16 on one NVIDIA GeForce RTX 5090
- Software: Python 3.12.14, PyTorch 2.7.1+cu128, Transformers 5.16.1
- Approximate five-epoch wall time: 738 seconds (12.3 minutes)

The cached pretrained CLIP backbone was loaded with Hugging Face network access
disabled. “From the beginning” here means a fresh fine-tune with newly initialized
LoRA adapters and classifier/fusion heads; the public CLIP backbone itself was not
retrained from random weights.

## Training and validation

Robust AUC is the unweighted mean of the six transform-family × level AUCs. The
worst-group metric is the minimum generator × level AUC among groups with at least
20 images.

| Epoch | Stage | Train loss | Val ROC-AUC | Val robust AUC | Val worst-group AUC | Val accuracy | Threshold | Mean gate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Heads | 0.3878 | 0.9794 | 0.9807 | 0.8403 | 0.9279 | 0.4637 | 0.456 |
| 2 | Joint | 0.2759 | 0.9894 | 0.9884 | **0.8819** | 0.9514 | 0.7050 | 0.397 |
| 3 | Joint | 0.1904 | 0.9915 | 0.9910 | 0.8438 | 0.9563 | 0.6011 | 0.344 |
| 4 | Joint | 0.1484 | 0.9920 | 0.9913623 | 0.8611 | 0.9581 | 0.3919 | 0.330 |
| **5** | **Joint** | **0.1196** | **0.9921** | **0.9914147** | **0.8750** | **0.9590** | **0.4475** | **0.338** |

Epoch 5 beat epoch 4 on the predeclared robust selection metric by 0.0000524 and
was selected. Epoch 2 had the best validation worst-group AUC, illustrating a real
tradeoff between aggregate family/level robustness and the noisiest small subgroup.
The mean fusion gate remained far from the defined 0.05/0.95 collapse boundaries.

## Held-out test results

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

Bootstrap intervals use 2,000 deterministic percentile replicates sampled by
`dataset_pair_id`, keeping each real/AIGC pair together.

Real-image recall (also called the true-negative rate, TNR) is the fraction of
real images correctly classified as real: TN / (TN + FP).

### Confusion matrix at threshold 0.447460

| Actual / predicted | Real | AIGC |
|---|---:|---:|
| Real | **1,576** | 58 |
| AIGC | 64 | **1,570** |

Mean AIGC probability was 0.0452 for real images and 0.9498 for AIGC images. The
mean matched-pair score margin (`AIGC − Real`) was 0.9046; the median was 0.9919.

### By transform family and assigned level

| Family / level | Images | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | Errors |
|---|---:|---:|---:|---:|---:|---:|
| COCO / 1 | 700 | 0.9895 | 0.9486 | 0.9400 | 0.9571 | 36 |
| COCO / 3 | 604 | 0.9837 | 0.9404 | 0.9272 | 0.9536 | 36 |
| COCO / 5 | 608 | 0.9876 | 0.9474 | 0.9441 | 0.9507 | 32 |
| SID / 1 | 476 | 0.9998 | 0.9895 | 0.9958 | 0.9832 | 5 |
| SID / 3 | 436 | 0.9999 | 0.9931 | 1.0000 | 0.9862 | 3 |
| SID / 5 | 444 | 0.9970 | 0.9775 | 0.9865 | 0.9685 | 10 |

The macro average gives each row equal weight, preventing the larger SID/FLUX
groups from dominating the robustness selection score.

### By source dataset

| Source | Pairs | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | Errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepGuardDB | 89 | 0.9157 | 0.8202 | 0.7079 | 0.9326 | 32 |
| Defactify | 867 | 0.9918 | 0.9585 | 0.9608 | 0.9562 | 72 |
| SID_Set | 678 | 0.9990 | 0.9867 | 0.9941 | 0.9794 | 18 |

DeepGuardDB is the clearest domain weakness. It supplies the small IMAGEN and SD
groups plus part of DALLE3, and its lower AIGC recall indicates that its generated
images are more often mistaken for real.

### By generator-associated pair group

Each row contains AIGC images from that generator and their matched real images.

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
18-image group is reported for transparency but excluded from the formal
worst-group statistic by the configured 20-image minimum.

## Dedicated same-pair robustness evaluation

The frozen epoch-5 checkpoint and the same 0.447460 threshold were applied once to
956 identical pairs rendered at each condition. No threshold was fitted on this
suite.

| Condition | Images | ROC-AUC | Accuracy | AIGC recall | Real-image recall (TNR) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 1,912 | 0.9890 | 0.9493 | 0.9372 | 0.9613 | 37 | 60 |
| t1 | 1,912 | 0.9895 | 0.9508 | 0.9372 | 0.9644 | 34 | 60 |
| t3 | 1,912 | 0.9878 | 0.9477 | 0.9362 | 0.9592 | 39 | 61 |
| t5 | 1,912 | 0.9831 | 0.9357 | 0.9320 | 0.9393 | 58 | 65 |

The severity-5 transformation reduced ROC-AUC by 0.00596 and accuracy by 0.01360
relative to clean. The slight t1 improvement is small and plausible under benign
JPEG recompression. Across all four repeated conditions, Defactify ROC-AUC was
0.9923 while DeepGuardDB was 0.9189; the robustness suite has no SID_Set rows.

## Limitations

- This is an in-dataset held-out test. All eight generator labels occur in the
  training split, so it is not an unseen-generator benchmark.
- Generator and source are confounded: all FLUX pairs come from SID_Set, while
  IMAGEN and SD occur only in DeepGuardDB. Generator differences cannot be cleanly
  separated from source-domain differences.
- IMAGEN and SD contain only 30 test pairs each. Their per-level estimates have
  18–22 images and correspondingly high sampling uncertainty.
- The same-pair robustness suite excludes SID_Set and therefore cannot measure
  degradation sensitivity for the strongest/largest source group.
- The global threshold is validation-calibrated for this distribution. Screenshots,
  social-media recompression, new generators, or new camera domains can shift both
  calibration and error rates.
- A classifier score is evidence, not proof of provenance. It should not be the
  sole basis for legal, punitive, authorship, or moderation decisions.

## Handoff artifacts

- `final_handoff_checkpoint_glow_dataset.pt`: selected epoch-5 checkpoint,
  distributed next to—not inside—the repository archive
- `results/test_metrics.json`: complete overall and subgroup held-out metrics
- `results/robustness_same_pairs_metrics.json`: complete same-pair suite metrics
- `results/bootstrap_summary.json`: pair-cluster confidence intervals
- `results/test_analysis.json`: calibration, average precision, and matched-pair
  summaries
- `results/validation_history.csv`: all five validation epochs
- `results/run_manifest.json`: checkpoint, environment, and evaluation policy

Checkpoint SHA-256:

```text
28f28d7c65a6a96e546fa46e899b3766dd4c058e0302a8a8a49bf24f8f121271
```
