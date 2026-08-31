from __future__ import annotations

import argparse
import copy
import csv
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from ..metrics import compute_metrics
from ..runtime import (
    atomic_torch_save,
    configure_torch,
    resolve_device,
    seed_everything,
    write_json,
)
from . import BASELINE_NAMES
from .data import audit_pair_splits, build_loaders, resolve_image_root
from .models import BuiltBaseline, build_baseline, kl_divergence


def average_precision(labels: list[int], scores: list[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = int(labels_array.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores_array, kind="mergesort")
    sorted_labels = labels_array[order]
    sorted_scores = scores_array[order]
    cumulative_true = np.cumsum(sorted_labels)
    cumulative_false = np.cumsum(1 - sorted_labels)
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    precision = cumulative_true[ends] / (cumulative_true[ends] + cumulative_false[ends])
    recall = cumulative_true[ends] / positives
    recall_change = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_change * precision))


def _forward(
    model: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(images)
    if isinstance(output, tuple):
        logits, mu, std = output
        return logits.squeeze(1), kl_divergence(mu, std)
    return output.squeeze(1), None


def _run_epoch(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    threshold: float,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    collect_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_kl_loss = 0.0
    labels_all: list[int] = []
    scores_all: list[float] = []
    rows: list[dict[str, Any]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, kl_loss = _forward(model, images)
            classification_loss = criterion(logits, labels)
            loss = classification_loss
            if kl_loss is not None:
                loss = loss + beta * kl_loss
            if training:
                loss.backward()
                optimizer.step()
            size = int(labels.numel())
            total_loss += float(loss.detach()) * size
            total_classification_loss += float(classification_loss.detach()) * size
            total_kl_loss += (
                float(kl_loss.detach()) * size if kl_loss is not None else 0.0
            )
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().to(torch.int64).tolist()
            labels_all.extend(labels_cpu)
            scores_all.extend(scores)
            if collect_rows:
                rows.extend(
                    {
                        "pair_id": pair_id,
                        "role": role,
                        "path": path,
                        "label": label,
                        "score": score,
                        "prediction": int(score >= threshold),
                    }
                    for pair_id, role, path, label, score in zip(
                        batch["pair_id"],
                        batch["role"],
                        batch["path"],
                        labels_cpu,
                        scores,
                    )
                )
    metrics = compute_metrics(labels_all, scores_all, threshold)
    metrics.update(
        {
            "loss": total_loss / len(labels_all),
            "classification_loss": total_classification_loss / len(labels_all),
            "kl_loss": total_kl_loss / len(labels_all),
            "average_precision": average_precision(labels_all, scores_all),
        }
    )
    return metrics, rows


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    backbone = getattr(model, "backbone", None)
    frozen_backbone = backbone is not None and not any(
        parameter.requires_grad for parameter in backbone.parameters()
    )
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not (frozen_backbone and name.startswith("backbone."))
    }


def _load_partial_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise ValueError(f"Checkpoint contains unknown keys: {unknown[:5]}")
    current.update(state)
    model.load_state_dict(current, strict=True)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "role", "path", "label", "score", "prediction"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if (
        not isinstance(config, dict)
        or "common" not in config
        or "baselines" not in config
    ):
        raise ValueError("Baseline config requires 'common' and 'baselines' mappings")
    missing = set(BASELINE_NAMES) - set(config["baselines"])
    if missing:
        raise ValueError(f"Baseline config is missing: {sorted(missing)}")
    return config


def train_baseline(
    name: str,
    built: BuiltBaseline,
    config: dict[str, Any],
    image_root: Path,
    output_root: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    seed_everything(seed)
    model = built.model.to(device)
    batch_size = int(config["batch_size"])
    threshold = float(config["threshold"])
    beta = float(config.get("beta", 0.0))
    train_loader, validation_loader, test_loader = build_loaders(
        image_root,
        built.train_transform,
        built.evaluation_transform,
        batch_size,
        num_workers,
        pin_memory=device.type == "cuda",
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))
    scheduler = None
    if name == "npr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    history: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics, _ = _run_epoch(
            model, train_loader, device, threshold, beta, optimizer
        )
        validation_metrics, _ = _run_epoch(
            model, validation_loader, device, threshold, beta, None
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        validation_auc = float(validation_metrics["roc_auc"])
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(_trainable_state(model))
        if scheduler is not None:
            scheduler.step()
        print(
            f"{name} epoch {epoch}/{config['epochs']} | "
            f"train AUC={train_metrics['roc_auc']:.4f} | "
            f"validation AUC={validation_auc:.4f}"
        )
    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for {name}")
    _load_partial_state(model, best_state)
    test_metrics, predictions = _run_epoch(
        model, test_loader, device, threshold, beta, None, collect_rows=True
    )

    baseline_output = output_root / name
    baseline_output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "baseline": name,
        "model_metadata": built.model_metadata,
        "model_state_dict": best_state,
        "state_policy": built.model_metadata["checkpoint_state"],
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "config": config,
    }
    atomic_torch_save(checkpoint, baseline_output / "best.pt")
    write_json({"history": history}, baseline_output / "history.json")
    write_json(test_metrics, baseline_output / "test_metrics.json")
    _write_rows(baseline_output / "test_predictions.csv", predictions)
    return {
        "baseline": name,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "test_metrics": test_metrics,
        "model_metadata": built.model_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the four baselines adapted from baselines_original.ipynb"
    )
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=BASELINE_NAMES,
        help="Repeat to run multiple baselines; omitted means all four",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch()
    config = load_config(args.config)
    common = dict(config["common"])
    seed = int(args.seed if args.seed is not None else common["seed"])
    num_workers = int(
        args.num_workers if args.num_workers is not None else common["num_workers"]
    )
    device = resolve_device(args.device)
    image_root = resolve_image_root(args.data_root)
    image_counts = audit_pair_splits(image_root)
    output_root = Path(args.output_dir).resolve()
    names = list(args.baseline or BASELINE_NAMES)
    summaries: list[dict[str, Any]] = []
    for name in names:
        baseline_config = {**common, **dict(config["baselines"][name])}
        print(f"Building baseline: {name}")
        built = build_baseline(name, baseline_config)
        summaries.append(
            train_baseline(
                name,
                built,
                baseline_config,
                image_root,
                output_root,
                device,
                seed,
                num_workers,
            )
        )
    write_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_notebook": "baselines_original.ipynb",
            "source_notebook_sha256": common["source_notebook_sha256"],
            "seed": seed,
            "device": str(device),
            "python": platform.python_version(),
            "image_root": str(image_root),
            "image_counts": image_counts,
            "baselines": summaries,
        },
        output_root / "run_summary.json",
    )


if __name__ == "__main__":
    main()
