from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import ExperimentConfig, load_config
from .data import ImageManifestDataset, build_dataloader, validate_dataset_layout
from .inference import infer_loader, score_predictions, write_predictions
from .losses import glowclip_loss
from .model import GLowCLIP, load_checkpoint, split_outputs
from .runtime import (
    atomic_torch_save,
    autocast_context,
    configure_torch,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GLowCLIP for Real/AIGC classification"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a YAML setting; may be repeated",
    )
    parser.add_argument("--resume", help="Resume from a last.pt checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--max-train-batches", type=int, help="Limit batches for a smoke test"
    )
    parser.add_argument("--max-eval-batches", type=int, help="Limit validation batches")
    return parser.parse_args()


def _lr_scale(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    if step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _build_optimizer(model: GLowCLIP, config: ExperimentConfig, joint: bool) -> AdamW:
    groups: list[dict[str, Any]] = [
        {
            "params": list(model.head_parameters()),
            "lr": config.train.head_learning_rate,
            "base_lr": config.train.head_learning_rate,
            "name": "heads",
        }
    ]
    if joint:
        groups.append(
            {
                "params": list(model.lora_parameters()),
                "lr": config.train.lora_learning_rate,
                "base_lr": config.train.lora_learning_rate,
                "name": "lora",
            }
        )
    return AdamW(groups, weight_decay=config.train.weight_decay)


def _set_learning_rates(
    optimizer: AdamW, step: int, total_steps: int, warmup_fraction: float
) -> float:
    scale = _lr_scale(step, total_steps, warmup_fraction)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * scale
    return scale


def _consistency_scale(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    return min(1.0, step / warmup_steps)


def _train_epoch(
    model: GLowCLIP,
    forward_model: torch.nn.Module,
    loader,
    optimizer: AdamW,
    scaler,
    device: torch.device,
    precision: str,
    config: ExperimentConfig,
    global_step: int,
    total_steps: int,
    epoch: int,
    writer: SummaryWriter,
    max_batches: int | None,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = config.train.gradient_accumulation
    batch_limit = min(len(loader), max_batches) if max_batches else len(loader)
    running = {
        key: 0.0 for key in ("loss", "fused_cls", "aux_cls", "feat_cons", "pred_cons")
    }
    running["gate_mean"] = 0.0
    processed = 0

    progress = tqdm(
        loader, total=batch_limit, desc=f"Epoch {epoch + 1} train", leave=False
    )
    for batch_index, batch in enumerate(progress):
        if batch_index >= batch_limit:
            break
        labels = batch["label"].to(device, non_blocking=True)
        clean = batch["pixel_values"].to(device, non_blocking=True)
        degraded = batch["degraded_values"].to(device, non_blocking=True)
        consistency = _consistency_scale(
            global_step, total_steps, config.train.consistency_warmup_fraction
        )
        with autocast_context(device, precision):
            outputs = forward_model(torch.cat((clean, degraded), dim=0))
            clean_outputs, degraded_outputs = split_outputs(outputs, clean.shape[0])
            losses = glowclip_loss(
                clean_outputs,
                degraded_outputs,
                labels,
                lambda_aux=config.train.lambda_aux * consistency,
                lambda_feature=config.train.lambda_feature * consistency,
                lambda_prediction=config.train.lambda_prediction * consistency,
            )
            scaled_loss = losses["loss"] / accumulation
        scaler.scale(scaled_loss).backward()

        is_step = (
            batch_index + 1
        ) % accumulation == 0 or batch_index + 1 == batch_limit
        if is_step:
            _set_learning_rates(
                optimizer, global_step + 1, total_steps, config.train.lr_warmup_fraction
            )
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                config.train.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        for key in ("loss", "fused_cls", "aux_cls", "feat_cons", "pred_cons"):
            running[key] += float(losses[key].detach().float())
        gate_mean = 0.5 * (
            clean_outputs["gate"].detach().float().mean()
            + degraded_outputs["gate"].detach().float().mean()
        )
        running["gate_mean"] += float(gate_mean)
        processed += 1

        if global_step and is_step and global_step % config.train.log_every == 0:
            writer.add_scalar(
                "train/loss_step", float(losses["loss"].detach()), global_step
            )
            writer.add_scalar("train/gate_mean_step", float(gate_mean), global_step)
            for group in optimizer.param_groups:
                writer.add_scalar(
                    f"learning_rate/{group['name']}", group["lr"], global_step
                )
        progress.set_postfix(
            loss=f"{float(losses['loss'].detach()):.4f}", gate=f"{float(gate_mean):.3f}"
        )

    if processed == 0:
        raise RuntimeError("The training loader produced no batches")
    return {key: value / processed for key, value in running.items()}, global_step


def _checkpoint_payload(
    model: GLowCLIP,
    config: ExperimentConfig,
    optimizer: AdamW,
    scaler,
    epoch: int,
    stage: str,
    global_step: int,
    threshold: float,
    metrics: dict[str, Any],
    best_robust_auc: float,
    best_worst_group_auc: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.adapter_state_dict(),
        "model_config": config.to_dict()["model"],
        "experiment_config": config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "stage": stage,
        "global_step": global_step,
        "threshold": threshold,
        "validation_metrics": metrics,
        "best_robust_auc": best_robust_auc,
        "best_worst_group_auc": best_worst_group_auc,
    }


def train(config: ExperimentConfig, args: argparse.Namespace) -> None:
    configure_torch()
    seed_everything(config.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(config.train.precision, device)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.to_dict(), output_dir / "resolved_config.json")

    layout = validate_dataset_layout(config.data.image_root, config.data.manifest_root)
    write_json(layout, output_dir / "dataset_summary.json")
    train_dataset = ImageManifestDataset(
        Path(config.data.manifest_root) / "train.csv",
        config.data.image_root,
        config.model.image_size,
        paired_views=True,
        online_degradation=config.data.online_degradation,
    )
    validation_dataset = ImageManifestDataset(
        Path(config.data.manifest_root) / "validation.csv",
        config.data.image_root,
        config.model.image_size,
        paired_views=False,
    )
    train_loader = build_dataloader(
        train_dataset,
        config.data.batch_size,
        config.data.num_workers,
        config.seed,
        training=True,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )
    validation_loader = build_dataloader(
        validation_dataset,
        config.data.batch_size * 2,
        config.data.num_workers,
        config.seed + 1,
        training=False,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )

    model = GLowCLIP.from_pretrained(config.model)
    model.to(device)
    forward_model: torch.nn.Module = model
    if config.train.compile:
        forward_model = torch.compile(model)
    total_epochs = config.train.head_epochs + config.train.joint_epochs
    train_batches = (
        min(len(train_loader), args.max_train_batches)
        if args.max_train_batches
        else len(train_loader)
    )
    steps_per_epoch = math.ceil(train_batches / config.train.gradient_accumulation)
    total_steps = total_epochs * steps_per_epoch
    scaler = make_grad_scaler(device, precision)

    start_epoch = 0
    global_step = 0
    best_robust_auc = -math.inf
    best_worst_group_auc = -math.inf
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint = load_checkpoint(args.resume)
        model.load_adapter_state_dict(resume_checkpoint["model"])
        start_epoch = int(resume_checkpoint.get("next_epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_robust_auc = float(resume_checkpoint.get("best_robust_auc", -math.inf))
        best_worst_group_auc = float(
            resume_checkpoint.get("best_worst_group_auc", -math.inf)
        )
        if resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
    if start_epoch >= total_epochs:
        raise ValueError(
            f"Checkpoint already completed {start_epoch} epochs; configured total is {total_epochs}"
        )

    print(
        f"Device: {device}; precision: {precision}; train images: {len(train_dataset)}"
    )
    print(f"Model parameters: {model.parameter_summary()}")
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    run_started = time.time()
    try:
        previous_stage: str | None = None
        optimizer: AdamW | None = None
        for epoch in range(start_epoch, total_epochs):
            joint = epoch >= config.train.head_epochs
            stage = "joint" if joint else "heads"
            model.set_lora_trainable(joint)
            if stage != previous_stage:
                optimizer = _build_optimizer(model, config, joint)
                if (
                    resume_checkpoint is not None
                    and epoch == start_epoch
                    and resume_checkpoint.get("stage") == stage
                    and resume_checkpoint.get("optimizer")
                ):
                    optimizer.load_state_dict(resume_checkpoint["optimizer"])
                previous_stage = stage
            assert optimizer is not None

            train_metrics, global_step = _train_epoch(
                model,
                forward_model,
                train_loader,
                optimizer,
                scaler,
                device,
                precision,
                config,
                global_step,
                total_steps,
                epoch,
                writer,
                args.max_train_batches,
            )
            validation_predictions = infer_loader(
                model,
                validation_loader,
                device,
                precision,
                description=f"Epoch {epoch + 1} validation",
                max_batches=args.max_eval_batches,
            )
            validation_metrics, threshold = score_predictions(
                validation_predictions,
                config.evaluation.threshold,
                config.evaluation.threshold_strategy,
                config.evaluation.min_group_samples,
            )

            for key, value in train_metrics.items():
                writer.add_scalar(f"train_epoch/{key}", value, epoch + 1)
            for key in (
                "roc_auc",
                "robust_auc",
                "worst_group_auc",
                "accuracy",
                "mean_gate",
            ):
                writer.add_scalar(
                    f"validation/{key}", validation_metrics[key], epoch + 1
                )
            writer.flush()

            robust_auc = float(validation_metrics["robust_auc"])
            worst_group_auc = float(validation_metrics["worst_group_auc"])
            improved_robust = math.isfinite(robust_auc) and robust_auc > best_robust_auc
            improved_worst = (
                math.isfinite(worst_group_auc)
                and worst_group_auc > best_worst_group_auc
            )
            if improved_robust:
                best_robust_auc = robust_auc
            if improved_worst:
                best_worst_group_auc = worst_group_auc
            payload = _checkpoint_payload(
                model,
                config,
                optimizer,
                scaler,
                epoch,
                stage,
                global_step,
                threshold,
                validation_metrics,
                best_robust_auc,
                best_worst_group_auc,
            )
            atomic_torch_save(payload, output_dir / "last.pt")
            if improved_robust:
                atomic_torch_save(payload, output_dir / "best_robust_auc.pt")
            if improved_worst:
                atomic_torch_save(payload, output_dir / "best_worst_group.pt")
            write_json(
                validation_metrics, output_dir / f"validation_epoch_{epoch + 1}.json"
            )
            write_predictions(
                validation_predictions,
                output_dir / f"validation_epoch_{epoch + 1}.csv",
                threshold,
            )
            print(
                f"Epoch {epoch + 1}/{total_epochs} [{stage}] "
                f"loss={train_metrics['loss']:.4f} auc={validation_metrics['roc_auc']:.4f} "
                f"robust_auc={robust_auc:.4f} worst_group={worst_group_auc:.4f} "
                f"threshold={threshold:.4f} gate={validation_metrics['mean_gate']:.3f}"
            )
            if (
                validation_metrics["mean_gate"] < 0.05
                or validation_metrics["mean_gate"] > 0.95
            ):
                print(
                    "Warning: fusion gate appears collapsed; inspect branch losses and feature scales."
                )
            resume_checkpoint = None
    finally:
        writer.close()
    print(f"Training finished in {(time.time() - run_started) / 3600.0:.2f} hours")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    train(config, args)


if __name__ == "__main__":
    main()
