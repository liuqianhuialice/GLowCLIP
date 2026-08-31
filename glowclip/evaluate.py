from __future__ import annotations

import argparse
from pathlib import Path

from .config import ModelConfig
from .data import ImageManifestDataset, build_dataloader
from .inference import infer_loader, score_predictions, write_predictions
from .model import GLowCLIP, load_checkpoint
from .runtime import (
    configure_torch,
    resolve_device,
    resolve_precision,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GLowCLIP checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/test.csv")
    parser.add_argument("--image-root", default="data/images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--threshold",
        type=float,
        help="Defaults to the validation threshold in checkpoint",
    )
    parser.add_argument(
        "--fit-threshold",
        action="store_true",
        help="Fit Youden threshold on this manifest (use for validation, not final test reporting)",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch()
    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.get("experiment_config", {})
    seed = int(experiment.get("seed", 42))
    seed_everything(seed)
    model_config = ModelConfig(**checkpoint["model_config"])
    device = resolve_device(args.device)
    requested_precision = args.precision or experiment.get("train", {}).get(
        "precision", "bf16"
    )
    precision = resolve_precision(requested_precision, device)

    dataset = ImageManifestDataset(
        args.manifest,
        args.image_root,
        model_config.image_size,
        paired_views=False,
    )
    loader = build_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
        seed,
        training=False,
        pin_memory=device.type == "cuda",
    )
    model = GLowCLIP.from_pretrained(model_config)
    model.load_adapter_state_dict(checkpoint["model"])
    model.to(device)

    predictions = infer_loader(
        model,
        loader,
        device,
        precision,
        description="Evaluation",
        max_batches=args.max_batches,
    )
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("threshold", 0.5))
    minimum = int(experiment.get("evaluation", {}).get("min_group_samples", 20))
    metrics, threshold = score_predictions(
        predictions,
        threshold,
        "youden" if args.fit_threshold else "fixed",
        minimum,
    )

    split_name = Path(args.manifest).stem
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).parent / f"evaluation_{split_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics, output_dir / "metrics.json")
    write_predictions(predictions, output_dir / "predictions.csv", threshold)
    print(
        f"{split_name}: n={metrics['count']} auc={metrics['roc_auc']:.4f} "
        f"robust_auc={metrics['robust_auc']:.4f} worst_group={metrics['worst_group_auc']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} threshold={threshold:.4f}"
    )
    print(f"Wrote {output_dir / 'metrics.json'} and {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
