from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ModelConfig
from .model import GLowCLIP, load_checkpoint
from .runtime import (
    autocast_context,
    configure_torch,
    resolve_device,
    resolve_precision,
    write_json,
)
from .transforms import CLIPImageTransform, decode_rgb

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class PredictionDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.transform = CLIPImageTransform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        try:
            with Image.open(path) as opened:
                image = decode_rgb(opened)
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Failed to decode image: {path}") from error
        return {"pixel_values": self.transform(image), "path": str(path)}


def collect_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            raise FileNotFoundError(f"Not an image file or directory: {path}")
    unique = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("No supported images found")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Real/AIGC labels for images")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="JSON output path; stdout when omitted")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    configure_torch()
    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.get("experiment_config", {})
    model_config = ModelConfig(**checkpoint["model_config"])
    device = resolve_device(args.device)
    requested_precision = args.precision or experiment.get("train", {}).get(
        "precision", "bf16"
    )
    precision = resolve_precision(requested_precision, device)
    threshold = float(
        args.threshold
        if args.threshold is not None
        else checkpoint.get("threshold", 0.5)
    )

    paths = collect_paths(args.inputs)
    dataset = PredictionDataset(paths, model_config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = GLowCLIP.from_pretrained(model_config)
    model.load_adapter_state_dict(checkpoint["model"])
    model.to(device).eval()

    results: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Predicting", leave=False):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            outputs = model(pixels)
        scores = torch.sigmoid(outputs["fused_logit"].float()).cpu().tolist()
        gates = outputs["gate"].float().mean(dim=-1).cpu().tolist()
        for path, score, gate in zip(batch["path"], scores, gates):
            results.append(
                {
                    "path": path,
                    "fake_probability": round(float(score), 6),
                    "prediction": "AIGC" if score >= threshold else "Real",
                    "gate_mean": round(float(gate), 6),
                }
            )
    payload = {"threshold": threshold, "predictions": results}
    if args.output:
        write_json(payload, args.output)
        print(f"Wrote {len(results)} predictions to {args.output}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
