from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from glowclip.data import ImageManifestDataset
from glowclip.degradations import CompoundDegradation
from glowclip.transforms import CLIPImageTransform


def test_transform_and_degradation_shapes() -> None:
    image = Image.new("RGB", (91, 47), (40, 120, 220))
    transform = CLIPImageTransform(64)
    tensor = transform(image)
    assert tensor.shape == (3, 64, 64)
    assert torch.isfinite(tensor).all()
    result = CompoundDegradation()(image)
    assert result.image.mode == "RGB"
    assert result.operations


def test_manifest_packaging_prefix_is_remapped(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "train" / "pair_000001" / "real.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10), (1, 2, 3)).save(image_path)
    manifest = tmp_path / "train.csv"
    columns = [
        "dataset_pair_id",
        "generator",
        "transform_level",
        "split",
        "role",
        "label",
        "output_path",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "dataset_pair_id": "pair_000001",
                "generator": "unit-test",
                "transform_level": "1",
                "split": "train",
                "role": "real",
                "label": "0",
                "output_path": "final_dataset/images/train/pair_000001/real.jpg",
            }
        )
    dataset = ImageManifestDataset(manifest, image_root, image_size=32)
    item = dataset[0]
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["label"].item() == 0.0
