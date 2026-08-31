from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from .degradations import CompoundDegradation
from .transforms import CLIPImageTransform, decode_rgb

REQUIRED_COLUMNS = {
    "dataset_pair_id",
    "generator",
    "transform_level",
    "split",
    "role",
    "label",
    "output_path",
}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def resolve_image_path(row: dict[str, str], image_root: str | Path) -> Path:
    image_root = Path(image_root)
    manifest_path = PurePosixPath(row["output_path"])
    candidates = []
    if manifest_path.is_absolute():
        candidates.append(Path(str(manifest_path)))
    candidates.extend(
        [
            image_root / Path(*manifest_path.parts),
            image_root.parent / Path(*manifest_path.parts[1:])
            if manifest_path.parts and manifest_path.parts[0] == "final_dataset"
            else image_root / "__never__",
            image_root / row["split"] / row["dataset_pair_id"] / manifest_path.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve image for {row['dataset_pair_id']} ({row['role']}). Tried:\n  {rendered}"
    )


class ImageManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        image_size: int = 224,
        paired_views: bool = False,
        online_degradation: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.rows = read_manifest(self.manifest_path)
        self.paths = [resolve_image_path(row, self.image_root) for row in self.rows]
        self.transform = CLIPImageTransform(image_size)
        self.paired_views = paired_views
        self.degrader = (
            CompoundDegradation() if online_degradation and paired_views else None
        )

        invalid_labels = {row["label"] for row in self.rows} - {"0", "1"}
        if invalid_labels:
            raise ValueError(
                f"Unsupported labels in {manifest_path}: {sorted(invalid_labels)}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.paths[index]
        try:
            with Image.open(path) as opened:
                image = decode_rgb(opened)
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Failed to decode image: {path}") from error

        item: dict[str, Any] = {
            "pixel_values": self.transform(image),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "pair_id": row["dataset_pair_id"],
            "role": row["role"],
            "generator": row.get("generator", "unknown"),
            "source_dataset": row.get("source_dataset", row.get("source", "unknown")),
            "transform_family": row.get("transform_family", "unknown"),
            "transform_level": int(row.get("transform_level", 0) or 0),
            "path": str(path),
        }
        if self.paired_views:
            if self.degrader is None:
                degraded = image.copy()
                operations = ("identity",)
            else:
                result = self.degrader(image)
                degraded = result.image
                operations = result.operations
            item["degraded_values"] = self.transform(degraded)
            item["online_operations"] = " | ".join(operations)
        return item


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    training: bool,
    pin_memory: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_layout(
    image_root: str | Path,
    manifest_root: str | Path,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    image_root = Path(image_root)
    manifest_root = Path(manifest_root)
    split_rows: dict[str, list[dict[str, str]]] = {}
    split_pairs: dict[str, set[str]] = {}
    summary: dict[str, Any] = {"splits": {}}

    for split in ("train", "validation", "test"):
        path = manifest_root / f"{split}.csv"
        rows = read_manifest(path)
        split_rows[split] = rows
        split_pairs[split] = {row["dataset_pair_id"] for row in rows}
        labels = Counter(row["label"] for row in rows)
        roles_by_pair: dict[str, set[str]] = {}
        hash_failures: list[str] = []
        for row in rows:
            roles_by_pair.setdefault(row["dataset_pair_id"], set()).add(row["role"])
            resolved = resolve_image_path(row, image_root)
            if (
                verify_hashes
                and row.get("output_sha256")
                and _sha256(resolved) != row["output_sha256"]
            ):
                hash_failures.append(str(resolved))
        incomplete = sorted(
            pair_id
            for pair_id, roles in roles_by_pair.items()
            if roles != {"real", "ai"}
        )
        if incomplete:
            raise ValueError(f"{split} contains incomplete pairs: {incomplete[:5]}")
        if hash_failures:
            raise ValueError(
                f"SHA-256 mismatch for {len(hash_failures)} files: {hash_failures[:3]}"
            )
        summary["splits"][split] = {
            "images": len(rows),
            "pairs": len(roles_by_pair),
            "labels": {key: labels[key] for key in sorted(labels)},
            "transform_levels": dict(
                sorted(Counter(row["transform_level"] for row in rows).items())
            ),
        }

    overlaps: dict[str, int] = {}
    split_names = list(split_pairs)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_pairs[left] & split_pairs[right]
            overlaps[f"{left}-{right}"] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Pair leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )
    summary["pair_overlap"] = overlaps
    summary["verified_hashes"] = verify_hashes
    return summary


def write_validation_summary(
    summary: dict[str, Any], output: str | Path | None = None
) -> None:
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
    else:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
