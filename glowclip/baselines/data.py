from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
ROLES = (("real", 0), ("ai", 1))
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PairImage:
    pair_id: str
    role: str
    label: int
    path: Path


def resolve_image_root(path: str | Path) -> Path:
    """Accept either ``.../images`` or its immediate parent directory."""
    root = Path(path).expanduser().resolve()
    candidates = (root, root / "images")
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SPLITS):
            return candidate
    expected = "{train,validation,test}/pair_id/{real,ai}.jpg"
    raise FileNotFoundError(f"Could not find {expected} below {root}")


def _role_image(pair_dir: Path, role: str) -> Path | None:
    matches = [pair_dir / f"{role}{extension}" for extension in IMAGE_EXTENSIONS]
    found = [path for path in matches if path.is_file()]
    if len(found) > 1:
        raise ValueError(f"Multiple {role} files found in {pair_dir}: {found}")
    return found[0] if found else None


def scan_split(split_root: str | Path) -> list[PairImage]:
    split_root = Path(split_root)
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_root}")
    records: list[PairImage] = []
    pair_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not pair_dirs:
        raise ValueError(f"No pair directories found in {split_root}")
    for pair_dir in pair_dirs:
        role_paths = {role: _role_image(pair_dir, role) for role, _ in ROLES}
        missing = [role for role, path in role_paths.items() if path is None]
        if missing:
            raise ValueError(
                f"Incomplete pair {pair_dir.name} in {split_root.name}; "
                f"missing {', '.join(missing)}"
            )
        records.extend(
            PairImage(pair_dir.name, role, label, role_paths[role])
            for role, label in ROLES
        )
    return records


def audit_pair_splits(image_root: str | Path) -> dict[str, int]:
    root = resolve_image_root(image_root)
    pair_ids: dict[str, set[str]] = {}
    image_counts: dict[str, int] = {}
    for split in SPLITS:
        records = scan_split(root / split)
        pair_ids[split] = {record.pair_id for record in records}
        image_counts[split] = len(records)
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = pair_ids[left] & pair_ids[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"Pair IDs cross {left}/{right} splits: {examples}")
    return image_counts


class PairDataset(Dataset[dict[str, Any]]):
    """One independent labeled image per real/AI member of a complete pair."""

    def __init__(self, split_root: str | Path, transform: Any) -> None:
        self.records = scan_split(split_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            transformed = self.transform(image)
        return {
            "image": transformed,
            "label": record.label,
            "pair_id": record.pair_id,
            "role": record.role,
            "path": str(record.path),
        }


def build_loaders(
    image_root: str | Path,
    train_transform: Any,
    evaluation_transform: Any,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    root = resolve_image_root(image_root)
    datasets = {
        "train": PairDataset(root / "train", train_transform),
        "validation": PairDataset(root / "validation", evaluation_transform),
        "test": PairDataset(root / "test", evaluation_transform),
    }
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(datasets["train"], shuffle=True, **common),
        DataLoader(datasets["validation"], shuffle=False, **common),
        DataLoader(datasets["test"], shuffle=False, **common),
    )
