from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image

from glowclip.data import ImageManifestDataset
from glowclip.normalize_dataset import normalize_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifests = source / "final_dataset" / "manifests"
    images = source / "final_dataset" / "images"
    manifests.mkdir(parents=True)
    rows = []
    pairs = []
    for pair_index, split in enumerate(("train", "validation", "test"), start=1):
        pair_id = f"u{pair_index:06d}"
        pair_rows = []
        for role, label, value in (("real", "0", 40), ("ai", "1", 210)):
            relative = Path(split) / pair_id / f"{role}.jpg"
            image_path = images / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            channel = value + pair_index
            Image.new("RGB", (224, 224), (channel, channel, channel)).save(image_path)
            pair_rows.append(
                {
                    "unified_pair_id": pair_id,
                    "source_dataset": "unit",
                    "source_package": "unit-package",
                    "source_pair_id": str(pair_index),
                    "split": split,
                    "role": role,
                    "label": label,
                    "generator": "dummy",
                    "caption_group": f"caption-{pair_index}",
                    "transform_level": "1",
                    "transform_family": "unit_ladder",
                    "operations": "none",
                    "source_relative_path": f"original/{role}.jpg",
                    "output_path": f"final_dataset/images/{relative.as_posix()}",
                    "output_sha256": _sha256(image_path),
                }
            )
        rows.extend(pair_rows)
        pairs.append(
            {
                "unified_pair_id": pair_id,
                "source_dataset": "unit",
                "source_package": "unit-package",
                "source_pair_id": str(pair_index),
                "split": split,
                "generator": "dummy",
                "real_path": pair_rows[0]["output_path"],
                "ai_path": pair_rows[1]["output_path"],
            }
        )

    image_columns = list(rows[0])
    for name, subset in (
        ("all_images.csv", rows),
        ("train.csv", [row for row in rows if row["split"] == "train"]),
        (
            "validation.csv",
            [row for row in rows if row["split"] == "validation"],
        ),
        ("test.csv", [row for row in rows if row["split"] == "test"]),
    ):
        with (manifests / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=image_columns)
            writer.writeheader()
            writer.writerows(subset)
    with (manifests / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    robustness_manifests = source / "robustness_eval" / "manifests"
    robustness_images = source / "robustness_eval" / "images"
    robustness_manifests.mkdir(parents=True)
    robustness_rows = []
    for condition, level in (("clean", "0"), ("t1", "1"), ("t3", "3"), ("t5", "5")):
        for role, label, value in (("real", "0", 50), ("ai", "1", 200)):
            relative = Path(condition) / "u000003" / f"{role}.jpg"
            image_path = robustness_images / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224), (value, value, value)).save(image_path)
            robustness_rows.append(
                {
                    "unified_pair_id": "u000003",
                    "source_dataset": "unit",
                    "condition": condition,
                    "transform_level": level,
                    "role": role,
                    "label": label,
                    "generator": "dummy",
                    "output_path": f"robustness_eval/images/{relative.as_posix()}",
                    "output_sha256": _sha256(image_path),
                }
            )
    robustness_columns = list(robustness_rows[0])
    for name, subset in [
        ("all_conditions.csv", robustness_rows),
        *[
            (
                f"{condition}.csv",
                [row for row in robustness_rows if row["condition"] == condition],
            )
            for condition in ("clean", "t1", "t3", "t5")
        ],
    ]:
        with (robustness_manifests / name).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=robustness_columns)
            writer.writeheader()
            writer.writerows(subset)

    output = tmp_path / "normalized"
    summary = normalize_dataset(
        source, output, link_mode="hardlink", verify_hashes=True
    )
    assert summary["audit"]["images"] == 6
    assert summary["audit"]["pairs"] == 3
    assert summary["robustness_eval"]["audit"]["images"] == 8
    assert summary["robustness_eval"]["audit"]["pairs"] == 1
    source_image = images / "train" / "u000001" / "real.jpg"
    normalized_image = output / "images" / "train" / "u000001" / "real.jpg"
    assert source_image.stat().st_ino == normalized_image.stat().st_ino
    dataset = ImageManifestDataset(
        output / "manifests" / "train.csv", output / "images", image_size=32
    )
    assert dataset[0]["source_dataset"] == "unit"
    assert dataset[0]["transform_family"] == "unit_ladder"
    robustness_dataset = ImageManifestDataset(
        output / "robustness_eval" / "manifests" / "all_conditions.csv",
        output / "robustness_eval" / "images",
        image_size=32,
    )
    assert len(robustness_dataset) == 8
    assert robustness_dataset[0]["transform_family"] == "glow_robustness"
