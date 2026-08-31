from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .data import validate_dataset_layout

SPLITS = ("train", "validation", "test")
ROBUSTNESS_CONDITIONS = ("clean", "t1", "t3", "t5")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        return columns, list(reader)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_image_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path)
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("final_dataset", "images"):
            relative = PurePosixPath(*parts[index + 2 :])
            if len(relative.parts) >= 3:
                return relative
    raise ValueError(f"Expected a path below final_dataset/images: {raw_path}")


def _relative_robustness_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path)
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("robustness_eval", "images"):
            relative = PurePosixPath(*parts[index + 2 :])
            if len(relative.parts) >= 3:
                return relative
    raise ValueError(f"Expected a path below robustness_eval/images: {raw_path}")


def _normalize_image_rows(
    columns: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    required = {
        "unified_pair_id",
        "source_dataset",
        "split",
        "role",
        "label",
        "generator",
        "transform_level",
        "transform_family",
        "output_path",
        "output_sha256",
    }
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    output_columns = ["dataset_pair_id", "source", "source_path", *columns]
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["dataset_pair_id"] = row["unified_pair_id"]
        item["source"] = row["source_dataset"]
        item["source_path"] = row.get("source_relative_path", "")
        item["output_path"] = _relative_image_path(row["output_path"]).as_posix()
        normalized.append(item)
    return output_columns, normalized


def _normalize_pair_rows(
    columns: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    output_columns = ["dataset_pair_id", "source", *columns]
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["dataset_pair_id"] = row["unified_pair_id"]
        item["source"] = row["source_dataset"]
        item["real_path"] = _relative_image_path(row["real_path"]).as_posix()
        item["ai_path"] = _relative_image_path(row["ai_path"]).as_posix()
        normalized.append(item)
    return output_columns, normalized


def _normalize_robustness_rows(
    columns: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    required = {
        "unified_pair_id",
        "source_dataset",
        "condition",
        "transform_level",
        "role",
        "label",
        "generator",
        "output_path",
        "output_sha256",
    }
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Robustness manifest is missing columns: {sorted(missing)}")
    output_columns = [
        "dataset_pair_id",
        "source",
        "source_path",
        "split",
        "transform_family",
        *columns,
    ]
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["dataset_pair_id"] = row["unified_pair_id"]
        item["source"] = row["source_dataset"]
        item["source_path"] = ""
        item["split"] = row["condition"]
        item["transform_family"] = "glow_robustness"
        item["output_path"] = _relative_robustness_path(row["output_path"]).as_posix()
        normalized.append(item)
    return output_columns, normalized


def _link_images(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(
            os.path.relpath(source, destination.parent), target_is_directory=True
        )
        return
    copy_function = os.link if mode == "hardlink" else shutil.copy2
    shutil.copytree(source, destination, copy_function=copy_function)


def _audit_rows(
    rows: list[dict[str, str]], source_root: Path, verify_hashes: bool
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    caption_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    expected_paths: set[Path] = set()
    problems: list[str] = []

    for index, row in enumerate(rows, start=1):
        pair_id = row["dataset_pair_id"]
        pairs[pair_id].append(row)
        caption = row.get("caption_group", "").strip()
        if caption:
            caption_splits[(row["source_dataset"], caption)].add(row["split"])
        source_pair_splits[(row["source_package"], row["source_pair_id"])].add(
            row["split"]
        )
        hash_splits[row["output_sha256"]].add(row["split"])
        path = source_root / "final_dataset" / "images" / Path(row["output_path"])
        expected_paths.add(path)
        if not path.is_file():
            problems.append(f"missing image: {path}")
            continue
        if verify_hashes and _sha256(path) != row["output_sha256"]:
            problems.append(f"SHA-256 mismatch: {path}")
        if verify_hashes:
            with Image.open(path) as image:
                if image.size != (224, 224):
                    problems.append(f"unexpected dimensions {image.size}: {path}")
        if index % 5000 == 0:
            print(f"  audited {index:,}/{len(rows):,} images")

    for pair_id, pair_rows in pairs.items():
        roles = {row["role"] for row in pair_rows}
        labels = {row["role"]: row["label"] for row in pair_rows}
        splits = {row["split"] for row in pair_rows}
        if len(pair_rows) != 2 or roles != {"real", "ai"}:
            problems.append(f"incomplete pair {pair_id}: roles={sorted(roles)}")
        if labels != {"real": "0", "ai": "1"}:
            problems.append(f"invalid labels for {pair_id}: {labels}")
        if len(splits) != 1:
            problems.append(f"pair crosses splits {pair_id}: {sorted(splits)}")
        for field in ("generator", "transform_level", "transform_family", "operations"):
            values = {row[field] for row in pair_rows}
            if len(values) != 1:
                problems.append(
                    f"pair {pair_id} disagrees on {field}: {sorted(values)}"
                )

    actual_paths = {
        path
        for path in (source_root / "final_dataset" / "images").rglob("*")
        if path.is_file()
    }
    missing_paths = expected_paths - actual_paths
    extra_paths = actual_paths - expected_paths
    if missing_paths:
        problems.append(f"{len(missing_paths)} manifest image paths are missing")
    if extra_paths:
        problems.append(f"{len(extra_paths)} image files are absent from the manifest")

    spanning_captions = [
        key for key, splits in caption_splits.items() if len(splits) > 1
    ]
    spanning_source_pairs = [
        key for key, splits in source_pair_splits.items() if len(splits) > 1
    ]
    spanning_hashes = [key for key, splits in hash_splits.items() if len(splits) > 1]
    if spanning_captions:
        problems.append(f"{len(spanning_captions)} caption groups cross splits")
    if spanning_source_pairs:
        problems.append(f"{len(spanning_source_pairs)} source pairs cross splits")
    if spanning_hashes:
        problems.append(f"{len(spanning_hashes)} byte hashes cross splits")
    if problems:
        raise ValueError("Glow dataset audit failed:\n  " + "\n  ".join(problems[:30]))

    return {
        "images": len(rows),
        "pairs": len(pairs),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "sources": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "generators": dict(sorted(Counter(row["generator"] for row in rows).items())),
        "transform_families": dict(
            sorted(Counter(row["transform_family"] for row in rows).items())
        ),
        "transform_levels": dict(
            sorted(Counter(row["transform_level"] for row in rows).items())
        ),
        "caption_groups_crossing_splits": 0,
        "source_pairs_crossing_splits": 0,
        "hashes_crossing_splits": 0,
        "verified_hashes_and_dimensions": verify_hashes,
    }


def _audit_robustness_rows(
    rows: list[dict[str, str]], source_root: Path, verify_hashes: bool
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    condition_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    expected_paths: set[Path] = set()
    problems: list[str] = []

    for index, row in enumerate(rows, start=1):
        pair_id = row["dataset_pair_id"]
        condition = row["condition"]
        pairs[pair_id].append(row)
        condition_keys[condition].add((pair_id, row["role"]))
        path = source_root / "robustness_eval" / "images" / Path(row["output_path"])
        expected_paths.add(path)
        if not path.is_file():
            problems.append(f"missing robustness image: {path}")
            continue
        if verify_hashes and _sha256(path) != row["output_sha256"]:
            problems.append(f"SHA-256 mismatch: {path}")
        if verify_hashes:
            with Image.open(path) as image:
                if image.size != (224, 224):
                    problems.append(f"unexpected dimensions {image.size}: {path}")
        if index % 2000 == 0:
            print(f"  audited {index:,}/{len(rows):,} robustness images")

    expected_conditions = set(ROBUSTNESS_CONDITIONS)
    if set(condition_keys) != expected_conditions:
        problems.append(
            "unexpected robustness conditions: "
            f"{sorted(condition_keys)} (expected {sorted(expected_conditions)})"
        )
    if condition_keys:
        reference = next(iter(condition_keys.values()))
        for condition, keys in condition_keys.items():
            if keys != reference:
                problems.append(
                    f"condition {condition} does not contain the same pair/role rows"
                )

    for pair_id, pair_rows in pairs.items():
        by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in pair_rows:
            by_condition[row["condition"]].append(row)
        if set(by_condition) != expected_conditions:
            problems.append(f"pair {pair_id} has conditions {sorted(by_condition)}")
        for condition, condition_rows in by_condition.items():
            roles = {row["role"] for row in condition_rows}
            labels = {row["role"]: row["label"] for row in condition_rows}
            if len(condition_rows) != 2 or roles != {"real", "ai"}:
                problems.append(
                    f"pair {pair_id}/{condition} is incomplete: {sorted(roles)}"
                )
            if labels != {"real": "0", "ai": "1"}:
                problems.append(
                    f"pair {pair_id}/{condition} has invalid labels: {labels}"
                )

    actual_paths = {
        path
        for path in (source_root / "robustness_eval" / "images").rglob("*")
        if path.is_file()
    }
    if expected_paths != actual_paths:
        problems.append(
            "robustness image tree differs from manifest: "
            f"missing={len(expected_paths - actual_paths)}, "
            f"extra={len(actual_paths - expected_paths)}"
        )
    if problems:
        raise ValueError(
            "Glow robustness audit failed:\n  " + "\n  ".join(problems[:30])
        )

    return {
        "images": len(rows),
        "pairs": len(pairs),
        "conditions": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "sources": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "same_pairs_in_every_condition": True,
        "verified_hashes_and_dimensions": verify_hashes,
    }


def normalize_robustness_eval(
    source_root: str | Path,
    output_root: str | Path,
    link_mode: str = "hardlink",
    verify_hashes: bool = False,
) -> dict[str, Any] | None:
    """Normalize the optional same-pair suite without mixing it into training."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    source_eval = source_root / "robustness_eval"
    if not source_eval.is_dir():
        return None
    target_eval = output_root / "robustness_eval"
    if target_eval.exists():
        raise FileExistsError(f"Robustness output already exists: {target_eval}")
    if link_mode not in {"hardlink", "copy", "symlink"}:
        raise ValueError("link_mode must be hardlink, copy, or symlink")

    all_columns, all_source_rows = _read_csv(
        source_eval / "manifests" / "all_conditions.csv"
    )
    output_columns, all_rows = _normalize_robustness_rows(all_columns, all_source_rows)
    audit = _audit_robustness_rows(all_rows, source_root, verify_hashes)

    target_manifests = target_eval / "manifests"
    target_manifests.mkdir(parents=True)
    _write_csv(target_manifests / "all_conditions.csv", output_columns, all_rows)
    for condition in ROBUSTNESS_CONDITIONS:
        columns, source_rows = _read_csv(source_eval / "manifests" / f"{condition}.csv")
        _, normalized_rows = _normalize_robustness_rows(columns, source_rows)
        expected = [row for row in all_rows if row["condition"] == condition]
        if normalized_rows != expected:
            raise ValueError(
                f"Robustness {condition}.csv differs from all_conditions.csv"
            )
        _write_csv(
            target_manifests / f"{condition}.csv", output_columns, normalized_rows
        )
    _link_images(source_eval / "images", target_eval / "images", link_mode)

    summary = {
        "source_root": str(source_eval),
        "output_root": str(target_eval),
        "image_materialization": link_mode,
        "audit": audit,
        "use": "evaluation only; never included in training or model selection",
    }
    (target_eval / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def normalize_dataset(
    source_root: str | Path,
    output_root: str | Path,
    link_mode: str = "hardlink",
    verify_hashes: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    source_manifests = source_root / "final_dataset" / "manifests"
    source_images = source_root / "final_dataset" / "images"
    if not source_manifests.is_dir() or not source_images.is_dir():
        raise FileNotFoundError(
            f"Expected final_dataset/images and final_dataset/manifests below {source_root}"
        )
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists; choose a new location to avoid overwriting: {output_root}"
        )
    if link_mode not in {"hardlink", "copy", "symlink"}:
        raise ValueError("link_mode must be hardlink, copy, or symlink")

    all_columns, all_source_rows = _read_csv(source_manifests / "all_images.csv")
    image_columns, all_rows = _normalize_image_rows(all_columns, all_source_rows)
    audit = _audit_rows(all_rows, source_root, verify_hashes)

    output_root.mkdir(parents=True)
    output_manifests = output_root / "manifests"
    output_manifests.mkdir()
    for split in SPLITS:
        split_columns, split_source_rows = _read_csv(source_manifests / f"{split}.csv")
        _, split_rows = _normalize_image_rows(split_columns, split_source_rows)
        _write_csv(output_manifests / f"{split}.csv", image_columns, split_rows)
    _write_csv(output_manifests / "all_images.csv", image_columns, all_rows)

    pair_columns, pair_source_rows = _read_csv(source_manifests / "pairs.csv")
    normalized_pair_columns, pair_rows = _normalize_pair_rows(
        pair_columns, pair_source_rows
    )
    _write_csv(output_manifests / "pairs.csv", normalized_pair_columns, pair_rows)
    _link_images(source_images, output_root / "images", link_mode)

    robustness_summary = normalize_robustness_eval(
        source_root,
        output_root,
        link_mode=link_mode,
        verify_hashes=verify_hashes,
    )

    normalized_validation = validate_dataset_layout(
        output_root / "images", output_manifests, verify_hashes=False
    )
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "image_materialization": link_mode,
        "audit": audit,
        "normalized_layout_validation": normalized_validation,
        "training_tree": "assigned transform levels only",
        "excluded_from_training": ["images_clean", "robustness_eval"],
        "robustness_eval": robustness_summary,
    }
    (output_root / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize glow_dataset for the GLowCLIP pipeline"
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize the normalized image tree",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = normalize_dataset(
        args.source_root,
        args.output_root,
        link_mode=args.link_mode,
        verify_hashes=args.verify_hashes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
