from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from glowclip.baselines import BASELINE_NAMES
from glowclip.baselines.data import audit_pair_splits, resolve_image_root, scan_split
from glowclip.baselines.runner import average_precision, load_config


def _write_pair(root: Path, split: str, pair_id: str) -> None:
    pair = root / "images" / split / pair_id
    pair.mkdir(parents=True)
    for role, color in (("real", "blue"), ("ai", "red")):
        Image.new("RGB", (8, 8), color).save(pair / f"{role}.jpg")


def test_notebook_config_locks_all_four_baselines() -> None:
    config = load_config("configs/baselines.yaml")
    assert tuple(config["baselines"]) == BASELINE_NAMES
    assert config["baselines"]["resnet18"]["epochs"] == 10
    assert config["baselines"]["openclip_linear"]["pretrained"] == ("laion2b_s34b_b79k")
    assert config["baselines"]["npr"]["epochs"] == 25
    assert config["baselines"]["vib"]["beta"] == 0.0001


def test_pair_tree_is_complete_and_split_isolated(tmp_path: Path) -> None:
    for split, pair_id in zip(("train", "validation", "test"), ("p1", "p2", "p3")):
        _write_pair(tmp_path, split, pair_id)
    root = resolve_image_root(tmp_path)
    assert root == tmp_path / "images"
    assert len(scan_split(root / "train")) == 2
    assert audit_pair_splits(root) == {"train": 2, "validation": 2, "test": 2}


def test_incomplete_pairs_fail_loudly(tmp_path: Path) -> None:
    pair = tmp_path / "train" / "p1"
    pair.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(pair / "real.jpg")
    with pytest.raises(ValueError, match="Incomplete pair"):
        scan_split(tmp_path / "train")


def test_average_precision_matches_simple_ranking() -> None:
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0
    )
