from __future__ import annotations

from pathlib import Path

from PIL import Image

from glowclip.predict import PredictionDataset, collect_paths


def test_unlabeled_unpaired_flat_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "incoming-test-images"
    image_dir.mkdir()
    Image.new("RGB", (31, 19), (20, 80, 140)).save(image_dir / "IMG_0007.JPG")
    Image.new("RGBA", (17, 29), (180, 40, 90, 120)).save(
        image_dir / "arbitrary-upload.png"
    )
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    paths = collect_paths([str(image_dir)])
    assert [path.name for path in paths] == ["IMG_0007.JPG", "arbitrary-upload.png"]

    dataset = PredictionDataset(paths, image_size=32)
    assert len(dataset) == 2
    for item in dataset:
        assert set(item) == {"pixel_values", "path"}
        assert item["pixel_values"].shape == (3, 32, 32)
        assert Path(item["path"]).is_file()

    # Passing the folder and one of its files must not score that image twice.
    deduplicated = collect_paths([str(image_dir), str(paths[0])])
    assert deduplicated == paths
