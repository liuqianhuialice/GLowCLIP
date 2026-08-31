from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .data import validate_dataset_layout, write_validation_summary


def _extract_archive(archive: Path, destination: Path, force: bool) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            member = PurePosixPath(info.filename)
            if (
                info.is_dir()
                or "__MACOSX" in member.parts
                or member.name.startswith("._")
            ):
                continue
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe path in {archive}: {info.filename}")
            target = destination.joinpath(*member.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes destination: {info.filename}")
            if target.exists() and not force:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    return extracted, skipped


def prepare_dataset(
    images_zip: str | Path,
    manifests_zip: str | Path,
    output_dir: str | Path,
    force: bool = False,
    verify_hashes: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_counts = _extract_archive(Path(images_zip), output_dir, force)
    manifest_counts = _extract_archive(Path(manifests_zip), manifest_dir, force)
    summary = validate_dataset_layout(
        output_dir / "images", manifest_dir, verify_hashes
    )
    summary["extraction"] = {
        "images": {"extracted": image_counts[0], "skipped": image_counts[1]},
        "manifests": {"extracted": manifest_counts[0], "skipped": manifest_counts[1]},
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely extract and validate the GLowCLIP dataset"
    )
    parser.add_argument("--images-zip", default="images.zip")
    parser.add_argument("--manifests-zip", default="manifests.zip")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing extracted files"
    )
    parser.add_argument(
        "--verify-hashes", action="store_true", help="Verify every manifest SHA-256"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        args.images_zip,
        args.manifests_zip,
        args.output_dir,
        force=args.force,
        verify_hashes=args.verify_hashes,
    )
    write_validation_summary(summary)


if __name__ == "__main__":
    main()
