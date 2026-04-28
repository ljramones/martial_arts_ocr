"""Generate a local real-page extraction review manifest.

The default input is the original Donn Draeger notes corpus under
``data/corpora/donn_draeger/dfd_notes_master/original``. The default output is
``data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json``,
which is intentionally gitignored.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SOURCE_KINDS = {"original", "derived", "training", "unknown"}


def collect_image_paths(input_dir: Path, *, recursive: bool = False) -> list[Path]:
    """Return supported image files in deterministic order."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    candidates: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )


def build_manifest(
    *,
    input_dir: Path,
    output_path: Path | None = None,
    collection_name: str,
    source_kind: str = "original",
    recursive: bool = False,
    limit: int | None = None,
    assume_japanese: bool = True,
    assume_macrons: bool = True,
    assume_images: bool = True,
) -> dict:
    """Build a manifest dictionary without writing it."""
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")

    image_paths = collect_image_paths(input_dir, recursive=recursive)
    if limit is not None:
        image_paths = image_paths[: max(0, limit)]

    samples = []
    for path in image_paths:
        samples.append(
            {
                "id": sample_id_for_path(path, collection_name=collection_name),
                "path": manifest_path(path, output_path=output_path),
                "description": f"Donn Draeger lecture page from {collection_name}",
                "source": {
                    "collection": collection_name,
                    "kind": source_kind,
                    "derived": source_kind in {"derived", "training"},
                },
                "expected": {
                    "has_japanese": assume_japanese,
                    "has_macrons": assume_macrons,
                    "has_diagrams_or_images": assume_images,
                    "min_image_regions": 0,
                    "min_text_regions": 1,
                    "notes": [
                        "Review whether images/diagrams are detected correctly",
                        "Review whether English/Japanese text cleanup preserves terminology",
                        "Review reading order manually",
                    ],
                },
            }
        )

    return {"samples": samples}


def write_manifest(manifest: dict, output_path: Path, *, force: bool = False) -> None:
    """Write manifest JSON, refusing to overwrite unless force is true."""
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sample_id_for_path(path: Path, *, collection_name: str) -> str:
    """Create a stable sample id from collection and filename."""
    collection = slugify(collection_name)
    stem = slugify(path.stem)
    return f"{collection}_{stem}"


def manifest_path(path: Path, *, output_path: Path | None = None) -> str:
    """Prefer repo-relative paths when possible."""
    _ = output_path
    base = Path.cwd()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slugify(value: str) -> str:
    """Convert a filename or collection name into a stable id fragment."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "sample"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local real-page manifest from corpus images.",
    )
    parser.add_argument(
        "--input",
        default="data/corpora/donn_draeger/dfd_notes_master/original",
        help="Input image folder",
    )
    parser.add_argument(
        "--output",
        default="data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json",
        help="Output manifest path",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of image files to include")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    parser.add_argument("--collection-name", default=None, help="Source collection name")
    parser.add_argument(
        "--source-kind",
        choices=sorted(SOURCE_KINDS),
        default="original",
        help="Dataset lineage class",
    )
    parser.add_argument("--assume-japanese", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assume-macrons", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assume-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input)
    output_path = Path(args.output)
    collection_name = args.collection_name or input_dir.name

    manifest = build_manifest(
        input_dir=input_dir,
        output_path=output_path,
        collection_name=collection_name,
        source_kind=args.source_kind,
        recursive=args.recursive,
        limit=args.limit,
        assume_japanese=args.assume_japanese,
        assume_macrons=args.assume_macrons,
        assume_images=args.assume_images,
    )
    write_manifest(manifest, output_path, force=args.force)
    print(f"Wrote {len(manifest['samples'])} samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
