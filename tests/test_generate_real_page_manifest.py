import json
from pathlib import Path

import pytest

from scripts.generate_real_page_manifest import (
    build_manifest,
    collect_image_paths,
    main,
    write_manifest,
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image; manifest generation only checks extensions")
    return path


def test_generates_manifest_from_temp_image_folder(tmp_path):
    input_dir = tmp_path / "pages"
    touch(input_dir / "page_002.jpg")
    touch(input_dir / "page_001.png")
    touch(input_dir / "notes.txt")

    output = tmp_path / "samples" / "manifest.local.json"
    manifest = build_manifest(
        input_dir=input_dir,
        output_path=output,
        collection_name="DFD Notes",
    )

    assert [sample["id"] for sample in manifest["samples"]] == [
        "dfd_notes_page_001",
        "dfd_notes_page_002",
    ]
    assert manifest["samples"][0]["source"] == {
        "collection": "DFD Notes",
        "kind": "original",
        "derived": False,
    }
    assert manifest["samples"][0]["expected"]["min_text_regions"] == 1
    assert manifest["samples"][0]["expected"]["min_image_regions"] == 0


def test_collect_image_paths_sorts_deterministically(tmp_path):
    input_dir = tmp_path / "pages"
    touch(input_dir / "b.JPG")
    touch(input_dir / "a.png")
    touch(input_dir / "c.webp")

    assert [path.name for path in collect_image_paths(input_dir)] == ["a.png", "b.JPG", "c.webp"]


def test_write_manifest_does_not_overwrite_without_force(tmp_path):
    output = tmp_path / "manifest.local.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_manifest({"samples": []}, output)

    write_manifest({"samples": []}, output, force=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"samples": []}


def test_limit_and_recursive_mode(tmp_path):
    input_dir = tmp_path / "pages"
    touch(input_dir / "root.jpg")
    touch(input_dir / "nested" / "inner.png")

    non_recursive = build_manifest(
        input_dir=input_dir,
        collection_name="collection",
        recursive=False,
    )
    recursive = build_manifest(
        input_dir=input_dir,
        collection_name="collection",
        recursive=True,
        limit=1,
    )

    assert len(non_recursive["samples"]) == 1
    assert len(recursive["samples"]) == 1


def test_empty_folder_writes_empty_manifest(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output = tmp_path / "manifest.local.json"

    exit_code = main([
        "--input",
        str(input_dir),
        "--output",
        str(output),
    ])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"samples": []}


def test_cli_supports_assumption_flags_and_source_kind(tmp_path):
    input_dir = tmp_path / "pages"
    output = tmp_path / "manifest.local.json"
    touch(input_dir / "page.jpg")

    exit_code = main([
        "--input",
        str(input_dir),
        "--output",
        str(output),
        "--source-kind",
        "training",
        "--no-assume-japanese",
        "--no-assume-macrons",
        "--no-assume-images",
    ])

    manifest = json.loads(output.read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    assert exit_code == 0
    assert sample["source"]["kind"] == "training"
    assert sample["source"]["derived"] is True
    assert sample["expected"]["has_japanese"] is False
    assert sample["expected"]["has_macrons"] is False
    assert sample["expected"]["has_diagrams_or_images"] is False
