from pathlib import Path


def test_data_layout_docs_exist():
    required_paths = [
        Path("data/README.md"),
        Path("data/corpora/donn_draeger/dfd_notes_master/README.md"),
        Path("data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.example.json"),
        Path("data/evaluation/real_page_review/README.md"),
        Path("docs/dataset-inventory.md"),
    ]

    for path in required_paths:
        assert path.exists(), f"missing data-layout documentation: {path}"


def test_data_layout_placeholders_exist_without_private_data_requirement():
    required_dirs = [
        Path("data/corpora/donn_draeger/dfd_notes_master/original"),
        Path("data/corpora/donn_draeger/dfd_notes_master/augmented"),
        Path("data/training/image_layout"),
        Path("data/training/orientation"),
        Path("data/evaluation/real_page_review/notes"),
    ]

    for path in required_dirs:
        assert path.exists(), f"missing data-layout directory placeholder: {path}"
