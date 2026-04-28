import json
from pathlib import Path


def test_notebooks_are_valid_json():
    notebook_paths = sorted(Path("notebooks").glob("*.ipynb"))

    assert notebook_paths

    for path in notebook_paths:
        with path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)
        assert notebook["nbformat"] == 4
        assert isinstance(notebook.get("cells"), list)
