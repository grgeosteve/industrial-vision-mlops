from pathlib import Path
from typing import Any

import pytest

from src.utils import coco_ops


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param({"path": "scenes/001.jpg"}, Path("scenes/001.jpg"), id="loco-path"),
        pytest.param({"path": "/scenes/001.jpg"}, Path("scenes/001.jpg"), id="loco-path-leading-slash"),
        pytest.param({"path": "scenes/001.jpg", "file_name": "001.jpg"}, Path("scenes/001.jpg"),
                     id="path-precedes-file-name"),
        pytest.param({"path": "", "file_name": "001.jpg"}, Path("images/001.jpg"), id="empty-path-falls-through"),
        pytest.param({"path": 123, "file_name": "001.jpg"}, Path("images/001.jpg"), id="non-str-path-falls-through"),
        pytest.param({"path": "///", "file_name": "001.jpg"}, Path("images/001.jpg"),
                     id="slash-only-path-falls-through"),
        pytest.param({"path": None, "file_name": "001.jpg"}, Path("images/001.jpg"), id="none-path-falls-through"),
        pytest.param({"path": ""}, None, id="empty-path"),
        pytest.param({"path": 123}, None, id="non-str-path"),
        pytest.param({"path": "///"}, None, id="slash-only-path"),
        pytest.param({"path": None}, None, id="none-path"),
        pytest.param({"path": " "}, Path(" "), id="whitespace-path-passes-through"),
        pytest.param({"file_name": "001.jpg"}, Path("images/001.jpg"), id="standard-file-name"),
        pytest.param({"file_name": "scenes/001.jpg"}, Path("images/scenes/001.jpg"), id="sub-path-file-name"),
        pytest.param({"file_name": "/001.jpg"}, Path("images/001.jpg"), id="leading-slash-file-name"),
        pytest.param({"file_name": "///"}, None, id="slash-only-file-name"),
        pytest.param({"file_name": 123}, None, id="non-str-file-name"),
        pytest.param({"file_name": None}, None, id="none-file-name"),
        pytest.param({"file_name": ""}, None, id="empty-file-name"),
        pytest.param({"file_name": " "}, Path("images") / " ", id="whitespace-file-name-passes-through"),
        pytest.param({"path": "", "file_name": ""}, None, id="empty-strings"),
        pytest.param({}, None, id="both-absent"),
    ]
)
def test_resolve_image_relpath(image: dict[str, Any], expected: Path | None) -> None:
    """Test the resolve_image_relpath function with various image scenarios."""

    assert coco_ops.resolve_image_relpath(image) == expected
