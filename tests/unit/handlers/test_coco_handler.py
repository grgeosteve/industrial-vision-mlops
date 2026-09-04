import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from src.converters.base_converter import BaseAnnotationConverter
from src.datatypes import DatasetConfigModel, SplitConfig
from src.handlers.coco_handler import CocoFormatHandler

CocoDict: TypeAlias = dict[str, Any]
MakeHandler: TypeAlias = Callable[..., CocoFormatHandler]
WriteAnno: TypeAlias = Callable[..., Path]

YIELD_PAYLOAD = ("images/stub.jpg", "0 0.5 0.5 1.0 1.0\n")


class _StubConverter(BaseAnnotationConverter[CocoDict]):
    """Records what the handler passed in and yields a fixed pair per call."""

    def __init__(self) -> None:
        self.received: list[CocoDict] = []

    def convert(self, raw_data: CocoDict) -> Iterator[tuple[str, str | None]]:
        self.received.append(raw_data)
        yield YIELD_PAYLOAD

@pytest.fixture
def converter() -> _StubConverter:
    return _StubConverter()

@pytest.fixture
def make_handler(tmp_path: Path, converter: _StubConverter) -> MakeHandler:
    def _make(splits: SplitConfig | None = None, **extra: Any) -> CocoFormatHandler:
        cfg = DatasetConfigModel(
            classes={"forklift": {"coco_id": 5, "yolo_id": 0}},
            splits=splits or {"train": ["train.json"], "val": ["val.json"]},
            path=Path("dataset"),
            type="object_detection",
            format="coco",
            name="test",
            **extra,
        )
        return CocoFormatHandler(tmp_path, cfg, converter)

    return _make

@pytest.fixture
def write_anno(tmp_path: Path) -> WriteAnno:
    def _write(fname: str, payload: CocoDict, dirname: str = "annotations") -> Path:
        anno_path = tmp_path / dirname / fname
        anno_path.parent.mkdir(parents=True, exist_ok=True)
        anno_path.write_text(json.dumps(payload))

        return anno_path

    return _write

def valid_coco() -> CocoDict:
    """Well-formed COCO; copy and inject one defect per case."""
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/b.jpg", "width": 200, "height": 200},
        ],
        "categories": [{"id": 5, "name": "forklift"}, {"id": 7, "name": "pallet"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 2, "category_id": 7, "bbox": [0, 0, 10, 10]},
        ],
    }

# --------------------------------------------------------------------------- #
# get_available_splits
# --------------------------------------------------------------------------- #
def test_get_available_splits_returns_configured_splits(make_handler: MakeHandler) -> None:
    assert make_handler().get_available_splits() == ["train", "val"]

def test_get_available_splits_allows_test_split(make_handler: MakeHandler) -> None:
    handler = make_handler(splits={
        "train": ["train.json"],
        "val": ["val.json"],
        "test": ["test.json"]
    })

    assert handler.get_available_splits() == ["train", "val", "test"]

@pytest.mark.parametrize("split_config", [
    pytest.param({"train": ["train.json"], "trian": ["trian.json"]}, id="typo-in-name"),
    pytest.param({"train": ["train.json"], "VAL": ["val.json"]}, id="uppercase-name"),
    pytest.param({"train": ["train.json"], "testing": ["testing.json"]}, id="long-name"),
    pytest.param({"train": ["train.json"], "extra": ["extra.json"]}, id="extra"),
])
def test_get_available_splits_invalid_name_raises(make_handler: MakeHandler, split_config: SplitConfig) -> None:
    handler = make_handler(splits=split_config)

    with pytest.raises(ValueError, match="Invalid splits found:"):
        handler.get_available_splits()

def test_get_available_splits_missing_mandatory_raises(make_handler: MakeHandler) -> None:
    handler = make_handler(splits={
        "val": ["val.json"],
        "test": ["test.json"]
    })

    with pytest.raises(ValueError, match="Missing mandatory split"):
        handler.get_available_splits()

def test_get_available_splits_allows_train_only(make_handler: MakeHandler) -> None:
    handler = make_handler(splits={
        "train": ["train.json"],
    })

    assert handler.get_available_splits() == ["train"]

# --------------------------------------------------------------------------- #
# process_split
# --------------------------------------------------------------------------- #
def test_process_split_yields_converter_output(make_handler: MakeHandler,
                                               write_anno: WriteAnno,
                                               converter: _StubConverter) -> None:
    write_anno(fname="train.json", payload=valid_coco())
    assert list(make_handler().process_split("train")) == [YIELD_PAYLOAD]
    assert converter.received == [valid_coco()]

def test_process_split_reads_multiple_annotation_files(make_handler: MakeHandler,
                                                       write_anno: WriteAnno,
                                                       converter: _StubConverter) -> None:
    num_anno_files = 3
    handler = make_handler(splits={"train": [f"train_{i}.json" for i in range(num_anno_files)]})

    payloads = []
    for i in range(num_anno_files):
        coco = valid_coco()
        coco["images"][0]["file_name"] = f"a_{i}.jpg"
        payloads.append(coco)
        write_anno(fname=f"train_{i}.json", payload=coco)

    expected = num_anno_files * [YIELD_PAYLOAD]

    assert list(handler.process_split("train")) == expected
    assert converter.received == payloads

def test_process_split_accepts_string_filename(make_handler: MakeHandler,
                                               write_anno: WriteAnno,
                                               converter: _StubConverter) -> None:
    handler = make_handler(splits={"train": "train.json"})
    write_anno(fname="train.json", payload=valid_coco())

    assert list(handler.process_split("train")) == [YIELD_PAYLOAD]
    assert converter.received == [valid_coco()]

def test_process_split_honours_annotations_path_override(make_handler: MakeHandler,
                                                         write_anno: WriteAnno,
                                                         converter: _StubConverter) -> None:
    handler = make_handler(splits={"train": "train.json"}, annotations_path="labels")
    write_anno(fname="train.json", payload=valid_coco(), dirname="labels")

    assert list(handler.process_split("train")) == [YIELD_PAYLOAD]
    assert converter.received == [valid_coco()]

def test_process_split_non_json_filename_raises(make_handler: MakeHandler, write_anno: WriteAnno) -> None:
    handler = make_handler(splits={"train": ["train.txt"]})
    write_anno(fname="train.txt", payload=valid_coco())

    with pytest.raises(ValueError, match="Only JSON annotations are supported for COCO format."):
        list(handler.process_split("train"))

def test_process_split_missing_file_raises(make_handler: MakeHandler) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        list(make_handler().process_split("train"))

def test_process_split_invalid_coco_document_raises(make_handler: MakeHandler, write_anno: WriteAnno) -> None:
    coco = valid_coco()
    coco.pop("images")

    handler = make_handler(splits={"train": ["train.json"]})
    write_anno(fname="train.json", payload=coco)

    with pytest.raises(ValueError, match="is not a valid COCO JSON."):
        list(handler.process_split("train"))

def test_process_split_empty_annotations_raises(make_handler: MakeHandler, write_anno: WriteAnno) -> None:
    coco = valid_coco()
    coco["annotations"] = []

    handler = make_handler(splits={"train": ["train.json"]})
    write_anno(fname="train.json", payload=coco)

    with pytest.raises(ValueError, match="Invalid 'train' COCO JSON: No 'annotations' were found"):
        list(handler.process_split("train"))

def test_process_split_test_split_without_annotations_yields(make_handler: MakeHandler,
                                                             write_anno: WriteAnno,
                                                             converter: _StubConverter,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    coco = valid_coco()
    coco.pop("annotations")
    handler = make_handler(splits={"train": "train.json", "test": "test.json"})
    write_anno(fname="test.json", payload=coco)

    caplog.set_level(logging.INFO)

    assert list(handler.process_split("test")) == [YIELD_PAYLOAD]
    assert converter.received == [coco]
    assert "No annotations found" in caplog.text
