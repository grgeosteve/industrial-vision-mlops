from collections.abc import Callable
from typing import Any, TypeAlias

import pytest

from src.converters.coco_to_yolo import CocoBBoxEntry, CocoToYoloDetectionConverter
from src.datatypes import ClassConfigMapping

CocoDict: TypeAlias = dict[str, Any]
MakeConverter: TypeAlias = Callable[..., CocoToYoloDetectionConverter]
Mutate: TypeAlias = Callable[[CocoDict], Any]


@pytest.fixture
def make_converter() -> MakeConverter:
    def _make(class_mapping: ClassConfigMapping | None = None) -> CocoToYoloDetectionConverter:
        return CocoToYoloDetectionConverter(class_mapping or {
            "forklift": {"coco_id": 5, "yolo_id": 0},
            "pallet": {"coco_id": 7, "yolo_id": 1},
        })

    return _make

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
# __init__
# --------------------------------------------------------------------------- #
def test_builds_class_lookup(make_converter: MakeConverter) -> None:
    assert make_converter().coco_to_yolo == {5: 0, 7: 1}

# --------------------------------------------------------------------------- #
# _index_images
# --------------------------------------------------------------------------- #
def test_index_images_clean(make_converter: MakeConverter) -> None:
    indexed = make_converter()._index_images(valid_coco())
    assert indexed[1] == {"path": "images/a.jpg", "height": 100, "width": 100}
    assert indexed[2] == {"path": "images/b.jpg", "height": 200, "width": 200}

@pytest.mark.parametrize("mutate, exception, expected", [
    pytest.param(lambda c: c["images"][0].pop("id"),
                 KeyError, "Image ID is missing", id="missing-id"),
    pytest.param(lambda c: c["images"][0].pop("file_name"),
                 KeyError, "missing both 'path' and 'file_name' keys", id="missing-path-and-file-name"),
    pytest.param(lambda c: c["images"][0].pop("height"),
                 KeyError, "missing 'height' or 'width' keys", id="missing-height"),
    pytest.param(lambda c: c["images"][0].pop("width"),
                 KeyError, "missing 'height' or 'width' keys", id="missing-width"),
    pytest.param(lambda c: c["images"][0].update({"width": -1}),
                 ValueError, "has non-positive dimensions", id="negative-width"),
    pytest.param(lambda c: c["images"][0].update({"height": 0}),
                 ValueError, "has non-positive dimensions", id="zero-height"),
])
def test_index_images_raises(make_converter: MakeConverter,
                             mutate: Mutate,
                             exception: type[Exception],
                             expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    with pytest.raises(exception, match=expected):
        make_converter()._index_images(coco)

# --------------------------------------------------------------------------- #
# _index_annotations
# --------------------------------------------------------------------------- #
def test_index_annotations_clean(make_converter: MakeConverter) -> None:
    converter = make_converter()

    coco = valid_coco()
    image_dict = converter._index_images(coco)
    image_bbox_dict = converter._index_annotations(coco, image_dict)

    assert image_bbox_dict[1] == [
        {"bbox": [10, 20, 30, 40], "class_id": 5, "image_width": 100, "image_height": 100}
    ]
    assert image_bbox_dict[2] == [
        {"bbox": [0, 0, 10, 10], "class_id": 7, "image_width": 200, "image_height": 200}
    ]

@pytest.mark.parametrize("mutate, exception, expected", [
    pytest.param(lambda c: c["annotations"][0].pop("image_id"),
                 KeyError, "Image ID not present in 'annotations'", id="missing-image-id"),
    pytest.param(lambda c: c["annotations"][0].pop("category_id"),
                 KeyError, "Category ID not present in 'annotations'", id="missing-category-id"),
])
def test_index_annotations_raises(make_converter: MakeConverter,
                                  mutate: Mutate,
                                  exception: type[Exception],
                                  expected: str) -> None:
    converter = make_converter()

    coco = valid_coco()
    mutate(coco)

    image_dict = converter._index_images(coco)

    with pytest.raises(exception, match=expected):
        converter._index_annotations(coco, image_dict)

def test_index_annotations_skips_orphan(make_converter: MakeConverter, caplog: pytest.LogCaptureFixture) -> None:
    converter = make_converter()
    coco = valid_coco()

    # Remove first image (image id = 1)
    coco["images"].pop(0)

    image_dict = converter._index_images(coco)
    image_bbox_dict = converter._index_annotations(coco, image_dict)

    assert len(image_bbox_dict) == 1
    assert image_bbox_dict[2] == [
        {"bbox": [0, 0, 10, 10], "class_id": 7, "image_width": 200, "image_height": 200}
    ]
    assert "Bounding box found for a non referenced image in 'annotations'." in caplog.text

@pytest.mark.parametrize("mutate", [
    pytest.param(lambda c: c["annotations"][0].pop("bbox"), id="missing-bbox-key"),
    pytest.param(lambda c: c["annotations"][0].update({"bbox": None}), id="null-bbox"),
    pytest.param(lambda c: c["annotations"][0].update({"bbox": []}), id="empty-bbox"),
])
def test_index_annotations_skips_absent_bbox(make_converter: MakeConverter,
                                             mutate: Mutate,
                                             caplog: pytest.LogCaptureFixture) -> None:
    converter = make_converter()
    coco = valid_coco()
    mutate(coco)

    image_dict = converter._index_images(coco)
    image_bbox_dict = converter._index_annotations(coco, image_dict)

    assert len(image_bbox_dict) == 1
    assert image_bbox_dict[2] == [
        {"bbox": [0, 0, 10, 10], "class_id": 7, "image_width": 200, "image_height": 200}
    ]
    assert "No bounding box found for annotation entry." in caplog.text

# --------------------------------------------------------------------------- #
# _normalise_bbox
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bbox, width, height, expected", [
    pytest.param((10, 20, 30, 40), 100, 100, (0.25, 0.4, 0.3, 0.4), id="square"),
    pytest.param((0, 0, 10, 10), 200, 200, (0.025, 0.025, 0.05, 0.05), id="origin-corner"),
    pytest.param((0, 0, 100, 100), 100, 100, (0.5, 0.5, 1.0, 1.0), id="full-image"),
    pytest.param((10, 20, 30, 40), 100, 200, (0.25, 0.2, 0.3, 0.2), id="non-square"),
])
def test_normalise_bbox(make_converter: MakeConverter,
                        bbox: tuple[float, float, float, float],
                        width: int,
                        height: int,
                        expected: tuple[float, float, float, float]) -> None:
    entry = CocoBBoxEntry(bbox=bbox, class_id=5, image_width=width, image_height=height)
    assert make_converter()._normalise_bbox(entry) == pytest.approx(expected)

# --------------------------------------------------------------------------- #
# _convert_image_annotations
# --------------------------------------------------------------------------- #
def test_convert_image_annotations_format(make_converter: MakeConverter) -> None:
    entries = [
        CocoBBoxEntry(bbox=(10, 20, 30, 40), class_id=5, image_width=100, image_height=100),
        CocoBBoxEntry(bbox=(0, 0, 10, 10), class_id=7, image_width=200, image_height=200),
    ]
    assert make_converter()._convert_image_annotations(entries) == (
        "0 0.25 0.4 0.3 0.4\n"
        "1 0.025 0.025 0.05 0.05\n"
    )

def test_convert_image_annotations_empty(make_converter: MakeConverter) -> None:
    """Empty image annotations convert to empty string so the downstream writer doesn't write anything"""
    assert make_converter()._convert_image_annotations([]) == ""

def test_convert_image_annotations_unmapped_class_raises(make_converter: MakeConverter) -> None:
    entry = CocoBBoxEntry(bbox=(0, 0, 10, 10), class_id=99, image_width=100, image_height=100)
    with pytest.raises(ValueError, match="does not have a YOLO mapping"):
        make_converter()._convert_image_annotations([entry])

# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #
def test_convert_yields_paths_and_labels(make_converter: MakeConverter) -> None:
    results = list(make_converter().convert(valid_coco()))

    assert results == [
        ("images/a.jpg", "0 0.25 0.4 0.3 0.4\n"),
        ("images/b.jpg", "1 0.025 0.025 0.05 0.05\n"),
    ]

def test_convert_no_annotations_yields_none(make_converter: MakeConverter) -> None:
    coco = valid_coco()
    coco.pop("annotations")

    results = list(make_converter().convert(coco))

    assert results == [
        ("images/a.jpg", None),
        ("images/b.jpg", None),
    ]

def test_convert_unannotated_image_yields_none(make_converter: MakeConverter) -> None:
    coco = valid_coco()
    coco["annotations"].pop(1)

    results = list(make_converter().convert(coco))

    assert results == [
        ("images/a.jpg", "0 0.25 0.4 0.3 0.4\n"),
        ("images/b.jpg", None),
    ]

def test_convert_invalid_bbox_contract_raises(make_converter: MakeConverter) -> None:
    coco = valid_coco()
    coco["annotations"][0].update({"bbox": [0, 10, 10]})

    with pytest.raises(ValueError, match="Invalid image bounding box mapping."):
        list(make_converter().convert(coco))
