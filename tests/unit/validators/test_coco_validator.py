import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from src.datatypes import ClassConfigMapping, DatasetConfigModel, SplitConfig
from src.validators.base_validator import DatasetValidationError
from src.validators.coco_validator import CocoDatasetValidator, SplitAnnotations

CocoDict: TypeAlias = dict[str, Any]
MakeValidator: TypeAlias = Callable[..., CocoDatasetValidator]
Mutate: TypeAlias = Callable[[CocoDict], Any]


@pytest.fixture
def make_validator(tmp_path: Path) -> MakeValidator:
    def _make(classes: ClassConfigMapping | None = None, splits: SplitConfig | None = None) -> CocoDatasetValidator:
        cfg = DatasetConfigModel(
            classes=classes or {"forklift": {"coco_id": 5, "yolo_id": 0}},
            splits=splits or {"train": ["train.json"], "val": ["val.json"]},
            path=Path("dataset"),
            type="object_detection",
            format="coco",
            name="test",
        )
        return CocoDatasetValidator(tmp_path, cfg, log_path=tmp_path / "validation.log")

    return _make

def wrap(coco: CocoDict, split: str = 'train', fname: str = "train.json") -> SplitAnnotations:
    """Wrap one COCO dict into the SplitAnnotations shape the checks consume."""
    return {split: {Path(fname): coco}}

def valid_coco() -> CocoDict:
    """Well-formed COCO; copy and inject one defect per case."""
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/b.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 5, "name": "forklift"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }

@pytest.fixture
def dataset_on_disk(tmp_path: Path) -> Path:
    """Writes a complete COCO dataset that satisfies every check."""
    def _coco(prefix: str, count: int) -> CocoDict:
        return {
            "images": [
                {"id": i, "file_name": f"{prefix}_{i}.jpg", "width": 100, "height": 100}
                for i in range(1, count + 1)
            ],
            "categories": [{"id": 5, "name": "forklift"}],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 5, "bbox": [0, 0, 10, 10]}
                for i in range(1, count + 1)
            ]
        }

    (tmp_path / "annotations").mkdir()
    (tmp_path / "images").mkdir()

    for split, count in (("train", 2), ("val", 1)):
        (tmp_path / "annotations" / f"{split}.json").write_text(json.dumps(_coco(split, count)))
        for i in range(1, count + 1):
            (tmp_path / "images" / f"{split}_{i}.jpg").touch()

    return tmp_path

# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("dataset_on_disk")
def test_validate_passes_on_valid_dataset(make_validator: MakeValidator) -> None:
    v = make_validator()
    v.validate()  # must not raise
    assert v.log_path.read_text() == ""  # ERROR-only handler empty

def test_validate_raises_on_invalid_dataset(make_validator: MakeValidator, dataset_on_disk: Path) -> None:
    (dataset_on_disk / "images" / "train_1.jpg").unlink()
    v = make_validator()

    with pytest.raises(DatasetValidationError):
        v.validate()

    assert "does not exist" in v.log_path.read_text()

# --------------------------------------------------------------------------- #
# _get_annotation_paths_by_split
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("splits, expected_names", [
    pytest.param({"train": "single.json"}, ["single.json"], id="bare-string"),
    pytest.param({"train": ["a.json", "b.json"]}, ["a.json", "b.json"], id="list")
])
def test_annotation_paths_normalise_split_config(
    make_validator: MakeValidator, tmp_path: Path, splits: SplitConfig, expected_names: list[str]) -> None:
    paths = make_validator(splits=splits)._get_annotation_paths_by_split()
    assert paths["train"] == [tmp_path / "annotations" / name for name in expected_names]

def test_annotation_paths_follow_annotation_path_override(tmp_path: Path) -> None:
    cfg = DatasetConfigModel.model_validate({
        "classes": {"forklift": {"coco_id": 5, "yolo_id": 0}},
        "splits": {"train": ["train.json"]},
        "path": Path("dataset"),
        "type": "object_detection",
        "format": "coco",
        "name": "test",
        "annotations_path": "labels",
    })
    v = CocoDatasetValidator(tmp_path, cfg, log_path=tmp_path / "validation.log")

    assert v._get_annotation_paths_by_split()["train"] == [tmp_path / "labels" / "train.json"]

# --------------------------------------------------------------------------- #
# _run_checks
# --------------------------------------------------------------------------- #
def test_run_checks_aborts_on_load_error(make_validator: MakeValidator, tmp_path: Path) -> None:
    v = make_validator(splits={"train": ["train.json"]})
    anno_dir = tmp_path / "annotations"
    anno_dir.mkdir()
    (anno_dir / "train.json").write_text("{not a valid json")

    errors = v._run_checks()

    assert errors

    # every error came from the loader
    assert all(e.startswith("_load_annotations_by_split") for e in errors)

# --------------------------------------------------------------------------- #
# _load_annotations_by_split
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [
    pytest.param({"images": 123, "categories": [{"id": 5, "name": "f"}]}, id="images-int"),
    pytest.param({"images": "abc", "categories": [{"id": 5, "name": "f"}]}, id="images-str"),
    pytest.param({"images": ["abc"], "categories": [{"id": 5, "name": "f"}]}, id="images-list-of-non-dicts"),
    pytest.param({"categories": [{"id": 5, "name": "f"}]}, id="images-key-missing"),
    pytest.param({"images": [{"id": 1}], "categories": 123}, id="categories-int"),
    pytest.param({"images": [{"id": 1}], "categories": "abc"}, id="categories-str"),
    pytest.param({"images": [{"id": 1}], "categories": ["abc"]}, id="categories-list-of-non-dicts"),
    pytest.param({"images": [{"id": 1}]}, id="categories-key-missing"),
    pytest.param({"images": [{"id": 1}], "categories": [{"id": 5, "name": "f"}],
                 "annotations": None}, id="annotations-null"),
])
def test_load_rejects_invalid_coco_structure(make_validator: MakeValidator, tmp_path: Path, payload: CocoDict) -> None:
    v = make_validator(splits={"train": ["train.json"]})
    anno_dir = tmp_path / "annotations"
    fpath = anno_dir / "train.json"
    fpath.parent.mkdir()

    fpath.write_text(json.dumps(payload))

    split_annotations, errors = v._load_annotations_by_split()

    assert any("is not a valid COCO annotation file" in e for e in errors)
    assert split_annotations["train"] == {}  # rejected file is not stored

def test_load_valid_file(make_validator: MakeValidator, tmp_path: Path) -> None:
    v = make_validator(splits={"train": ["train.json"]})
    anno_dir = tmp_path / "annotations"
    fpath = anno_dir / "train.json"
    fpath.parent.mkdir()

    fpath.write_text(json.dumps(valid_coco()))

    split_annotations, errors = v._load_annotations_by_split()

    assert errors == []
    assert split_annotations["train"][fpath] == valid_coco()

# --------------------------------------------------------------------------- #
# _extract_config_class_mapping
# --------------------------------------------------------------------------- #
def test_extract_config_class_mapping_skips_missing_coco_id(make_validator: MakeValidator) -> None:
    v = make_validator(classes={
        "forklift": {"coco_id": 5, "yolo_id": 0},
        "unused": {"yolo_id": 1}, # no coco_id -> dropped
    })
    assert v._extract_config_class_mapping() == {"forklift": 5}

def test_extract_config_class_mapping(make_validator: MakeValidator) -> None:
    v = make_validator(classes={
        "forklift": {"coco_id": 5, "yolo_id": 0},
        "pallet": {"coco_id": 7, "yolo_id": 1},
    })
    assert v._extract_config_class_mapping() == {"forklift": 5, "pallet": 7}

# --------------------------------------------------------------------------- #
# _check_annotation_structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
    pytest.param(lambda c: c.pop("images"),         "Key 'images' is missing",        id="missing_images"),
    pytest.param(lambda c: c.update(images=[]),     "'images' field is empty",        id="empty_images"),
    pytest.param(lambda c: c.pop("categories"),     "Key 'categories' is missing",    id="missing_categories"),
    pytest.param(lambda c: c.update(categories=[]), "'categories' field is empty",    id="empty_categories"),
    pytest.param(lambda c: c.pop("annotations"),    "split contains no annotations.", id="missing-annotations"),
    pytest.param(lambda c: c.update(annotations=[]), "'annotations' field for split train is empty.",
                 id="empty-annotations"),
    ]
)
def test_check_annotation_structure(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    errors = make_validator()._check_annotation_structure(wrap(coco))
    assert any(expected in e for e in errors)

def test_check_annotation_structure_clean(make_validator: MakeValidator) -> None:
    assert make_validator()._check_annotation_structure(wrap(valid_coco())) == []

# --------------------------------------------------------------------------- #
# _check_image_dimensions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, expected", [
    pytest.param(lambda c: c["images"][0].pop("width"),               "has no 'width'", id="missing-width"),
    pytest.param(lambda c: c["images"][0].update(width=float("nan")), "non-integer",    id="nan-width"),
    pytest.param(lambda c: c["images"][0].update(width=True),         "non-integer",    id="bool-width"),
    pytest.param(lambda c: c["images"][0].update(width=""),           "non-integer",    id="empty-str-width"),
    pytest.param(lambda c: c["images"][0].update(width="100"),        "non-integer",    id="str-width"),
    pytest.param(lambda c: c["images"][0].update(width=100.5),        "non-integer",    id="float-pixel-width"),
    pytest.param(lambda c: c["images"][0].update(width=-5),           "non-positive",   id="negative-width"),
    pytest.param(lambda c: c["images"][0].update(width=0),            "non-positive",   id="zero-width"),
    pytest.param(lambda c: c["images"][0].pop("height"), "has no 'width' and/or 'height'", id="missing-height"),
    pytest.param(lambda c: c["images"][0].update(height=float("nan")), "non-integer",    id="nan-height"),
    pytest.param(lambda c: c["images"][0].update(height=True),         "non-integer",    id="bool-height"),
    pytest.param(lambda c: c["images"][0].update(height=""),           "non-integer",    id="empty-str-height"),
    pytest.param(lambda c: c["images"][0].update(height="100"),        "non-integer",    id="str-height"),
    pytest.param(lambda c: c["images"][0].update(height=100.5),        "non-integer",    id="float-pixel-height"),
    pytest.param(lambda c: c["images"][0].update(height=-5),           "non-positive",   id="negative-height"),
    pytest.param(lambda c: c["images"][0].update(height=0),            "non-positive",   id="zero-height"),
    pytest.param(lambda c: c["images"][0].update(height=100.5, width=float("nan")),
                 "non-integer", id="float-pixel-width-nan-height"),
])
def test_check_image_dimensions_errors(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    errors = make_validator()._check_image_dimensions(wrap(coco))
    assert any(expected in e for e in errors)

def test_check_image_dimensions_clean(make_validator: MakeValidator) -> None:
    assert make_validator()._check_image_dimensions(wrap(valid_coco())) == []

# --------------------------------------------------------------------------- #
# _check_bbox_validity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, expected", [
    pytest.param(lambda c: c["annotations"][0].pop("bbox"),                  "is missing",   id="bbox-missing"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[]),              "is malformed", id="bbox-empty"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[0, 0, 1]),       "is malformed", id="bbox-malformed"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[0, 0, -1, 1]),   "non-positive", id="bbox-non_positive"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[0, 0, 0, 0]),    "non-positive", id="bbox-all-zeros"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[0, 0, 1, "x"]),  "non-numeric", id="bbox-str-coord"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[0, 0, 1, True]), "non-numeric", id="bbox-bool-coord"),
    pytest.param(lambda c: c["annotations"][0].update(bbox=[0, 0, 1, float("nan")]),
                 "non-numeric", id="bbox-nan-coord"),
])
def test_check_bbox_validity(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    errors = make_validator()._check_bbox_validity(wrap(coco))
    assert any(expected in e for e in errors)

def test_check_bbox_validity_clean(make_validator: MakeValidator) -> None:
    assert make_validator()._check_bbox_validity(wrap(valid_coco())) == []

# --------------------------------------------------------------------------- #
# _check_annotation_completeness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, expected", [
    pytest.param(lambda c: c["images"][0].pop("id"), "image_id is missing", id="missing-image-id"),
    pytest.param(lambda c: c["images"][0].pop("file_name"), "Image path for image_id", id="missing-file-name-and-path"),
    pytest.param(
        lambda c: c["images"].append({"id": 1, "file_name": "dup.jpg", "width": 100, "height": 100}),
        "DUPLICATE IDS", id="duplicate-image-ids",
    ),
    pytest.param(
        lambda c: c["images"][1].update({"file_name": "a.jpg", "path": None}),
        "DUPLICATE IMAGE PATHS", id="duplicate-image-filenames"
    ),
    pytest.param(
        lambda c: c["images"][0].update({"path": "images/b.jpg"}),
        "DUPLICATE IMAGE PATHS", id="duplicate-image-paths"
    ),
    pytest.param(
        lambda c: c["annotations"].append({"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]}),
        "DUPLICATE ANNOTATION IDS", id="duplicate-annotation-ids"
    ),
    pytest.param(
        lambda c: c["images"].append({"id": 3, "file_name": "dup.jpg", "width": 100, "height": 100}),
        "IMAGES NOT ANNOTATED", id="unannotated-images"
    ),
    pytest.param(
        lambda c: c["annotations"].append({"id": 3, "image_id": 99, "category_id": 5, "bbox": [0, 0, 10, 10]}),
        "ORPHANED IMAGE REFERENCES", id="orphaned-image-references"
    )
])
def test_check_annotation_completeness(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    errors = make_validator()._check_annotation_completeness(wrap(coco))
    assert any(expected in e for e in errors)

def test_check_annotation_completeness_clean(make_validator: MakeValidator) -> None:
    assert make_validator()._check_annotation_completeness(wrap(valid_coco())) == []

# --------------------------------------------------------------------------- #
# _check_class_mapping_coverage  (config vs annotation categories)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, expected", [
    pytest.param(
        lambda c: c["categories"].pop(0),
        "Class coverage mismatch.", id="class-coverage-mismatch-dropped"
    ),
    pytest.param(
        lambda c: c["categories"][0].update({"name": "new_forklift"}),
        "Class coverage mismatch.", id="class-coverage-mismatch-name"
    ),
    pytest.param(
        lambda c: c["categories"][0].update({"id": 99}),
        "Class ID mismatch", id="class-id-mismatch"
    )
])
def test_coverage_mismatch(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    errors = make_validator()._check_class_mapping_coverage(wrap(coco))
    assert any(expected in e for e in errors)

def test_check_class_mapping_coverage_clean(make_validator: MakeValidator) -> None:
    assert make_validator()._check_class_mapping_coverage(wrap(valid_coco())) == []

# --------------------------------------------------------------------------- #
# _check_class_consistency  (multi-file: differing categories between files)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, expected", [
    pytest.param(
        lambda c: c["categories"][0].update({"name": "new_forklift"}),
        "has different classes than", id="different-class-name"
    ),
    pytest.param(
        lambda c: c["categories"][0].update({"id": 99}),
        "has different classes than", id="different-id"
    ),
    pytest.param(
        lambda c: c["categories"].append({"name": "new_forklift", "id": 99}),
        "has different classes than", id="extra-class"
    ),
])
def test_class_consistency_mismatch(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco_train_1 = valid_coco()
    coco_train_2 = valid_coco()
    mutate(coco_train_2)

    split_annotations = wrap(coco_train_1, split='train', fname="train1.json")
    split_annotations["train"].update({Path('train2.json'): coco_train_2})

    errors = make_validator()._check_class_consistency(split_annotations)
    assert any(expected in e for e in errors)

def test_check_class_consistency_clean(make_validator: MakeValidator) -> None:
    split_annotations = wrap(valid_coco(), split='train', fname="train1.json")
    split_annotations["train"].update({Path('train2.json'): valid_coco()})
    split_annotations.update(wrap(valid_coco(), split='test', fname="test.json"))

    assert make_validator()._check_class_consistency(split_annotations) == []

# --------------------------------------------------------------------------- #
# _check_split_filename_uniqueness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, expected", [
    pytest.param(
        lambda c: c["images"].append({"id": 99, "file_name": "a.jpg", "width": 100, "height": 100}),
        "Filename conflict between 'id':", id="filename-filename-conflict"
    ),
    pytest.param(
        lambda c: c["images"].append({"id": 99, "file_name": "b.jpg", "width": 100, "height": 100}),
        "Filename conflict between 'id':", id="filename-path-conflict"
    )
])
def test_filename_conflict(make_validator: MakeValidator, mutate: Mutate, expected: str) -> None:
    coco = valid_coco()
    mutate(coco)
    errors = make_validator()._check_split_filename_uniqueness(wrap(coco))
    assert any(expected in e for e in errors)

def test_filename_conflict_across_files(make_validator: MakeValidator) -> None:
    split_annotations = wrap(valid_coco(), split="train", fname="train1.json")
    split_annotations["train"].update({Path("train2.json"): valid_coco()})

    errors = make_validator()._check_split_filename_uniqueness(split_annotations)
    assert any("Filename conflict between 'id':" in e for e in errors)

def test_filename_benign_duplicate_warns(make_validator: MakeValidator, caplog: pytest.LogCaptureFixture) -> None:
    v = make_validator()
    coco = valid_coco()
    existing_entry = dict(coco["images"][0])
    coco["images"].append(existing_entry)
    errors = v._check_split_filename_uniqueness(wrap(coco))
    assert len(errors) == 0
    assert "is duplicated." in caplog.text

def test_check_filename_uniqueness_clean(make_validator: MakeValidator) -> None:
    coco_train_1 = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/b.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 5, "name": "forklift"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }

    coco_train_2 = {
        "images": [
            {"id": 1, "file_name": "c.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/d.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 5, "name": "forklift"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }

    split_annotations = wrap(coco_train_1, split="train", fname="train1.json")
    split_annotations["train"].update({Path("train2.json"): coco_train_2})

    assert make_validator()._check_split_filename_uniqueness(split_annotations) == []


# --------------------------------------------------------------------------- #
# _check_missing_images  (needs files on disk; file_name -> images/<name>)
# --------------------------------------------------------------------------- #
def test_missing_image_on_disk(make_validator: MakeValidator, tmp_path: Path) -> None:
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/b.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 5, "name": "forklift"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }

    v = make_validator()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").touch()  # a.jpg present, b.jpg absent
    errors = v._check_missing_images(wrap(coco))
    assert any("for image_id 2 does not exist" in e for e in errors)

def test_check_missing_images_clean(make_validator: MakeValidator, tmp_path: Path) -> None:
    coco = valid_coco()
    for image in coco["images"]:
        path = image.get("path")
        filename = image.get("file_name")

        if path is not None:
            fpath = tmp_path / Path(path)
        elif filename is not None:
            fpath = tmp_path / "images" / filename
        else:
            continue

        fpath.parent.mkdir(exist_ok=True)
        fpath.touch()

    assert make_validator()._check_missing_images(wrap(coco)) == []

# --------------------------------------------------------------------------- #
# _check_data_leakage  (multi-split: same image in two splits)
# --------------------------------------------------------------------------- #
def test_cross_split_leakage(make_validator: MakeValidator) -> None:
    v = make_validator()
    coco = valid_coco()
    split_annotations = wrap(coco, split='train', fname="train.json")
    split_annotations.update(wrap(coco, split='test', fname="test.json"))
    errors = v._check_data_leakage(split_annotations)
    assert any("Data leakage detected" in e for e in errors)

def test_data_leakage_clean(make_validator: MakeValidator) -> None:
    coco_train = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/b.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 5, "name": "forklift"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }

    coco_test = {
        "images": [
            {"id": 1, "file_name": "c.jpg", "width": 100, "height": 100},
            {"id": 2, "path": "images/d.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 5, "name": "forklift"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10]},
        ],
    }

    split_annotations = wrap(coco_train, split="train", fname="train.json")
    split_annotations.update(wrap(coco_test, split="test", fname="test.json"))

    assert make_validator()._check_data_leakage(split_annotations) == []


# --------------------------------------------------------------------------- #
# _load_annotations_by_split
# --------------------------------------------------------------------------- #
def test_load_unloadable_file(make_validator: MakeValidator, tmp_path: Path) -> None:
    v = make_validator(splits={"train": ["train.json"]})
    anno_dir = tmp_path / "annotations"
    anno_dir.mkdir()
    (anno_dir / "train.json").write_text("{not valid json")
    _, errors = v._load_annotations_by_split()
    assert any("Failed to load" in e for e in errors)


def test_load_top_level_list(make_validator: MakeValidator, tmp_path: Path) -> None:
    v = make_validator(splits={"train": ["train.json"]})
    anno_dir = tmp_path / "annotations"
    anno_dir.mkdir()
    (anno_dir / "train.json").write_text("[1, 2, 3]")
    _, errors = v._load_annotations_by_split()
    assert any("Expected a dict" in e for e in errors)

# --------------------------------------------------------------------------- #
# test-split exemptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("check, mutate, marker", [
    pytest.param("_check_annotation_structure",
                 lambda c: c.update(annotations=[]),
                 "'annotations' field for split", id="structure-empty-annotations"),
    pytest.param("_check_annotation_structure",
                 lambda c: c.pop("annotations", None),
                 "split contains no annotations", id="structure-none-annotations"),
    pytest.param("_check_annotation_completeness",
                 lambda c: c.update(annotations=[]),
                 "No annotations found in annotation file", id="completeness-no-annotations"),
    pytest.param("_check_annotation_completeness",
                 lambda c: c["images"].append({"id": 99, "file_name": "c.jpg", "width": 100, "height": 100}),
                 "IMAGES NOT ANNOTATED: ", id="completeness-unannotated-images"),
    pytest.param("_check_bbox_validity",
                 lambda c: c["annotations"][0].pop("bbox", None),
                 "bbox for annotation 'id'", id="bbox-validity-none-bbox")
])
def test_test_split_exemptions(make_validator: MakeValidator, check: str, mutate: Mutate, marker: str) -> None:
    coco = valid_coco()
    mutate(coco)
    v = make_validator()

    # control - training split
    assert any(marker in e for e in getattr(v, check)(wrap(coco, split="train")))

    # exemption - error is not firing on test split
    assert not any(marker in e for e in getattr(v, check)(wrap(coco, split="test")))
