import builtins
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from src.datatypes import ClassConfigMapping
from src.utils.file_ops import load_yaml_config
from src.writers.yolo_writer import YoloWriter

IMAGE_BYTES = b"fake-image-bytes"
DATASET_DIRNAME = "loco"
MakeWriter: TypeAlias = Callable[..., YoloWriter]


@pytest.fixture
def make_writer(tmp_path: Path) -> MakeWriter:
    def _make(class_mapping: ClassConfigMapping | None = None) -> YoloWriter:
        return YoloWriter(tmp_path / DATASET_DIRNAME, class_mapping or {
            "forklift": {"coco_id": 5, "yolo_id": 0},
            "pallet": {"coco_id": 7, "yolo_id": 1},
        })

    return _make

@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    """A real file on disk."""
    image_path = tmp_path / "raw" / "a.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(IMAGE_BYTES)

    return image_path

# --------------------------------------------------------------------------- #
# _extract_classes
# --------------------------------------------------------------------------- #
def test_extract_classes_builds_mapping(make_writer: MakeWriter) -> None:
    assert make_writer()._extract_classes() == {
        0: "forklift",
        1: "pallet",
    }

def test_extract_classes_sorts_by_yolo_id(make_writer: MakeWriter) -> None:
    w = make_writer(class_mapping={
        "pallet": {"coco_id": 7, "yolo_id": 1},
        "forklift": {"coco_id": 5, "yolo_id": 0},
    })

    assert w._extract_classes() == {
        0: "forklift",
        1: "pallet",
    }

def test_extract_classes_missing_yolo_id_raises(make_writer: MakeWriter) -> None:
    with pytest.raises(ValueError, match="YOLO ID missing for class 'pallet'"):
        make_writer(class_mapping={
            "pallet": {"coco_id": 7},
            "forklift": {"coco_id": 5, "yolo_id": 0},
        })

# --------------------------------------------------------------------------- #
# setup_split
# --------------------------------------------------------------------------- #
def test_setup_split_creates_image_and_label_dirs(make_writer: MakeWriter) -> None:
    w = make_writer()

    splits = ("train", "val")
    data_dirs = ("images", "labels")
    for split in splits:
        w.setup_split(split)

        for data_dir in data_dirs:
            assert (w.output_dir / data_dir / split).is_dir()

def test_setup_split_test_has_no_label_dir(make_writer: MakeWriter) -> None:
    w = make_writer()

    w.setup_split("test")

    assert (w.output_dir / "images" / "test").is_dir()
    assert not (w.output_dir / "labels" / "test").exists()

def test_setup_split_is_idempotent(make_writer: MakeWriter) -> None:
    w = make_writer()

    w.setup_split("train")
    w.setup_split("train")

    assert (w.output_dir / "images" / "train").is_dir()
    assert (w.output_dir / "labels" / "train").is_dir()

@pytest.mark.parametrize("split_name", [
    pytest.param("", id="empty-string"),
    pytest.param("trian", id="typo"),
    pytest.param("TRAIN", id="uppercase"),
    pytest.param("training", id="long-format"),
    pytest.param("invalid", id="invalid-split-name")
])
def test_setup_split_invalid_name_raises(make_writer: MakeWriter, split_name: str) -> None:
    with pytest.raises(ValueError, match="Invalid split name"):
        make_writer().setup_split(split_name)

def test_setup_split_directory_failure_raises(make_writer: MakeWriter,
                                              caplog: pytest.LogCaptureFixture) -> None:
    w = make_writer()
    w.output_dir.mkdir()

    # A file where the images directory must go, so mkdir cannot create it.
    (w.output_dir / "images").write_text("not a directory")

    with pytest.raises(RuntimeError, match="Could not create directory structure"):
        w.setup_split("train")

    assert "Filesystem error while initialising split" in caplog.text

# --------------------------------------------------------------------------- #
# write_item
# --------------------------------------------------------------------------- #
def test_write_item_writes_image_and_label(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)
    w.write_item(source_image, target_filename, label_content, split_name)

    output_image_path = w.output_dir / 'images' / split_name / target_filename
    output_label_path = w.output_dir / 'labels' / split_name / Path(target_filename).with_suffix('.txt')

    assert output_image_path.read_bytes() == IMAGE_BYTES
    assert output_label_path.read_text() == label_content

def test_write_item_no_label_content_writes_image_only(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    target_filename = "a.jpg"
    label_content = None
    split_name = "val"

    w.setup_split(split_name)
    w.write_item(source_image, target_filename, label_content, split_name)

    output_image_path = w.output_dir / 'images' / split_name / target_filename
    output_label_path = w.output_dir / 'labels' / split_name / Path(target_filename).with_suffix('.txt')

    assert output_image_path.read_bytes() == IMAGE_BYTES
    assert not output_label_path.exists()

def test_write_item_test_split_skips_label(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "test"

    w.setup_split(split_name)
    w.write_item(source_image, target_filename, label_content, split_name)

    output_image_path = w.output_dir / 'images' / split_name / target_filename
    output_label_path = w.output_dir / 'labels' / split_name / Path(target_filename).with_suffix('.txt')

    assert output_image_path.read_bytes() == IMAGE_BYTES
    assert not output_label_path.exists()

def test_write_item_missing_source_raises(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)
    source_image.unlink()

    with pytest.raises(FileNotFoundError, match="Source image missing"):
        w.write_item(source_image, target_filename, label_content, split_name)

def test_write_item_existing_image_target_raises(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)

    output_image_path = w.output_dir / 'images' / split_name / target_filename
    output_image_path.touch()

    with pytest.raises(FileExistsError, match="Target image already exists"):
        w.write_item(source_image, target_filename, label_content, split_name)

def test_write_item_existing_label_target_raises(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)

    output_label_path = w.output_dir / 'labels' / split_name / Path(target_filename).with_suffix('.txt')
    output_label_path.touch()

    with pytest.raises(FileExistsError, match="Target label file already exists"):
        w.write_item(source_image, target_filename, label_content, split_name)


def test_write_item_removes_orphan_when_label_write_fails(make_writer: MakeWriter,
                                                          source_image: Path,
                                                          monkeypatch: pytest.MonkeyPatch,
                                                          caplog: pytest.LogCaptureFixture) -> None:
    w = make_writer()
    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)

    real_open = builtins.open

    def failing_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        """Fail only the label write. shutil.copy2 needs open for the image."""
        if str(file).endswith(".txt"):
            raise OSError("Simulated disk failure")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)

    with pytest.raises(RuntimeError, match="Disk failure or permission issue"):
        w.write_item(source_image, target_filename, label_content, split_name)

    assert not (w.output_dir / "images" / split_name / target_filename).exists()
    assert "removing orphaned image" in caplog.text

@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="Root bypasses directory write permissions")
def test_write_item_permission_error_raises(make_writer: MakeWriter,
                                            source_image: Path,
                                            caplog: pytest.LogCaptureFixture) -> None:
    w = make_writer()
    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)

    # Read only image directory, so copy is denied
    image_dir = w.output_dir / "images" / split_name
    original_mode = stat.S_IMODE(image_dir.stat().st_mode)
    image_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        with pytest.raises(PermissionError):
            w.write_item(source_image, target_filename, label_content, split_name)
    finally:
        image_dir.chmod(original_mode)

    assert "Access error processing" in caplog.text

# --------------------------------------------------------------------------- #
# finalise
# --------------------------------------------------------------------------- #
def test_finalise_writes_dataset_yaml(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    label_content = "0 0.25 0.4 0.3 0.4\n"
    splits = ["train", "val"]

    for split in splits:
        w.setup_split(split)
        w.write_item(source_image, f"{split}.jpg", label_content, split)

    w.finalise(splits)

    config = load_yaml_config(w.output_dir / "dataset.yaml")

    assert config == {
        "path": DATASET_DIRNAME,
        "names": {0: "forklift", 1: "pallet"},
        "train": "images/train",
        "val": "images/val",
    }

def test_finalise_omits_absent_splits(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    label_content = "0 0.25 0.4 0.3 0.4\n"
    splits = ["train", "val"]

    for split in splits:
        w.setup_split(split)
        w.write_item(source_image, f"{split}.jpg", label_content, split)

    w.finalise(["train"])

    config = load_yaml_config(w.output_dir / "dataset.yaml")

    assert config == {
        "path": DATASET_DIRNAME,
        "names": {0: "forklift", 1: "pallet"},
        "train": "images/train",
    }

def test_finalise_empty_splits_raises(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    label_content = "0 0.25 0.4 0.3 0.4\n"
    splits = ["train", "val"]

    for split in splits:
        w.setup_split(split)
        w.write_item(source_image, f"{split}.jpg", label_content, split)

    with pytest.raises(RuntimeError, match="Cannot finalise an empty dataset."):
        w.finalise([])

def test_finalise_missing_split_dir_raises(make_writer: MakeWriter) -> None:
    w = make_writer()

    splits = ["train"]

    with pytest.raises(RuntimeError, match=f"Dataset integrity check failed: Split '{splits[0]}' is missing."):
        w.finalise(splits)

def test_finalise_empty_split_dir_raises(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    label_content = "0 0.25 0.4 0.3 0.4\n"
    splits = ["train", "val"]

    for split in splits:
        w.setup_split(split)
        w.write_item(source_image, f"{split}.jpg", label_content, split)

    (w.output_dir / "images" / "train" / f"{splits[0]}.jpg").unlink()

    with pytest.raises(RuntimeError, match=f"Dataset integrity check failed: Split '{splits[0]}' is missing."):
        w.finalise(splits)

def test_finalise_no_train_split_raises(make_writer: MakeWriter, source_image: Path) -> None:
    w = make_writer()

    label_content = "0 0.25 0.4 0.3 0.4\n"
    splits = ["val"]

    for split in splits:
        w.setup_split(split)
        w.write_item(source_image, f"{split}.jpg", label_content, split)

    with pytest.raises(RuntimeError, match="No 'train' split provided"):
        w.finalise(splits)

def test_finalise_config_write_failure_raises(make_writer: MakeWriter,
                                              source_image: Path,
                                              caplog: pytest.LogCaptureFixture) -> None:
    w = make_writer()
    target_filename = "a.jpg"
    label_content = "0 0.25 0.4 0.3 0.4\n"
    split_name = "train"

    w.setup_split(split_name)
    w.write_item(source_image, target_filename, label_content, split_name)

    # A directory where dataset.yaml must go, so open() cannot write it
    (w.output_dir / "dataset.yaml").mkdir()

    with pytest.raises(RuntimeError, match="Critical failure during dataset finalisation"):
        w.finalise([split_name])

    assert "Failed to write dataset.yaml" in caplog.text
