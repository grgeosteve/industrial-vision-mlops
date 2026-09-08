import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias, cast

import pytest
import yaml

from src import paths
from src.data.process_dataset import get_pipeline_components, get_validator, main, process_dataset
from src.datatypes import ClassConfigMapping, DatasetConfig, DatasetConfigModel, DatasetConfigs
from src.handlers.base_handler import BaseFormatHandler
from src.handlers.coco_handler import CocoFormatHandler
from src.utils import coco_ops, file_ops
from src.validators.base_validator import BaseDatasetValidator, DatasetValidationError
from src.validators.coco_validator import CocoDatasetValidator
from src.writers.base_writer import BaseDatasetWriter
from src.writers.yolo_writer import YoloWriter

CocoDict: TypeAlias = dict[str, Any]
WriteConfigFile: TypeAlias = Callable[..., Path]
WriteRawDataset: TypeAlias = Callable[..., Path]
PipelineComponents: TypeAlias = tuple[type[BaseFormatHandler], type[BaseDatasetWriter]]

ARGPARSE_USAGE_ERROR = 2  # argparse exits with 2 on usage error

DATASET_NAME = "loco"
TARGET_FORMAT = "yolo"
CONFIG_FILENAME = "dataset_config.yaml"
NUM_CLASSES = 3
CLASS_NAMES = [f"category_{i}" for i in range(1, NUM_CLASSES + 1)]
CLASS_IDS = list(range(1, NUM_CLASSES + 1))

NUM_TRAIN_ANNOTATION_FILES = 3
NUM_VAL_ANNOTATION_FILES = 2
IMAGES_PER_ANNOTATION_FILE = 2

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_BYTES = b"fake-image-bytes"

BBOX = (0, 0, 10, 10)
# BBOX normalised against IMAGE_WIDTH x IMAGE_HEIGHT: centre (0.05, 0.05), size 0.1 x 0.1
NORMALISED_BBOX = "0.05 0.05 0.1 0.1"


@pytest.fixture(autouse=True)
def pipeline_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirects the pipeline data directories and the validation log into tmp_path."""
    monkeypatch.setattr(paths, "EXTERNAL_DATA_DIR", tmp_path / "external")
    monkeypatch.setattr(paths, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")

    # BaseDatasetValidator binds its default log path at import
    # Patching it needs to replace the default itself
    monkeypatch.setattr(BaseDatasetValidator.__init__, "__defaults__", (tmp_path / "logs" / "validation.log",))

def make_config_data(source_format: str = "coco",
                     dataset_type: str = "object_detection",
                     target_format: str = "yolo") -> dict[str, DatasetConfigs]:
    """Returns the dataset configuration file structure."""
    classes: ClassConfigMapping = {}
    for i in range(NUM_CLASSES):
        classes[CLASS_NAMES[i]] = {f"{source_format}_id": CLASS_IDS[i],
                                   f"{target_format}_id": i}

    annotations: list[str] = [f"train_{(i + 1)}.json" for i in range(NUM_TRAIN_ANNOTATION_FILES)]
    annotations.extend([f"val_{(i + 1)}.json" for i in range(NUM_VAL_ANNOTATION_FILES)])

    return {
        "datasets": {
            DATASET_NAME: {
                "name": "LOCO",
                "type": dataset_type,
                "format": source_format,
                "path": "dataset",
                "annotations_path": "annotations",
                "annotations": annotations,
                "splits": {
                    "train": [f"train_{(i + 1)}.json" for i in range(NUM_TRAIN_ANNOTATION_FILES)],
                    "val": [f"val_{(i + 1)}.json" for i in range(NUM_VAL_ANNOTATION_FILES)]
                },
                "classes": classes
            }
        }
    }

def make_dataset_config_data(source_format: str = "coco",
                             dataset_type: str = "object_detection",
                             target_format: str = "yolo") -> DatasetConfig:
    """Returns a single dataset entry from the configuration file structure."""
    datasets: DatasetConfigs = make_config_data(source_format, dataset_type, target_format)["datasets"]

    return datasets[DATASET_NAME]

def make_config(source_format: str = "coco",
                dataset_type: str = "object_detection",
                target_format: str = "yolo") -> DatasetConfigModel:
    """Returns the validated configuration model for a single dataset."""
    config_data = make_config_data(source_format, dataset_type, target_format)

    return DatasetConfigModel(**config_data["datasets"][DATASET_NAME])

def valid_coco(prefix: str, count: int, image_dir: str | None = None) -> CocoDict:
    """Returns a well-formed COCO annotation document.

    An image directory yields the LOCO 'path' field, None the COCO 'file_name' layout.
    """
    # image_dir=None covers the plain COCO layout. raw_dataset always supplies the LOCO 'path'
    def _image(index: int) -> dict[str, Any]:
        file_name = f"{prefix}_{index}.jpg"
        image = {"id": index, "file_name": file_name, "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT}
        if image_dir is not None:
            image["path"] = f"/{image_dir}/{file_name}"

        return image

    return {
        "images": [_image(i + 1) for i in range(count)],
        "categories": [
            {"id": CLASS_IDS[i], "name": CLASS_NAMES[i]}
            for i in range(NUM_CLASSES)
        ],
        "annotations": [
            {
                "id": (i + 1),
                "image_id": (i // NUM_CLASSES) + 1,
                "category_id": CLASS_IDS[(i % NUM_CLASSES)],
                "bbox": BBOX
            }
            for i in range(NUM_CLASSES * count)
        ],
    }

@pytest.fixture
def raw_dataset(pipeline_paths: None) -> WriteRawDataset:
    """Returns a factory that writes a complete COCO dataset that satisfies every check."""
    # The format parameters mirror config_file's. valid_coco is the only builder until a
    # second format is added, at which point the dispatch goes here
    def _write(source_format: str = "coco",
               dataset_type: str = "object_detection",
               target_format: str = "yolo") -> Path:
        dataset_config = make_dataset_config_data(source_format, dataset_type, target_format)
        dataset_raw_dir = paths.RAW_DATA_DIR / DATASET_NAME

        annotations_dir = dataset_raw_dir / dataset_config.get("annotations_path", "annotations")
        annotations_dir.mkdir(parents=True, exist_ok=True)

        for filenames in dataset_config["splits"].values():
            for filename in filenames:
                coco = valid_coco(Path(filename).stem, IMAGES_PER_ANNOTATION_FILE, dataset_config.get("path"))
                (annotations_dir / filename).write_text(json.dumps(coco))

                for image in coco["images"]:
                    # _image always sets 'file_name', so the helper never returns None here
                    image_relpath = cast(Path, coco_ops.resolve_image_relpath(image))
                    image_path = dataset_raw_dir / image_relpath
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(IMAGE_BYTES)

        return dataset_raw_dir

    return _write

@pytest.fixture
def config_file(pipeline_paths: None) -> WriteConfigFile:
    """Returns a factory that writes a complete dataset configuration file that satisfies every check."""
    def _write(source_format: str = "coco",
               dataset_type: str = "object_detection",
               target_format: str = "yolo") -> Path:
        config_path = paths.CONFIG_DIR / CONFIG_FILENAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(make_config_data(source_format, dataset_type, target_format)))

        return config_path

    return _write

def cli_argv(config_path: Path, *extra: str) -> list[str]:
    """Returns the argv list for a process_dataset command line invocation."""
    return ["process_dataset.py",
            "--dataset", DATASET_NAME,
            "--config", str(config_path),
            "--format", TARGET_FORMAT,
            *extra]

# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #
def test_project_paths_are_isolated() -> None:
    """Requests no fixture: pipeline_paths is autouse, so isolation must hold regardless."""
    for project_dir in (paths.EXTERNAL_DATA_DIR, paths.RAW_DATA_DIR, paths.PROCESSED_DATA_DIR,
                        paths.CONFIG_DIR, paths.LOGS_DIR):
        assert not project_dir.is_relative_to(paths.PROJECT_ROOT)

    validator_log_path = cast(tuple[Path, ...], BaseDatasetValidator.__init__.__defaults__)[0]
    assert not validator_log_path.is_relative_to(paths.PROJECT_ROOT)

# --------------------------------------------------------------------------- #
# get_pipeline_components
# --------------------------------------------------------------------------- #
SUPPORTED_PIPELINES = [
    pytest.param("coco", "object_detection", "yolo", (CocoFormatHandler, YoloWriter), id="coco-detection-yolo"),
]

# One case per dimension the factory checks
UNSUPPORTED_PIPELINES = [
    pytest.param("unsupported", "object_detection", "yolo", id="source-format"),
    pytest.param("coco", "unsupported", "yolo", id="dataset-type"),
    pytest.param("coco", "object_detection", "unsupported", id="target-format"),
]

@pytest.mark.parametrize("source_format, dataset_type, target_format, expected", SUPPORTED_PIPELINES)
def test_get_pipeline_components_returns_coco_yolo_pair(tmp_path: Path,
                                                        source_format: str,
                                                        dataset_type: str,
                                                        target_format: str,
                                                        expected: PipelineComponents) -> None:
    cfg = make_config(source_format, dataset_type, target_format)
    handler, writer = get_pipeline_components(cfg, target_format, tmp_path, tmp_path)

    assert (type(handler), type(writer)) == expected

@pytest.mark.parametrize("source_format, dataset_type, target_format", UNSUPPORTED_PIPELINES)
def test_get_pipeline_components_unsupported_combination_raises(tmp_path: Path,
                                                                source_format: str,
                                                                dataset_type: str,
                                                                target_format: str) -> None:
    cfg = make_config(source_format, dataset_type, target_format)

    with pytest.raises(NotImplementedError,
                       match=f"Conversion from '{cfg.format}' to '{target_format}' for {cfg.type} "
                             "is currently not supported."):
        get_pipeline_components(cfg, target_format, tmp_path, tmp_path)

# --------------------------------------------------------------------------- #
# get_validator
# --------------------------------------------------------------------------- #
SUPPORTED_VALIDATORS = [
    pytest.param("coco", "object_detection", CocoDatasetValidator, id="coco-detection"),
]

# One case per dimension the factory checks
UNSUPPORTED_VALIDATORS = [
    pytest.param("unsupported", "object_detection", id="source-format"),
    pytest.param("coco", "unsupported", id="dataset-type"),
]

@pytest.mark.parametrize("source_format, dataset_type, expected", SUPPORTED_VALIDATORS)
def test_get_validator_returns_coco_validator(tmp_path: Path,
                                              source_format: str,
                                              dataset_type: str,
                                              expected: type[BaseDatasetValidator]) -> None:
    cfg = make_config(source_format, dataset_type)

    assert type(get_validator(source_format, dataset_type, tmp_path, cfg)) is expected

@pytest.mark.parametrize("source_format, dataset_type", UNSUPPORTED_VALIDATORS)
def test_get_validator_unsupported_raises(tmp_path: Path,
                                          source_format: str,
                                          dataset_type: str) -> None:
    cfg = make_config(source_format, dataset_type)

    with pytest.raises(NotImplementedError, match="is currently not supported."):
        get_validator(source_format, dataset_type, tmp_path, cfg)

# --------------------------------------------------------------------------- #
# process_dataset
# --------------------------------------------------------------------------- #
def test_process_dataset_writes_images_and_labels(config_file: WriteConfigFile,
                                                  raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

    processed_dir = paths.PROCESSED_DATA_DIR / DATASET_NAME

    for split, filenames in make_dataset_config_data()["splits"].items():
        images = sorted(p.name for p in (processed_dir / "images" / split).iterdir())
        labels = sorted(p.name for p in (processed_dir / "labels" / split).iterdir())

        assert len(images) == len(filenames) * IMAGES_PER_ANNOTATION_FILE
        assert [Path(name).stem for name in labels] == [Path(name).stem for name in images]

def test_process_dataset_writes_label_content(config_file: WriteConfigFile,
                                              raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

    labels_dir = paths.PROCESSED_DATA_DIR / DATASET_NAME / "labels" / "train"
    label_lines = sorted(labels_dir.iterdir())[0].read_text().splitlines()

    # valid_coco annotates every image once per class, all with the same box
    assert label_lines == [f"{yolo_id} {NORMALISED_BBOX}" for yolo_id in range(NUM_CLASSES)]

def test_process_dataset_writes_dataset_yaml(config_file: WriteConfigFile,
                                             raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

    processed_dir = paths.PROCESSED_DATA_DIR / DATASET_NAME
    processed_config_path = processed_dir / "dataset.yaml"
    assert processed_config_path.is_file()

    config = file_ops.load_yaml_config(processed_config_path)
    assert config == {
        "path": str(processed_dir.resolve()),
        "names": dict(enumerate(CLASS_NAMES)),
        "train": "images/train",
        "val": "images/val",
    }

def test_process_dataset_unknown_dataset_raises(config_file: WriteConfigFile) -> None:
    config_path = config_file()

    with pytest.raises(ValueError, match="No such dataset found in the configuration"):
        process_dataset("unknown", config_path, TARGET_FORMAT)

def test_process_dataset_invalid_config_raises(raw_dataset: WriteRawDataset) -> None:
    raw_dataset()

    config_data = make_config_data()

    # 'classes' is required by DatasetConfigModel, so dropping it makes the model reject the config
    config_data["datasets"][DATASET_NAME].pop("classes")

    config_path = paths.CONFIG_DIR / CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config_data))

    with pytest.raises(ValueError, match=f"Invalid configuration for {DATASET_NAME}"):
        process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

def test_process_dataset_missing_raw_dir_raises(config_file: WriteConfigFile) -> None:
    config_path = config_file()

    with pytest.raises(FileNotFoundError, match="Raw data directory"):
        process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

def test_process_dataset_validation_failure_raises(config_file: WriteConfigFile,
                                                   raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    dataset_raw_dir = raw_dataset()

    # Delete an image from the raw dataset to introduce a validation failure
    image_path = next(iter(dataset_raw_dir.rglob("*.jpg")))
    image_path.unlink()

    with pytest.raises(DatasetValidationError, match=f"Validation failed for dataset {DATASET_NAME}"):
        process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

def test_process_dataset_existing_output_raises(config_file: WriteConfigFile,
                                                raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

    with pytest.raises(FileExistsError, match="Processed dataset already exists"):
        process_dataset(DATASET_NAME, config_path, TARGET_FORMAT, overwrite=False)

def test_process_dataset_overwrite_replaces_existing_output(config_file: WriteConfigFile,
                                                            raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    process_dataset(DATASET_NAME, config_path, TARGET_FORMAT)

    processed_dir = paths.PROCESSED_DATA_DIR / DATASET_NAME

    # Dummy file sentinel: proves the output directory was removed, not written over in place
    dummy_path = processed_dir / "dummy.txt"
    dummy_path.write_text("dummy")

    process_dataset(DATASET_NAME, config_path, TARGET_FORMAT, overwrite=True)

    assert not dummy_path.exists()
    assert (processed_dir / "dataset.yaml").is_file()

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def test_main_processes_dataset(monkeypatch: pytest.MonkeyPatch,
                                config_file: WriteConfigFile,
                                raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    monkeypatch.setattr(sys, "argv", cli_argv(config_path))
    main()

    assert (paths.PROCESSED_DATA_DIR / DATASET_NAME / "dataset.yaml").is_file()

def test_main_passes_overwrite_flag(monkeypatch: pytest.MonkeyPatch,
                                    config_file: WriteConfigFile,
                                    raw_dataset: WriteRawDataset) -> None:
    config_path = config_file()
    raw_dataset()

    monkeypatch.setattr(sys, "argv", cli_argv(config_path))
    main()

    # Dummy file sentinel: proves the output directory was removed, not written over in place
    dummy_path = paths.PROCESSED_DATA_DIR / DATASET_NAME / "dummy.txt"
    dummy_path.write_text("dummy")

    monkeypatch.setattr(sys, "argv", cli_argv(config_path, "--overwrite"))
    main()

    assert not dummy_path.exists()

def test_main_reraises_pipeline_failure(monkeypatch: pytest.MonkeyPatch,
                                        config_file: WriteConfigFile) -> None:
    # No raw dataset is written, so process_dataset fails and main() must not suppress the error
    config_path = config_file()
    monkeypatch.setattr(sys, "argv", cli_argv(config_path))

    with pytest.raises(FileNotFoundError, match="Raw data directory"):
        main()

def test_main_requires_dataset_argument(monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["process_dataset.py"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR
    assert "--dataset" in capsys.readouterr().err

def test_main_rejects_unknown_format(monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["process_dataset.py",
                                      "--dataset", DATASET_NAME,
                                      "--format", "voc"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR
    assert "--format" in capsys.readouterr().err
