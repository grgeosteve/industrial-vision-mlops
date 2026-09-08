import argparse
import logging
import shutil
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from src import paths
from src.converters.coco_to_yolo import CocoToYoloDetectionConverter
from src.datatypes import DatasetConfigModel
from src.handlers.base_handler import BaseFormatHandler
from src.handlers.coco_handler import CocoFormatHandler
from src.utils import file_ops
from src.validators.base_validator import BaseDatasetValidator, DatasetValidationError
from src.validators.coco_validator import CocoDatasetValidator
from src.writers.base_writer import BaseDatasetWriter
from src.writers.yolo_writer import YoloWriter

logger = logging.getLogger(__name__)

def get_pipeline_components(dataset_config_model: DatasetConfigModel,
                            target_format: str,
                            dataset_raw_dir: Path,
                            output_dir: Path) -> tuple[BaseFormatHandler, BaseDatasetWriter]:
    """Factory for dataset format handler and dataset writer.

    Args:
        dataset_config_model (DatasetConfigModel): Dataset configuration pydantic model.
        target_format (str): Processed dataset format (currently only YOLO is supported).
        dataset_raw_dir (Path): Root directory of the raw dataset.
        output_dir (Path): Processed dataset directory.

    Returns:
        tuple[BaseFormatHandler, BaseDatasetWriter]: Tuple of dataset format handler and writer.

    Raises:
        NotImplementedError: If the source format, dataset type, and target format combination is not supported.
        NotImplementedError: If output format is not supported.
    """
    # 1. Resolve the handler / converter (Input side)
    if (dataset_config_model.format == 'coco'
            and dataset_config_model.type == 'object_detection'
            and target_format == 'yolo'):
        converter = CocoToYoloDetectionConverter(dataset_config_model.classes)
        handler = CocoFormatHandler(dataset_raw_dir, dataset_config_model, converter)
    else:
        raise NotImplementedError(f"Conversion from '{dataset_config_model.format}' to '{target_format}'"
                                  f" for {dataset_config_model.type} is currently not supported.")

    # 2. Resolve the writer (Output side). Unreachable while YOLO is the only target, but
    # a new handler branch above must not be able to leave 'writer' unbound.
    if target_format == 'yolo':
        writer = YoloWriter(output_dir, dataset_config_model.classes)
    else:
        raise NotImplementedError(f"Output format {target_format} not supported.")

    return handler, writer

def get_validator(source_format: str,
                  dataset_type: str,
                  dataset_raw_dir: Path,
                  dataset_config: DatasetConfigModel) -> BaseDatasetValidator:
    """Factory for the appropriate validator for a given source format.

    Args:
        source_format (str): Raw dataset format (currently only COCO is supported).
        dataset_type (str): Dataset task type (currently only 'object_detection' is supported).
        dataset_raw_dir (Path): Root directory of the raw dataset.
        dataset_config (DatasetConfigModel): Dataset configuration pydantic model.

    Returns:
        BaseDatasetValidator: Validator for the given source format.

    Raises:
        NotImplementedError: If the source format or dataset type is not supported.
    """
    if source_format == 'coco' and dataset_type == 'object_detection':
        return CocoDatasetValidator(dataset_raw_dir, dataset_config)
    else:
        raise NotImplementedError(f"Input format {source_format} for the "
                                  f"task of {dataset_type} is currently not supported.")

def process_dataset(dataset_name: str, config_path: Path, target_format: str, overwrite: bool = False) -> None:
    """Processes the dataset by routing it through the appropriate format handler.

    Args:
        dataset_name (str): The name of the dataset to be processed.
        config_path (Path): The path to the configuration file.
        target_format (str): Processed dataset format (currently only YOLO is supported).
        overwrite (bool): Overwrite existing processed data (default: False).

    Raises:
        ValueError: If the dataset is not present in the configuration file.
        ValueError: If the dataset configuration is invalid.
        FileNotFoundError: If the raw data directory or a source image does not exist.
        PermissionError: If a source image or an output file cannot be accessed.
        NotImplementedError: If the dataset format, type, and target format combination is not supported.
        FileExistsError: If the processed dataset already exists and overwrite is not set.
        DatasetValidationError: If the dataset fails pre-flight validation.
        RuntimeError: If the writer cannot create, populate, or finalise the output dataset.
    """
    logger.info(f"Initiating processing pipeline for dataset: {dataset_name}")

    # 1. Load config
    full_config = file_ops.load_yaml_config(config_path)
    if 'datasets' not in full_config or dataset_name not in full_config['datasets']:
        raise ValueError(f"No such dataset found in the configuration: {dataset_name}")

    # 2. Get the dataset config and validate it
    dataset_config = full_config['datasets'][dataset_name]

    try:
        dataset_config_model = DatasetConfigModel(**dataset_config)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for {dataset_name}") from e

    # 3. Verify the raw data and validate the dataset
    dataset_raw_dir = paths.RAW_DATA_DIR / dataset_name
    if not dataset_raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory {dataset_raw_dir} for dataset {dataset_name} doesn't exist.")

    validator = get_validator(dataset_config_model.format,
                              dataset_config_model.type,
                              dataset_raw_dir,
                              dataset_config_model)
    try:
        validator.validate()
    except DatasetValidationError as e:
        raise DatasetValidationError(f"Validation failed for dataset {dataset_name}") from e

    # 4. Get the format handler and writer
    handler, writer = get_pipeline_components(dataset_config_model,
                                              target_format,
                                              dataset_raw_dir,
                                              paths.PROCESSED_DATA_DIR / dataset_name)

    # 5. Refuse to write over an existing dataset unless replacement is requested
    if writer.output_dir.exists() and any(writer.output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Processed dataset already exists at {writer.output_dir}. "
                                  f"Re-run with --overwrite to replace it.")
        logger.warning(f"Removing existing processed dataset at {writer.output_dir}")
        shutil.rmtree(writer.output_dir)

    # 6. Convert the data
    splits = handler.get_available_splits()
    for split in splits:
        logger.info(f"Processing split: {split}")

        writer.setup_split(split)

        count = 0
        for img_path, label_content in tqdm(handler.process_split(split),
                                            desc=f"Converting {split}",
                                            unit=" img",
                                            disable=None):
            writer.write_item(source_img_path=dataset_raw_dir / img_path,
                              target_filename=Path(img_path).name,
                              label_content=label_content,
                              split_name=split)

            count += 1

        logger.info(f"Split {split}: {count} images processed.")

    # 7. Finalise the dataset
    writer.finalise(splits)
    logger.info(f"Dataset processing for dataset {dataset_name} in {target_format} format complete.")
    logger.info(f"Processed data is located at: {writer.output_dir}")

def main() -> None:
    """Parses the command line arguments and runs the dataset processing pipeline."""
    default_config_path = paths.CONFIG_DIR / "data_config.yaml"

    parser = argparse.ArgumentParser(description="Process a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to process")
    parser.add_argument("--config", type=Path, default=default_config_path,
                        help="Path to the data configuration YAML file")
    parser.add_argument("--format", type=str, choices=["yolo"], default="yolo", help="Format for the output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed data")
    args = parser.parse_args()

    try:
        process_dataset(args.dataset, args.config, args.format, args.overwrite)
        logger.info(f"Data processing completed successfully for dataset {args.dataset}.")
    except Exception:
        logger.exception(f"Pipeline failed while processing data for dataset {args.dataset}.")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
