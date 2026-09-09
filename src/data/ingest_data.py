import argparse
import logging
import shutil
import sys
from pathlib import Path

from src import paths
from src.utils import file_ops

logger = logging.getLogger(__name__)

def ingest_data(dataset_name: str, config_path: str | Path) -> None:
    """Ingest data based on the dataset name specified in the config."""

    config_path = Path(config_path).resolve()
    full_config = file_ops.load_yaml_config(config_path)
    if 'datasets' not in full_config or dataset_name not in full_config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

    dataset_config = full_config['datasets'][dataset_name]

    # Path resolution and directory setup
    external_data_dir =  paths.EXTERNAL_DATA_DIR / dataset_name
    dataset_raw_dir = paths.RAW_DATA_DIR / dataset_name

    external_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_raw_dir.mkdir(parents=True, exist_ok=True)

    # Extract the dataset
    archives = dataset_config.get("archives", {})
    if not archives:
        raise ValueError("No archives defined in the configuration.")

    for filename, url in archives.items():
        # Download the archive file
        archive_path = external_data_dir / filename
        file_ops.download_data(url, archive_path)

        # Extract data
        extract_target = dataset_raw_dir
        file_ops.extract_archive(archive_path, extract_target)

    # External metadata handling
    metadata_urls = dataset_config.get("metadata_urls")
    if metadata_urls:
        logger.info(f"Downloading external metadata for {dataset_name}...")
        for meta_name, meta_url in metadata_urls.items():
            meta_path = external_data_dir / meta_name
            file_ops.download_data(meta_url, meta_path)

    # Handle annotations if specified
    annotations = dataset_config.get("annotations", [])
    if annotations:
        logger.info(f"Collecting annotations for {dataset_name}...")
        annotations_raw_dir = dataset_raw_dir / "annotations"
        annotations_raw_dir.mkdir(parents=True, exist_ok=True)

        for ann_path in annotations:
            full_ann_path = external_data_dir / ann_path
            if not full_ann_path.exists():
                logger.warning(f"Annotation file not found: {full_ann_path}")
                continue

            # Copy annotation file to raw data directory
            dest_ann_path = annotations_raw_dir / full_ann_path.name
            if dest_ann_path.exists():
                logger.info(f"Annotation file already exists at {dest_ann_path}. Skipping copy.")
            else:
                try:
                    shutil.copy(full_ann_path, dest_ann_path)
                    logger.info(f"Copied annotation file to {dest_ann_path}")
                except Exception:
                    logger.exception("Error copying annotation file")
                    raise

def main() -> None:
    default_config_path = paths.CONFIG_DIR / "data_config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset to ingest")
    parser.add_argument("--config", type=Path, default=default_config_path,
                        help="Path to the data configuration YAML file")
    args = parser.parse_args()

    try:
        ingest_data(args.dataset, args.config)
        logger.info(f"Data ingestion completed successfully for dataset '{args.dataset}'.")
    except Exception:
        logger.exception(f"Pipeline failed while ingesting data for dataset '{args.dataset}'")
        sys.exit(1)

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
