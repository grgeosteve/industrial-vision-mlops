import logging
from collections.abc import Iterator

from pydantic import ValidationError

from src.datatypes import CocoDocument
from src.handlers.base_handler import BaseFormatHandler
from src.utils import file_ops

logger = logging.getLogger(__name__)

class CocoFormatHandler(BaseFormatHandler):
    """Handles file discovery and loading for COCO format datasets."""

    def get_available_splits(self) -> list[str]:
        """Returns a list of available splits in the dataset.

        Returns:
            list[str]: A list of strings representing the available splits.

        Raises:
            ValueError: If the configured split names are not recognised.
            ValueError: If a mandatory split is missing.
        """
        split_config = self.dataset_config.splits
        all_allowed_splits = {'train', 'val', 'test'}

        extracted_splits = list(split_config.keys())
        extracted_set = set(extracted_splits)

        # 1. Check for invalid split names
        if not extracted_set.issubset(all_allowed_splits):
            invalid = extracted_set.difference(all_allowed_splits)
            raise ValueError(f"Invalid splits found: {invalid}. Allowed splits are: {all_allowed_splits}")

        # 2. Check that train split is present
        if 'train' not in extracted_set:
            raise ValueError("Missing mandatory split: 'train' must be defined.")

        return extracted_splits

    def process_split(self, split_name: str) -> Iterator[tuple[str, str | None]]:
        """Processes and converts a single split into the target format.

        Args:
            split_name (str): The name of the split to be converted.

        Yields:
            tuple[str, str | None]: A tuple containing the path to the image and its corresponding label if it exists.

        Raises:
            ValueError: If the split does not appear in the dataset configuration file.
            ValueError: If annotation file is not a JSON file or it doesn't exist.
            ValueError: If the annotation file is not a valid COCO document.
            ValueError: If a non-test split has no annotations.
        """
        split_config = self.dataset_config.splits
        configs = split_config.get(split_name)
        if not configs:
            raise ValueError(f"Split {split_name} does not exist in dataset configuration")

        if isinstance(configs, str):
            configs = [configs]

        # Get annotation dir
        anno_dir_name = (self.dataset_config.model_extra or {}).get("annotations_path", "annotations")
        anno_dir = self.dataset_raw_dir / anno_dir_name

        # Iterate over config files
        for config in configs:
            config_path = anno_dir / config
            if not config.endswith(".json"):
                raise ValueError("Only JSON annotations are supported for COCO format.")

            if not config_path.exists():
                raise ValueError(f"Annotation file {config_path} does not exist")

            # Get annotation data
            anno_data = file_ops.load_json_config(config_path)

            # Annotation data structural validation
            try:
                document = CocoDocument.model_validate(anno_data)
            except ValidationError as e:
                raise ValueError(f"Annotation file {config_path} is not a valid COCO JSON.") from e

            if not document.annotations and split_name == "test":
                logger.info(f"Test set detected in {config_path}. No annotations found. Proceeding only with images.")
            elif not document.annotations:
                raise ValueError(f"Invalid '{split_name}' COCO JSON: No 'annotations' were found.")

            yield from self.converter.convert(anno_data)
