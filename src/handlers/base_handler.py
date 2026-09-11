"""Abstract format handler contract for discovering and loading raw dataset splits.

Concrete handlers must subclass BaseFormatHandler, own a BaseAnnotationConverter, and yield
image path and label content pairs one split at a time.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from src.converters.base_converter import BaseAnnotationConverter
from src.datatypes import DatasetConfigModel


class BaseFormatHandler(ABC):
    """Abstract contract for all source format file handlers."""

    def __init__(self,
                 dataset_raw_dir: Path,
                 dataset_config: DatasetConfigModel,
                 converter: BaseAnnotationConverter) -> None:
        """Binds the handler to one raw dataset and the converter for its target format.

        Args:
            dataset_raw_dir (Path): Root directory of the raw dataset.
            dataset_config (DatasetConfigModel): Validated dataset configuration.
            converter (BaseAnnotationConverter): Converter for the target annotation format.
        """
        self.dataset_raw_dir = dataset_raw_dir
        self.converter = converter
        self.dataset_config = dataset_config

    @abstractmethod
    def get_available_splits(self) -> list[str]:
        """Returns a list of declared splits in the dataset.

        Returns:
            list[str]: The split names declared for the dataset.
        """
        pass

    @abstractmethod
    def process_split(self, split_name: str) -> Iterator[tuple[str, str | None]]:
        """Loads and converts one split, yielding image path and label content pairs.

        Args:
            split_name (str): The name of the split to process.

        Yields:
            tuple[str, str | None]: Image path relative to the dataset root, and its label content,
                                    or None when the image has no annotations.
        """
        pass
