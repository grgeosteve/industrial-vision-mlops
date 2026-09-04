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
        self.dataset_raw_dir = dataset_raw_dir
        self.converter = converter
        self.dataset_config = dataset_config

    @abstractmethod
    def get_available_splits(self) -> list[str]:
        """Discovers and returns the names of the splits to be processed."""
        pass

    @abstractmethod
    def process_split(self, split_name: str) -> Iterator[tuple[str, str | None]]:
        """Loads and converts one split, yielding image path and label content pairs."""
        pass
