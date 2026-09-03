from abc import ABC, abstractmethod
from pathlib import Path

from src.datatypes import ClassConfigMapping


class BaseDatasetWriter(ABC):
    """Abstract contract for all target format writers."""

    def __init__(self, output_dir: Path, class_mapping: ClassConfigMapping):
        self.output_dir = output_dir
        self.class_mapping = class_mapping

    @abstractmethod
    def setup_split(self, split_name: str) -> None:
        """Prepares the directory structure for a new split."""
        pass

    @abstractmethod
    def write_item(self,
                   source_img_path: Path,
                   target_filename: str,
                   label_content: str | None,
                   split_name: str) -> None:
        """Saves a single image and its corresponding annotation."""
        pass

    @abstractmethod
    def finalise(self, splits_present: list[str]) -> None:
        """Generates root configuration files or finalises metadata."""
        pass
