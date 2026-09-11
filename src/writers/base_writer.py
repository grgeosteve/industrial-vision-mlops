"""Abstract dataset writer contract for writing datasets on disk.

Concrete writers must subclass BaseDatasetWriter and be called in the order:
setup_split, write_item once per image, then finalise.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.datatypes import ClassConfigMapping


class BaseDatasetWriter(ABC):
    """Abstract contract for all target format writers."""

    def __init__(self, output_dir: Path, class_mapping: ClassConfigMapping) -> None:
        """Prepares the writer with an output directory and the dataset class mapping.

        Args:
            output_dir (Path): Root directory for the processed dataset.
            class_mapping (ClassConfigMapping): Class names mapped to their source and target ids.
        """
        self.output_dir = output_dir
        self.class_mapping = class_mapping

    @abstractmethod
    def setup_split(self, split_name: str) -> None:
        """Prepares the directory structure for a new split.

        Args:
            split_name (str): The name of the new split.
        """
        pass

    @abstractmethod
    def write_item(self,
                   source_img_path: Path,
                   target_filename: str,
                   label_content: str | None,
                   split_name: str) -> None:
        """Saves a single image and its corresponding label, if it exists, to disk.

        Args:
            source_img_path (Path): The path to the image file.
            target_filename (str): Output filename for the image.
            label_content (str | None): The label content, or None if the image has no annotations.
            split_name (str): The name of the split the item belongs to.
        """
        pass

    @abstractmethod
    def finalise(self, splits_present: list[str]) -> None:
        """Generates root configuration files or finalises metadata.

        Args:
            splits_present (list[str]): The split names present in the dataset.
        """
        pass
