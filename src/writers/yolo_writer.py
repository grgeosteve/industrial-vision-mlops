import logging
import shutil
from pathlib import Path

import yaml

from src.datatypes import ClassConfigMapping
from src.writers.base_writer import BaseDatasetWriter

logger = logging.getLogger(__name__)

class YoloWriter(BaseDatasetWriter):
    """Writes converted image and label pairs into a YOLO dataset directory structure."""

    def __init__(self, output_dir: Path, class_mapping: ClassConfigMapping) -> None:
        """Prepares the writer for a YOLO dataset at the given output directory.

        Args:
            output_dir (Path): Root directory for the processed YOLO dataset.
            class_mapping (ClassConfigMapping): Class names mapped to their COCO and YOLO ids.

        Raises:
            ValueError: If a class does not have a YOLO id.
        """
        super().__init__(output_dir, class_mapping)

        self.classes = self._extract_classes()

    def setup_split(self, split_name: str) -> None:
        """Prepares the directory structure for a new split.

        Args:
            split_name (str): The name of the new split.

        Raises:
            ValueError: If the split name is invalid.
            RuntimeError: If the directory structure cannot be created.
        """
        if split_name not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split name '{split_name}'. Only 'train', 'val', 'test' are allowed.")
        try:
            image_dir = self.output_dir / 'images' / split_name
            image_dir.mkdir(parents=True, exist_ok=True)

            if split_name != 'test':
                labels_dir = self.output_dir / 'labels' / split_name
                labels_dir.mkdir(parents=True, exist_ok=True)

        except OSError as e:
            logger.error(f"Filesystem error while initialising split '{split_name}' at {self.output_dir}: {e}")
            raise RuntimeError(f"Critical failure: Could not create directory structure: {e}") from e

    def write_item(self,
                   source_img_path: Path,
                   target_filename: str,
                   label_content: str | None,
                   split_name: str) -> None:
        """Writes a single image and its corresponding label, if it exists, to the split directory.

        Ensures that if the label write fails, the image is not left orphaned.

        Args:
            source_img_path (Path): The path to the image file.
            target_filename (str): Output filename for the image. The label reuses it with a '.txt' extension.
            label_content (str | None): The content of the label file,
                                        or None if no annotation is available for this image.
            split_name (str): The name of the split the item belongs to. Labels are not written for 'test'.

        Raises:
            FileNotFoundError: If the source file does not exist.
            PermissionError: If user doesn't have sufficient access permissions for the target file.
            FileExistsError: If the target file already exists.
            RuntimeError: If there is a disk failure or permission error during target file write.
        """
        img_target = self.output_dir / 'images' / split_name / target_filename
        label_target = self.output_dir / 'labels' / split_name / Path(target_filename).with_suffix('.txt')

        try:
            if not source_img_path.exists():
                raise FileNotFoundError(f"Source image missing: {source_img_path}")

            if img_target.exists():
                raise FileExistsError(f"Target image already exists: {img_target}.")
            if label_target.exists():
                raise FileExistsError(f"Target label file already exists: {label_target}")

            # Copy image first
            shutil.copy2(source_img_path, img_target)

            if split_name != 'test' and label_content:
                try:
                    with open(label_target, 'w', encoding='utf-8') as f:
                        f.write(label_content)
                except OSError as e:
                    logger.error(f"Label write failed for {target_filename}, removing orphaned image: {e}")
                    img_target.unlink(missing_ok=True)
                    raise

        except FileExistsError:
            raise
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Access error processing {target_filename}: {e}")
            raise
        except OSError as e:
            logger.error(f"System error writing item {target_filename} to {split_name}: {e}")
            raise RuntimeError(f"Disk failure or permission issue during write: {e}") from e

    def _extract_classes(self) -> dict[int, str]:
        """Builds the YOLO class id to class name mapping from the dataset class mapping.

        Returns:
            dict[int, str]: A dictionary mapping class IDs to class names.

        Raises:
            ValueError: If a class does not have a YOLO id.
        """
        # Extract the mapping into a list of tuples
        extracted = []
        for name, config in self.class_mapping.items():
            yolo_id = config.get('yolo_id')
            if yolo_id is None:
                raise ValueError(f"YOLO ID missing for class '{name}'")

            extracted.append((yolo_id, name))

        # Sort explicitly by the yolo_id
        sorted_mapping = sorted(extracted, key=lambda x: x[0])

        return {yolo_id: name for yolo_id, name in sorted_mapping}

    def finalise(self, splits_present: list[str]) -> None:
        """Generates the dataset.yaml configuration file for the YOLO dataset.

        Validates that every declared split exists on disk and that a train split is present.

        Args:
            splits_present (list[str]): A list of split names present in the dataset.

        Raises:
            RuntimeError: If no splits were processed.
            RuntimeError: If a split is missing or empty on disk.
            RuntimeError: If the train split of the dataset is not present.
            RuntimeError: If the configuration file cannot be written.
        """
        if not splits_present:
            logger.error("Finalise called but no splits were processed. Dataset is empty")
            raise RuntimeError("Cannot finalise an empty dataset.")

        # Strict physical verification
        for split in splits_present:
            image_dir = self.output_dir / 'images' / split

            # If a split was requested but doesn't exist or is empty finalisation fails
            if not (image_dir.exists() and any(image_dir.iterdir())):
                logger.error(
                    f"Consistency Error: Split '{split}' was requested by pipeline but is missing/empty on disk.")
                raise RuntimeError(f"Dataset integrity check failed: Split '{split}' is missing.")

        # Critical requirement: Training set must always exist
        if 'train' not in splits_present:
            raise RuntimeError("Dataset integrity check failed: No 'train' split provided. Training is not possible.")

        # Construct the YOLO config dictionary
        dataset_config = {
            'path': str(self.output_dir.resolve()),
            'names': self.classes,
        }
        for split in ('train', 'val', 'test'):
            if split in splits_present:
                dataset_config[split] = f"images/{split}"

        # Write to disk
        yolo_config_path = self.output_dir / 'dataset.yaml'
        try:
            with open(yolo_config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(dataset_config, f, default_flow_style=False)
            logger.info(f"Successfully wrote YOLO config to {yolo_config_path}")
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Failed to write dataset.yaml: {e}")
            raise RuntimeError(f"Critical failure during dataset finalisation: {e}") from e
