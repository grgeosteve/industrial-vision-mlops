import itertools
import logging
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.datatypes import CocoDocument
from src.utils import coco_ops, file_ops
from src.validators.base_validator import BaseDatasetValidator

logger = logging.getLogger(__name__)

SplitAnnotations = dict[str, dict[Path, dict[str, Any]]]


class CocoDatasetValidator(BaseDatasetValidator):
    """Validates a COCO format dataset before processing."""

    def _get_annotation_paths_by_split(self) -> dict[str, list[Path]]:
        """
        Resolves annotation file paths for each split from the dataset config.
        Returns a dict of split_name -> list of absolute annotation file Paths.
        """
        anno_dir_name = (self.dataset_config.model_extra or {}).get("annotations_path", "annotations")
        anno_dir = self.dataset_raw_dir / anno_dir_name

        split_config = self.dataset_config.splits
        result = {}

        for split_name, fnames in split_config.items():
            filenames = fnames
            if isinstance(fnames, str):
                filenames = [fnames]
            result[split_name] = [anno_dir / fname for fname in filenames]

        return result

    def _run_checks(self) -> list[str]:
        logger.info(f"Running checks for {self.dataset_config.name} dataset...")

        split_annotations, load_errors = self._load_annotations_by_split()
        if load_errors:
            logger.warning(f"Aborting checks: {len(load_errors)} annotation file(s) failed to load.")
            return load_errors

        errors = []
        errors.extend(self._check_annotation_structure(split_annotations))
        errors.extend(self._check_class_consistency(split_annotations))
        errors.extend(self._check_class_mapping_coverage(split_annotations))
        errors.extend(self._check_annotation_completeness(split_annotations))
        errors.extend(self._check_image_dimensions(split_annotations))
        errors.extend(self._check_split_filename_uniqueness(split_annotations))
        errors.extend(self._check_missing_images(split_annotations))
        errors.extend(self._check_bbox_validity(split_annotations))
        errors.extend(self._check_data_leakage(split_annotations))

        logger.info(f"All checks completed with {len(errors)} errors.")

        return errors

    def _load_annotations_by_split(self) -> tuple[SplitAnnotations, list[str]]:
        """Load every annotation file exactly once. Returns the parsed data per split
        and any load errors; a non-empty error list aborts validation before the checks."""
        error_prefix = self._check_name()
        errors = []
        split_annotations: SplitAnnotations = {}

        for split, annotation_filepaths in self._get_annotation_paths_by_split().items():
            split_annotations[split] = {}
            for anno_fpath in annotation_filepaths:
                logger.info(f"{error_prefix}: Loading {split} annotation file {anno_fpath}...")
                try:
                    anno_data = file_ops.load_json_config(anno_fpath)
                    if not isinstance(anno_data, dict):
                        errors.append(f"{error_prefix}: Annotation file {anno_fpath} loaded, but returned a {type(anno_data).__name__}. Expected a dict.")
                    else:
                        _ = CocoDocument(**anno_data)
                        split_annotations[split][anno_fpath] = anno_data
                except ValidationError as e:
                    errors.append(f"{error_prefix}: JSON file: {anno_fpath} is not a valid COCO annotation file: {e}")
                except Exception as e:
                    errors.append(f"{error_prefix}: Failed to load annotation file: {anno_fpath}: {e}")

        return split_annotations, errors

    def _extract_config_class_mapping(self) -> dict[str, int]:
        """Extract the class mapping from the dataset config file."""

        classes = self.dataset_config.classes
        class_mapping = {}
        for class_name, ids in classes.items():
            coco_id = ids.get('coco_id')
            if coco_id is None:
                # Checking the config is not the scope of this validator
                continue

            class_mapping[class_name] = coco_id

        return class_mapping


    def _check_annotation_structure(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that each annotation file has non-empty images and categories.
        Non-empty annotations are required for train and val splits only -
        test splits may have no annotations key or an empty annotations list."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking annotation structure...")

        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                # Check 'images' field
                images = data.get('images')
                if images is None:
                    errors.append(f"{error_prefix}: Invalid format for COCO format annotation file: {anno_fpath}. Key 'images' is missing.")
                elif len(images) == 0:
                    errors.append(f"{error_prefix}: Invalid format for COCO format annotation file: {anno_fpath}. 'images' field is empty.")

                # Check 'categories' field
                categories = data.get('categories')
                if categories is None:
                    errors.append(f"{error_prefix}: Invalid format for COCO format annotation file: {anno_fpath}. Key 'categories' is missing.")
                elif len(categories) == 0:
                    errors.append(f"{error_prefix}: Invalid format for COCO format annotation file: {anno_fpath}. 'categories' field is empty.")

                # Check 'annotations' field
                if split != 'test':
                    annotations = data.get('annotations')
                    if annotations is None:
                        errors.append(f"{error_prefix}: Invalid format for COCO format annotation file: {anno_fpath}. {split} split contains no annotations.")
                    elif len(annotations) == 0:
                        errors.append(f"{error_prefix}: Invalid format for COCO format annotation file: {anno_fpath}. 'annotations' field for split {split} is empty.")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors

    def _check_class_consistency(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that all annotation files define identical class id/name pairs."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking class consistency between annotation files...")

        class_dict = {}
        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                categories = data.get('categories', [])
                if not categories:
                    continue # Ignore. This is beyond the scope of this validator

                classes = [(cat.get('id'), cat.get('name')) for cat in categories]
                if any(class_id is None or class_name is None for class_id, class_name in classes):
                    # Ignore. Malformed category entries are beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Malformed 'categories' entries in {anno_fpath}. Skipping class comparison for this file.")
                    continue

                class_dict[anno_fpath] = sorted(classes, key=lambda c: (str(c[0]), str(c[1])))

        if class_dict:
            comparison_classes = next(iter(class_dict.values()))
            for anno_fpath, classes in class_dict.items():
                if classes != comparison_classes:
                    errors.append(f"{error_prefix}: {anno_fpath} has different classes than {comparison_classes}")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors

    def _check_class_mapping_coverage(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check strict full class coverage between class config mapping and categories present in the annotation files."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking class mapping coverage...")

        class_config_mapping = self._extract_config_class_mapping()

        class_dict = {}
        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                categories = data.get('categories', [])
                if not categories:
                    continue # Ignore. This is beyond the scope of this validator

                classes = [(cat.get('id'), cat.get('name')) for cat in categories]
                if any(class_id is None or class_name is None for class_id, class_name in classes):
                    # Ignore. Malformed category entries are beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Malformed 'categories' entries in {anno_fpath}. Skipping class comparison for this file.")
                    continue

                for class_id, class_name in classes:
                    class_dict[class_name] = class_id

        # Check coverage
        if class_dict != class_config_mapping:
            config_class_set = set(class_config_mapping.keys())
            class_set = set(class_dict.keys())

            difference_set = config_class_set.symmetric_difference(class_set)
            if difference_set:
                errors.append(f"{error_prefix}: Class coverage mismatch. Difference between config and raw annotation classes: {difference_set}.")
            else:
                id_mismatches = {cls: (class_config_mapping[cls], class_dict[cls]) for cls in class_dict.keys() if class_config_mapping[cls] != class_dict[cls]}
                for cls, (config_id, anno_id) in id_mismatches.items():
                    errors.append(f"{error_prefix}: Class ID mismatch between config and annotations: {cls}: {config_id} (config) vs {anno_id} (annotations)")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors


    def _check_annotation_completeness(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check unique image IDs, unique annotation IDs, unique image paths,
        all images annotated, and no orphaned annotation references."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking annotation completeness...")

        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                images = data.get('images', [])
                if not images:
                    # This is not handled as an error. This check is beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Invalid format for COCO format annotation file {anno_fpath}. No images found.")

                # Check that image ids and image filenames / paths are unique
                image_ids = set()
                image_paths = set()
                duplicate_ids = []
                duplicate_paths = []
                for image in images:
                    image_id = image.get('id')
                    if image_id is None or image_id == '':
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. image_id is missing.")
                        continue

                    if image_id in image_ids:
                        duplicate_ids.append(image_id)
                    else:
                        image_ids.add(image_id)

                    rel_path = coco_ops.resolve_image_relpath(image)
                    if rel_path is None:
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. Image path for image_id {image_id} is missing.")
                        continue
                    path = rel_path.as_posix()

                    if path in image_paths:
                        duplicate_paths.append(path)
                    else:
                        image_paths.add(path)


                # Build a set of annotation ids and a list of image ids referenced in the annotations
                annotations = data.get('annotations', [])
                if len(annotations) == 0 and split != 'test':
                    errors.append(f"{error_prefix}: No annotations found in annotation file {anno_fpath}")

                # Check that annotation ids are unique
                annotated_image_ids = []
                annotation_ids = set()
                duplicate_annotation_ids = []

                for anno in annotations:
                    annotation_id = anno.get('id')
                    if annotation_id is None or annotation_id == '':
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. Annotation 'id' is missing.")
                        continue

                    if annotation_id in annotation_ids:
                        duplicate_annotation_ids.append(annotation_id)
                    else:
                        annotation_ids.add(annotation_id)

                    annotation_image_id = anno.get('image_id')
                    if annotation_image_id is None or annotation_image_id == '':
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. image_id for annotation 'id' {annotation_id} is missing.")
                    else:
                        annotated_image_ids.append(annotation_image_id)

                # Check for unannotated images, orphaned annotation references
                annotated_image_set = set(annotated_image_ids)

                all_images_annotated = image_ids.issubset(annotated_image_set)
                all_annotated_images_referenced = annotated_image_set.issubset(image_ids)

                # Final report for the annotation file
                # Duplicate ids
                if len(duplicate_ids) > 0:
                    errors.append(f"{error_prefix}: DUPLICATE IDS: {len(duplicate_ids)} duplicate ids found in {anno_fpath}: {duplicate_ids}")

                # Duplicate paths
                if len(duplicate_paths) > 0:
                    errors.append(f"{error_prefix}: DUPLICATE IMAGE PATHS: {len(duplicate_paths)} duplicate paths found in {anno_fpath}: {duplicate_paths}")

                # Duplicate annotation ids
                if len(duplicate_annotation_ids) > 0:
                    errors.append(f"{error_prefix}: DUPLICATE ANNOTATION IDS: {len(duplicate_annotation_ids)} duplicate annotation ids found in {anno_fpath}: {duplicate_annotation_ids}")

                # Images not annotated
                if not all_images_annotated and split != 'test':
                    missing_image_ids = image_ids.difference(annotated_image_set)
                    errors.append(f"{error_prefix}: IMAGES NOT ANNOTATED: {len(missing_image_ids)} images in {anno_fpath} are not referenced by any annotations: {missing_image_ids}")

                # Orphaned references
                if not all_annotated_images_referenced:
                    orphaned_image_ids = annotated_image_set.difference(image_ids)
                    errors.append(f"{error_prefix}: ORPHANED IMAGE REFERENCES: {len(orphaned_image_ids)} images in {anno_fpath} are referenced by annotations but are not present in the 'images' list: {orphaned_image_ids}")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors

    def _check_image_dimensions(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that all image dimensions in the annotation files are valid

        Check that all image width and height are present, have integer values and are greater than 0"""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking image dimensions validity...")

        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                images = data.get('images', [])
                if not images:
                    # This is not handled as an error. This check is beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Invalid format for COCO format annotation file {anno_fpath}. No images found.")

                for image in images:
                    image_id = image.get('id')
                    if image_id is None or image_id == '':
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. image 'id' is missing.")
                        continue

                    # Check if width and height information exists and is valid
                    width = image.get('width')
                    height = image.get('height')
                    if width is None or height is None:
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. Image with 'id'={image_id} has no 'width' and/or 'height'")
                        continue

                    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (width, height)):
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. Image with 'id'={image_id} has non-integer values for 'width' or 'height'")
                        continue

                    if width <= 0 or height <= 0:
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. Image with 'id'={image_id} has non-positive width or height")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors

    def _check_split_filename_uniqueness(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that no two annotation files within the same split reference the same filename.
        Prevents silent image overwrites when multiple annotation files exist per split."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking filename uniqueness per split...")

        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            split_filename_dict: dict[str, dict[str, Any]] = {}

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                images = data.get('images', [])
                if not images:
                    # This is not handled as an error. This check is beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Invalid format for COCO format annotation file {anno_fpath}. No images found.")

                for image in images:
                    image_id = image.get('id')
                    if image_id is None or image_id == '':
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. image_id is missing.")
                        continue

                    rel_path = coco_ops.resolve_image_relpath(image)
                    if rel_path is None:
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. Image path for image_id {image_id} is missing.")
                        continue

                    filename = rel_path.name
                    path = rel_path.as_posix()
                    anno_fname = anno_fpath.name

                    if filename in split_filename_dict:
                        # Check for duplicate entries and conflicts
                        existing_entry = split_filename_dict[filename]
                        if {'annotation_file': anno_fname, 'image_id': image_id, 'image_path': path} == existing_entry:
                            logger.warning(f"{logger_prefix}: Warning: {anno_fpath}. Image filename {filename} for 'id' {image_id} is duplicated.")
                        else: # CONFLICT!
                            errors.append(f"{error_prefix}: Error in {anno_fpath}. Filename conflict between 'id': {image_id} and image entry in {existing_entry['annotation_file']}: 'id': {existing_entry['image_id']}")
                    else:
                        split_filename_dict[filename] = {'annotation_file': anno_fname, 'image_id': image_id, 'image_path': path}


        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors

    def _check_missing_images(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that every image path referenced in the annotation files exists on disk."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking for missing image files...")

        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")
                missing_images = []

                images = data.get('images', [])
                if not images:
                    # This is not handled as an error. This check is beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Invalid format for COCO format annotation file {anno_fpath}. No images found.")

                for image in images:
                    image_id = image.get('id')
                    if image_id is None or image_id == '':
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. image_id is missing.")
                        continue

                    rel_path = coco_ops.resolve_image_relpath(image)
                    if rel_path is None:
                        # Only a warning is sent to the terminal. This error is not in the scope of this validator
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. Image path for image_id {image_id} is missing.")
                        continue

                    path = (self.dataset_raw_dir / rel_path).resolve()

                    # Check if the image exists
                    if not path.is_file():
                        errors.append(f"{error_prefix}: Missing image error in {anno_fpath}. Image file {path} for image_id {image_id} does not exist.")
                        missing_images.append(path)

                if len(missing_images) > 0:
                    errors.append(f"{error_prefix}: Missing image summary for {anno_fpath}: {len(missing_images)} image files do not exist")
                    logger.warning(f"{logger_prefix}: {len(missing_images)} image files from {anno_fpath} do not exist. See log for details.")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors

    def _check_bbox_validity(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that no bounding box has zero or negative width or height.
        Scoped to detection format only - checks the bbox field exclusively.
        Segmentation keys are ignored and bbox inference from segmentation masks is out of scope."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking bounding box validity for dataset {self.dataset_config.name}...")

        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking bounding box validity for split {split}...")
            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking bounding box validity for annotation file {anno_fpath}...")

                annotations = data.get('annotations', [])
                # If annotations aren't there just skip - the scope of this check is only bbox validation
                if len(annotations) == 0:
                    continue

                for anno in annotations:
                    annotation_id = anno.get('id')
                    if annotation_id is None or annotation_id == '':
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. Annotation 'id' is missing.")
                        continue

                    bbox = anno.get('bbox')
                    if bbox is None:
                        if split != 'test':
                            errors.append(f"{error_prefix}: Error in {anno_fpath}. bbox for annotation 'id' {annotation_id} is missing.")
                        continue

                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. bbox for annotation 'id' {annotation_id} is malformed, expected [x, y, width, height]: {bbox}")
                        continue

                    if not all(isinstance(v, (int, float)) and not math.isnan(v) and not isinstance(v, bool) for v in bbox):
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. bbox for annotation 'id' {annotation_id} has non-numeric values: {bbox}")
                        continue

                    _, _, width, height = bbox
                    if width <= 0 or height <= 0:
                        errors.append(f"{error_prefix}: Error in {anno_fpath}. bbox for annotation 'id' {annotation_id} has non-positive width or height: {bbox}")

        if not errors:
            logger.info(f"{logger_prefix}: Bounding box validity check completed for dataset {self.dataset_config.name} without errors.")
        else:
            logger.info(f"{logger_prefix}: Bounding box validity check completed for dataset {self.dataset_config.name} with {len(errors)} errors.")

        return errors

    def _check_data_leakage(self, split_annotations: SplitAnnotations) -> list[str]:
        """Check that no image appears in more than one split."""

        error_prefix = self._check_name()
        logger_prefix = error_prefix
        errors = []

        logger.info(f"{logger_prefix}: Checking for data leakage (image overlap) between splits...")

        split_image_paths = {}
        for split, annotation_data in split_annotations.items():
            logger.info(f"{logger_prefix}: Checking split {split}...")

            image_paths = set()
            for anno_fpath, data in annotation_data.items():
                logger.info(f"{logger_prefix}: {split} | Checking annotation file {anno_fpath}...")

                images = data.get('images', [])
                if not images:
                    # This is not handled as an error. This check is beyond the scope of this validator
                    logger.warning(f"{logger_prefix}: Invalid format for COCO format annotation file {anno_fpath}. No images found.")

                for image in images:
                    image_id = image.get('id')
                    if image_id is None or image_id == '':
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. image_id is missing.")
                        continue

                    rel_path = coco_ops.resolve_image_relpath(image)
                    if rel_path is None:
                        # Skipping. This check is beyond the scope of this validator
                        logger.warning(f"{logger_prefix}: Error in {anno_fpath}. Image path for image_id {image_id} is missing. Skipping")
                        continue

                    path = (self.dataset_raw_dir / rel_path).resolve().as_posix()
                    image_paths.add(path)

            split_image_paths[split] = image_paths

        # Check for cross split leakage
        for split, other_split in itertools.combinations(split_image_paths, 2):
            overlap = split_image_paths[split].intersection(split_image_paths[other_split])
            if overlap:
                errors.append(f"{error_prefix}: Error: Data leakage detected between splits {split} and {other_split}. Overlapping images: {overlap}.")
                logger.warning(f"{logger_prefix}: FATAL ERROR: Data leakage detected between splits {split} and {other_split}. Number of overlapping images: {len(overlap)}.")

        if len(errors) == 0:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset without errors.")
        else:
            logger.info(f"{logger_prefix}: Completed validation of {self.dataset_config.name} dataset with {len(errors)} errors.")

        return errors
