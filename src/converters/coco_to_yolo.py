import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from src.converters.base_converter import BaseAnnotationConverter
from src.datatypes import ClassConfigMapping
from src.utils.coco_ops import resolve_image_relpath

logger = logging.getLogger(__name__)

class CocoBBoxEntry(BaseModel):
    """Validates a single COCO bbox entry."""
    bbox: tuple[float, float, float, float]
    class_id: int
    image_height: int
    image_width: int

class ImageBBoxMapping(RootModel):
    """Validates the internal image_bbox_dict mapping."""
    root: dict[int, list[CocoBBoxEntry]]

class CocoToYoloDetectionConverter(BaseAnnotationConverter[dict[str, Any]]):
    """Converts COCO object detection dictionaries into normalised YOLO strings."""

    def __init__(self, class_mapping: ClassConfigMapping) -> None:
        """Builds the COCO to YOLO class id lookup for the dataset class mapping.

        Args:
            class_mapping (ClassConfigMapping): Class names mapped to their COCO and YOLO ids.

        Raises:
            KeyError: If any class is missing its 'coco_id' or 'yolo_id'.
            KeyError: If two classes declare the same 'coco_id'.
        """
        self.coco_to_yolo: dict[int, int] = {}
        for _, class_ids in class_mapping.items():
            coco_id = class_ids.get('coco_id')
            yolo_id = class_ids.get('yolo_id')

            if coco_id is None or yolo_id is None:
                raise KeyError("COCO and YOLO IDs are required in Class Mapping for COCO to Yolo conversion.")

            if coco_id in self.coco_to_yolo:
                raise KeyError(f"Class mapping contains multiple entries for COCO ID: {coco_id}.")

            self.coco_to_yolo[coco_id] = yolo_id

    def _index_images(self, raw_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """Index every image entry by id, resolving its path and dimensions.

        Args:
            raw_data (dict[str, Any]): Raw COCO annotation file data.

        Returns:
            dict[int, dict[str, Any]]: Image entries keyed by image id,
                                       each with its resolved path, height, and width.

        Raises:
            KeyError: If image entry in the COCO data is missing the image 'id' key.
            KeyError: If image entry in the COCO data is missing both 'path' and 'file_name' keys.
            KeyError: If image entry in the COCO data is missing 'height' or 'width' keys.
            ValueError: If an image entry has non-positive 'height' or 'width'.
        """
        image_dict: dict[int, dict[str, Any]] = {}
        images = raw_data.get('images', [])

        for image in images:
            # First check for image id. This is essential information for every COCO dataset.
            image_id = image.get('id')
            if image_id is None:
                raise KeyError("Image ID is missing from raw COCO data.")

            image_path = resolve_image_relpath(image)
            if image_path is None:
                raise KeyError(f"Image ID {image_id} is missing both 'path' and 'file_name' keys.")

            image_height = image.get('height')
            image_width = image.get('width')
            if image_height is None or image_width is None:
                raise KeyError(f"Image with ID {image_id} is missing 'height' or 'width' keys.")
            elif image_height <= 0 or image_width <= 0:
                raise ValueError(
                    f"Image with ID {image_id} has non-positive dimensions: {image_width}x{image_height}.")

            image_dict[image_id] = {'path': image_path.as_posix(), 'height': image_height, 'width': image_width}

        return image_dict

    def _index_annotations(self,
                           raw_data: dict[str, Any],
                           image_dict: dict[int, dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        """Group every usable annotation based on its image id.

        Args:
            raw_data (dict[str, Any]): Raw COCO annotation file data.
            image_dict (dict[int, dict[str, Any]]): Image entries keyed by image id,
                                                    each with path, height, and width information.

        Returns:
            dict[int, list[dict[str, Any]]]: Dictionary of annotation entries grouped by image id,
                                             each including bbox, class_id,
                                             and the parent image dimensions.

        Raises:
            KeyError: If 'image_id' key is missing in any entry in 'annotations'.
            KeyError: If 'category_id' key is missing in any entry in 'annotations'.
        """
        image_bbox_dict: dict[int, list[dict[str, Any]]] = {}
        annotations = raw_data.get('annotations', [])
        for anno in annotations:
            image_id = anno.get('image_id')
            if image_id is None:
                raise KeyError("Image ID not present in 'annotations'. This is required information in COCO datasets.")

            if image_id not in image_dict:
                logger.warning("Bounding box found for a non referenced image in 'annotations'. Skipping this entry.")
                continue

            class_id = anno.get('category_id')
            if class_id is None:
                raise KeyError(
                    "Category ID not present in 'annotations'. This is required information in COCO datasets.")

            # If bbox is not present for an annotation entry just continue
            # We assume that a detection dataset should contain bbox coordinates
            bbox = anno.get('bbox')
            if not bbox:
                logger.warning("No bounding box found for annotation entry. Skipping this entry.")
                continue

            if image_id not in image_bbox_dict:
                image_bbox_dict[image_id] = []

            image_bbox_dict[image_id].append({'bbox': bbox,
                                              'class_id': class_id,
                                              'image_width': image_dict[image_id]['width'],
                                              'image_height': image_dict[image_id]['height']})

        return image_bbox_dict

    def _normalise_bbox(self, bbox_data: CocoBBoxEntry) -> tuple[float, float, float, float]:
        """Converts COCO bounding box coordinates into normalised YOLO coordinates.

        Args:
            bbox_data (CocoBBoxEntry): A single COCO annotation bounding box entry.

        Returns:
            tuple[float, float, float, float]: The converted YOLO bounding box coordinates.
        """
        imgh = bbox_data.image_height
        imgw = bbox_data.image_width
        x, y, w, h = bbox_data.bbox

        # Apply the yolo conversion and bbox normalisation
        cx = (x + w / 2.0) / imgw
        cy = (y + h / 2.0) / imgh
        w = w / imgw
        h = h / imgh

        return (cx, cy, w, h)

    def _convert_image_annotations(self, bbox_data: list[CocoBBoxEntry]) -> str:
        """Converts the COCO image bounding box annotations to YOLO string formatted representations.

        Args:
            bbox_data (list[CocoBBoxEntry]): A list of COCO annotation bounding box entries.

        Returns:
            str: The formatted YOLO strings.

        Raises:
            ValueError: If YOLO mapping for any COCO class id is missing.
        """
        formatted_annotations = ""
        for bbox in bbox_data:
            yolo_bbox = self._normalise_bbox(bbox)
            coco_class_id = bbox.class_id
            if coco_class_id not in self.coco_to_yolo:
                raise ValueError(
                    f"COCO class id {coco_class_id} does not have a YOLO mapping. Check the class mapping.")

            yolo_class_id = self.coco_to_yolo[coco_class_id]

            formatted_annotations += f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n"

        return formatted_annotations

    def _convert_no_annotations(self, raw_data: dict[str, Any]) -> Iterator[tuple[str, str | None]]:
        """Yields every image path with no label, for annotation-free file data.

        Args:
            raw_data (dict[str, Any]): Raw COCO annotation file data.

        Yields:
            tuple[str, str | None]: Image path and None, as there are no annotations to convert.
        """
        image_dict = self._index_images(raw_data)
        for image_data in image_dict.values():
            yield image_data['path'], None

    def _convert_annotations(self, raw_data: dict[str, Any]) -> Iterator[tuple[str, str | None]]:
        """Yields every image path with its converted YOLO label block.

        Args:
            raw_data (dict[str, Any]): Raw COCO annotation file data.

        Yields:
            tuple[str, str | None]: Image path and its YOLO label string, or None if the image has no annotations.

        Raises:
            ValueError: If the image bounding box mapping is invalid.
        """
        image_dict = self._index_images(raw_data)
        image_bbox_dict = self._index_annotations(raw_data, image_dict)

        # Validate dict against the Pydantic contract
        try:
            image_bbox_mapping = ImageBBoxMapping.model_validate(image_bbox_dict)
        except ValidationError as e:
            raise ValueError(
                "Invalid image bounding box mapping. Please check the image bounding box mapping.") from e

        for image_id, image_data in image_dict.items():
            if image_id in image_bbox_dict:
                yield image_data['path'], self._convert_image_annotations(image_bbox_mapping.root[image_id])
            else:
                yield image_data['path'], None

    def convert(self, raw_data: dict[str, Any]) -> Iterator[tuple[str, str | None]]:
        """Converts one COCO annotation file into YOLO image path and label pairs.

        Args:
            raw_data (dict[str, Any]): Raw COCO annotation file data.

        Yields:
            tuple[str, str | None]: Image path relative to the dataset root, and its YOLO
                                    label string, or None when the image has no annotations.

        Raises:
            KeyError: If the COCO data is missing required image or annotation keys.
            ValueError: If an image has non-positive dimensions, the annotations fail the bounding box contract,
                        or a COCO class id has no YOLO mapping.
        """
        annotations = raw_data.get('annotations', [])
        if not annotations:
            yield from self._convert_no_annotations(raw_data)
        else:
            yield from self._convert_annotations(raw_data)
