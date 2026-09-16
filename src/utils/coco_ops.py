"""Shared helpers for reading COCO annotation structures."""

from pathlib import Path
from typing import Any


def resolve_image_relpath(image: dict[str, Any], image_root: Path | None = None) -> Path | None:
    """Resolves the image path relative to the dataset root.

    Prefers the non-standard LOCO 'path' field, falls back to COCO 'file_name'.

    Args:
        image (dict[str, Any]): A single COCO 'images' entry.
        image_root (Path | None): Directory holding the images named by 'file_name',
                                  relative to the dataset root, or None when they sit at the root.

    Returns:
        Path | None: The image path relative to the dataset root, or None if neither
                     'path' nor 'file_name' holds a usable value.
    """
    image_path = image.get('path')
    if image_path and isinstance(image_path, str):
        stripped = image_path.lstrip("/")
        if stripped != "":
            return Path(stripped)

    image_fname = image.get('file_name')
    if image_fname and isinstance(image_fname, str):
        stripped_fname = image_fname.lstrip("/")
        if stripped_fname != "":
            return image_root / stripped_fname if image_root else Path(stripped_fname)
    return None

def get_image_root(config_extra: dict[str, Any] | None, annotation_name: str) -> Path | None:
    """Looks up the configured image root for one annotation file.

    Reads the optional 'image_roots' mapping, which pairs an annotation filename with the
    directory holding its images. A dataset whose annotations carry explicit image paths
    declares no roots.

    Args:
        config_extra (dict[str, Any] | None): The dataset configuration extra fields.
        annotation_name (str): Filename of the annotation file being processed.

    Returns:
        Path | None: The declared image root relative to the dataset root, or None
                     when the dataset declares no root for this annotation file.
    """
    image_roots = (config_extra or {}).get("image_roots") or {}
    root = image_roots.get(annotation_name)
    if root and isinstance(root, str):
        stripped = root.strip("/")
        if stripped != "":
            return Path(stripped)
    return None
