"""Shared helpers for reading COCO annotation structures."""

from pathlib import Path
from typing import Any


def resolve_image_relpath(image: dict[str, Any]) -> Path | None:
    """Resolves the image path relative to the dataset root.

    Prefers the non-standard LOCO 'path' field, falls back to COCO 'file_name'.

    Args:
        image (dict[str, Any]): A single COCO 'images' entry.

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
            return Path("images") / stripped_fname
    return None
