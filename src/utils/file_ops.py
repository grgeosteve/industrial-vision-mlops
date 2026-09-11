"""Filesystem helpers for configuration loading, archive extraction and downloads.

Archive extraction is path-traversal safe: every member is resolved and checked against
the destination directory before anything is written.
"""

import json
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

def is_safe_path(base_dir: str | Path, target_path: str | Path) -> bool:
    """Reports whether target_path resolves inside base_dir, to prevent path traversal.

    Both paths are resolved before comparison, so symlinks are followed.

    Args:
        base_dir (str | Path): The directory the target must stay within.
        target_path (str | Path): The path to test.

    Returns:
        bool: True if target_path is inside base_dir, False otherwise.
    """
    base_path = Path(base_dir).resolve()
    target_path = Path(target_path).resolve()

    return target_path.is_relative_to(base_path)

def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Loads configuration from a YAML file.

    Args:
        config_path (str | Path): The path to the YAML file to load.

    Returns:
        dict[str, Any]: The parsed configuration mapping.

    Raises:
        ValueError: If the file does not parse as YAML.
        ValueError: If the parsed document is not a mapping.
    """
    config_path = Path(config_path).resolve()

    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML configuration format in file: {config_path}, "
                             f"expected dict but found {type(config).__name__}")

        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {config_path}") from e

def load_json_config(file_path: str | Path) -> dict[str, Any] | list[Any]:
    """Safely loads a JSON config file and handles common errors.

    Args:
        file_path (str | Path): The path to the JSON file to load.

    Returns:
        dict[str, Any] | list[Any]: The loaded JSON data as a dictionary or list.

    Raises:
        ValueError: If the file does not parse as JSON.
        ValueError: If the parsed document is neither a dict nor a list.
    """
    file_path = Path(file_path).resolve()
    try:
        with open(file_path, encoding='utf-8') as f:
            config = json.load(f)

        if not isinstance(config, (dict, list)):
            raise ValueError(f"Invalid JSON format in file: {file_path}, "
                             f"expected dict or list but found {type(config).__name__}")

        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e

def secure_zip_extract(archive_path: str | Path, dest_dir: str | Path) -> None:
    """Extracts a ZIP archive securely rejecting content that escapes the destination directory.

    Args:
        archive_path (str | Path): The ZIP archive to extract.
        dest_dir (str | Path): The directory to extract into.

    Raises:
        PermissionError: If any of the contents resolves outside dest_dir.
    """
    archive_path = Path(archive_path).resolve()
    dest_dir = Path(dest_dir).resolve()

    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            member_target = (dest_dir / member.filename).resolve()

            if not is_safe_path(dest_dir, member_target):
                raise PermissionError(f"Unsafe path detected in ZIP archive: {member}")

        zip_ref.extractall(dest_dir)

def secure_tar_extract(archive_path: str | Path, dest_dir: str | Path) -> None:
    """Extracts a TAR archive securely rejecting content that escapes the destination directory.

    Args:
        archive_path (str | Path): The TAR archive to extract.
        dest_dir (str | Path): The directory to extract into.

    Raises:
        PermissionError: If any of the contents is a symbolic or hard link.
        PermissionError: If any of the contents resolves outside dest_dir.
    """
    archive_path = Path(archive_path).resolve()
    dest_dir = Path(dest_dir).resolve()

    with tarfile.open(archive_path, 'r:*') as tar:
        for member in tar.getmembers():
            member_target = (dest_dir / member.name).resolve()

            if member.issym() or member.islnk():
                raise PermissionError(f"Symbolic links are not allowed in TAR archives: {member.name}")

            if not is_safe_path(dest_dir, member_target):
                raise PermissionError(f"Unsafe path detected in TAR archive: {member.name}")

        tar.extractall(dest_dir)

def extract_archive(archive_path: str | Path, extract_path: str | Path) -> None:
    """Extracts an archive file to the specified path, skipping if the destination is populated.

    Args:
        archive_path (str | Path): The archive to extract.
        extract_path (str | Path): The directory to extract into.

    Raises:
        ValueError: If the archive is neither a TAR nor a ZIP file.
    """
    archive_path = Path(archive_path).resolve()
    extract_path = Path(extract_path).resolve()

    extract_path.mkdir(parents=True, exist_ok=True)

    if not any(extract_path.iterdir()):
        logger.info(f"Extracting {archive_path} to {extract_path}...")
        if tarfile.is_tarfile(archive_path):
            secure_tar_extract(archive_path, extract_path)
        elif zipfile.is_zipfile(archive_path):
            secure_zip_extract(archive_path, extract_path)
        else:
            raise ValueError("Unsupported archive format")
        logger.info(f"Extraction completed for {archive_path}")
    else:
        logger.info(f"Extracted data already exists at {extract_path}. Skipping extraction.")

def download_data(url: str, destination_path: str | Path) -> None:
    """Downloads data from a URL to a specified path, skipping if the file already exists.

    Args:
        url (str): The URL to download.
        destination_path (str | Path): The destination path.
    """
    destination_path = Path(destination_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        logger.info(f"File already exists at {destination_path}. Skipping download.")
        return

    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()  # Check if the request was successful

    # Extract total file size from headers for progress tracking
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192

    with tqdm(total=total_size if total_size > 0 else None,
              unit='B', unit_scale=True, desc=destination_path.name) as pbar:
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)

    logger.info(f"Downloaded file saved to {destination_path}")
