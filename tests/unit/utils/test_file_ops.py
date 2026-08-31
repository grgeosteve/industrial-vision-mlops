import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import responses

from src.utils import file_ops


def test_is_safe_path() -> None:
    """Test the is_safe_path function for various path scenarios."""
    base = Path("/tmp/data")
    assert file_ops.is_safe_path(base, "/tmp/data/dataset.zip") is True
    assert file_ops.is_safe_path(base, "/tmp/evil_dir/dataset.zip") is False
    assert file_ops.is_safe_path(base, "/tmp/data/../../etc/passwd") is False

@responses.activate
def test_download_data(tmp_path: Path) -> None:
    """Test the download_data function with a mocked HTTP response."""

    url = "http://example.com/dataset.zip"
    dest = tmp_path / "dataset.zip"
    payload = b"fake_archive_data"

    # Intercept the HTTP request and return a fake response
    responses.add(responses.GET, url, body=payload, status=200)

    file_ops.download_data(url, dest)

    # Assert the file was written to the temporary directory
    assert dest.exists()
    assert dest.read_bytes() == payload

def test_extract_archive_zip(tmp_path: Path) -> None:
    """Test the extract_archive function with a sample zip file."""

    file1_str = "This is file 1"
    file2_str = "This is file 2"

    # Create a sample zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.writestr("file1.txt", file1_str)
        zipf.writestr("file2.txt", file2_str)

    # Extract the archive
    extract_target = tmp_path / "extracted"
    file_ops.extract_archive(zip_path, extract_target)

    # Assert the files were extracted correctly
    assert (extract_target / "file1.txt").exists()
    assert (extract_target / "file2.txt").exists()
    assert (extract_target / "file1.txt").read_text() == file1_str
    assert (extract_target / "file2.txt").read_text() == file2_str

def test_extract_archive_tar(tmp_path: Path) -> None:
    """Test the extract_archive function with a sample tar file."""

    archive_path = tmp_path / "test.tar"
    extract_path = tmp_path / "extracted"

    dummy_text = "Hello, World!"
    dummy_file = tmp_path / "hello.txt"
    dummy_file.write_text(dummy_text)

    # Create a sample tar file
    with tarfile.open(archive_path, "w") as tar:
        tar.add(dummy_file, arcname="hello.txt")

    # Extract the archive
    file_ops.extract_archive(archive_path, extract_path)

    # Assert the files were extracted correctly
    assert (extract_path / "hello.txt").exists()
    assert (extract_path / "hello.txt").read_text() == dummy_text

def test_extract_archive_tar_gz(tmp_path: Path) -> None:
    """Test the extract_archive function with a sample tar.gz file."""

    archive_path = tmp_path / "test.tar.gz"
    extract_path = tmp_path / "extracted"

    dummy_text = "Hello, World!"
    dummy_file = tmp_path / "hello.txt"
    dummy_file.write_text(dummy_text)

    # Create a sample tar.gz file
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dummy_file, arcname="hello.txt")

    # Extract the archive
    file_ops.extract_archive(archive_path, extract_path)

    # Assert the files were extracted correctly
    assert (extract_path / "hello.txt").exists()
    assert (extract_path / "hello.txt").read_text() == dummy_text

def test_extract_archive_unsupported_format(tmp_path: Path) -> None:
    """Test that extract_archive raises an error for unsupported formats."""

    archive_path = tmp_path / "fake_archive.rar"
    extract_path = tmp_path / "extracted"

    archive_path.write_text("This is not a real archive")

    with pytest.raises(ValueError, match="Unsupported archive format"):
        file_ops.extract_archive(archive_path, extract_path)

def test_extract_archive_nested_folder(tmp_path: Path) -> None:
    """Test that extract_archive correctly handles nested folders in a zip file."""

    archive_path = tmp_path / "nested.zip"
    extract_path = tmp_path / "extracted"

    fake_data = "fake_image_data"
    fake_metadata = '{"key": "value"}'

    # Create a sample zip file with nested folders
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        zipf.writestr("dataset/images/sample.jpg", fake_data)
        zipf.writestr("dataset/metadata.json", fake_metadata)

    # Extract the archive
    extract_target = tmp_path / "extracted"
    file_ops.extract_archive(archive_path, extract_target)

    # Assert the files were extracted correctly
    assert (extract_path / "dataset").is_dir()
    assert (extract_target / "dataset/images/sample.jpg").exists()
    assert (extract_target / "dataset/metadata.json").exists()
    assert (extract_target / "dataset/images/sample.jpg").read_text() == fake_data
    assert (extract_target / "dataset/metadata.json").read_text() == fake_metadata

def test_tar_symlink_rejection(tmp_path: Path) -> None:
    """Test that secure_tar_extract rejects symbolic links in tar files."""

    archive_path = tmp_path / "malicious.tar"
    extract_path = tmp_path / "extracted"

    # Create a safe "forbidden" file outside the extraction directory
    forbidden_target = tmp_path / "forbidden_secret.txt"
    forbidden_target.write_text("sensitive data")

    with tarfile.open(archive_path, "w") as tar:
        tarinfo = tarfile.TarInfo(name="symlink")
        tarinfo.type = tarfile.SYMTYPE

        # Point the symlink to a sandbox forbidden file
        tarinfo.linkname = str(forbidden_target)
        tar.addfile(tarinfo)

    with pytest.raises(PermissionError, match="Symbolic links are not allowed in TAR archives"):
        file_ops.extract_archive(archive_path, extract_path)

def test_zip_path_traversal_rejection(tmp_path: Path) -> None:
    """Test that secure_zip_extract rejects path traversal attempts in zip files."""

    archive_path = tmp_path / "malicious.zip"
    extract_path = tmp_path / "extracted"

    with zipfile.ZipFile(archive_path, 'w') as zipf:
        zipf.writestr("../evil.txt", "This should not be extracted")

    with pytest.raises(PermissionError, match="Unsafe path detected in ZIP archive"):
        file_ops.extract_archive(archive_path, extract_path)

def test_tar_path_traversal_rejection(tmp_path: Path) -> None:
    """Test that secure_tar_extract rejects path traversal attempts in tar files."""

    archive_path = tmp_path / "malicious.tar"
    extract_path = tmp_path / "extracted"

    with tarfile.open(archive_path, "w") as tar:
        tarinfo = tarfile.TarInfo(name="../evil.txt")
        tarinfo.size = len("This should not be extracted")
        tar.addfile(tarinfo, fileobj=io.BytesIO(b"This should not be extracted"))

    with pytest.raises(PermissionError, match="Unsafe path detected in TAR archive"):
        file_ops.extract_archive(archive_path, extract_path)

def test_load_json_config_valid(tmp_path: Path) -> None:
    """Test that load_json_config correctly loads a valid JSON file."""

    config_data = {"dataset": "coco", "classes": [1, 2, 3]}
    config_path = tmp_path / "config.json"
    config_path.write_text('{"dataset": "coco", "classes": [1, 2, 3]}')

    loaded_config = file_ops.load_json_config(config_path)
    assert loaded_config == config_data

def test_load_json_config_list(tmp_path: Path) -> None:
    """Test that load_json_config correctly loads a JSON file containing a list."""

    config_data = [1, 2, 3]
    config_path = tmp_path / "config.json"
    config_path.write_text('[1, 2, 3]')

    loaded_config = file_ops.load_json_config(config_path)
    assert loaded_config == config_data

def test_load_json_config_wrong_type(tmp_path: Path) -> None:
    """Test that load_json_config raises an error if the JSON is not a dict or list."""

    config_path_string = tmp_path / "wrong_type_config_string.json"
    config_path_string.write_text('"This is a string, not a dict or list"')

    config_path_number = tmp_path / "wrong_type_config_number.json"
    config_path_number.write_text('42')

    with pytest.raises(ValueError, match="Invalid JSON format in file"):
        file_ops.load_json_config(config_path_string)
    with pytest.raises(ValueError, match="Invalid JSON format in file"):
        file_ops.load_json_config(config_path_number)

def test_load_json_config_invalid_json(tmp_path: Path) -> None:
    """Test that load_json_config raises an error for invalid JSON format."""

    config_path = tmp_path / "invalid.json"
    config_path.write_text('{"key": "value",}')  # Invalid JSON due to trailing comma

    with pytest.raises(ValueError, match="Invalid JSON format in file"):
        file_ops.load_json_config(config_path)

def test_load_json_config_nonexistent_file(tmp_path: Path) -> None:
    """Test that load_json_config raises an error when the file does not exist."""

    config_path = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError):
        file_ops.load_json_config(config_path)

def test_load_yaml_config_valid(tmp_path: Path) -> None:
    """Test that load_yaml_config correctly loads a valid YAML file."""

    config_data = {"dataset": "coco", "classes": [1, 2, 3]}
    config_path = tmp_path / "config.yaml"
    config_path.write_text('dataset: coco\nclasses:\n  - 1\n  - 2\n  - 3\n')

    loaded_config = file_ops.load_yaml_config(config_path)
    assert loaded_config == config_data

def test_load_yaml_config_file_not_found(tmp_path: Path) -> None:
    """Test that load_yaml_config raises an error when the file does not exist."""

    config_path = tmp_path / "nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        file_ops.load_yaml_config(config_path)

def test_load_yaml_config_invalid_yaml(tmp_path: Path) -> None:
    """Test that load_yaml_config raises an error for invalid YAML format."""

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("dataset: [loco\n- malformed")

    with pytest.raises(ValueError, match="Error parsing YAML configuration file"):
        file_ops.load_yaml_config(config_path)

def test_load_yaml_config_not_a_dict(tmp_path: Path) -> None:
    """Test that load_yaml_config raises an error if the YAML does not contain a dict."""

    config_path = tmp_path / "not_a_dict.yaml"
    config_path.write_text('- item1\n- item2\n- item3\n')  # This is a list, not a dict

    with pytest.raises(ValueError, match="Invalid YAML configuration format in file"):
        file_ops.load_yaml_config(config_path)

def test_load_yaml_config_empty_file(tmp_path: Path) -> None:
    """Test that load_yaml_config raises an error for an empty YAML file."""

    config_path = tmp_path / "empty.yaml"
    config_path.write_text('')  # Empty file

    with pytest.raises(ValueError, match="Invalid YAML configuration format in file"):
        file_ops.load_yaml_config(config_path)
