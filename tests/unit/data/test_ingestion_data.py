from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.ingest_data import ingest_data


@pytest.fixture
def mock_config() -> dict:
    """Create a temporary YAML config file for testing."""
    return {
        "datasets": {
            "test_dataset": {
                "archives": {
                    "test.zip": "https://example.com/test.zip"
                },
                "metadata_urls": {
                    "ann1.json": "https://example.com/ann1.json",
                    "ann2.json": "https://example.com/ann2.json"
                },
                "annotations": [
                    "ann1.json",
                    "ann2.json"
                ]
            }
        }
    }

@patch("src.data.ingest_data.shutil")
@patch("src.data.ingest_data.file_ops")
@patch("src.data.ingest_data.paths")
def test_ingest_data(mock_paths: MagicMock,
                     mock_file_ops: MagicMock,
                     mock_shutil: MagicMock,
                     mock_config: dict,
                     tmp_path: Path) -> None:
    """Test the ingest_data function with a mocked configuration."""

    # Setup mock paths
    mock_paths.EXTERNAL_DATA_DIR = tmp_path / "external"
    mock_paths.RAW_DATA_DIR = tmp_path / "raw"
    mock_paths.CONFIG_DIR = tmp_path / "configs"

    # Mock the load_yaml_config to return our test config
    mock_file_ops.load_yaml_config.return_value = mock_config

    # Create mock download by blocking "shutil.copy" and
    # creating dummy files instead
    def mock_download(url: str, dest: str | Path) -> None:
        dest = Path(dest).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    mock_file_ops.download_data.side_effect = mock_download

    # Execute the function
    ingest_data("test_dataset", mock_paths.CONFIG_DIR / "data_config.yaml")

    # Assert main archive ingestion steps were called
    mock_file_ops.download_data.assert_any_call(
        "https://example.com/test.zip",
        mock_paths.EXTERNAL_DATA_DIR / "test_dataset" / "test.zip"
    )
    mock_file_ops.extract_archive.assert_called_once_with(
        mock_paths.EXTERNAL_DATA_DIR / "test_dataset" / "test.zip",
        mock_paths.RAW_DATA_DIR / "test_dataset"
    )

    # Assert metadata download was called
    mock_file_ops.download_data.assert_any_call(
        "https://example.com/ann1.json",
        mock_paths.EXTERNAL_DATA_DIR / "test_dataset" / "ann1.json"
    )
    mock_file_ops.download_data.assert_any_call(
        "https://example.com/ann2.json",
        mock_paths.EXTERNAL_DATA_DIR / "test_dataset" / "ann2.json"
    )

    # Assert annotation handling (copying) was called
    mock_shutil.copy.assert_any_call(
        mock_paths.EXTERNAL_DATA_DIR / "test_dataset" / "ann1.json",
        mock_paths.RAW_DATA_DIR / "test_dataset" / "annotations" / "ann1.json"
    )
    mock_shutil.copy.assert_any_call(
        mock_paths.EXTERNAL_DATA_DIR / "test_dataset" / "ann2.json",
        mock_paths.RAW_DATA_DIR / "test_dataset" / "annotations" / "ann2.json"
    )
