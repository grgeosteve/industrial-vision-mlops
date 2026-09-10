from pathlib import Path

import pytest

from src import paths
from src.validators.base_validator import BaseDatasetValidator


@pytest.fixture(autouse=True)
def pipeline_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirects the pipeline data directories and the validation log into tmp_path."""
    monkeypatch.setattr(paths, "EXTERNAL_DATA_DIR", tmp_path / "external")
    monkeypatch.setattr(paths, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")

    # BaseDatasetValidator binds its default log path at import
    # Patching it needs to replace the default itself
    monkeypatch.setattr(BaseDatasetValidator.__init__, "__defaults__", (tmp_path / "logs" / "validation.log",))
