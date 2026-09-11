"""Project directory paths, anchored to the repository root.

Resolves PROJECT_ROOT by walking up from this file to the directory containing
pyproject.toml, then derives every data, config and log directory from it.
"""

from pathlib import Path


def _get_project_root() -> Path:
    """Resolves the project root by locating the directory containing pyproject.toml.

    Returns:
        Path: The first parent directory of this file that contains pyproject.toml.

    Raises:
        FileNotFoundError: If no parent directory contains pyproject.toml.
    """
    anchor_file = "pyproject.toml"
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / anchor_file).exists():
            return parent

    raise FileNotFoundError(f"Could not find {anchor_file} in any parent directory of {current_path}")

PROJECT_ROOT = _get_project_root()

# Centralised paths
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
