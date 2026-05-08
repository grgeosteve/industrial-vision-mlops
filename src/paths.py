from pathlib import Path

def _get_project_root() -> Path:
    """Get the project root directory."""

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