from pathlib import Path
from typing import Any, TypeAlias

from pydantic import BaseModel

# Key: Format identifier
# Value: Integer ID
ClassInfo: TypeAlias = dict[str, int]

# Key: Class Name
# Value: ClassInfo
ClassConfigMapping: TypeAlias = dict[str, ClassInfo]

# Split config: split name -> annotation filename(s)
SplitConfig: TypeAlias = dict[str, list[str] | str]
DatasetConfig: TypeAlias = dict[str, Any]
DatasetConfigs: TypeAlias = dict[str, DatasetConfig]

class DatasetConfigModel(BaseModel):
    """
    Validates the DatasetConfig dictionary contains the necessary keys
    """

    # Prevents crashing when encountering extra information
    model_config = {"extra": "allow"}

    classes: ClassConfigMapping
    splits: SplitConfig
    path: Path
    type: str
    format: str
    name: str
