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

class CocoDocument(BaseModel):
    """
    Top level structural contract for a COCO annotation file
    """
    model_config = {"extra": "allow"}
    images: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    annotations: list[dict[str, Any]] = []  # Key may be absent for test splits; null is rejected
