from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar('T')

class BaseAnnotationConverter(ABC, Generic[T]):
    """Abstract base class for all annotation converters."""

    @abstractmethod
    def convert(self, raw_data: T) -> Iterator[tuple[str, str | None]]:
        pass
