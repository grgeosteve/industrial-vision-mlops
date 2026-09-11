"""Abstract contract for annotation format converters.

Concrete converters must subclass BaseAnnotationConverter, parametrised by the raw data type
they accept, and yield image path and label content pairs.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar('T')

class BaseAnnotationConverter(ABC, Generic[T]):
    """Abstract base class for all annotation converters."""

    @abstractmethod
    def convert(self, raw_data: T) -> Iterator[tuple[str, str | None]]:
        """Converts one annotation file's raw data into image path and label pairs.

        Args:
            raw_data (T): Raw annotation data in the source format the converter handles.

        Yields:
            tuple[str, str | None]: Image path relative to the dataset root, and its label
                                    content, or None when the image carries no annotations.
        """
        pass
