import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src import paths
from src.datatypes import DatasetConfigModel

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = paths.LOGS_DIR / "validation.log"


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class DatasetValidationError(ValueError):
    """Raised when a dataset fails pre-flight validation."""


class BaseDatasetValidator(ABC):
    """Abstract contract for all source format dataset validators."""

    def __init__(self, dataset_raw_dir: Path, dataset_config: DatasetConfigModel,
                 log_path: Path = _DEFAULT_LOG_PATH) -> None:
        self.dataset_raw_dir = dataset_raw_dir
        self.dataset_config = dataset_config
        self.log_path = log_path

    @contextmanager
    def _logging_session(self) -> Iterator[None]:
        """Attach ERROR->file and INFO/WARNING->stream handlers to the validators
        package logger for the duration of one validate() call."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(fmt)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.addFilter(_MaxLevelFilter(logging.WARNING))
        stream_handler.setFormatter(fmt)

        subsystem_logger = logging.getLogger(__package__)
        prev_level, prev_propagate = subsystem_logger.level, subsystem_logger.propagate

        subsystem_logger.setLevel(logging.INFO)
        subsystem_logger.propagate = False
        subsystem_logger.addHandler(file_handler)
        subsystem_logger.addHandler(stream_handler)
        try:
            yield
        finally:
            subsystem_logger.removeHandler(file_handler)
            subsystem_logger.removeHandler(stream_handler)
            subsystem_logger.setLevel(prev_level)
            subsystem_logger.propagate = prev_propagate
            file_handler.close()
            stream_handler.close()

    def validate(self) -> None:
        """Runs all checks, logs every failure to file, then raises if any are found."""
        with self._logging_session():
            try:
                errors = self._run_checks()
            except Exception:
                logger.exception("Validation aborted by an unexpected error.")
                raise

            if errors:
                for error in errors:
                    logger.error(error)
                raise DatasetValidationError(f"Dataset validation failed with {len(errors)} error(s). See {self.log_path} for details.")

            logger.info("Dataset validation passed.")

    @abstractmethod
    def _run_checks(self) -> list[str]:
        """Run all format-specific checks. Return a list of error messages, empty if valid."""
        pass

    @staticmethod
    def _check_name() -> str:
        """Gets the name of the calling check method."""
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            return "<unknown>"
        return frame.f_back.f_code.co_name
