import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest

from src.datatypes import DatasetConfigModel
from src.validators.base_validator import BaseDatasetValidator, DatasetValidationError

PKG = "src.validators"

class _StubValidator(BaseDatasetValidator):
    """Minimal concrete validator

    Returns errors on demand
    """
    errors_to_return: list[str] = []
    handlers_during: list[logging.Handler] | None = None

    def _run_checks(self) -> list[str]:
        # A logger inside the package, so session routing applies
        log = logging.getLogger(f"{PKG}.stub")

        self.handlers_during = list(logging.getLogger(PKG).handlers)

        log.info("INFO_MARKER")
        log.warning("WARNING_MARKER")

        return list(self.errors_to_return)

    def named_probe(self) -> str:
        return self._check_name()

MakeStub: TypeAlias = Callable[[], _StubValidator]

@pytest.fixture
def make_stub(tmp_path: Path) -> MakeStub:
    def _make() -> _StubValidator:
        cfg = DatasetConfigModel(
            classes={"forklift": {"coco_id": 5, "yolo_id": 0}},
            splits={"train": ["train.json"], "val": ["val.json"]},
            path=Path("dataset"),
            type="object_detection",
            format="coco",
            name="test",
        )
        return _StubValidator(tmp_path, cfg, log_path=tmp_path / "validation.log")

    return _make


def test_check_name_returns_calling_method(make_stub: MakeStub) -> None:
    assert make_stub().named_probe() == "named_probe"


def test_session_attaches_then_restores(make_stub: MakeStub) -> None:
    pkg = logging.getLogger(PKG)
    before = (list(pkg.handlers), pkg.level, pkg.propagate)

    v = make_stub()
    v.validate()

    assert v.handlers_during is not None
    assert {type(h) for h in v.handlers_during} == {logging.FileHandler, logging.StreamHandler}
    assert (list(pkg.handlers), pkg.level, pkg.propagate) == before  # fully restored after

def test_validate_is_repeatable(make_stub: MakeStub) -> None:
    pkg = logging.getLogger(PKG)
    before = (list(pkg.handlers), pkg.level, pkg.propagate)

    v = make_stub()
    v.validate()
    v.validate()

    assert v.handlers_during is not None
    assert {type(h) for h in v.handlers_during} == {logging.FileHandler, logging.StreamHandler}
    assert (list(pkg.handlers), pkg.level, pkg.propagate) == before  # fully restored after

def test_errors_go_to_file_not_terminal(make_stub: MakeStub, capsys: pytest.CaptureFixture[str]) -> None:
    v = make_stub()
    v.errors_to_return = ["ERROR_MARKER: test error"]

    with pytest.raises(DatasetValidationError, match="1 error"):
        v.validate()

    err = capsys.readouterr().err
    log_text = v.log_path.read_text()

    assert "ERROR_MARKER" in log_text       # ERROR -> file
    assert "ERROR_MARKER" not in err        # not in terminal
    assert "WARNING_MARKER" in err          # WARNING -> terminal
    assert "WARNING_MARKER" not in log_text # not in file
    assert "INFO_MARKER" in err             # INFO -> terminal
    assert "INFO_MARKER" not in log_text    # not in file


def test_validate_passes(make_stub: MakeStub) -> None:
    v = make_stub()
    v.validate()  # must not raise
    assert v.log_path.read_text() == ""  # ERROR-only handler, empty file


def test_unexpected_error_logged_and_reraised(make_stub: MakeStub, monkeypatch: pytest.MonkeyPatch) -> None:
    v = make_stub()

    def error() -> None:
        raise RuntimeError("error")

    monkeypatch.setattr(v, "_run_checks", error)
    with pytest.raises(RuntimeError):
        v.validate()

    assert "Validation aborted by an unexpected error." in v.log_path.read_text()
