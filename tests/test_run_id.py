import pytest

from src.helper import validate_run_id


@pytest.mark.parametrize(
    "run_id",
    ["run-123", "20260715_ab.cd", "A1"],
)
def test_validate_run_id_accepts_safe_single_components(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "nested/run", r"nested\\run", "run id", "/tmp/run"],
)
def test_validate_run_id_rejects_unsafe_components(run_id: str) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        validate_run_id(run_id)
