import pytest
from pydantic import ValidationError

from src.typing.config import ToolContractConfig


def test_tool_contract_config_defaults_to_fail() -> None:
    cfg = ToolContractConfig()

    assert cfg.enforcement == "fail"
    assert cfg.unsafe_reason is None


@pytest.mark.parametrize("enforcement", ["warn", "skip"])
def test_tool_contract_config_requires_reason_for_unsafe_modes(
    enforcement: str,
) -> None:
    with pytest.raises(ValidationError, match="unsafe_reason"):
        ToolContractConfig(enforcement=enforcement)


def test_tool_contract_config_accepts_reason_for_warn() -> None:
    cfg = ToolContractConfig(enforcement="warn", unsafe_reason="testing custom parser")

    assert cfg.enforcement == "warn"
    assert cfg.unsafe_reason == "testing custom parser"
