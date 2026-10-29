"""Tests for _normalize_tools in src/genner/OAI.py."""

from __future__ import annotations

from src.genner.OAI import _normalize_tools


class TestNormalizeTools:
    """Unit tests for the _normalize_tools helper."""

    def test_normalize_flat_msagent_shape(self) -> None:
        """ms-agent Tool TypedDict shape -> OpenAI function-call shape."""
        flat = [
            {
                "server_name": "nsl",
                "tool_name": "nsl---deploy_attack_sol",
                "description": "Compile and run an Attack.sol contract",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        ]
        result = _normalize_tools(flat)

        assert len(result) == 1
        entry = result[0]
        assert entry["type"] == "function"
        assert "function" in entry
        assert entry["function"]["name"] == "nsl---deploy_attack_sol"
        assert entry["function"]["description"] == "Compile and run an Attack.sol contract"
        assert entry["function"]["parameters"]["type"] == "object"

    def test_normalize_already_correct(self) -> None:
        """OpenAI-shaped tools pass through unchanged."""
        correct = [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "Does stuff",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        result = _normalize_tools(correct)
        assert result is correct  # identity, not just equality

    def test_normalize_multiple_tools(self) -> None:
        flat = [
            {"tool_name": "tool_a", "description": "A", "parameters": {}},
            {"tool_name": "tool_b", "description": "B", "parameters": {}},
        ]
        result = _normalize_tools(flat)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"

    def test_normalize_drops_extra_fields(self) -> None:
        """Only tool_name/description/parameters are mapped; server_name etc. are dropped."""
        flat = [
            {
                "server_name": "nsl",
                "tool_name": "deploy",
                "description": "Deploy contract",
                "parameters": {"type": "object"},
                "extra_field": "should_be_dropped",
            },
        ]
        result = _normalize_tools(flat)

        func = result[0]["function"]
        assert set(func.keys()) == {"name", "description", "parameters"}

    def test_normalize_missing_optional_fields(self) -> None:
        """Handles tools with minimal fields (only tool_name required in ms-agent)."""
        flat = [{"tool_name": "minimal"}]
        result = _normalize_tools(flat)

        func = result[0]["function"]
        assert func["name"] == "minimal"
        assert func["description"] == ""
        assert func["parameters"] == {}
