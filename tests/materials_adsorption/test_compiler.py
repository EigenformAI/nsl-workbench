from __future__ import annotations

import pytest

from tasks.materials_adsorption.compiler import (
    RuleCompiler,
    RuleEvaluationError,
    RuleValidationError,
)
from tasks.materials_adsorption.dataset import CandidateState
from tasks.materials_adsorption.descriptors import DescriptorRegistry, DescriptorSpec


@pytest.fixture
def registry() -> DescriptorRegistry:
    return DescriptorRegistry(
        version="test-v1",
        descriptors=(
            DescriptorSpec("binding_en", "number", "dimensionless", "chemistry"),
            DescriptorSpec("coordination", "number", "count", "site"),
            DescriptorSpec("site_distance", "number", "angstrom", "site"),
            DescriptorSpec("is_hollow", "boolean", "boolean", "site"),
        ),
    )


def _candidates() -> list[CandidateState]:
    return [
        CandidateState(
            alias="cfg_b",
            system_alias="sys_1",
            descriptors={
                "binding_en": 2.0,
                "coordination": 2.0,
                "site_distance": 1.5,
                "is_hollow": False,
            },
        ),
        CandidateState(
            alias="cfg_a",
            system_alias="sys_1",
            descriptors={
                "binding_en": 3.0,
                "coordination": 4.0,
                "site_distance": 2.0,
                "is_hollow": True,
            },
        ),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').getenv('SECRET')",
        "open('/tmp/private').read()",
        "candidate_id",
        "{'Pt': 1.0}.get('Pt', 0.0)",
        "[binding_en for _ in range(2)][0]",
        "binding_en.__class__",
        "globals()",
    ],
)
def test_forbidden_constructs_and_unknown_names_fail(
    registry: DescriptorRegistry, source: str
) -> None:
    with pytest.raises(RuleValidationError):
        RuleCompiler(registry).compile(source)


def test_compilation_is_label_free_deterministic_and_order_independent(
    registry: DescriptorRegistry,
) -> None:
    compiler = RuleCompiler(registry)
    rule = compiler.compile(
        "binding_en * coordination",
        applicable="coordination > 0",
        uncertainty="1 / (1 + binding_en)",
    )

    artifact = rule.evaluate(_candidates(), dataset_hash="data", universe_hash="universe")
    replay = rule.evaluate(
        list(reversed(_candidates())),
        dataset_hash="data",
        universe_hash="universe",
    )

    assert artifact.artifact_hash == replay.artifact_hash
    assert artifact.rule_hash == replay.rule_hash
    assert [entry.candidate_alias for entry in artifact.entries] == ["cfg_a", "cfg_b"]
    assert [entry.value for entry in artifact.entries] == [12.0, 4.0]
    assert artifact.support_count == 2


def test_whitespace_equivalent_rules_share_hash(registry: DescriptorRegistry) -> None:
    compiler = RuleCompiler(registry)

    assert compiler.compile("binding_en+1.0").rule_hash == compiler.compile(
        "binding_en + 1.0"
    ).rule_hash


def test_complexity_is_monotonic_for_padded_rule(registry: DescriptorRegistry) -> None:
    compiler = RuleCompiler(registry)
    simple = compiler.compile("binding_en")
    padded = compiler.compile("binding_en + 0.0 * binding_en")

    assert padded.complexity > simple.complexity


def test_complexity_charges_only_the_feature_expression(
    registry: DescriptorRegistry,
) -> None:
    compiler = RuleCompiler(registry)

    assert compiler.compile("binding_en").complexity == 1
    assert compiler.compile(
        "binding_en", applicable="coordination > 0", uncertainty="coordination"
    ).complexity == 1


def test_unit_incompatible_expression_is_rejected(registry: DescriptorRegistry) -> None:
    with pytest.raises(RuleValidationError, match="unit"):
        RuleCompiler(registry).compile("coordination + site_distance")


def test_intermediate_magnitude_limit_stops_nested_power(
    registry: DescriptorRegistry,
) -> None:
    rule = RuleCompiler(registry).compile("(binding_en ** 16) ** 16")

    with pytest.raises(RuleEvaluationError, match="resource limit"):
        rule.evaluate(_candidates(), dataset_hash="data", universe_hash="universe")
