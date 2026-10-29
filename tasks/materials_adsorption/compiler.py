from __future__ import annotations

import ast
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Mapping

from .dataset import CandidateState
from .descriptors import DescriptorRegistry, _canonical_hash

__all__ = [
    "CompiledRule",
    "ConsequenceArtifact",
    "ConsequenceEntry",
    "RuleCompiler",
    "RuleEvaluationError",
    "RuleValidationError",
    "rule_expression_contract",
]


class RuleValidationError(ValueError):
    pass


class RuleEvaluationError(ValueError):
    pass


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


_FUNCTIONS: Mapping[str, Callable[..., float]] = {
    "abs": abs,
    "clip": _clip,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "sqrt": math.sqrt,
}
_ARITIES = {
    "abs": (1, 1),
    "clip": (3, 3),
    "exp": (1, 1),
    "log": (1, 1),
    "log10": (1, 1),
    "max": (2, 4),
    "min": (2, 4),
    "sqrt": (1, 1),
}
_ALLOWED_NODES = (
    ast.Expression,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.IfExp,
    ast.Call,
)
_MAX_SOURCE_LENGTH = 1_000
_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 16
_MAX_CONSTANTS = 32
_MAX_CONSTANT_MAGNITUDE = 1_000_000.0
_MAX_INTERMEDIATE_MAGNITUDE = 1_000_000_000_000.0

Unit = tuple[tuple[str, Fraction], ...]


def rule_expression_contract() -> str:
    """Return the public expression grammar enforced by ``RuleCompiler``."""

    functions = ", ".join(sorted(_FUNCTIONS))
    return (
        "Expression grammar: use one restricted Python expression with declared "
        "descriptor names, finite numeric or boolean constants, arithmetic "
        "operators +, -, *, /, %, and **, boolean operators and/or/not, comparison "
        "operators ==, !=, <, <=, >, and >=, and `value_if_true if condition else "
        f"value_if_false`. Allowlisted functions: {functions}. Function arguments "
        "are positional only. Numeric addition, subtraction, modulo, min, max, and "
        "comparisons require compatible descriptor units; exp/log/log10 require a "
        "dimensionless argument. The feature expression and uncertainty expression "
        "must return numbers; applicable must return a boolean. Attribute access, "
        "subscripting, comprehensions, containers, imports, files, network access, "
        "and all other calls are forbidden. Limits: 1000 source characters, 128 AST "
        "nodes, depth 16, 32 numeric constants, and power exponents bounded by 16."
    )


@dataclass(frozen=True, slots=True)
class _Inferred:
    kind: str
    unit: Unit = ()
    literal: bool = False


def _unit_for_name(unit: str) -> Unit:
    return () if unit == "dimensionless" else ((unit, Fraction(1)),)


def _combine_units(left: Unit, right: Unit, sign: int = 1) -> Unit:
    powers = dict(left)
    for name, exponent in right:
        powers[name] = powers.get(name, Fraction(0)) + sign * exponent
    return tuple(sorted((name, exponent) for name, exponent in powers.items() if exponent))


def _power_unit(unit: Unit, exponent: Fraction) -> Unit:
    return tuple((name, power * exponent) for name, power in unit if power * exponent)


def _node_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_node_depth(child) for child in children)


class _Validator:
    def __init__(self, registry: DescriptorRegistry) -> None:
        self.registry = registry
        self.references: set[str] = set()

    def validate(self, source: str, expected_type: str) -> tuple[ast.Expression, str, int]:
        if not source or len(source) > _MAX_SOURCE_LENGTH:
            raise RuleValidationError("rule expression is empty or too long")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise RuleValidationError(f"invalid rule expression: {exc.msg}") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > _MAX_AST_NODES or _node_depth(tree) > _MAX_AST_DEPTH:
            raise RuleValidationError("rule expression exceeds the AST resource limit")
        forbidden = next((node for node in nodes if not isinstance(node, _ALLOWED_NODES)), None)
        if forbidden is not None:
            raise RuleValidationError(f"forbidden syntax: {type(forbidden).__name__}")
        constants = [
            node
            for node in nodes
            if isinstance(node, ast.Constant) and type(node.value) in (int, float)
        ]
        if len(constants) > _MAX_CONSTANTS:
            raise RuleValidationError("too many numerical constants")
        for constant in constants:
            value = float(constant.value)
            if not math.isfinite(value) or abs(value) > _MAX_CONSTANT_MAGNITUDE:
                raise RuleValidationError("numerical constant is out of bounds")
        actual_type = self._infer(tree.body)
        if actual_type.kind != expected_type:
            raise RuleValidationError(
                f"expression must return {expected_type}, not {actual_type.kind}"
            )
        normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
        complexity = sum(
            1
            for node in nodes
            if isinstance(
                node,
                (ast.Name, ast.Constant, ast.operator, ast.unaryop, ast.boolop, ast.cmpop),
            )
        ) + 2 * sum(isinstance(node, (ast.Call, ast.IfExp)) for node in nodes)
        return tree, normalized, complexity

    def _infer(self, node: ast.AST) -> _Inferred:
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return _Inferred("boolean", literal=True)
            if type(node.value) in (int, float):
                return _Inferred("number", literal=True)
            raise RuleValidationError("only boolean and numerical constants are allowed")
        if isinstance(node, ast.Name):
            descriptor = self.registry.get(node.id)
            if descriptor is None:
                raise RuleValidationError(f"unknown descriptor: {node.id!r}")
            self.references.add(node.id)
            return _Inferred(
                descriptor.value_type,
                _unit_for_name(descriptor.unit) if descriptor.value_type == "number" else (),
            )
        if isinstance(node, ast.BinOp):
            left = self._infer(node.left)
            right = self._infer(node.right)
            self._require(left, "number", node)
            self._require(right, "number", node)
            if isinstance(node.op, ast.Pow) and (
                not isinstance(node.right, ast.Constant)
                or type(node.right.value) not in (int, float)
                or abs(float(node.right.value)) > 16
            ):
                raise RuleValidationError("power exponents must be constants bounded by 16")
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mod)):
                return _Inferred(
                    "number",
                    self._compatible_unit(left, right, node),
                    left.literal and right.literal,
                )
            if isinstance(node.op, ast.Mult):
                return _Inferred(
                    "number",
                    _combine_units(left.unit, right.unit),
                    left.literal and right.literal,
                )
            if isinstance(node.op, ast.Div):
                return _Inferred(
                    "number",
                    _combine_units(left.unit, right.unit, -1),
                    left.literal and right.literal,
                )
            exponent = Fraction(str(node.right.value))
            return _Inferred(
                "number",
                _power_unit(left.unit, exponent),
                left.literal,
            )
        if isinstance(node, ast.UnaryOp):
            expected = "boolean" if isinstance(node.op, ast.Not) else "number"
            inferred = self._infer(node.operand)
            self._require(inferred, expected, node)
            return _Inferred(expected, inferred.unit, inferred.literal)
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._require(self._infer(value), "boolean", node)
            return _Inferred("boolean")
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            inferred = [self._infer(operand) for operand in operands]
            operand_types = [item.kind for item in inferred]
            if len(set(operand_types)) != 1:
                raise RuleValidationError("comparison operands must have matching types")
            if operand_types[0] == "boolean" and any(
                not isinstance(operator, (ast.Eq, ast.NotEq)) for operator in node.ops
            ):
                raise RuleValidationError("booleans support equality comparisons only")
            if operand_types[0] == "number":
                for left, right in zip(inferred, inferred[1:]):
                    self._compatible_unit(left, right, node)
            return _Inferred("boolean")
        if isinstance(node, ast.IfExp):
            self._require(self._infer(node.test), "boolean", node)
            body_type = self._infer(node.body)
            other_type = self._infer(node.orelse)
            if other_type.kind != body_type.kind:
                raise RuleValidationError("conditional branches must have matching types")
            if body_type.kind == "number":
                return _Inferred(
                    "number", self._compatible_unit(body_type, other_type, node)
                )
            return body_type
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise RuleValidationError("only allowlisted direct function calls are allowed")
            if node.keywords:
                raise RuleValidationError("keyword arguments are not allowed")
            minimum, maximum = _ARITIES[node.func.id]
            if not minimum <= len(node.args) <= maximum:
                raise RuleValidationError(f"invalid argument count for {node.func.id}")
            arguments = [self._infer(argument) for argument in node.args]
            for argument in arguments:
                self._require(argument, "number", node)
            function = node.func.id
            if function in {"exp", "log", "log10"}:
                if arguments[0].unit:
                    raise RuleValidationError(f"{function} requires a dimensionless argument")
                return _Inferred("number")
            if function == "sqrt":
                return _Inferred("number", _power_unit(arguments[0].unit, Fraction(1, 2)))
            if function == "abs":
                return arguments[0]
            current = arguments[0]
            for argument in arguments[1:]:
                current = _Inferred(
                    "number",
                    self._compatible_unit(current, argument, node),
                    current.literal and argument.literal,
                )
            return current
        raise RuleValidationError(f"forbidden syntax: {type(node).__name__}")

    @staticmethod
    def _require(actual: _Inferred, expected: str, node: ast.AST) -> None:
        if actual.kind != expected:
            raise RuleValidationError(
                f"{type(node).__name__} requires {expected} operands, not {actual.kind}"
            )

    @staticmethod
    def _compatible_unit(left: _Inferred, right: _Inferred, node: ast.AST) -> Unit:
        if left.literal:
            return right.unit
        if right.literal:
            return left.unit
        if left.unit != right.unit:
            raise RuleValidationError(
                f"{type(node).__name__} operands have incompatible units"
            )
        return left.unit


def _checked_number(value: Any) -> float:
    if type(value) not in (int, float):
        raise TypeError("numeric expression produced a non-number")
    number = float(value)
    if not math.isfinite(number) or abs(number) > _MAX_INTERMEDIATE_MAGNITUDE:
        raise RuleEvaluationError("expression exceeds the runtime resource limit")
    return number


def _evaluate(node: ast.AST, values: Mapping[str, Any]) -> bool | float:
    """Evaluate the frozen AST with bounded-size floating-point intermediates."""
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is bool else _checked_number(node.value)
    if isinstance(node, ast.Name):
        value = values[node.id]
        return value if type(value) is bool else _checked_number(value)
    if isinstance(node, ast.BinOp):
        left = _checked_number(_evaluate(node.left, values))
        right = _checked_number(_evaluate(node.right, values))
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(left) > 1.0 and right > 0.0:
                if right * math.log(abs(left)) > math.log(_MAX_INTERMEDIATE_MAGNITUDE):
                    raise RuleEvaluationError(
                        "expression exceeds the runtime resource limit"
                    )
            result = math.pow(left, right)
        else:  # pragma: no cover - validator and evaluator are kept in lockstep.
            raise RuleEvaluationError("unsupported binary operator")
        return _checked_number(result)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        number = _checked_number(value)
        return number if isinstance(node.op, ast.UAdd) else _checked_number(-number)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for item in node.values:
                if not bool(_evaluate(item, values)):
                    return False
            return True
        for item in node.values:
            if bool(_evaluate(item, values)):
                return True
        return False
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, values)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, values)
            if isinstance(operator, ast.Eq):
                matched = left == right
            elif isinstance(operator, ast.NotEq):
                matched = left != right
            elif isinstance(operator, ast.Lt):
                matched = left < right
            elif isinstance(operator, ast.LtE):
                matched = left <= right
            elif isinstance(operator, ast.Gt):
                matched = left > right
            else:
                matched = left >= right
            if not matched:
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if bool(_evaluate(node.test, values)) else node.orelse
        return _evaluate(branch, values)
    if isinstance(node, ast.Call):
        arguments = [_checked_number(_evaluate(item, values)) for item in node.args]
        return _checked_number(_FUNCTIONS[node.func.id](*arguments))
    raise RuleEvaluationError(f"unsupported expression node: {type(node).__name__}")


@dataclass(frozen=True, slots=True)
class ConsequenceEntry:
    candidate_alias: str
    applicable: bool
    value: float | None
    uncertainty: float | None
    branch_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_alias": self.candidate_alias,
            "applicable": self.applicable,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "branch_label": self.branch_label,
        }


@dataclass(frozen=True, slots=True)
class ConsequenceArtifact:
    rule_hash: str
    compiler_hash: str
    descriptor_registry_hash: str
    dataset_hash: str
    universe_hash: str
    complexity: int
    entries: tuple[ConsequenceEntry, ...]
    support_count: int
    behavior_hash: str
    artifact_hash: str

    @property
    def descriptor_hash(self) -> str:
        return self.descriptor_registry_hash

    @classmethod
    def create(
        cls,
        *,
        rule_hash: str,
        compiler_hash: str,
        descriptor_hash: str,
        dataset_hash: str,
        universe_hash: str,
        complexity: int,
        entries: list[ConsequenceEntry] | tuple[ConsequenceEntry, ...],
    ) -> ConsequenceArtifact:
        ordered = tuple(sorted(entries, key=lambda entry: entry.candidate_alias))
        aliases = [entry.candidate_alias for entry in ordered]
        if len(set(aliases)) != len(aliases):
            raise ValueError("consequence candidate aliases must be unique")
        payload = {
            "schema_version": 1,
            "rule_hash": rule_hash,
            "compiler_hash": compiler_hash,
            "descriptor_registry_hash": descriptor_hash,
            "dataset_hash": dataset_hash,
            "universe_hash": universe_hash,
            "complexity": complexity,
            "entries": [entry.to_dict() for entry in ordered],
        }
        behavior_payload = {
            "schema_version": 1,
            "compiler_hash": compiler_hash,
            "descriptor_registry_hash": descriptor_hash,
            "dataset_hash": dataset_hash,
            "universe_hash": universe_hash,
            "entries": [
                {
                    "candidate_alias": entry.candidate_alias,
                    "applicable": entry.applicable,
                    "value": entry.value,
                }
                for entry in ordered
            ],
        }
        return cls(
            rule_hash=rule_hash,
            compiler_hash=compiler_hash,
            descriptor_registry_hash=descriptor_hash,
            dataset_hash=dataset_hash,
            universe_hash=universe_hash,
            complexity=complexity,
            entries=ordered,
            support_count=sum(entry.applicable for entry in ordered),
            behavior_hash=_canonical_hash(behavior_payload),
            artifact_hash=_canonical_hash(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_hash": self.rule_hash,
            "compiler_hash": self.compiler_hash,
            "descriptor_registry_hash": self.descriptor_registry_hash,
            "dataset_hash": self.dataset_hash,
            "universe_hash": self.universe_hash,
            "complexity": self.complexity,
            "entries": [entry.to_dict() for entry in self.entries],
            "support_count": self.support_count,
            "behavior_hash": self.behavior_hash,
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConsequenceArtifact:
        artifact = cls.create(
            rule_hash=str(payload["rule_hash"]),
            compiler_hash=str(payload["compiler_hash"]),
            descriptor_hash=str(payload["descriptor_registry_hash"]),
            dataset_hash=str(payload["dataset_hash"]),
            universe_hash=str(payload["universe_hash"]),
            complexity=int(payload["complexity"]),
            entries=[
                ConsequenceEntry(
                    candidate_alias=str(item["candidate_alias"]),
                    applicable=bool(item["applicable"]),
                    value=None if item.get("value") is None else float(item["value"]),
                    uncertainty=(
                        None
                        if item.get("uncertainty") is None
                        else float(item["uncertainty"])
                    ),
                    branch_label=item.get("branch_label"),
                )
                for item in payload["entries"]
            ],
        )
        expected = payload.get("artifact_hash")
        if expected is not None and artifact.artifact_hash != expected:
            raise ValueError("consequence artifact hash mismatch")
        expected_behavior = payload.get("behavior_hash")
        if expected_behavior is not None and artifact.behavior_hash != expected_behavior:
            raise ValueError("consequence behavior hash mismatch")
        return artifact


@dataclass(frozen=True, slots=True)
class CompiledRule:
    rule_hash: str
    compiler_hash: str
    descriptor_registry_hash: str
    complexity: int
    normalized_rule: str
    normalized_applicable: str
    normalized_uncertainty: str
    referenced_descriptors: frozenset[str]
    _registry: DescriptorRegistry = field(repr=False, compare=False)
    _rule_tree: ast.Expression = field(repr=False, compare=False)
    _applicable_tree: ast.Expression = field(repr=False, compare=False)
    _uncertainty_tree: ast.Expression = field(repr=False, compare=False)

    def evaluate(
        self,
        candidates: list[CandidateState] | tuple[CandidateState, ...],
        *,
        dataset_hash: str,
        universe_hash: str,
    ) -> ConsequenceArtifact:
        if not dataset_hash or not universe_hash:
            raise ValueError("dataset_hash and universe_hash must be non-empty")
        ordered = sorted(candidates, key=lambda candidate: candidate.alias)
        aliases = [candidate.alias for candidate in ordered]
        if len(set(aliases)) != len(aliases):
            raise RuleEvaluationError("candidate aliases must be unique")
        entries: list[ConsequenceEntry] = []
        for candidate in ordered:
            try:
                self._registry.validate_values(candidate.descriptors)
                applicable = _evaluate(self._applicable_tree.body, candidate.descriptors)
                if type(applicable) is not bool:
                    raise RuleEvaluationError("applicability must evaluate to boolean")
                if not applicable:
                    entries.append(ConsequenceEntry(candidate.alias, False, None, None))
                    continue
                value = _evaluate(self._rule_tree.body, candidate.descriptors)
                uncertainty = _evaluate(
                    self._uncertainty_tree.body, candidate.descriptors
                )
                if type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise RuleEvaluationError("rule value must be a finite number")
                if (
                    type(uncertainty) not in (int, float)
                    or not math.isfinite(float(uncertainty))
                    or uncertainty < 0
                ):
                    raise RuleEvaluationError("uncertainty must be a finite non-negative number")
            except RuleEvaluationError:
                raise
            except (ArithmeticError, NameError, TypeError, ValueError) as exc:
                raise RuleEvaluationError(
                    f"rule evaluation failed for {candidate.alias}: {exc}"
                ) from exc
            entries.append(
                ConsequenceEntry(candidate.alias, True, float(value), float(uncertainty))
            )
        entries_tuple = tuple(entries)
        return ConsequenceArtifact.create(
            rule_hash=self.rule_hash,
            compiler_hash=self.compiler_hash,
            descriptor_hash=self.descriptor_registry_hash,
            dataset_hash=dataset_hash,
            universe_hash=universe_hash,
            complexity=self.complexity,
            entries=entries_tuple,
        )


@dataclass(frozen=True, slots=True)
class RuleCompiler:
    registry: DescriptorRegistry

    @property
    def compiler_hash(self) -> str:
        return _canonical_hash(
            {
                "schema_version": 1,
                "language": "materials-adsorption-expression",
                "python_ast_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "allowed_functions": sorted(_FUNCTIONS),
                "limits": {
                    "source_length": _MAX_SOURCE_LENGTH,
                    "ast_nodes": _MAX_AST_NODES,
                    "ast_depth": _MAX_AST_DEPTH,
                    "constants": _MAX_CONSTANTS,
                    "constant_magnitude": _MAX_CONSTANT_MAGNITUDE,
                    "intermediate_magnitude": _MAX_INTERMEDIATE_MAGNITUDE,
                },
                "complexity": "feature-expression-nodes-v1",
            }
        )

    def compile(
        self,
        source: str,
        *,
        applicable: str = "True",
        uncertainty: str = "0.0",
    ) -> CompiledRule:
        validator = _Validator(self.registry)
        rule_tree, normalized_rule, rule_complexity = validator.validate(source, "number")
        applicable_tree, normalized_applicable, _ = validator.validate(
            applicable, "boolean"
        )
        uncertainty_tree, normalized_uncertainty, _ = validator.validate(
            uncertainty, "number"
        )
        payload = {
            "schema_version": 1,
            "compiler_hash": self.compiler_hash,
            "descriptor_registry_hash": self.registry.registry_hash,
            "rule": normalized_rule,
            "applicable": normalized_applicable,
            "uncertainty": normalized_uncertainty,
        }
        return CompiledRule(
            rule_hash=_canonical_hash(payload),
            compiler_hash=self.compiler_hash,
            descriptor_registry_hash=self.registry.registry_hash,
            complexity=rule_complexity,
            normalized_rule=normalized_rule,
            normalized_applicable=normalized_applicable,
            normalized_uncertainty=normalized_uncertainty,
            referenced_descriptors=frozenset(validator.references),
            _registry=self.registry,
            _rule_tree=rule_tree,
            _applicable_tree=applicable_tree,
            _uncertainty_tree=uncertainty_tree,
        )
