"""Versioned optimization variables, problems, budgets, and run evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..units import normalize_unit, parse_unit


OPTIMIZATION_PROBLEM_SCHEMA = "circuit_ai.optimization_problem"
OPTIMIZATION_RESULT_SCHEMA = "circuit_ai.optimization_result"
OPTIMIZATION_SCHEMA_VERSION = 1


class OptimizationContractError(ValueError):
    pass


class VariableKind(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    DERIVED = "derived"


class VariableScale(str, Enum):
    LINEAR = "linear"
    LOG10 = "log10"


class OptimizationStatus(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


@dataclass(frozen=True)
class OptimizationVariable:
    variable_id: str
    kind: VariableKind
    unit: str = "1"
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()
    initial: Any = None
    scale: VariableScale = VariableScale.LINEAR
    expression: str | None = None
    source_path: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variable_id.strip():
            raise OptimizationContractError("optimization variable id must not be empty")
        kind = self.kind if isinstance(self.kind, VariableKind) else VariableKind(self.kind)
        scale = self.scale if isinstance(self.scale, VariableScale) else VariableScale(self.scale)
        unit = normalize_unit(self.unit)
        parse_unit(unit)
        choices = tuple(self.choices)
        if kind in {VariableKind.CONTINUOUS, VariableKind.INTEGER}:
            if self.lower is None or self.upper is None:
                raise OptimizationContractError(
                    f"variable {self.variable_id!r} requires finite lower and upper bounds"
                )
            lower, upper = float(self.lower), float(self.upper)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise OptimizationContractError(
                    f"variable {self.variable_id!r} has invalid bounds"
                )
            if scale is VariableScale.LOG10 and lower <= 0.0:
                raise OptimizationContractError(
                    f"log-scaled variable {self.variable_id!r} requires positive bounds"
                )
            if kind is VariableKind.INTEGER:
                if scale is not VariableScale.LINEAR:
                    raise OptimizationContractError("integer variables currently require linear scale")
                if math.ceil(lower) > math.floor(upper):
                    raise OptimizationContractError(
                        f"integer variable {self.variable_id!r} has no integral value in its bounds"
                    )
            if choices:
                raise OptimizationContractError("numeric variables must not define choices")
        elif kind is VariableKind.CATEGORICAL:
            if not choices:
                raise OptimizationContractError("categorical variables require at least one choice")
            if self.lower is not None or self.upper is not None:
                raise OptimizationContractError("categorical variables use choices instead of bounds")
            if scale is not VariableScale.LINEAR:
                raise OptimizationContractError("categorical variables currently require linear scale")
        else:
            if not str(self.expression or "").strip():
                raise OptimizationContractError("derived variables require an expression")
            if self.lower is not None or self.upper is not None or choices:
                raise OptimizationContractError(
                    "derived variables must not define optimizer bounds or choices"
                )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))
        if self.initial is not None:
            self.validate_value(self.initial)

    @property
    def fixed(self) -> bool:
        if self.kind in {VariableKind.CONTINUOUS, VariableKind.INTEGER}:
            return float(self.lower) == float(self.upper)
        return self.kind is VariableKind.CATEGORICAL and len(self.choices) == 1

    @property
    def decision(self) -> bool:
        return self.kind is not VariableKind.DERIVED and not self.fixed

    @property
    def encoded_bounds(self) -> tuple[float, float]:
        if self.kind is VariableKind.CATEGORICAL:
            return 0.0, float(len(self.choices) - 1)
        if self.kind is VariableKind.DERIVED:
            raise OptimizationContractError("derived variables have no encoded bounds")
        assert self.lower is not None and self.upper is not None
        if self.scale is VariableScale.LOG10:
            return math.log10(self.lower), math.log10(self.upper)
        return float(self.lower), float(self.upper)

    @property
    def integral_coordinate(self) -> bool:
        return self.kind in {VariableKind.INTEGER, VariableKind.CATEGORICAL}

    def default_value(self) -> Any:
        if self.initial is not None:
            return self.initial
        if self.kind is VariableKind.CATEGORICAL:
            return self.choices[0]
        if self.kind is VariableKind.DERIVED:
            return None
        assert self.lower is not None and self.upper is not None
        if self.kind is VariableKind.INTEGER:
            return int(round((math.ceil(self.lower) + math.floor(self.upper)) / 2.0))
        if self.scale is VariableScale.LOG10:
            return float(math.sqrt(self.lower * self.upper))
        return float((self.lower + self.upper) / 2.0)

    def encode(self, value: Any) -> float:
        self.validate_value(value)
        if self.kind is VariableKind.CATEGORICAL:
            return float(self.choices.index(value))
        if self.kind is VariableKind.DERIVED:
            raise OptimizationContractError("derived variables are not encoded")
        numeric = float(value)
        return math.log10(numeric) if self.scale is VariableScale.LOG10 else numeric

    def decode(self, coordinate: float) -> Any:
        if not math.isfinite(float(coordinate)):
            raise OptimizationContractError("optimizer coordinates must be finite")
        if self.kind is VariableKind.CATEGORICAL:
            index = int(round(float(coordinate)))
            if not 0 <= index < len(self.choices):
                raise OptimizationContractError(
                    f"categorical coordinate for {self.variable_id!r} lies outside its choices"
                )
            return self.choices[index]
        if self.kind is VariableKind.DERIVED:
            raise OptimizationContractError("derived variables are not decoded from coordinates")
        encoded_lower, encoded_upper = self.encoded_bounds
        coordinate = float(coordinate)
        coordinate_tolerance = 1e-12 * max(1.0, abs(encoded_lower), abs(encoded_upper))
        if (
            coordinate < encoded_lower - coordinate_tolerance
            or coordinate > encoded_upper + coordinate_tolerance
        ):
            raise OptimizationContractError(
                f"optimizer coordinate for {self.variable_id!r} lies outside its bounds"
            )
        coordinate = min(max(coordinate, encoded_lower), encoded_upper)
        value = 10.0 ** coordinate if self.scale is VariableScale.LOG10 else coordinate
        assert self.lower is not None and self.upper is not None
        if math.isclose(value, self.lower, rel_tol=1e-12, abs_tol=0.0):
            value = self.lower
        elif math.isclose(value, self.upper, rel_tol=1e-12, abs_tol=0.0):
            value = self.upper
        if self.kind is VariableKind.INTEGER:
            value = int(round(value))
        self.validate_value(value)
        return value

    def validate_value(self, value: Any, *, tolerance: float = 0.0) -> None:
        if self.kind is VariableKind.CATEGORICAL:
            if value not in self.choices:
                raise OptimizationContractError(
                    f"{value!r} is not a choice for variable {self.variable_id!r}"
                )
            return
        if self.kind is VariableKind.DERIVED:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OptimizationContractError(f"variable {self.variable_id!r} requires a number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise OptimizationContractError(f"variable {self.variable_id!r} requires a finite value")
        assert self.lower is not None and self.upper is not None
        if numeric < self.lower - tolerance or numeric > self.upper + tolerance:
            raise OptimizationContractError(
                f"variable {self.variable_id!r} value {numeric} lies outside its bounds"
            )
        if self.kind is VariableKind.INTEGER and not math.isclose(numeric, round(numeric), abs_tol=tolerance):
            raise OptimizationContractError(f"variable {self.variable_id!r} requires an integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable_id": self.variable_id,
            "kind": self.kind.value,
            "unit": self.unit,
            "lower": self.lower,
            "upper": self.upper,
            "choices": list(self.choices),
            "initial": self.initial,
            "scale": self.scale.value,
            "expression": self.expression,
            "source_path": self.source_path,
            "attributes": _thaw(self.attributes),
        }


DerivedResolver = Callable[[OptimizationVariable, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class OptimizationProblem:
    problem_id: str
    variables: tuple[OptimizationVariable, ...]
    objective_ids: tuple[str, ...]
    constraint_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = OPTIMIZATION_PROBLEM_SCHEMA
    schema_version: int = OPTIMIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != OPTIMIZATION_PROBLEM_SCHEMA or self.schema_version != OPTIMIZATION_SCHEMA_VERSION:
            raise OptimizationContractError("unsupported OptimizationProblem schema")
        if not self.problem_id.strip() or not self.objective_ids:
            raise OptimizationContractError("optimization problem id and objectives are required")
        if len({item.variable_id for item in self.variables}) != len(self.variables):
            raise OptimizationContractError("optimization variable ids must be unique")
        if len(set(self.objective_ids)) != len(self.objective_ids):
            raise OptimizationContractError("optimization objective ids must be unique")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def decision_variables(self) -> tuple[OptimizationVariable, ...]:
        return tuple(item for item in self.variables if item.decision)

    @property
    def encoded_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(item.encoded_bounds for item in self.decision_variables)

    @property
    def integrality(self) -> tuple[bool, ...]:
        return tuple(item.integral_coordinate for item in self.decision_variables)

    @property
    def has_explicit_initial(self) -> bool:
        return any(item.initial is not None for item in self.decision_variables)

    @property
    def initial_vector(self) -> tuple[float, ...]:
        return tuple(item.encode(item.default_value()) for item in self.decision_variables)

    def encode(self, values: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple(item.encode(values[item.variable_id]) for item in self.decision_variables)

    def decode(
        self,
        coordinates: Sequence[float],
        *,
        derived_resolver: DerivedResolver | None = None,
    ) -> dict[str, Any]:
        decision = self.decision_variables
        if len(coordinates) != len(decision):
            raise OptimizationContractError(
                f"problem expects {len(decision)} optimizer coordinates, got {len(coordinates)}"
            )
        values: dict[str, Any] = {}
        for variable in self.variables:
            if variable.fixed:
                values[variable.variable_id] = variable.default_value()
        for variable, coordinate in zip(decision, coordinates):
            values[variable.variable_id] = variable.decode(float(coordinate))
        if derived_resolver is not None:
            for variable in self.variables:
                if variable.kind is VariableKind.DERIVED:
                    values[variable.variable_id] = derived_resolver(variable, MappingProxyType(values))
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "variables": [item.as_dict() for item in self.variables],
            "objective_ids": list(self.objective_ids),
            "constraint_ids": list(self.constraint_ids),
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationBudget:
    max_evaluations: int | None = None
    max_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise OptimizationContractError("max_evaluations must be positive")
        if self.max_time_s is not None and (not math.isfinite(self.max_time_s) or self.max_time_s <= 0.0):
            raise OptimizationContractError("max_time_s must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_evaluations": self.max_evaluations,
            "max_time_s": self.max_time_s,
        }


@dataclass(frozen=True)
class OptimizationRunResult:
    problem_id: str
    solver_id: str
    solver_version: str
    status: OptimizationStatus
    success: bool
    values: Mapping[str, Any]
    objective_values: Mapping[str, float]
    search_objective: float | None
    truth_objective: float | None
    evaluations: int
    truth_evaluations: int
    cache_hits: int
    duration_s: float
    message: str
    budget: EvaluationBudget = field(default_factory=EvaluationBudget)
    schema: str = OPTIMIZATION_RESULT_SCHEMA
    schema_version: int = OPTIMIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, OptimizationStatus) else OptimizationStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", _freeze_mapping(self.values))
        object.__setattr__(self, "objective_values", _freeze_mapping(self.objective_values))

    @property
    def truth_evaluated(self) -> bool:
        return self.truth_evaluations > 0 and self.truth_objective is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "solver_id": self.solver_id,
            "solver_version": self.solver_version,
            "status": self.status.value,
            "success": self.success,
            "values": _thaw(self.values),
            "objective_values": _thaw(self.objective_values),
            "search_objective": self.search_objective,
            "truth_objective": self.truth_objective,
            "truth_evaluated": self.truth_evaluated,
            "evaluations": self.evaluations,
            "truth_evaluations": self.truth_evaluations,
            "cache_hits": self.cache_hits,
            "duration_s": self.duration_s,
            "message": self.message,
            "budget": self.budget.as_dict(),
        }


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_thaw(item) for item in value]
    return value
