"""Declarative, unit-safe constraint contracts and evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from ..metrics import MetricSpec, MetricStatus, MetricValue
from ..units import normalize_unit, parse_unit


CONSTRAINT_SPEC_SCHEMA = "circuit_ai.constraint_spec"
CONSTRAINT_REPORT_SCHEMA = "circuit_ai.constraint_report"
CONSTRAINT_SCHEMA_VERSION = 1


class ConstraintContractError(ValueError):
    pass


class ConstraintSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    OBJECTIVE = "objective"


class ConstraintOperator(str, Enum):
    EQUAL = "equal"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    INTERVAL = "interval"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    OBSERVED = "observed"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class FeasibilityStatus(str, Enum):
    VERIFIED_FEASIBLE = "verified_feasible"
    KNOWN_INFEASIBLE = "known_infeasible"
    INDETERMINATE = "indeterminate"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class ToleranceSpec:
    absolute: float = 0.0
    relative: float = 0.0
    unit: str | None = None
    policy: str = "explicit"

    def __post_init__(self) -> None:
        if not math.isfinite(self.absolute) or self.absolute < 0.0:
            raise ConstraintContractError("absolute tolerance must be finite and non-negative")
        if not math.isfinite(self.relative) or self.relative < 0.0:
            raise ConstraintContractError("relative tolerance must be finite and non-negative")
        if self.unit is not None:
            parse_unit(self.unit)
            object.__setattr__(self, "unit", normalize_unit(self.unit))
        if not self.policy.strip():
            raise ConstraintContractError("tolerance policy must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "absolute": self.absolute,
            "relative": self.relative,
            "unit": self.unit,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ToleranceSpec":
        data = data or {}
        return cls(
            absolute=float(data.get("absolute", 0.0)),
            relative=float(data.get("relative", 0.0)),
            unit=(str(data["unit"]) if data.get("unit") is not None else None),
            policy=str(data.get("policy", "explicit")),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    constraint_id: str
    metric: MetricSpec
    operator: ConstraintOperator
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    value: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    tolerance: ToleranceSpec = field(default_factory=ToleranceSpec)
    weight: float = 1.0
    source_path: str = ""
    description: str = ""
    required_evidence_rank: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CONSTRAINT_SPEC_SCHEMA
    schema_version: int = CONSTRAINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != CONSTRAINT_SPEC_SCHEMA or self.schema_version != CONSTRAINT_SCHEMA_VERSION:
            raise ConstraintContractError("unsupported ConstraintSpec schema")
        if not self.constraint_id.strip():
            raise ConstraintContractError("constraint id must not be empty")
        operator = self.operator if isinstance(self.operator, ConstraintOperator) else ConstraintOperator(self.operator)
        severity = self.severity if isinstance(self.severity, ConstraintSeverity) else ConstraintSeverity(self.severity)
        unit = normalize_unit(self.unit if self.unit is not None else self.metric.unit)
        parse_unit(unit)
        for label, number in (("value", self.value), ("minimum", self.minimum), ("maximum", self.maximum)):
            if number is not None and not math.isfinite(float(number)):
                raise ConstraintContractError(f"constraint {label} must be finite")
        if operator is ConstraintOperator.EQUAL and self.value is None:
            raise ConstraintContractError("equal constraint requires value")
        if operator is ConstraintOperator.MINIMUM and self.minimum is None:
            raise ConstraintContractError("minimum constraint requires minimum")
        if operator is ConstraintOperator.MAXIMUM and self.maximum is None:
            raise ConstraintContractError("maximum constraint requires maximum")
        if operator is ConstraintOperator.INTERVAL:
            if self.minimum is None or self.maximum is None or self.minimum > self.maximum:
                raise ConstraintContractError("interval constraint requires ordered minimum and maximum")
        if operator in {ConstraintOperator.MINIMIZE, ConstraintOperator.MAXIMIZE}:
            severity = ConstraintSeverity.OBJECTIVE
        if severity is ConstraintSeverity.OBJECTIVE and operator not in {
            ConstraintOperator.MINIMIZE,
            ConstraintOperator.MAXIMIZE,
        }:
            raise ConstraintContractError("objective constraints require minimize or maximize")
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ConstraintContractError("constraint weight must be finite and non-negative")
        if self.required_evidence_rank < 0:
            raise ConstraintContractError("required evidence rank must be non-negative")
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "attributes", dict(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "constraint_id": self.constraint_id,
            "metric": self.metric.as_dict(),
            "operator": self.operator.value,
            "severity": self.severity.value,
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "tolerance": self.tolerance.as_dict(),
            "weight": self.weight,
            "source_path": self.source_path,
            "description": self.description,
            "required_evidence_rank": self.required_evidence_rank,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ConstraintEvaluation:
    constraint_id: str
    metric_id: str
    severity: ConstraintSeverity
    operator: ConstraintOperator
    status: ConstraintStatus
    actual: float | None
    actual_unit: str
    requirement: Mapping[str, float | None]
    requirement_unit: str
    effective_tolerance: float
    normalized_violation: float | None
    weighted_penalty: float | None
    source_path: str
    provider_id: str
    evidence_level: str
    evidence_rank: int
    evidence: tuple[dict[str, Any], ...]
    message: str

    @property
    def passed(self) -> bool:
        return self.status in {ConstraintStatus.PASSED, ConstraintStatus.OBSERVED}

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "metric_id": self.metric_id,
            "severity": self.severity.value,
            "operator": self.operator.value,
            "status": self.status.value,
            "passed": self.passed,
            "required": dict(self.requirement),
            "required_unit": self.requirement_unit,
            "actual": self.actual,
            "actual_unit": self.actual_unit,
            "effective_tolerance": self.effective_tolerance,
            "normalized_violation": self.normalized_violation,
            "weighted_penalty": self.weighted_penalty,
            "source_path": self.source_path,
            "provider_id": self.provider_id,
            "evidence_level": self.evidence_level,
            "evidence_rank": self.evidence_rank,
            "evidence": list(self.evidence),
            "message": self.message,
        }


@dataclass(frozen=True)
class ConstraintReport:
    evaluations: tuple[ConstraintEvaluation, ...]
    feasibility: FeasibilityStatus
    metric_values: Mapping[str, MetricValue]
    soft_penalty: float = 0.0
    objective_values: Mapping[str, float] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    schema: str = CONSTRAINT_REPORT_SCHEMA
    schema_version: int = CONSTRAINT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.feasibility is FeasibilityStatus.VERIFIED_FEASIBLE

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(
            item.message
            for item in self.evaluations
            if item.severity is ConstraintSeverity.HARD and not item.passed
        )

    @property
    def checks(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": item.constraint_id,
                "passed": item.passed,
                "detail": item.message,
                "status": item.status.value,
            }
            for item in self.evaluations
        )

    @property
    def hard_violation(self) -> float:
        return float(
            sum(
                item.normalized_violation or 0.0
                for item in self.evaluations
                if item.severity is ConstraintSeverity.HARD
            )
        )

    @property
    def unavailable_hard_count(self) -> int:
        return sum(
            item.severity is ConstraintSeverity.HARD
            and item.status in {
                ConstraintStatus.MISSING,
                ConstraintStatus.UNSUPPORTED,
                ConstraintStatus.ERROR,
            }
            for item in self.evaluations
        )

    @property
    def feasibility_rank(self) -> int:
        return {
            FeasibilityStatus.VERIFIED_FEASIBLE: 0,
            FeasibilityStatus.KNOWN_INFEASIBLE: 1,
            FeasibilityStatus.INDETERMINATE: 2,
            FeasibilityStatus.EXECUTION_FAILED: 3,
        }[self.feasibility]

    @property
    def cost_cny(self) -> float:
        return self._metric_number("resource.bom_cost")

    @property
    def area_mm2(self) -> float:
        return self._metric_number("resource.area")

    def _metric_number(self, metric_id: str) -> float:
        metric = self.metric_values.get(metric_id)
        return float(metric.value) if metric is not None and metric.available else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "passed": self.passed,
            "feasibility": self.feasibility.value,
            "feasibility_rank": self.feasibility_rank,
            "hard_violation": self.hard_violation,
            "unavailable_hard_count": self.unavailable_hard_count,
            "soft_penalty": self.soft_penalty,
            "objective_values": dict(self.objective_values),
            "diagnostics": list(self.diagnostics),
            "evaluations": [item.as_dict() for item in self.evaluations],
            "checks": list(self.checks),
            "issues": list(self.issues),
            "cost_cny": self.cost_cny,
            "area_mm2": self.area_mm2,
            "metrics": {key: value.as_dict() for key, value in sorted(self.metric_values.items())},
        }


def constraint_status_for_metric(metric: MetricValue) -> ConstraintStatus:
    return {
        MetricStatus.MISSING: ConstraintStatus.MISSING,
        MetricStatus.UNSUPPORTED: ConstraintStatus.UNSUPPORTED,
        MetricStatus.ERROR: ConstraintStatus.ERROR,
    }.get(metric.status, ConstraintStatus.ERROR)
