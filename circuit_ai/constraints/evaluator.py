"""Generic unit-aware comparison of metrics against declarative constraints."""

from __future__ import annotations

from typing import Iterable

from ..metrics import MetricSet, MetricStatus, MetricValue
from ..units import UnitContractError, convert_scalar
from .contracts import (
    ConstraintEvaluation,
    ConstraintOperator,
    ConstraintReport,
    ConstraintSeverity,
    ConstraintSpec,
    ConstraintStatus,
    FeasibilityStatus,
    constraint_status_for_metric,
)


class ConstraintEvaluator:
    def evaluate(
        self,
        constraints: Iterable[ConstraintSpec],
        metrics: MetricSet,
    ) -> ConstraintReport:
        specs = tuple(sorted(constraints, key=lambda item: item.constraint_id))
        if len({item.constraint_id for item in specs}) != len(specs):
            raise ValueError("constraint ids must be unique")
        evaluations = tuple(self._evaluate_one(spec, metrics.get(spec.metric.metric_id)) for spec in specs)
        hard = tuple(item for item in evaluations if item.severity is ConstraintSeverity.HARD)
        unavailable = tuple(
            item
            for item in hard
            if item.status in {
                ConstraintStatus.MISSING,
                ConstraintStatus.UNSUPPORTED,
                ConstraintStatus.ERROR,
            }
        )
        if any(item.status is ConstraintStatus.ERROR for item in unavailable):
            feasibility = FeasibilityStatus.EXECUTION_FAILED
        elif unavailable:
            feasibility = FeasibilityStatus.INDETERMINATE
        elif any(item.status is ConstraintStatus.FAILED for item in hard):
            feasibility = FeasibilityStatus.KNOWN_INFEASIBLE
        else:
            feasibility = FeasibilityStatus.VERIFIED_FEASIBLE
        soft_penalty = float(
            sum(
                item.weighted_penalty or 0.0
                for item in evaluations
                if item.severity is ConstraintSeverity.SOFT
            )
        )
        objectives = {
            item.constraint_id: float(item.weighted_penalty)
            for item in evaluations
            if item.severity is ConstraintSeverity.OBJECTIVE
            and item.weighted_penalty is not None
        }
        return ConstraintReport(
            evaluations=evaluations,
            feasibility=feasibility,
            metric_values=metrics.values,
            soft_penalty=soft_penalty,
            objective_values=objectives,
            diagnostics=metrics.diagnostics,
        )

    def _evaluate_one(
        self,
        spec: ConstraintSpec,
        metric: MetricValue | None,
    ) -> ConstraintEvaluation:
        requirement = {
            "value": spec.value,
            "minimum": spec.minimum,
            "maximum": spec.maximum,
        }
        if metric is None:
            metric = MetricValue(
                metric_id=spec.metric.metric_id,
                status=MetricStatus.MISSING,
                value=None,
                unit=spec.metric.unit,
                provider_id=spec.metric.provider_id,
                source=spec.metric.source,
                diagnostics=("metric was not produced",),
            )
        if not metric.available:
            return self._unavailable(spec, metric, requirement, constraint_status_for_metric(metric))
        if metric.evidence_rank < spec.required_evidence_rank:
            return self._unavailable(
                spec,
                metric,
                requirement,
                ConstraintStatus.UNSUPPORTED,
                message=(
                    f"metric evidence rank {metric.evidence_rank} is below required "
                    f"rank {spec.required_evidence_rank}"
                ),
            )
        try:
            actual = convert_scalar(float(metric.value), metric.unit, spec.unit)
            tolerance = self._effective_tolerance(spec)
        except UnitContractError as exc:
            return self._unavailable(
                spec,
                metric,
                requirement,
                ConstraintStatus.ERROR,
                message=str(exc),
            )
        status, violation, message = self._compare(spec, actual, tolerance)
        penalty = self._penalty(spec, actual, violation)
        return ConstraintEvaluation(
            constraint_id=spec.constraint_id,
            metric_id=spec.metric.metric_id,
            severity=spec.severity,
            operator=spec.operator,
            status=status,
            actual=actual,
            actual_unit=spec.unit,
            requirement=requirement,
            requirement_unit=spec.unit,
            effective_tolerance=tolerance,
            normalized_violation=violation,
            weighted_penalty=penalty,
            source_path=spec.source_path,
            provider_id=metric.provider_id,
            evidence_level=metric.evidence_level,
            evidence_rank=metric.evidence_rank,
            evidence=tuple(item.as_dict() for item in metric.evidence),
            message=message,
        )

    def _effective_tolerance(self, spec: ConstraintSpec) -> float:
        reference = spec.value
        if reference is None and spec.minimum is not None and spec.maximum is not None:
            reference = max(abs(spec.minimum), abs(spec.maximum))
        elif reference is None:
            reference = spec.minimum if spec.minimum is not None else spec.maximum
        reference = abs(float(reference or 0.0))
        absolute = spec.tolerance.absolute
        if spec.tolerance.unit is not None:
            absolute = convert_scalar(absolute, spec.tolerance.unit, spec.unit)
        return float(absolute + spec.tolerance.relative * reference)

    def _compare(
        self,
        spec: ConstraintSpec,
        actual: float,
        tolerance: float,
    ) -> tuple[ConstraintStatus, float, str]:
        op = spec.operator
        if op is ConstraintOperator.MINIMIZE:
            return ConstraintStatus.OBSERVED, 0.0, f"observed {actual:.9g} {spec.unit}; minimize"
        if op is ConstraintOperator.MAXIMIZE:
            return ConstraintStatus.OBSERVED, 0.0, f"observed {actual:.9g} {spec.unit}; maximize"
        if op is ConstraintOperator.EQUAL:
            assert spec.value is not None
            excess = max(0.0, abs(actual - spec.value) - tolerance)
            scale = max(abs(spec.value), tolerance, 1e-12)
            passed = excess == 0.0
            message = (
                f"actual={actual:.9g} {spec.unit}, required={spec.value:.9g} {spec.unit}, "
                f"tolerance={tolerance:.9g} {spec.unit}"
            )
        elif op is ConstraintOperator.MINIMUM:
            assert spec.minimum is not None
            excess = max(0.0, spec.minimum - actual - tolerance)
            scale = max(abs(spec.minimum), tolerance, 1e-12)
            passed = excess == 0.0
            message = (
                f"actual={actual:.9g} {spec.unit}, minimum={spec.minimum:.9g} {spec.unit}, "
                f"tolerance={tolerance:.9g} {spec.unit}"
            )
        elif op is ConstraintOperator.MAXIMUM:
            assert spec.maximum is not None
            excess = max(0.0, actual - spec.maximum - tolerance)
            scale = max(abs(spec.maximum), tolerance, 1e-12)
            passed = excess == 0.0
            message = (
                f"actual={actual:.9g} {spec.unit}, maximum={spec.maximum:.9g} {spec.unit}, "
                f"tolerance={tolerance:.9g} {spec.unit}"
            )
        else:
            assert spec.minimum is not None and spec.maximum is not None
            excess = max(0.0, spec.minimum - actual - tolerance, actual - spec.maximum - tolerance)
            scale = max(abs(spec.minimum), abs(spec.maximum), tolerance, 1e-12)
            passed = excess == 0.0
            message = (
                f"actual={actual:.9g} {spec.unit}, interval=[{spec.minimum:.9g}, "
                f"{spec.maximum:.9g}] {spec.unit}, tolerance={tolerance:.9g} {spec.unit}"
            )
        return (
            ConstraintStatus.PASSED if passed else ConstraintStatus.FAILED,
            float(excess / scale),
            message,
        )

    @staticmethod
    def _penalty(spec: ConstraintSpec, actual: float, violation: float) -> float:
        if spec.severity is ConstraintSeverity.OBJECTIVE:
            directed = actual if spec.operator is ConstraintOperator.MINIMIZE else -actual
            return float(spec.weight * directed)
        return float(spec.weight * violation)

    @staticmethod
    def _unavailable(
        spec: ConstraintSpec,
        metric: MetricValue,
        requirement,
        status: ConstraintStatus,
        *,
        message: str | None = None,
    ) -> ConstraintEvaluation:
        detail = message or "; ".join(metric.diagnostics) or f"metric status is {metric.status.value}"
        return ConstraintEvaluation(
            constraint_id=spec.constraint_id,
            metric_id=spec.metric.metric_id,
            severity=spec.severity,
            operator=spec.operator,
            status=status,
            actual=None,
            actual_unit=metric.unit,
            requirement=requirement,
            requirement_unit=spec.unit,
            effective_tolerance=0.0,
            normalized_violation=None,
            weighted_penalty=None,
            source_path=spec.source_path,
            provider_id=metric.provider_id,
            evidence_level=metric.evidence_level,
            evidence_rank=metric.evidence_rank,
            evidence=tuple(item.as_dict() for item in metric.evidence),
            message=detail,
        )
