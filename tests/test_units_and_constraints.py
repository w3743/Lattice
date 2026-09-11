from __future__ import annotations

from hypothesis import given, strategies as st
import pytest

from circuit_ai.constraints import (
    ConstraintEvaluator,
    ConstraintOperator,
    ConstraintSeverity,
    ConstraintSpec,
    ConstraintStatus,
    FeasibilityStatus,
    ToleranceSpec,
)
from circuit_ai.metrics import MetricSet, MetricSpec, MetricStatus, MetricValue
from circuit_ai.units import UnitContractError, convert_scalar, normalize_unit, units_compatible


def _metric_spec(metric_id: str = "port.output.voltage") -> MetricSpec:
    return MetricSpec(
        metric_id=metric_id,
        provider_id="simulation.scalar",
        source="output_voltage_v",
        quantity="voltage",
        unit="V",
    )


def _value(value: float, unit: str = "V", *, status=MetricStatus.AVAILABLE) -> MetricValue:
    return MetricValue(
        metric_id="port.output.voltage",
        status=status,
        value=value if status is MetricStatus.AVAILABLE else None,
        unit=unit,
        provider_id="simulation.scalar",
        source="output_voltage_v",
        diagnostics=() if status is MetricStatus.AVAILABLE else ("not produced",),
    )


def test_project_unit_aliases_and_dimensional_compatibility() -> None:
    assert normalize_unit("mm2") == "mm**2"
    assert convert_scalar(1000.0, "mV", "V") == pytest.approx(1.0)
    assert convert_scalar(1.0, "cm2", "mm2") == pytest.approx(100.0)
    assert units_compatible("CNY", "CNY")
    assert not units_compatible("V", "A")
    with pytest.raises(UnitContractError):
        convert_scalar(1.0, "V", "A")


def test_logarithmic_units_must_match_exactly() -> None:
    assert convert_scalar(-3.0, "dB", "dB") == -3.0
    with pytest.raises(UnitContractError):
        convert_scalar(-3.0, "dB", "1")


def test_metric_spec_round_trip_preserves_normalized_units() -> None:
    spec = MetricSpec(
        metric_id="resource.area",
        provider_id="graph.resource",
        source="area",
        quantity="area",
        unit="mm2",
        selectors={"exclude_roles": ["external_load"]},
    )
    restored = MetricSpec.from_dict(spec.as_dict())
    assert restored == spec
    assert restored.unit == "mm**2"


def test_equal_constraint_converts_units_and_combines_tolerances() -> None:
    spec = ConstraintSpec(
        constraint_id="output_voltage",
        metric=_metric_spec(),
        operator=ConstraintOperator.EQUAL,
        value=1.0,
        unit="V",
        tolerance=ToleranceSpec(absolute=10.0, relative=0.01, unit="mV"),
    )
    report = ConstraintEvaluator().evaluate(
        (spec,), MetricSet({"port.output.voltage": _value(1019.0, "mV")})
    )
    evaluation = report.evaluations[0]
    assert report.passed
    assert evaluation.effective_tolerance == pytest.approx(0.02)
    assert evaluation.actual == pytest.approx(1.019)


@pytest.mark.parametrize(
    ("operator", "kwargs", "actual", "expected"),
    [
        (ConstraintOperator.MINIMUM, {"minimum": 5.0}, 4.0, ConstraintStatus.FAILED),
        (ConstraintOperator.MAXIMUM, {"maximum": 5.0}, 4.0, ConstraintStatus.PASSED),
        (
            ConstraintOperator.INTERVAL,
            {"minimum": 4.0, "maximum": 6.0},
            5.0,
            ConstraintStatus.PASSED,
        ),
    ],
)
def test_bound_operators(operator, kwargs, actual, expected) -> None:
    spec = ConstraintSpec(
        constraint_id="bound",
        metric=_metric_spec(),
        operator=operator,
        **kwargs,
    )
    report = ConstraintEvaluator().evaluate(
        (spec,), MetricSet({"port.output.voltage": _value(actual)})
    )
    assert report.evaluations[0].status is expected


def test_hard_soft_and_objective_have_distinct_semantics() -> None:
    metric = _metric_spec()
    constraints = (
        ConstraintSpec("hard", metric, ConstraintOperator.MINIMUM, minimum=9.0),
        ConstraintSpec(
            "soft",
            metric,
            ConstraintOperator.MAXIMUM,
            severity=ConstraintSeverity.SOFT,
            maximum=9.5,
            weight=2.0,
        ),
        ConstraintSpec(
            "minimize_voltage",
            metric,
            ConstraintOperator.MINIMIZE,
            severity=ConstraintSeverity.OBJECTIVE,
            weight=0.5,
        ),
    )
    report = ConstraintEvaluator().evaluate(
        constraints, MetricSet({metric.metric_id: _value(10.0)})
    )
    assert report.feasibility is FeasibilityStatus.VERIFIED_FEASIBLE
    assert report.soft_penalty > 0.0
    assert report.objective_values["minimize_voltage"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("metric_status", "constraint_status", "feasibility"),
    [
        (MetricStatus.MISSING, ConstraintStatus.MISSING, FeasibilityStatus.INDETERMINATE),
        (
            MetricStatus.UNSUPPORTED,
            ConstraintStatus.UNSUPPORTED,
            FeasibilityStatus.INDETERMINATE,
        ),
        (MetricStatus.ERROR, ConstraintStatus.ERROR, FeasibilityStatus.EXECUTION_FAILED),
    ],
)
def test_unavailable_hard_metric_is_never_silently_passed(
    metric_status, constraint_status, feasibility
) -> None:
    spec = ConstraintSpec(
        "output_voltage",
        _metric_spec(),
        ConstraintOperator.EQUAL,
        value=10.0,
    )
    metric = _value(0.0, status=metric_status)
    report = ConstraintEvaluator().evaluate((spec,), MetricSet({metric.metric_id: metric}))
    assert not report.passed
    assert report.evaluations[0].status is constraint_status
    assert report.feasibility is feasibility


def test_constraint_order_does_not_change_report_semantics() -> None:
    metric = _metric_spec()
    first = ConstraintSpec("a", metric, ConstraintOperator.MINIMUM, minimum=9.0)
    second = ConstraintSpec("b", metric, ConstraintOperator.MAXIMUM, maximum=11.0)
    values = MetricSet({metric.metric_id: _value(10.0)})
    forward = ConstraintEvaluator().evaluate((first, second), values).as_dict()
    reverse = ConstraintEvaluator().evaluate((second, first), values).as_dict()
    assert forward == reverse


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_voltage_conversion_round_trip(value: float) -> None:
    assert convert_scalar(convert_scalar(value, "V", "mV"), "mV", "V") == pytest.approx(value)
