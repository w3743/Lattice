"""Solver-independent simulation tasks backed by the v1 metric contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .constraints import (
    ConstraintEvaluator,
    ConstraintSpec,
    ConstraintStatus,
    compile_constraint_program,
)
from .ir import UnifiedIR
from .metrics import (
    MetricContext,
    MetricEngine,
    MetricSet,
    MetricSpec,
    MetricStatus,
    MetricValue,
)
from .simulation.contracts import SimulationResult


@dataclass(frozen=True)
class SimulationTask:
    name: str
    analysis_kind: str
    source_port: str | None
    response_port: str | None
    conditions: dict[str, Any]
    sweep: dict[str, Any]
    metrics: tuple[MetricSpec, ...]
    constraints: tuple[ConstraintSpec, ...] = ()
    excitations: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "analysis_kind": self.analysis_kind,
            "source_port": self.source_port,
            "response_port": self.response_port,
            "conditions": dict(self.conditions),
            "sweep": dict(self.sweep),
            "metrics": [metric.as_dict() for metric in self.metrics],
            "constraints": [constraint.as_dict() for constraint in self.constraints],
            "excitations": [dict(item) for item in self.excitations],
        }

    def to_requests(
        self,
        graph,
        *,
        parameter_values: Mapping[str, Any] | None = None,
        fidelity: str = "unspecified",
    ):
        from .simulation import simulation_requests_for_task

        return simulation_requests_for_task(
            self,
            graph,
            parameter_values=parameter_values,
            fidelity=fidelity,
        )


@dataclass(frozen=True)
class MetricObservation:
    name: str
    value: float | None
    unit: str
    passed: bool
    message: str
    status: str = "passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "passed": self.passed,
            "message": self.message,
            "status": self.status,
        }


@dataclass(frozen=True)
class SimulationTaskEvaluation:
    task_name: str
    analysis_kind: str
    passed: bool
    observations: tuple[MetricObservation, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "analysis_kind": self.analysis_kind,
            "passed": self.passed,
            "observations": [item.as_dict() for item in self.observations],
            "issues": list(self.issues),
        }


def simulation_tasks_from_ir(ir: UnifiedIR) -> tuple[SimulationTask, ...]:
    program = compile_constraint_program(ir)
    tasks: list[SimulationTask] = []
    for index, analysis in enumerate(ir.analyses):
        target = ir.targets[index] if index < len(ir.targets) else {}
        kind = str(analysis.get("kind", "unknown"))
        source_port = _optional_text(analysis.get("source_port"))
        response_port = _optional_text(
            analysis.get("output_port", analysis.get("response_port", analysis.get("port")))
        )
        constraints = _task_constraints(program.constraints, index, kind)
        metrics = tuple(
            {item.metric.metric_id: item.metric for item in constraints}[key]
            for key in sorted({item.metric.metric_id for item in constraints})
        )
        sweep = {
            key: value
            for key, value in analysis.items()
            if key not in {"kind", "source_port", "output_port", "response_port", "port"}
        }
        tasks.append(
            SimulationTask(
                name=f"{ir.name}__{index + 1}_{kind}",
                analysis_kind=kind,
                source_port=source_port,
                response_port=response_port,
                conditions=dict(ir.operating_point),
                sweep=sweep,
                metrics=metrics,
                constraints=constraints,
                excitations=_task_excitations(ir, source_port, target, kind),
            )
        )
    return tuple(tasks)


def evaluate_simulation_task(
    task: SimulationTask,
    measurements: Mapping[str, Any] | Any,
) -> SimulationTaskEvaluation:
    results = _simulation_results(measurements)
    if results is not None:
        metric_set = MetricEngine().extract(
            task.metrics,
            MetricContext(simulation_results=results),
        )
    else:
        values = _measurement_mapping(measurements)
        metric_values: dict[str, MetricValue] = {}
        for metric in task.metrics:
            raw = values.get(metric.source)
            try:
                actual = float(raw)
            except (TypeError, ValueError):
                actual = None
            available = actual is not None and math.isfinite(actual)
            metric_values[metric.metric_id] = MetricValue(
                metric_id=metric.metric_id,
                status=MetricStatus.AVAILABLE if available else MetricStatus.MISSING,
                value=actual if available else None,
                unit=metric.unit,
                provider_id=metric.provider_id,
                source=metric.source,
                diagnostics=() if available else (f"measurement {metric.source!r} is missing or non-finite",),
            )
        metric_set = MetricSet(metric_values)
    report = ConstraintEvaluator().evaluate(task.constraints, metric_set)
    observations = tuple(
        MetricObservation(
            name=item.constraint_id,
            value=item.actual,
            unit=item.actual_unit,
            passed=item.passed,
            message=item.message,
            status=item.status.value,
        )
        for item in report.evaluations
    )
    issues = tuple(
        item.message
        for item in report.evaluations
        if item.status
        not in {ConstraintStatus.PASSED, ConstraintStatus.OBSERVED}
    )
    complete = not any(
        item.status in {
            ConstraintStatus.MISSING,
            ConstraintStatus.UNSUPPORTED,
            ConstraintStatus.ERROR,
        }
        for item in report.evaluations
    )
    return SimulationTaskEvaluation(
        task_name=task.name,
        analysis_kind=task.analysis_kind,
        passed=report.passed and complete,
        observations=observations,
        issues=issues,
    )


def _task_constraints(
    constraints: tuple[ConstraintSpec, ...],
    target_index: int,
    analysis_kind: str,
) -> tuple[ConstraintSpec, ...]:
    normalized = _normalize_analysis_kind(analysis_kind)
    simulation_providers = {
        "simulation.scalar",
        "dc_port_metrics",
        "frequency_response",
        "transient_ripple",
        "device_stress",
    }
    selected = []
    for constraint in constraints:
        if constraint.metric.provider_id not in simulation_providers:
            continue
        target_owned = constraint.source_path.startswith(f"targets[{target_index}]")
        analysis_owned = constraint.metric.analysis_kind == normalized
        if target_owned or analysis_owned:
            selected.append(constraint)
    return tuple(sorted(selected, key=lambda item: item.constraint_id))


def _simulation_results(value: Any) -> tuple[SimulationResult, ...] | None:
    if isinstance(value, SimulationResult):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, SimulationResult) for item in value):
        return tuple(value)
    return None


def _measurement_mapping(measurements: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(measurements, Mapping):
        return dict(measurements)
    if isinstance(measurements, (tuple, list)):
        merged: dict[str, Any] = {}
        for item in measurements:
            merged.update(_measurement_mapping(item))
        return merged
    if hasattr(measurements, "scalars"):
        return {
            str(key): getattr(quantity, "value", quantity)
            for key, quantity in measurements.scalars.items()
        }
    if hasattr(measurements, "as_dict"):
        return dict(measurements.as_dict())
    return dict(vars(measurements))


def _normalize_analysis_kind(kind: str) -> str:
    return {
        "dc_operating_point": "dc_transfer",
        "voltage_transfer": "small_signal_ac",
        "transimpedance": "small_signal_ac",
        "input_impedance": "small_signal_ac",
        "output_impedance": "small_signal_ac",
        "impedance": "small_signal_ac",
    }.get(kind, kind)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _task_excitations(
    ir: UnifiedIR,
    source_port: str | None,
    target: Mapping[str, Any],
    analysis_kind: str,
) -> tuple[dict[str, Any], ...]:
    excitations: list[dict[str, Any]] = []
    port = next((item for item in ir.ports if item.name == source_port), None)
    if port is not None and port.excitation:
        waveform = port.excitation.get("waveform", port.excitation)
        if isinstance(waveform, Mapping) and waveform.get("kind"):
            excitations.append(
                {
                    "id": f"{source_port}_{waveform['kind']}",
                    "kind": str(waveform["kind"]),
                    "target_port": source_port,
                    **{key: value for key, value in waveform.items() if key != "kind"},
                }
            )
    if analysis_kind in {"dc_transfer", "dc_operating_point"} and target.get(
        "input_voltage_v"
    ) is not None:
        return (
            {
                "id": f"{source_port or 'input'}_dc",
                "kind": "dc_voltage",
                "target_port": source_port or "input",
                "value": float(target["input_voltage_v"]),
                "unit": "V",
            },
        )
    return tuple(excitations)


__all__ = [
    "MetricObservation",
    "MetricSpec",
    "SimulationTask",
    "SimulationTaskEvaluation",
    "evaluate_simulation_task",
    "simulation_tasks_from_ir",
]
