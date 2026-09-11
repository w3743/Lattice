"""Compile specification-facing SimulationTask objects into backend requests."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..graph import CircuitGraph
from .contracts import (
    AnalysisSpec,
    Excitation,
    ObservableSpec,
    OperatingConditions,
    SimulationRequest,
    SweepSpec,
)


_AC_ANALYSES = {
    "voltage_transfer",
    "transimpedance",
    "impedance",
    "input_impedance",
    "output_impedance",
    "small_signal_ac",
}


def simulation_requests_for_task(
    task: Any,
    graph: CircuitGraph,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    fidelity: str = "unspecified",
) -> tuple[SimulationRequest, ...]:
    """Compile one task to one or more independently executable requests."""

    analysis_kind = _canonical_analysis_kind(task.analysis_kind)
    sweeps = _sweeps_from_task(task, analysis_kind)
    observables = _observables_from_task(task, analysis_kind)
    excitations = _excitations_from_task(task, analysis_kind)
    temperature = float(task.conditions.get("temperature_c", 25.0))
    corner = str(task.conditions.get("corner", "typical"))
    condition_variables = {
        key: value
        for key, value in task.conditions.items()
        if key not in {"temperature_c", "corner"}
    }
    request = SimulationRequest(
        request_id=f"{task.name}__request_1",
        analysis=AnalysisSpec(
            kind=analysis_kind,
            sweeps=sweeps,
            options={"source_analysis_kind": task.analysis_kind},
        ),
        graph_id=graph.graph_id,
        parameter_values=dict(parameter_values or {}),
        excitations=excitations,
        conditions=OperatingConditions(
            temperature_c=temperature,
            corner=corner,
            variables=condition_variables,
        ),
        requested_observables=observables,
        fidelity=fidelity,
        metadata={
            "simulation_task_name": task.name,
            "metric_sources": [item.source for item in task.metrics],
        },
    )
    return (request,)


def simulation_requests_for_tasks(
    tasks: Iterable[Any],
    graph: CircuitGraph,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    fidelity: str = "unspecified",
) -> tuple[SimulationRequest, ...]:
    return tuple(
        request
        for task in tasks
        for request in simulation_requests_for_task(
            task,
            graph,
            parameter_values=parameter_values,
            fidelity=fidelity,
        )
    )


def legacy_ac_simulation_request(
    graph: CircuitGraph,
    analysis_request: Any,
    frequencies_hz: Iterable[float],
    *,
    parameter_values: Mapping[str, Any] | None = None,
    request_id: str | None = None,
) -> SimulationRequest:
    """Adapt the legacy AC analysis object at the compatibility boundary."""

    quantity, observable_id, unit, target_port = {
        "voltage_transfer": ("voltage_gain", "voltage_transfer", "1", "output"),
        "transimpedance": ("transimpedance", "transimpedance", "ohm", "output"),
        "input_impedance": ("input_impedance", "input_impedance", "ohm", "input"),
        "output_impedance": ("output_impedance", "output_impedance", "ohm", "output"),
    }.get(
        analysis_request.kind,
        (analysis_request.kind, analysis_request.kind, "", "output"),
    )
    source_kind = (
        "ac_current"
        if analysis_request.kind in {"transimpedance", "output_impedance"}
        else "ac_voltage"
    )
    return SimulationRequest(
        request_id=request_id or f"{graph.graph_id}__small_signal_ac",
        analysis=AnalysisSpec(
            kind="small_signal_ac",
            sweeps=(
                SweepSpec(
                    variable="frequency",
                    unit="Hz",
                    values=tuple(float(item) for item in frequencies_hz),
                ),
            ),
            options={
                "source_analysis_kind": analysis_request.kind,
                "allow_native_sweep_approximation": True,
            },
        ),
        graph_id=graph.graph_id,
        parameter_values=_bound_parameter_values(graph, parameter_values),
        excitations=(
            Excitation(
                excitation_id=f"{analysis_request.source_name}_ac",
                kind=source_kind,
                target_kind="component",
                target_id=analysis_request.source_name,
                parameters={"magnitude": 1.0, "phase_deg": 0.0},
            ),
        ),
        requested_observables=(
            ObservableSpec(
                observable_id=observable_id,
                quantity=quantity,
                unit=unit,
                target_kind="port",
                target_id=target_port,
                source_id=analysis_request.source_name,
                representation="complex",
            ),
        ),
        fidelity="linear_frequency_domain",
        metadata={"compatibility_adapter": "AnalysisRequest"},
    )


def _canonical_analysis_kind(kind: str) -> str:
    return "small_signal_ac" if kind in _AC_ANALYSES else kind


def _bound_parameter_values(
    graph: CircuitGraph,
    parameter_values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep only request values that are actually bound by the graph.

    Legacy template graphs materialize optimized values directly in their
    ``ParameterBinding`` objects.  Passing the legacy parameter dictionary to
    the variable-oriented graph adapter made those fixed graphs fail with an
    ``unknown or unused`` diagnostic.  Variable-backed graphs still receive
    their explicitly bound values unchanged.
    """

    values = dict(parameter_values or {})
    bound = {
        parameter.variable_id
        for component in graph.components
        for parameter in component.parameters
        if parameter.variable_id is not None
    }
    return {key: value for key, value in values.items() if key in bound}


def _sweeps_from_task(task: Any, analysis_kind: str) -> tuple[SweepSpec, ...]:
    if analysis_kind == "small_signal_ac":
        values = task.sweep.get("frequency_hz", (10.0, 1e6, 160))
        return (_range_or_explicit_sweep("frequency", "Hz", values, scale="log"),)
    if analysis_kind in {"transient", "periodic_steady_state"}:
        values = task.sweep.get("time_s")
        if values is not None:
            return (_range_or_explicit_sweep("time", "s", values, scale="linear"),)
    dc_sweep = task.sweep.get("dc_sweep")
    if analysis_kind == "dc_transfer" and isinstance(dc_sweep, Mapping):
        return (
            SweepSpec(
                variable=str(dc_sweep.get("variable", "input_voltage")),
                unit=str(dc_sweep.get("unit", "V")),
                start=float(dc_sweep["start"]),
                stop=float(dc_sweep["stop"]),
                points=int(dc_sweep["points"]),
                scale=str(dc_sweep.get("scale", "linear")),
            ),
        )
    return ()


def _range_or_explicit_sweep(
    variable: str,
    unit: str,
    raw: Any,
    *,
    scale: str,
) -> SweepSpec:
    values = tuple(raw) if isinstance(raw, (list, tuple)) else (float(raw),)
    if len(values) == 3 and _integer_like(values[2]):
        return SweepSpec(
            variable=variable,
            unit=unit,
            start=float(values[0]),
            stop=float(values[1]),
            points=int(values[2]),
            scale=scale,
        )
    return SweepSpec(variable=variable, unit=unit, values=tuple(float(item) for item in values))


def _observables_from_task(task: Any, analysis_kind: str) -> tuple[ObservableSpec, ...]:
    source_port = task.source_port or "input"
    response_port = task.response_port or "output"
    if analysis_kind == "small_signal_ac":
        legacy_kind = task.analysis_kind
        if legacy_kind == "transimpedance":
            return (
                ObservableSpec(
                    "transimpedance",
                    "transimpedance",
                    "ohm",
                    "port",
                    response_port,
                    reference_id=source_port,
                    representation="complex",
                ),
            )
        if legacy_kind in {"impedance", "input_impedance", "output_impedance"}:
            observable_id = (
                legacy_kind
                if legacy_kind in {"input_impedance", "output_impedance"}
                else "output_impedance"
                if response_port == "output"
                else "input_impedance"
            )
            quantity = observable_id
            return (
                ObservableSpec(
                    observable_id,
                    quantity,
                    "ohm",
                    "port",
                    task.response_port or task.source_port or "input",
                    representation="complex",
                ),
            )
        return (
            ObservableSpec(
                "voltage_transfer",
                "voltage_gain",
                "1",
                "port",
                response_port,
                reference_id=source_port,
                representation="complex",
            ),
        )

    observables: list[ObservableSpec] = []
    for metric in task.metrics:
        target_id = source_port if metric.source.startswith("input_") else response_port
        quantity = _metric_quantity(metric.source)
        target_kind = "design" if quantity == "efficiency" else "port"
        if metric.source.startswith("inductor_"):
            target_kind, target_id = "component_role", "energy_storage"
        observables.append(
            ObservableSpec(
                observable_id=metric.source,
                quantity=quantity,
                unit=metric.unit,
                target_kind=target_kind,
                target_id=target_id,
                representation="scalar",
            )
        )
    return tuple(_unique_observables(observables))


def _metric_quantity(source: str) -> str:
    if source.endswith("voltage_v"):
        return "voltage"
    if source.endswith("current_a"):
        return "current"
    if source.endswith("power_w"):
        return "power"
    if "ripple" in source and source.endswith("mv"):
        return "voltage_ripple"
    if "ripple" in source and source.endswith("a"):
        return "current_ripple"
    return "efficiency" if source == "efficiency" else source


def _excitations_from_task(task: Any, analysis_kind: str) -> tuple[Excitation, ...]:
    source_port = task.source_port or "input"
    raw_excitations = tuple(getattr(task, "excitations", ()))
    if analysis_kind in {"dc_transfer", "dc_operating_point"}:
        for raw in raw_excitations:
            if str(raw.get("kind", "")) == "dc_voltage":
                return (_excitation_from_mapping(raw, source_port),)
        return ()

    compatible = [
        _excitation_from_mapping(raw, source_port)
        for raw in raw_excitations
        if str(raw.get("kind", "")) not in {"dc_voltage", "dc_current"}
    ]
    if compatible:
        return tuple(compatible)
    if analysis_kind == "small_signal_ac":
        current_driven = task.analysis_kind in {"transimpedance", "output_impedance"}
        return (
            Excitation(
                excitation_id="default_ac_source",
                kind="ac_current" if current_driven else "ac_voltage",
                target_kind="port",
                target_id=source_port,
                parameters={"magnitude": 1.0, "phase_deg": 0.0},
            ),
        )
    return ()


def _excitation_from_mapping(raw: Mapping[str, Any], default_port: str) -> Excitation:
    target_id = str(raw.get("target_port", raw.get("target_id", default_port)))
    parameters = {
        key: value
        for key, value in raw.items()
        if key not in {"id", "kind", "target_kind", "target_id", "target_port"}
    }
    return Excitation(
        excitation_id=str(raw.get("id", f"{raw.get('kind', 'source')}_{target_id}")),
        kind=str(raw.get("kind", "source")),
        target_kind=str(raw.get("target_kind", "port")),
        target_id=target_id,
        parameters=parameters,
    )


def _unique_observables(items: list[ObservableSpec]) -> list[ObservableSpec]:
    result: list[ObservableSpec] = []
    seen: set[str] = set()
    for item in items:
        if item.observable_id not in seen:
            result.append(item)
            seen.add(item.observable_id)
    return result


def _integer_like(value: Any) -> bool:
    try:
        return float(value).is_integer() and int(value) > 0
    except (TypeError, ValueError):
        return False
