"""Compile PBDL/UnifiedIR requirements into metric queries and constraints."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..graph import CircuitGraph
from ..ir import IRPort, UnifiedIR
from ..metrics import MetricSpec
from .contracts import (
    ConstraintOperator,
    ConstraintSeverity,
    ConstraintSpec,
    ToleranceSpec,
)


class ConstraintCompilationError(ValueError):
    pass


@dataclass(frozen=True)
class ConstraintProgram:
    metrics: tuple[MetricSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    diagnostics: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": [item.as_dict() for item in self.metrics],
            "constraints": [item.as_dict() for item in self.constraints],
            "diagnostics": list(self.diagnostics),
        }


def compile_constraint_program(
    ir: UnifiedIR,
    graph: CircuitGraph | None = None,
) -> ConstraintProgram:
    compiler = _ConstraintCompiler(ir, graph)
    compiler.compile()
    metrics = tuple(compiler.metrics[key] for key in sorted(compiler.metrics))
    constraints = tuple(sorted(compiler.constraints, key=lambda item: item.constraint_id))
    if len({item.constraint_id for item in constraints}) != len(constraints):
        raise ConstraintCompilationError("compiled constraint ids are not unique")
    return ConstraintProgram(metrics, constraints, tuple(compiler.diagnostics))


class _ConstraintCompiler:
    def __init__(self, ir: UnifiedIR, graph: CircuitGraph | None) -> None:
        self.ir = ir
        self.graph = graph
        self.metrics: dict[str, MetricSpec] = {}
        self.constraints: list[ConstraintSpec] = []
        self.diagnostics: list[str] = []

    def compile(self) -> None:
        self._compile_resource_metrics()
        for index, target in enumerate(self.ir.targets):
            analysis = self.ir.analyses[index] if index < len(self.ir.analyses) else self.ir.primary_analysis
            if str(target.get("target_kind", "")) == "dc":
                self._compile_dc_target(index, target, analysis)
            else:
                self._compile_frequency_target(index, target, analysis)
        self._compile_port_constraints()
        self._compile_structural_constraints()
        self._compile_explicit_constraints()
        if self.graph is not None:
            self._compile_parameter_ranges()
            self._compile_ratings()

    def _compile_resource_metrics(self) -> None:
        for metric_id, source, quantity, unit in (
            ("resource.component_count", "component_count", "count", "1"),
            ("resource.bom_cost", "bom_cost", "cost", "CNY"),
            ("resource.area", "area", "area", "mm2"),
        ):
            self._metric(MetricSpec(metric_id, "graph.resource", source, quantity, unit))

    def _compile_dc_target(
        self,
        index: int,
        target: Mapping[str, Any],
        analysis: Mapping[str, Any],
    ) -> None:
        source_port, output_port = _analysis_ports(self.ir, analysis)
        mappings = (
            ("input_voltage_v", source_port, "voltage", "input_voltage_v", "V", "equal"),
            ("input_current_a", source_port, "current", "input_current_a", "A", "equal"),
            ("output_voltage_v", output_port, "voltage", "output_voltage_v", "V", "equal"),
            ("output_current_a", output_port, "current", "output_current_a", "A", "equal"),
            ("output_power_w", output_port, "power", "output_power_w", "W", "equal"),
            ("efficiency", "design", "efficiency", "efficiency", "1", "minimum"),
            ("ripple_mv", output_port, "voltage_ripple", "predicted_ripple_mv", "mV", "maximum"),
        )
        for target_key, port, quantity, source, unit, operator in mappings:
            if target.get(target_key) is None:
                continue
            metric_id = (
                f"port.{port}.{quantity}.dc"
                if port != "design"
                else f"design.{quantity}.dc"
            )
            metric = self._metric(
                MetricSpec(
                    metric_id,
                    "dc_port_metrics",
                    source,
                    quantity,
                    unit,
                    analysis_kind=_normalize_analysis_kind(str(analysis.get("kind", "dc_transfer"))),
                    attributes={"legacy_name": target_key},
                )
            )
            value = float(target[target_key])
            tolerance = _tolerance_from(
                target.get("tolerance"),
                unit,
                default_for_equal=operator == "equal",
            )
            self._add(
                ConstraintSpec(
                    constraint_id=f"target.{index}.{target_key}",
                    metric=metric,
                    operator=ConstraintOperator(operator),
                    severity=_severity(target.get("severity", "hard")),
                    value=value if operator == "equal" else None,
                    minimum=value if operator == "minimum" else None,
                    maximum=value if operator == "maximum" else None,
                    unit=unit,
                    tolerance=tolerance,
                    weight=float(target.get("weight", 1.0)),
                    source_path=f"targets[{index}].{target_key}",
                    description=str(target.get("description", "")),
                )
            )

    def _compile_frequency_target(
        self,
        index: int,
        target: Mapping[str, Any],
        analysis: Mapping[str, Any],
    ) -> None:
        frequencies = _frequency_grid(target, analysis)
        if frequencies is None:
            self.diagnostics.append(
                f"targets[{index}] has no frequency grid; frequency-response objectives were not compiled"
            )
            return
        try:
            module = importlib.import_module("端口描述语言.targets")
            target_model = module.target_from_dict(dict(target))
            target_db = tuple(float(item) for item in target_model.evaluate_magnitude_db(frequencies))
        except Exception as exc:
            self.diagnostics.append(f"targets[{index}] could not be evaluated: {exc}")
            return
        source = _ac_observable_id(
            str(analysis.get("kind", "voltage_transfer")),
            analysis,
        )
        metric = self._metric(
            MetricSpec(
                metric_id=f"target.{index}.rmse_db",
                provider_id="frequency_response",
                source=source,
                quantity="magnitude_error",
                unit="dB",
                reduction="rmse_db",
                analysis_kind="small_signal_ac",
                selectors={
                    "frequency_hz": tuple(float(item) for item in frequencies),
                    "target_magnitude_db": target_db,
                },
            )
        )
        self._add(
            ConstraintSpec(
                constraint_id=f"target.{index}.minimize_rmse_db",
                metric=metric,
                operator=ConstraintOperator.MINIMIZE,
                severity=ConstraintSeverity.OBJECTIVE,
                unit="dB",
                weight=float(target.get("weight", 1.0)),
                source_path=f"targets[{index}]",
                description="minimize frequency-response magnitude RMSE",
            )
        )
        limit = _target_tolerance_limit(target, target_db)
        if limit is None:
            return
        self._add(
            ConstraintSpec(
                constraint_id=f"target.{index}.rmse_db_within_tolerance",
                metric=metric,
                operator=ConstraintOperator.MAXIMUM,
                severity=ConstraintSeverity.HARD,
                maximum=limit.value,
                unit="dB",
                weight=float(target.get("weight", 1.0)),
                source_path=f"targets[{index}].tolerance",
                description=(
                    f"frequency-response magnitude RMSE must stay within {limit.basis}; "
                    "a stated tolerance is an acceptance limit, and a limit that is "
                    "never evaluated is not a limit"
                ),
            )
        )

    def _compile_port_constraints(self) -> None:
        for port_index, port in enumerate(self.ir.ports):
            for index, raw in enumerate(port.constraints):
                analysis = raw.get("analysis", {"kind": "dc_operating_point"})
                analysis_map = {"kind": analysis} if isinstance(analysis, str) else dict(analysis)
                kind = _normalize_analysis_kind(str(analysis_map.get("kind", "dc_operating_point")))
                operator_text = str(raw.get("operator", "equal"))
                constraint_id = str(raw.get("id") or f"port.{port.name}.{index}.{raw.get('variable', '')}")
                if operator_text in {"function", "samples"}:
                    self._compile_sampled_port_constraint(
                        constraint_id, port, raw, analysis_map, port_index, index
                    )
                    continue
                try:
                    operator = ConstraintOperator(operator_text)
                except ValueError as exc:
                    raise ConstraintCompilationError(
                        f"unsupported operator {operator_text!r} at ports[{port_index}].constraints[{index}]"
                    ) from exc
                variable = str(raw.get("variable", ""))
                unit = str(raw.get("unit") or _port_variable_unit(port, variable))
                metric = self._port_metric(port, variable, unit, kind, analysis_map, raw, index)
                self._add(
                    ConstraintSpec(
                        constraint_id=constraint_id,
                        metric=metric,
                        operator=operator,
                        severity=_severity(raw.get("severity", raw.get("kind", "hard"))),
                        value=_optional_float(raw.get("value")),
                        minimum=_optional_float(raw.get("minimum")),
                        maximum=_optional_float(raw.get("maximum")),
                        unit=unit,
                        tolerance=_tolerance_from(
                            raw.get("tolerance"),
                            unit,
                            default_for_equal=operator is ConstraintOperator.EQUAL,
                        ),
                        weight=float(raw.get("weight", 1.0)),
                        source_path=f"ports[{port_index}].variable_constraints[{index}]",
                        required_evidence_rank=int(raw.get("required_evidence_rank", 0)),
                        attributes={"port_id": port.name, "variable": variable},
                    )
                )

    def _port_metric(
        self,
        port: IRPort,
        variable: str,
        unit: str,
        kind: str,
        analysis: Mapping[str, Any],
        raw: Mapping[str, Any],
        index: int,
    ) -> MetricSpec:
        quantity = _port_variable_quantity(port, variable)
        metric_id = f"port.{port.name}.{variable}.{kind}.{index}"
        if kind in {"dc_transfer", "dc_operating_point"}:
            source = _dc_port_source(self.ir, port.name, variable)
            return self._metric(
                MetricSpec(
                    metric_id,
                    "dc_port_metrics",
                    source,
                    quantity,
                    unit,
                    analysis_kind=kind,
                    attributes={
                        "port_id": port.name,
                        "variable": variable,
                        "legacy_name": f"{port.name}.{variable}",
                    },
                )
            )
        reduction = str(raw.get("reduction") or _frequency_reduction(unit, quantity))
        selectors = dict(raw.get("data", {}))
        coordinate = _single_frequency(analysis, selectors)
        if coordinate is not None:
            selectors.setdefault("frequency", coordinate)
        return self._metric(
            MetricSpec(
                metric_id,
                "frequency_response",
                _ac_observable_id(kind, analysis),
                quantity,
                unit,
                reduction=reduction,
                analysis_kind="small_signal_ac",
                selectors=selectors,
                attributes={
                    "port_id": port.name,
                    "variable": variable,
                    "legacy_name": f"{port.name}.{variable}",
                },
            )
        )

    def _compile_sampled_port_constraint(
        self,
        constraint_id: str,
        port: IRPort,
        raw: Mapping[str, Any],
        analysis: Mapping[str, Any],
        port_index: int,
        index: int,
    ) -> None:
        data = dict(raw.get("data", {}))
        target_db = data.get("magnitude_db")
        if not target_db:
            raise ConstraintCompilationError(
                f"sampled constraint {constraint_id!r} requires data.magnitude_db"
            )
        unit = "dB"
        metric = self._metric(
            MetricSpec(
                f"port.{port.name}.{raw.get('variable', 'v')}.sample_rmse.{index}",
                "frequency_response",
                _ac_observable_id(
                    str(analysis.get("kind", "small_signal_ac")),
                    analysis,
                ),
                "magnitude_error",
                unit,
                reduction="rmse_db",
                analysis_kind="small_signal_ac",
                selectors={"target_magnitude_db": tuple(float(item) for item in target_db)},
            )
        )
        tolerance = _tolerance_from(raw.get("tolerance"), unit, default_for_equal=False)
        maximum = tolerance.absolute if tolerance.absolute > 0.0 else float(data.get("max_rmse_db", 0.0))
        self._add(
            ConstraintSpec(
                constraint_id=constraint_id,
                metric=metric,
                operator=ConstraintOperator.MAXIMUM,
                severity=_severity(raw.get("severity", "hard")),
                maximum=maximum,
                unit=unit,
                weight=float(raw.get("weight", 1.0)),
                source_path=f"ports[{port_index}].variable_constraints[{index}]",
            )
        )

    def _compile_structural_constraints(self) -> None:
        constraints = self.ir.constraints
        allowed = tuple(str(item) for item in constraints.get("element_types", ()))
        if allowed:
            metric = self._metric(
                MetricSpec(
                    "structure.disallowed_element_count",
                    "graph.resource",
                    "disallowed_element_count",
                    "count",
                    "1",
                    selectors={"allowed": allowed},
                )
            )
            self._add(
                ConstraintSpec(
                    "structure.allowed_elements",
                    metric,
                    ConstraintOperator.MAXIMUM,
                    maximum=0.0,
                    source_path="constraints.element_types",
                )
            )
        required = tuple(str(item) for item in constraints.get("required_elements", ()))
        if required:
            metric = self._metric(
                MetricSpec(
                    "structure.missing_required_element_count",
                    "graph.resource",
                    "missing_required_element_count",
                    "count",
                    "1",
                    selectors={"required": required},
                )
            )
            self._add(
                ConstraintSpec(
                    "structure.required_elements",
                    metric,
                    ConstraintOperator.MAXIMUM,
                    maximum=0.0,
                    source_path="constraints.required_elements",
                )
            )
        max_count = constraints.get("max_component_count")
        if max_count is not None:
            self._add(
                ConstraintSpec(
                    "resource.max_component_count",
                    self.metrics["resource.component_count"],
                    ConstraintOperator.MAXIMUM,
                    maximum=float(max_count),
                    source_path="constraints.max_component_count",
                )
            )
        if constraints.get("max_cost_cny") is not None:
            self._add(
                ConstraintSpec(
                    "resource.max_cost",
                    self.metrics["resource.bom_cost"],
                    ConstraintOperator.MAXIMUM,
                    maximum=float(constraints["max_cost_cny"]),
                    unit="CNY",
                    source_path="constraints.max_cost_cny",
                )
            )
        if constraints.get("max_area_mm2") is not None:
            self._add(
                ConstraintSpec(
                    "resource.max_area",
                    self.metrics["resource.area"],
                    ConstraintOperator.MAXIMUM,
                    maximum=float(constraints["max_area_mm2"]),
                    unit="mm2",
                    source_path="constraints.max_area_mm2",
                )
            )
        if constraints.get("max_power_mw") is not None:
            metric = self._metric(
                MetricSpec(
                    "resource.power_dissipation",
                    "dc_port_metrics",
                    "power_dissipation_w",
                    "power",
                    "mW",
                )
            )
            self._add(
                ConstraintSpec(
                    "resource.max_power",
                    metric,
                    ConstraintOperator.MAXIMUM,
                    maximum=float(constraints["max_power_mw"]),
                    unit="mW",
                    source_path="constraints.max_power_mw",
                )
            )

    def _compile_explicit_constraints(self) -> None:
        declarations = tuple(self.ir.constraints.get("metric_constraints", ()))
        objectives = tuple(self.ir.constraints.get("objectives", ()))
        for index, declaration in enumerate((*declarations, *objectives)):
            raw = dict(declaration)
            is_objective = index >= len(declarations)
            constraint_id = str(raw.get("id") or f"declaration.{index}")
            metric = self._explicit_metric(raw)
            value, unit = _value_and_unit(raw.get("value"), raw.get("unit", metric.unit))
            minimum, _ = _value_and_unit(raw.get("minimum"), unit)
            maximum, _ = _value_and_unit(raw.get("maximum"), unit)
            operator_text = str(
                raw.get("operator")
                or raw.get("direction")
                or ("minimize" if is_objective else "equal")
            )
            self._add(
                ConstraintSpec(
                    constraint_id,
                    metric,
                    ConstraintOperator(operator_text),
                    severity=ConstraintSeverity.OBJECTIVE if is_objective else _severity(raw.get("severity", "hard")),
                    value=value,
                    minimum=minimum,
                    maximum=maximum,
                    unit=unit,
                    tolerance=_tolerance_from(raw.get("tolerance"), unit, default_for_equal=operator_text == "equal"),
                    weight=float(raw.get("weight", 1.0) if raw.get("weight") is not None else 1.0),
                    source_path=f"constraints.{'objectives' if is_objective else 'metric_constraints'}[{index - len(declarations) if is_objective else index}]",
                    description=str(raw.get("description", "")),
                    required_evidence_rank=int(raw.get("required_evidence_rank", 0)),
                )
            )

    def _explicit_metric(self, raw: Mapping[str, Any]) -> MetricSpec:
        metric_value = raw.get("metric")
        if isinstance(metric_value, Mapping):
            return self._metric(MetricSpec.from_dict(metric_value))
        metric_id = str(metric_value or raw.get("metric_id") or "")
        if not metric_id:
            raise ConstraintCompilationError("explicit constraint requires metric")
        provider = str(raw.get("provider_id", ""))
        source = str(raw.get("source", ""))
        quantity = str(raw.get("quantity", metric_id.rsplit(".", 1)[-1]))
        unit = str(raw.get("unit", "1"))
        reduction = str(raw.get("reduction", "scalar"))
        selectors = dict(raw.get("selectors", {}))
        if metric_id == "resource.bom_cost":
            provider, source, quantity, unit = "graph.resource", "bom_cost", "cost", "CNY"
        elif metric_id in {"resource.area", "resource.area_mm2"}:
            metric_id, provider, source, quantity, unit = "resource.area", "graph.resource", "area", "area", "mm2"
        elif metric_id == "resource.component_count":
            provider, source, quantity, unit = "graph.resource", "component_count", "count", "1"
        elif metric_id.startswith("port.") and not provider:
            parts = metric_id.split(".")
            if len(parts) >= 4:
                port_id, quantity, reduction = parts[1], parts[2], parts[3]
                variable = {"voltage": "v", "current": "i", "power": "p"}.get(quantity, quantity)
                source = _dc_port_source(self.ir, port_id, variable)
                provider = "dc_port_metrics" if reduction in {"dc", "mean", "scalar"} else "transient_ripple"
        if not provider:
            provider = "simulation.scalar"
        if not source:
            source = metric_id
        return self._metric(
            MetricSpec(
                metric_id,
                provider,
                source,
                quantity,
                unit,
                reduction=reduction,
                analysis_kind=(str(raw["analysis_kind"]) if raw.get("analysis_kind") else None),
                selectors=selectors,
            )
        )

    def _compile_parameter_ranges(self) -> None:
        assert self.graph is not None
        ranges = {str(key).casefold(): value for key, value in self.ir.constraints.get("parameter_ranges", {}).items()}
        for component in self.graph.components:
            if component.attributes.get("role") == "external_load" or component.model.kind in {
                "voltage_source",
                "current_source",
            }:
                continue
            raw_range = ranges.get(component.model.kind.casefold())
            if raw_range is None:
                continue
            for parameter in component.parameters:
                if parameter.value is None:
                    continue
                if parameter.name not in {"value", "turns_ratio"}:
                    continue
                metric_id = f"component.{component.instance_id}.{parameter.name}"
                metric = self._metric(
                    MetricSpec(
                        metric_id,
                        "graph.parameter",
                        f"{component.instance_id}.{parameter.name}",
                        "component_parameter",
                        parameter.unit,
                    )
                )
                self._add(
                    ConstraintSpec(
                        f"parameter_range.{component.instance_id}.{parameter.name}",
                        metric,
                        ConstraintOperator.INTERVAL,
                        minimum=float(raw_range[0]),
                        maximum=float(raw_range[1]),
                        unit=parameter.unit,
                        source_path=f"constraints.parameter_ranges.{component.model.kind}",
                    )
                )

    def _compile_ratings(self) -> None:
        assert self.graph is not None
        for component in self.graph.components:
            for index, rating in enumerate(component.ratings):
                source = f"device.{component.instance_id}.{rating.quantity}"
                reduction = "maximum_abs" if rating.maximum is not None else "minimum"
                metric = self._metric(
                    MetricSpec(
                        f"{source}.{reduction}",
                        "device_stress",
                        source,
                        rating.quantity,
                        rating.unit,
                        reduction=reduction,
                    )
                )
                if rating.maximum is not None:
                    self._add(
                        ConstraintSpec(
                            f"rating.{component.instance_id}.{index}.maximum",
                            metric,
                            ConstraintOperator.MAXIMUM,
                            maximum=rating.maximum,
                            unit=rating.unit,
                            source_path=f"graph.components.{component.instance_id}.ratings[{index}]",
                        )
                    )
                if rating.minimum is not None:
                    self._add(
                        ConstraintSpec(
                            f"rating.{component.instance_id}.{index}.minimum",
                            metric,
                            ConstraintOperator.MINIMUM,
                            minimum=rating.minimum,
                            unit=rating.unit,
                            source_path=f"graph.components.{component.instance_id}.ratings[{index}]",
                        )
                    )

    def _metric(self, metric: MetricSpec) -> MetricSpec:
        previous = self.metrics.get(metric.metric_id)
        if previous is not None and previous != metric:
            raise ConstraintCompilationError(
                f"metric id {metric.metric_id!r} has incompatible definitions"
            )
        self.metrics[metric.metric_id] = metric
        return metric

    def _add(self, constraint: ConstraintSpec) -> None:
        self.constraints.append(constraint)


def _analysis_ports(ir: UnifiedIR, analysis: Mapping[str, Any]) -> tuple[str, str]:
    source = str(analysis.get("source_port") or (ir.ports[0].name if ir.ports else "input"))
    output = str(
        analysis.get("output_port")
        or analysis.get("response_port")
        or (ir.ports[-1].name if ir.ports else "output")
    )
    return source, output


def _dc_port_source(ir: UnifiedIR, port_name: str, variable: str) -> str:
    source_port, output_port = _analysis_ports(ir, ir.primary_analysis)
    prefix = "input" if port_name == source_port else "output" if port_name == output_port else f"port.{port_name}"
    suffix = {"v": "voltage_v", "i": "current_a", "p": "power_w"}.get(variable, variable)
    if variable == "p" and prefix == "input":
        return "input_power_w"
    if variable == "p" and prefix == "output":
        return "output_power_w"
    return f"{prefix}_{suffix}" if not prefix.startswith("port.") else f"{prefix}.{suffix}"


def _port_variable_unit(port: IRPort, variable: str) -> str:
    for item in port.variables:
        if str(item.get("name")) == variable and item.get("unit") is not None:
            return str(item["unit"])
    return {"v": "V", "i": "A", "p": "W"}.get(variable, "1")


def _port_variable_quantity(port: IRPort, variable: str) -> str:
    for item in port.variables:
        if str(item.get("name")) == variable:
            return str(item.get("quantity", variable))
    return {"v": "voltage", "i": "current", "p": "power"}.get(variable, variable)


def _normalize_analysis_kind(kind: str) -> str:
    return {
        "dc_operating_point": "dc_transfer",
        "voltage_transfer": "small_signal_ac",
        "transimpedance": "small_signal_ac",
        "input_impedance": "small_signal_ac",
        "output_impedance": "small_signal_ac",
        "impedance": "small_signal_ac",
    }.get(kind, kind)


def _ac_observable_id(kind: str, analysis: Mapping[str, Any] | None = None) -> str:
    analysis = analysis or {}
    if kind == "impedance":
        port = str(analysis.get("port", "input"))
        return "output_impedance" if port == "output" else "input_impedance"
    return {
        "small_signal_ac": "voltage_transfer",
        "voltage_transfer": "voltage_transfer",
        "transimpedance": "transimpedance",
        "impedance": "input_impedance",
        "input_impedance": "input_impedance",
        "output_impedance": "output_impedance",
    }.get(kind, "voltage_transfer")


def _frequency_reduction(unit: str, quantity: str) -> str:
    if unit == "dB":
        return "magnitude_db_at"
    if unit in {"deg", "degree"} or quantity == "phase":
        return "phase_deg_at"
    return "magnitude_at"


def _single_frequency(analysis: Mapping[str, Any], data: Mapping[str, Any]) -> float | None:
    for value in (data.get("frequency_hz"), analysis.get("frequency_hz")):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _frequency_grid(target: Mapping[str, Any], analysis: Mapping[str, Any]) -> np.ndarray | None:
    raw = analysis.get("frequency_hz")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        start, stop, points = float(raw[0]), float(raw[1]), int(raw[2])
        if start > 0 and stop > start and points > 1:
            return np.geomspace(start, stop, points)
    samples = target.get("frequency_hz") or target.get("samples_frequency_hz")
    if isinstance(samples, (list, tuple)) and samples:
        return np.asarray(samples, dtype=float)
    return None


def _tolerance_from(raw: Any, unit: str, *, default_for_equal: bool) -> ToleranceSpec:
    if isinstance(raw, (int, float)):
        return ToleranceSpec(absolute=float(raw), unit=unit)
    data = dict(raw or {})
    if data:
        return ToleranceSpec(
            absolute=float(data.get("absolute", 0.0) or 0.0),
            relative=float(data.get("relative", 0.0) or 0.0),
            unit=str(data.get("unit", unit)),
            policy=str(data.get("policy", "explicit")),
        )
    if default_for_equal:
        return ToleranceSpec(
            absolute=1e-12,
            relative=1e-6,
            unit=unit,
            policy="conservative_default",
        )
    return ToleranceSpec(unit=unit)


def _severity(value: Any) -> ConstraintSeverity:
    try:
        return ConstraintSeverity(str(value))
    except ValueError as exc:
        raise ConstraintCompilationError(f"unsupported constraint severity {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _value_and_unit(value: Any, default_unit: Any) -> tuple[float | None, str]:
    if isinstance(value, Mapping):
        return _optional_float(value.get("value")), str(value.get("unit", default_unit))
    return _optional_float(value), str(default_unit)


@dataclass(frozen=True)
class _ToleranceLimit:
    """A tolerance turned into a dB RMSE limit, with the wording that explains it."""

    value: float
    basis: str


def _positive_finite(value: Any) -> float | None:
    """A usable positive number, or ``None``.

    ``bool`` is excluded deliberately: ``True`` is an ``int`` in Python and would
    otherwise read as a 1 dB tolerance.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _target_tolerance_limit(
    target: Mapping[str, Any],
    target_magnitude_db: Sequence[float],
) -> _ToleranceLimit | None:
    """Turn a filter target's declared tolerance into a hard RMSE limit.

    Why this exists: the frequency-target compiler used to emit only a soft
    MINIMIZE, so *any* design satisfied the acceptance gate. A 4th-order
    Butterworth request answered by a first-order R/C was reported
    ``verified_feasible`` at 32.9 dB RMSE against a declared 0.5 dB tolerance,
    because the tolerance was never compiled into anything.

    ``relative`` takes precedence over ``absolute`` when both are given, since a
    relative band scales with the requirement while an absolute error is a
    number the user had to guess. The conversion treats the band as symmetric
    around the target, so the limit is ``20*log10(1 + relative)`` and does not
    depend on the filter's own gain.
    """

    tolerance = target.get("tolerance")
    if not isinstance(tolerance, Mapping):
        return None

    relative = _positive_finite(tolerance.get("relative"))
    if relative is not None:
        if not 0.0 < relative < 1.0 or not target_magnitude_db:
            return None
        return _ToleranceLimit(
            value=float(20.0 * math.log10(1.0 + relative)),
            basis=f"relative {relative:g} of the target magnitude",
        )

    absolute = _positive_finite(tolerance.get("absolute"))
    if absolute is None:
        return None
    return _ToleranceLimit(value=absolute, basis=f"absolute {absolute:g} dB")
