"""Built-in metric providers over simulation evidence and CircuitGraph metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping, Protocol

import numpy as np

from ..graph import CircuitGraph
from ..simulation.contracts import SimulationRequest, SimulationResult, SimulationStatus
from ..units import UnitContractError, convert_scalar
from .contracts import (
    EvidenceReference,
    MetricSet,
    MetricSpec,
    MetricStatus,
    MetricValue,
)


@dataclass(frozen=True)
class MetricContext:
    graph: CircuitGraph | None = None
    ir: Any | None = None
    simulation_requests: tuple[SimulationRequest, ...] = ()
    simulation_results: tuple[SimulationResult, ...] = ()
    parameter_values: Mapping[str, Any] | None = None


class MetricProvider(Protocol):
    provider_id: str

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue: ...


class MetricEngine:
    def __init__(self, providers: Iterable[MetricProvider] | None = None) -> None:
        selected = tuple(providers) if providers is not None else builtin_metric_providers()
        self._providers = {item.provider_id: item for item in selected}
        if len(self._providers) != len(selected):
            raise ValueError("metric provider ids must be unique")

    def extract(
        self,
        specs: Iterable[MetricSpec],
        context: MetricContext,
    ) -> MetricSet:
        unique: dict[str, MetricSpec] = {}
        diagnostics: list[str] = []
        for spec in specs:
            previous = unique.get(spec.metric_id)
            if previous is not None and previous != spec:
                raise ValueError(f"metric id {spec.metric_id!r} has incompatible definitions")
            unique[spec.metric_id] = spec
        values: dict[str, MetricValue] = {}
        for metric_id in sorted(unique):
            spec = unique[metric_id]
            provider = self._providers.get(spec.provider_id)
            if provider is None:
                values[metric_id] = _unavailable(
                    spec,
                    MetricStatus.UNSUPPORTED,
                    f"metric provider {spec.provider_id!r} is not registered",
                )
                continue
            try:
                values[metric_id] = provider.extract(spec, context)
            except Exception as exc:  # provider failures become evidence, not orchestration failures
                message = f"metric provider {spec.provider_id!r} failed: {exc}"
                diagnostics.append(message)
                values[metric_id] = _unavailable(spec, MetricStatus.ERROR, message)
        return MetricSet(values, tuple(diagnostics))


class SimulationScalarProvider:
    provider_id = "simulation.scalar"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        results = _matching_results(spec, context)
        observations: list[tuple[float, str, SimulationResult]] = []
        for result in results:
            quantity = result.scalars.get(spec.source)
            if quantity is not None:
                observations.append((float(quantity.value), quantity.unit, result))
        if not observations:
            return _missing_from_results(spec, results)
        try:
            values = np.asarray(
                [convert_scalar(value, unit, spec.unit) for value, unit, _ in observations],
                dtype=float,
            )
            value = _reduce_real(values, spec.reduction)
        except (UnitContractError, ValueError) as exc:
            return _unavailable(spec, MetricStatus.ERROR, str(exc))
        evidence = tuple(_simulation_evidence(item[2]) for item in observations)
        return MetricValue(
            metric_id=spec.metric_id,
            status=MetricStatus.AVAILABLE,
            value=value,
            unit=spec.unit,
            provider_id=self.provider_id,
            source=spec.source,
            evidence=_unique_evidence(evidence),
            attributes={"observation_count": len(observations), "reduction": spec.reduction},
        )


class DcPortMetricProvider(SimulationScalarProvider):
    provider_id = "dc_port_metrics"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        direct = super().extract(spec, context)
        if direct.available or spec.source not in {"input_power_w", "power_dissipation_w"}:
            return direct
        derived_sources = (
            "input_voltage_v",
            "input_current_a",
            "output_power_w",
        )
        derived: dict[str, MetricValue] = {}
        for source in derived_sources:
            child = replace(
                spec,
                metric_id=f"{spec.metric_id}.__{source}",
                provider_id=self.provider_id,
                source=source,
                unit="V" if source.endswith("voltage_v") else "A" if source.endswith("current_a") else "W",
                reduction="scalar",
            )
            derived[source] = super().extract(child, context)
        needed = derived_sources[:2] if spec.source == "input_power_w" else derived_sources
        if not all(derived[source].available for source in needed):
            return direct
        input_power = float(derived["input_voltage_v"].value) * float(
            derived["input_current_a"].value
        )
        value = input_power
        if spec.source == "power_dissipation_w":
            value -= float(derived["output_power_w"].value)
            value = max(0.0, value)
        try:
            converted = convert_scalar(value, "W", spec.unit)
        except UnitContractError as exc:
            return _unavailable(spec, MetricStatus.ERROR, str(exc), provider_id=self.provider_id)
        evidence = tuple(
            item
            for source in needed
            for item in derived[source].evidence
        )
        return MetricValue(
            metric_id=spec.metric_id,
            status=MetricStatus.AVAILABLE,
            value=converted,
            unit=spec.unit,
            provider_id=self.provider_id,
            source=spec.source,
            evidence=_unique_evidence(evidence),
            attributes={"derived_from": needed},
        )


class FrequencyResponseMetricProvider:
    provider_id = "frequency_response"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        return _extract_waveform_metric(spec, context, self.provider_id, domain="frequency")


class TransientRippleMetricProvider:
    provider_id = "transient_ripple"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        return _extract_waveform_metric(spec, context, self.provider_id, domain="time")


class DeviceStressMetricProvider:
    provider_id = "device_stress"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        scalar_spec = replace(spec, provider_id="simulation.scalar")
        scalar = SimulationScalarProvider().extract(scalar_spec, context)
        if scalar.available:
            return replace(scalar, provider_id=self.provider_id)
        waveform = _extract_waveform_metric(spec, context, self.provider_id, domain=None)
        if waveform.available:
            return waveform
        statuses = {scalar.status, waveform.status}
        status = MetricStatus.ERROR if MetricStatus.ERROR in statuses else (
            MetricStatus.UNSUPPORTED if MetricStatus.UNSUPPORTED in statuses else MetricStatus.MISSING
        )
        return _unavailable(
            spec,
            status,
            "; ".join((*scalar.diagnostics, *waveform.diagnostics)),
            provider_id=self.provider_id,
        )


class ResourceMetricProvider:
    provider_id = "graph.resource"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        if context.graph is None:
            return _unavailable(spec, MetricStatus.MISSING, "CircuitGraph is unavailable")
        components = tuple(
            item
            for item in context.graph.components
            if item.attributes.get("role") != "external_load"
            and item.model.kind not in {"voltage_source", "current_source"}
        )
        kinds = tuple(item.model.kind for item in components)
        constraints = dict(getattr(context.ir, "constraints", {}) or {})
        costs = {str(key).casefold(): float(value) for key, value in constraints.get("unit_costs", {}).items()}
        areas = {
            str(key).casefold(): float(value)
            for key, value in constraints.get("unit_areas_mm2", {}).items()
        }
        if spec.source == "component_count":
            value, unit = float(len(components)), "1"
        elif spec.source == "bom_cost":
            value = sum(
                float(item.attributes.get("cost_cny", costs.get(item.model.kind.casefold(), 0.0)))
                for item in components
            )
            unit = "CNY"
        elif spec.source == "area":
            value = sum(
                float(item.attributes.get("area_mm2", areas.get(item.model.kind.casefold(), 0.0)))
                for item in components
            )
            unit = "mm2"
        elif spec.source == "element_count":
            element_type = str(spec.selectors.get("element_type", ""))
            value, unit = float(sum(kind.casefold() == element_type.casefold() for kind in kinds)), "1"
        elif spec.source == "disallowed_element_count":
            allowed = {
                str(item).casefold()
                for item in spec.selectors.get("allowed", constraints.get("element_types", ()))
            }
            value = float(
                sum(bool(allowed) and not _element_kind_matches(kind, allowed) for kind in kinds)
            )
            unit = "1"
        elif spec.source == "missing_required_element_count":
            required = {
                str(item).casefold()
                for item in spec.selectors.get("required", constraints.get("required_elements", ()))
            }
            value, unit = float(
                sum(not _element_kind_matches(kind, required) for kind in kinds)
            ), "1"
            if required:
                value = float(
                    len(required - {
                        alias
                        for kind in kinds
                        for alias in _element_kind_aliases(kind)
                    })
                )
        else:
            return _unavailable(
                spec,
                MetricStatus.UNSUPPORTED,
                f"unsupported graph resource metric {spec.source!r}",
            )
        try:
            converted = convert_scalar(float(value), unit, spec.unit)
        except UnitContractError as exc:
            return _unavailable(spec, MetricStatus.ERROR, str(exc))
        return MetricValue(
            metric_id=spec.metric_id,
            status=MetricStatus.AVAILABLE,
            value=converted,
            unit=spec.unit,
            provider_id=self.provider_id,
            source=spec.source,
            evidence=(_graph_evidence(context.graph, "declared_graph"),),
            attributes={"component_ids": tuple(item.instance_id for item in components)},
        )


class GraphParameterMetricProvider:
    provider_id = "graph.parameter"

    def extract(self, spec: MetricSpec, context: MetricContext) -> MetricValue:
        if context.graph is None:
            return _unavailable(spec, MetricStatus.MISSING, "CircuitGraph is unavailable")
        try:
            instance_id, parameter_name = spec.source.rsplit(".", 1)
        except ValueError:
            return _unavailable(
                spec, MetricStatus.ERROR, "parameter source must be '<instance>.<parameter>'"
            )
        component = next(
            (item for item in context.graph.components if item.instance_id == instance_id),
            None,
        )
        if component is None:
            return _unavailable(spec, MetricStatus.MISSING, f"component {instance_id!r} is absent")
        parameter = next((item for item in component.parameters if item.name == parameter_name), None)
        if parameter is None:
            return _unavailable(
                spec,
                MetricStatus.MISSING,
                f"component {instance_id!r} has no parameter {parameter_name!r}",
            )
        value = parameter.value
        if value is None and parameter.variable_id and context.parameter_values is not None:
            value = context.parameter_values.get(parameter.variable_id)
        if value is None:
            return _unavailable(
                spec,
                MetricStatus.MISSING,
                f"parameter {spec.source!r} is unresolved",
            )
        try:
            converted = convert_scalar(float(value), parameter.unit, spec.unit)
        except (TypeError, ValueError, UnitContractError) as exc:
            return _unavailable(spec, MetricStatus.ERROR, str(exc))
        return MetricValue(
            metric_id=spec.metric_id,
            status=MetricStatus.AVAILABLE,
            value=converted,
            unit=spec.unit,
            provider_id=self.provider_id,
            source=spec.source,
            evidence=(_graph_evidence(context.graph, "declared_graph"),),
        )


def builtin_metric_providers() -> tuple[MetricProvider, ...]:
    return (
        SimulationScalarProvider(),
        DcPortMetricProvider(),
        FrequencyResponseMetricProvider(),
        TransientRippleMetricProvider(),
        DeviceStressMetricProvider(),
        ResourceMetricProvider(),
        GraphParameterMetricProvider(),
    )


def _element_kind_aliases(kind: str) -> set[str]:
    normalized = str(kind).casefold()
    aliases = {normalized}
    if normalized in {"vcvs", "opamp", "ideal_opamp"}:
        aliases.update({"vcvs", "opamp", "ideal_opamp"})
    if normalized in {"switch", "ideal_switch"}:
        aliases.update({"switch", "ideal_switch"})
    if normalized in {"diode", "ideal_diode"}:
        aliases.update({"diode", "ideal_diode"})
    if normalized in {"transformer", "ideal_transformer"}:
        aliases.update({"transformer", "ideal_transformer"})
    return aliases


def _element_kind_matches(kind: str, allowed: set[str]) -> bool:
    return bool(_element_kind_aliases(kind) & allowed)


def _extract_waveform_metric(
    spec: MetricSpec,
    context: MetricContext,
    provider_id: str,
    *,
    domain: str | None,
) -> MetricValue:
    results = _matching_results(spec, context)
    found: list[tuple[Any, SimulationResult]] = []
    for result in results:
        waveform = result.waveforms.get(spec.source)
        if waveform is not None:
            found.append((waveform, result))
    if not found:
        return _missing_from_results(spec, results, provider_id=provider_id)
    if len(found) > 1 and spec.reduction not in {"first", "last"}:
        return _unavailable(
            spec,
            MetricStatus.ERROR,
            f"waveform metric {spec.source!r} is ambiguous across {len(found)} results",
            provider_id=provider_id,
        )
    waveform, result = found[-1] if spec.reduction == "last" else found[0]
    axis = waveform.axes[0]
    if domain is not None and axis.name != domain:
        return _unavailable(
            spec,
            MetricStatus.UNSUPPORTED,
            f"provider {provider_id!r} requires {domain!r} axis, got {axis.name!r}",
            provider_id=provider_id,
        )
    try:
        value, computed_unit = _reduce_waveform(spec, waveform)
        converted = convert_scalar(value, computed_unit, spec.unit)
    except (ValueError, UnitContractError) as exc:
        return _unavailable(spec, MetricStatus.ERROR, str(exc), provider_id=provider_id)
    return MetricValue(
        metric_id=spec.metric_id,
        status=MetricStatus.AVAILABLE,
        value=converted,
        unit=spec.unit,
        provider_id=provider_id,
        source=spec.source,
        evidence=(_simulation_evidence(result),),
        attributes={
            "reduction": spec.reduction,
            "axis": axis.name,
            "axis_unit": axis.unit,
            "sample_count": len(waveform.values),
        },
    )


def _reduce_waveform(spec: MetricSpec, waveform) -> tuple[float, str]:
    values = np.asarray(waveform.values)
    axis = np.asarray(waveform.axes[0].values, dtype=float)
    reduction = spec.reduction
    if reduction in {"first", "last"}:
        selected = values[0] if reduction == "first" else values[-1]
        return _real_or_magnitude(selected), waveform.unit
    if reduction in {"magnitude_at", "magnitude_db_at", "phase_deg_at"}:
        coordinate = _required_coordinate(spec, waveform.axes[0].name)
        selected = _interpolate(axis, values, coordinate, logarithmic=axis.min() > 0)
        if reduction == "magnitude_at":
            return float(abs(selected)), waveform.unit
        if reduction == "magnitude_db_at":
            return float(20.0 * math.log10(max(abs(selected), 1e-300))), "dB"
        return float(np.degrees(np.angle(selected))), "deg"
    if reduction in {"rmse_db", "max_abs_error_db", "phase_rmse_deg"}:
        if reduction == "phase_rmse_deg":
            target = np.asarray(spec.selectors.get("target_phase_deg", ()), dtype=float)
            actual = np.degrees(np.unwrap(np.angle(values)))
            unit = "deg"
        else:
            target = np.asarray(spec.selectors.get("target_magnitude_db", ()), dtype=float)
            actual = 20.0 * np.log10(np.maximum(np.abs(values), 1e-300))
            unit = "dB"
        if target.shape != actual.shape:
            raise ValueError(
                f"target sample shape {target.shape} does not match waveform shape {actual.shape}"
            )
        error = actual - target
        if reduction == "max_abs_error_db":
            return float(np.max(np.abs(error))), unit
        return float(np.sqrt(np.mean(np.square(error)))), unit
    if reduction in {"bandwidth_3db", "bandwidth_hz"}:
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(values), 1e-300))
        reference = float(spec.selectors.get("reference_db", magnitude_db[0]))
        indices = np.flatnonzero(magnitude_db >= reference - 3.0)
        if len(indices) == 0:
            raise ValueError("waveform never reaches the -3 dB pass region")
        return float(axis[indices[-1]]), waveform.axes[0].unit
    working = values
    if reduction.startswith("steady_state_"):
        fraction = float(spec.selectors.get("window_fraction", 0.2))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("steady-state window_fraction must be in (0, 1]")
        start = max(0, len(values) - max(1, int(math.ceil(len(values) * fraction))))
        working = values[start:]
        reduction = reduction.removeprefix("steady_state_")
    real = np.abs(working) if np.iscomplexobj(working) else np.asarray(working, dtype=float)
    if reduction == "mean":
        return float(np.mean(real)), waveform.unit
    if reduction == "rms":
        return float(np.sqrt(np.mean(np.square(real)))), waveform.unit
    if reduction == "peak_to_peak":
        return float(np.max(real) - np.min(real)), waveform.unit
    if reduction in {"maximum", "maximum_abs"}:
        return float(np.max(np.abs(real)) if reduction == "maximum_abs" else np.max(real)), waveform.unit
    if reduction in {"minimum", "minimum_abs"}:
        return float(np.min(np.abs(real)) if reduction == "minimum_abs" else np.min(real)), waveform.unit
    if reduction == "peak_magnitude_db":
        return float(np.max(20.0 * np.log10(np.maximum(np.abs(values), 1e-300)))), "dB"
    if reduction == "minimum_magnitude_db":
        return float(np.min(20.0 * np.log10(np.maximum(np.abs(values), 1e-300)))), "dB"
    raise ValueError(f"unsupported waveform reduction {spec.reduction!r}")


def _required_coordinate(spec: MetricSpec, axis_name: str) -> float:
    keys = (axis_name, f"{axis_name}_hz", "coordinate")
    for key in keys:
        if spec.selectors.get(key) is not None:
            return float(spec.selectors[key])
    raise ValueError(f"reduction {spec.reduction!r} requires a coordinate for axis {axis_name!r}")


def _interpolate(axis, values, coordinate: float, *, logarithmic: bool):
    if coordinate < axis.min() or coordinate > axis.max():
        raise ValueError(f"coordinate {coordinate} lies outside waveform axis")
    x = np.log10(axis) if logarithmic else axis
    point = math.log10(coordinate) if logarithmic else coordinate
    if np.iscomplexobj(values):
        return np.interp(point, x, values.real) + 1j * np.interp(point, x, values.imag)
    return float(np.interp(point, x, values))


def _reduce_real(values: np.ndarray, reduction: str) -> float:
    if reduction in {"scalar", "first"}:
        return float(values[0])
    if reduction == "last":
        return float(values[-1])
    if reduction == "minimum":
        return float(np.min(values))
    if reduction == "maximum":
        return float(np.max(values))
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "rms":
        return float(np.sqrt(np.mean(np.square(values))))
    if reduction == "peak_to_peak":
        return float(np.max(values) - np.min(values))
    raise ValueError(f"unsupported scalar reduction {reduction!r}")


def _real_or_magnitude(value) -> float:
    return float(abs(value) if isinstance(value, complex) else value)


def _matching_results(spec: MetricSpec, context: MetricContext) -> tuple[SimulationResult, ...]:
    if spec.analysis_kind is None:
        return context.simulation_results
    kinds = {item.request_id: item.analysis.kind for item in context.simulation_requests}
    return tuple(
        result
        for result in context.simulation_results
        if kinds.get(result.request_id) in {spec.analysis_kind, None}
    )


def _missing_from_results(
    spec: MetricSpec,
    results: tuple[SimulationResult, ...],
    *,
    provider_id: str | None = None,
) -> MetricValue:
    if not results:
        return _unavailable(
            spec,
            MetricStatus.MISSING,
            "no simulation result is available",
            provider_id=provider_id,
        )
    failed = tuple(item for item in results if item.status is not SimulationStatus.PASSED)
    if failed:
        unsupported = all(item.status is SimulationStatus.UNSUPPORTED for item in failed)
        diagnostics = [
            f"{item.request_id}: {item.status.value}"
            + (f" ({'; '.join(diag.code for diag in item.diagnostics)})" if item.diagnostics else "")
            for item in failed
        ]
        return _unavailable(
            spec,
            MetricStatus.UNSUPPORTED if unsupported else MetricStatus.ERROR,
            "; ".join(diagnostics),
            provider_id=provider_id,
        )
    return _unavailable(
        spec,
        MetricStatus.MISSING,
        f"observable {spec.source!r} is absent from successful simulation results",
        provider_id=provider_id,
    )


def _unavailable(
    spec: MetricSpec,
    status: MetricStatus,
    message: str,
    *,
    provider_id: str | None = None,
) -> MetricValue:
    return MetricValue(
        metric_id=spec.metric_id,
        status=status,
        value=None,
        unit=spec.unit,
        provider_id=provider_id or spec.provider_id,
        source=spec.source,
        diagnostics=(message,),
    )


def _simulation_evidence(result: SimulationResult) -> EvidenceReference:
    level, rank = _evidence_level(result.backend_id, result.fidelity)
    return EvidenceReference(
        source_kind="simulation_result",
        source_id=result.request_id,
        evidence_hash=result.evidence_hash,
        backend_id=result.backend_id,
        backend_version=result.backend_version,
        fidelity=result.fidelity,
        level=level,
        level_rank=rank,
    )


def _graph_evidence(graph: CircuitGraph, level: str) -> EvidenceReference:
    return EvidenceReference(
        source_kind="circuit_graph",
        source_id=graph.graph_id,
        evidence_hash=graph.document_hash,
        fidelity="declared",
        level=level,
        level_rank=70,
    )


def _evidence_level(backend_id: str, fidelity: str) -> tuple[str, int]:
    text = f"{backend_id} {fidelity}".casefold()
    if "measured" in text or "laboratory" in text:
        return "measured", 100
    if "ngspice" in text or "spice" in text:
        return "spice", 80
    if "linear_frequency_domain" in text or "numeric" in text or "mna" in text:
        return "numerical", 60
    if "differentiable" in text or "torch" in text:
        return "differentiable", 50
    if "analytic" in text or "ideal" in text or "reduced" in text:
        return "analytic", 40
    if "surrogate" in text or "learned" in text:
        return "surrogate", 20
    return "unverified", 0


def _unique_evidence(items: tuple[EvidenceReference, ...]) -> tuple[EvidenceReference, ...]:
    unique = {(item.source_kind, item.source_id, item.evidence_hash): item for item in items}
    return tuple(unique[key] for key in sorted(unique))
