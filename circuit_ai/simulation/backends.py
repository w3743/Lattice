"""Built-in backends that emit the unified SimulationResult contract."""

from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter
from typing import Any, Callable

import numpy as np

from ..analysis import AnalysisRequest, LinearACAnalyzer
from ..graph import CircuitGraph, ModelRef, graph_parameter_defaults, graph_to_linear_circuit
from ..mna import CompiledLinearMNA, LinearCircuit
from .contracts import (
    Diagnostic,
    Quantity,
    ResultAxis,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    Waveform,
)


class UnsupportedModelError(ValueError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    backend_version: str
    analysis_kinds: frozenset[str]
    model_kinds: frozenset[str]
    fidelities: frozenset[str]


@dataclass(frozen=True)
class LinearCompiledModel:
    graph: CircuitGraph
    circuit: LinearCircuit
    model_manifest: tuple[ModelRef, ...]
    mna_plan: CompiledLinearMNA | None = None


@dataclass(frozen=True)
class PowerCompiledModel:
    graph: CircuitGraph
    model_manifest: tuple[ModelRef, ...]


class LinearMNASimulatorBackend:
    capabilities = BackendCapabilities(
        backend_id="linear_mna",
        backend_version="1.0.0",
        analysis_kinds=frozenset({"small_signal_ac"}),
        model_kinds=frozenset(
            {"R", "C", "L", "voltage_source", "current_source", "vcvs"}
        ),
        fidelities=frozenset({"linear_frequency_domain", "reduced"}),
    )

    def __init__(self, analyzer: LinearACAnalyzer | None = None) -> None:
        self._analyzer = analyzer or LinearACAnalyzer()
        self._compiled: dict[str, LinearCompiledModel] = {}
        self._plans: dict[tuple[str, str, str, tuple[tuple[str, str], ...]], CompiledLinearMNA] = {}

    def compile(self, graph: CircuitGraph) -> LinearCompiledModel:
        graph.require_valid()
        cached = self._compiled.get(graph.graph_hash)
        if cached is not None:
            return cached
        model_kinds = {component.model.kind for component in graph.components}
        unsupported = model_kinds - self.capabilities.model_kinds
        if unsupported:
            raise UnsupportedModelError(
                f"linear_mna does not support models {sorted(unsupported)}"
            )
        circuit = graph_to_linear_circuit(graph, graph_parameter_defaults(graph))
        plan_key = _linear_plan_cache_key(graph, self.capabilities)
        plan = self._plans.get(plan_key)
        if plan is None:
            plan = CompiledLinearMNA.compile(circuit)
            self._plans[plan_key] = plan
        compiled = LinearCompiledModel(
            graph=graph,
            circuit=circuit,
            model_manifest=_model_manifest(graph),
            mna_plan=plan,
        )
        self._compiled[graph.graph_hash] = compiled
        return compiled

    def simulate(
        self,
        compiled: LinearCompiledModel,
        request: SimulationRequest,
    ) -> SimulationResult:
        started = perf_counter()
        if request.graph_id != compiled.graph.graph_id:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.FAILED,
                "graph_mismatch",
                f"request graph_id={request.graph_id!r} does not match compiled graph",
                started,
            )
        if request.analysis.kind not in self.capabilities.analysis_kinds:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNSUPPORTED,
                "unsupported_analysis",
                f"linear_mna does not support analysis {request.analysis.kind!r}",
                started,
            )
        try:
            frequencies = _frequency_coordinates(request)
            if not request.requested_observables:
                raise UnsupportedModelError("small_signal_ac request has no observables")
            circuit = graph_to_linear_circuit(compiled.graph, request.parameter_values)
            analysis_requests = tuple(
                _linear_analysis_request(compiled.graph, request, observable)
                for observable in request.requested_observables
            )
            analyses = self._analyzer.analyze_many(
                circuit,
                analysis_requests,
                np.asarray(frequencies, dtype=float),
                compiled_mna=compiled.mna_plan,
            )
            waveforms: dict[str, Waveform] = {}
            for observable, analysis in zip(request.requested_observables, analyses):
                waveforms[observable.observable_id] = Waveform(
                    observable_id=observable.observable_id,
                    unit=observable.unit,
                    axes=(ResultAxis("frequency", "Hz", frequencies),),
                    values=tuple(complex(item) for item in analysis.values),
                    attributes={"analysis_kind": analysis.request.kind},
                )
        except UnsupportedModelError as exc:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNSUPPORTED,
                "unsupported_observable",
                str(exc),
                started,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            text = str(exc)
            code = "singular" if isinstance(exc, np.linalg.LinAlgError) else (
                "invalid_parameters" if "parameter" in text.casefold() else "analysis_failed"
            )
            return self._error_result(
                compiled,
                request,
                SimulationStatus.FAILED,
                code,
                str(exc),
                started,
            )
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            model_manifest=compiled.model_manifest,
            status=SimulationStatus.PASSED,
            waveforms=waveforms,
            runtime_s=perf_counter() - started,
            fidelity=request.fidelity,
            statistics={
                "matrix_solves": len(frequencies)
                * int(getattr(self._analyzer, "last_statistics", {}).get("solve_groups", 1)),
                "batched_matrix_solves": int(
                    getattr(self._analyzer, "last_statistics", {}).get("solve_groups", 1)
                ),
                "observable_extractions": len(waveforms),
                "compiled_cache_entries": len(self._compiled),
                "compiled_plan_cache_entries": len(self._plans),
            },
        )

    def _error_result(
        self,
        compiled: LinearCompiledModel,
        request: SimulationRequest,
        status: SimulationStatus,
        code: str,
        message: str,
        started: float,
    ) -> SimulationResult:
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            model_manifest=compiled.model_manifest,
            status=status,
            diagnostics=(Diagnostic(code, "error", message),),
            runtime_s=perf_counter() - started,
            fidelity=request.fidelity,
        )


class TorchMNASimulatorBackend:
    capabilities = BackendCapabilities(
        backend_id="torch_mna",
        backend_version="1.0.0",
        analysis_kinds=frozenset({"small_signal_ac"}),
        model_kinds=frozenset(
            {"R", "C", "L", "voltage_source", "current_source", "vcvs"}
        ),
        fidelities=frozenset({"differentiable", "linear_frequency_domain"}),
    )

    def __init__(self) -> None:
        self._compiled: dict[str, LinearCompiledModel] = {}
        self._plans: dict[tuple[str, str, str, tuple[tuple[str, str], ...]], CompiledLinearMNA] = {}

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "PyTorch is not installed"
        return True, "PyTorch is available"

    def compile(self, graph: CircuitGraph) -> LinearCompiledModel:
        graph.require_valid()
        cached = self._compiled.get(graph.graph_hash)
        if cached is not None:
            return cached
        model_kinds = {component.model.kind for component in graph.components}
        unsupported = model_kinds - self.capabilities.model_kinds
        if unsupported:
            raise UnsupportedModelError(
                f"torch_mna does not support models {sorted(unsupported)}"
            )
        circuit = graph_to_linear_circuit(graph, graph_parameter_defaults(graph))
        plan_key = _linear_plan_cache_key(graph, self.capabilities)
        plan = self._plans.get(plan_key)
        if plan is None:
            plan = CompiledLinearMNA.compile(circuit)
            self._plans[plan_key] = plan
        compiled = LinearCompiledModel(
            graph=graph,
            circuit=circuit,
            model_manifest=_model_manifest(graph),
            mna_plan=plan,
        )
        self._compiled[graph.graph_hash] = compiled
        return compiled

    def simulate(
        self,
        compiled: LinearCompiledModel,
        request: SimulationRequest,
    ) -> SimulationResult:
        started = perf_counter()
        if request.analysis.kind not in self.capabilities.analysis_kinds:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNSUPPORTED,
                "unsupported_analysis",
                f"torch_mna does not support analysis {request.analysis.kind!r}",
                started,
            )
        try:
            import torch

            from ..differentiable import TorchLinearCircuitMNA

            frequencies = _frequency_coordinates(request)
            if not request.requested_observables:
                raise UnsupportedModelError("small_signal_ac request has no observables")
            circuit = graph_to_linear_circuit(compiled.graph, request.parameter_values)
            waveforms: dict[str, Waveform] = {}
            for observable in request.requested_observables:
                legacy_request = _linear_analysis_request(compiled.graph, request, observable)
                if legacy_request.kind not in {"voltage_transfer", "transimpedance"}:
                    raise UnsupportedModelError(
                        f"torch_mna does not support {legacy_request.kind!r}"
                    )
                template = _FixedLinearTemplate(circuit, legacy_request)
                model = TorchLinearCircuitMNA(
                    template,
                    np.asarray(frequencies, dtype=float),
                    np.zeros(len(frequencies), dtype=np.complex128),
                    legacy_request,
                )
                with torch.no_grad():
                    values = model.response(torch.empty(0, dtype=torch.float64))
                waveforms[observable.observable_id] = Waveform(
                    observable_id=observable.observable_id,
                    unit=observable.unit,
                    axes=(ResultAxis("frequency", "Hz", frequencies),),
                    values=tuple(complex(item) for item in values.detach().cpu().numpy()),
                    attributes={"analysis_kind": legacy_request.kind, "differentiable": True},
                )
        except ImportError as exc:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNAVAILABLE,
                "backend_unavailable",
                str(exc),
                started,
            )
        except UnsupportedModelError as exc:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNSUPPORTED,
                "unsupported_observable",
                str(exc),
                started,
            )
        except (RuntimeError, ValueError) as exc:
            text = str(exc)
            folded = text.casefold()
            if "singular" in folded:
                code = "singular"
            elif "parameter" in folded:
                code = "invalid_parameters"
            else:
                code = "analysis_failed"
            return self._error_result(
                compiled,
                request,
                SimulationStatus.FAILED,
                code,
                text,
                started,
            )
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            model_manifest=compiled.model_manifest,
            status=SimulationStatus.PASSED,
            waveforms=waveforms,
            runtime_s=perf_counter() - started,
            fidelity=request.fidelity,
            statistics={
                "batched_matrix_solves": len(waveforms),
                "frequency_points": len(frequencies),
                "compiled_cache_entries": len(self._compiled),
                "compiled_plan_cache_entries": len(self._plans),
            },
        )

    def _error_result(
        self,
        compiled: LinearCompiledModel,
        request: SimulationRequest,
        status: SimulationStatus,
        code: str,
        message: str,
        started: float,
    ) -> SimulationResult:
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            model_manifest=compiled.model_manifest,
            status=status,
            diagnostics=(Diagnostic(code, "error", message),),
            runtime_s=perf_counter() - started,
            fidelity=request.fidelity,
        )


class AnalyticPowerSimulatorBackend:
    """Parameterized adapter for one registered ideal power equation set."""

    def __init__(
        self,
        *,
        backend_id: str,
        solver_id: str,
        parameter_type: type,
        dc_solver: Callable[[float, Any], Any],
        model_kinds: frozenset[str],
        version: str = "1.0.0",
    ) -> None:
        self.solver_id = solver_id
        self.parameter_type = parameter_type
        self.dc_solver = dc_solver
        self.capabilities = BackendCapabilities(
            backend_id=backend_id,
            backend_version=version,
            analysis_kinds=frozenset({"dc_operating_point", "dc_transfer"}),
            model_kinds=model_kinds,
            fidelities=frozenset({"analytic", "ideal_averaged"}),
        )
        self._compiled: dict[str, PowerCompiledModel] = {}

    def compile(self, graph: CircuitGraph) -> PowerCompiledModel:
        graph.require_valid()
        cached = self._compiled.get(graph.graph_hash)
        if cached is not None:
            return cached
        if graph.preferred_solver_ids and self.solver_id not in graph.preferred_solver_ids:
            raise UnsupportedModelError(
                f"graph does not declare solver {self.solver_id!r}"
            )
        model_kinds = {component.model.kind for component in graph.components}
        unsupported = model_kinds - self.capabilities.model_kinds
        if unsupported:
            raise UnsupportedModelError(
                f"{self.solver_id} does not support models {sorted(unsupported)}"
            )
        compiled = PowerCompiledModel(graph=graph, model_manifest=_model_manifest(graph))
        self._compiled[graph.graph_hash] = compiled
        return compiled

    def simulate(
        self,
        compiled: PowerCompiledModel,
        request: SimulationRequest,
    ) -> SimulationResult:
        started = perf_counter()
        if request.graph_id != compiled.graph.graph_id:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.FAILED,
                "graph_mismatch",
                "request graph does not match compiled power model",
                started,
            )
        if request.analysis.kind not in self.capabilities.analysis_kinds:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNSUPPORTED,
                "unsupported_analysis",
                f"{self.solver_id} does not support analysis {request.analysis.kind!r}",
                started,
            )
        try:
            parameters = self._parameters(request)
            input_values = _power_input_values(request)
            operating_points = tuple(self.dc_solver(value, parameters) for value in input_values)
            scalars, waveforms = _power_outputs(request, input_values, operating_points)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.FAILED,
                "invalid_parameters",
                str(exc),
                started,
            )
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            model_manifest=compiled.model_manifest,
            status=SimulationStatus.PASSED,
            scalars=scalars,
            waveforms=waveforms,
            runtime_s=perf_counter() - started,
            fidelity=request.fidelity,
            statistics={
                "equation_evaluations": len(input_values),
                "compiled_cache_entries": len(self._compiled),
            },
        )

    def _parameters(self, request: SimulationRequest):
        # A parameter record may declare fields that do not apply to every
        # stage it describes (for example a transformer turns ratio on a
        # non-isolated converter). Such a field is passed through whenever the
        # request provides it, and tolerated as absent otherwise so the record
        # can fall back to its own default instead of inventing a value.
        optional = set(getattr(self.parameter_type, "optional_parameter_fields", frozenset()))
        names = {item.name for item in fields(self.parameter_type)}
        values = {
            name: request.parameter_values[name]
            for name in names
            if name in request.parameter_values
        }
        missing = (names - optional) - values.keys()
        if missing:
            raise ValueError(f"missing power parameters {sorted(missing)}")
        return self.parameter_type(**values)

    def _error_result(
        self,
        compiled: PowerCompiledModel,
        request: SimulationRequest,
        status: SimulationStatus,
        code: str,
        message: str,
        started: float,
    ) -> SimulationResult:
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            model_manifest=compiled.model_manifest,
            status=status,
            diagnostics=(Diagnostic(code, "error", message),),
            runtime_s=perf_counter() - started,
            fidelity=request.fidelity,
        )


def _frequency_coordinates(request: SimulationRequest) -> tuple[float, ...]:
    sweep = next(
        (item for item in request.analysis.sweeps if item.variable == "frequency"),
        None,
    )
    if sweep is None:
        raise UnsupportedModelError("small_signal_ac requires a frequency sweep")
    return sweep.materialize()


def _linear_analysis_request(graph: CircuitGraph, request: SimulationRequest, observable) -> AnalysisRequest:
    target_node, target_reference = _port_nodes(graph, observable.target_id)
    source_name = _source_name(graph, request, observable.quantity)
    if observable.quantity == "voltage_gain":
        return AnalysisRequest.voltage_transfer(
            output_node=target_node,
            source_name=source_name,
            reference_node=target_reference,
        )
    if observable.quantity == "transimpedance":
        return AnalysisRequest.transimpedance(
            output_node=target_node,
            source_name=source_name,
            reference_node=target_reference,
        )
    if observable.quantity in {"input_impedance", "impedance"}:
        port = next(item for item in graph.ports if item.port_id == observable.target_id)
        if observable.quantity == "impedance" and port.direction == "output":
            return AnalysisRequest.output_impedance(
                output_node=target_node,
                source_name=observable.source_id or "Itest",
                reference_node=target_reference,
            )
        return AnalysisRequest.input_impedance(
            input_node=target_node,
            source_name=source_name,
            reference_node=target_reference,
        )
    if observable.quantity == "output_impedance":
        return AnalysisRequest.output_impedance(
            output_node=target_node,
            source_name=observable.source_id or "Itest",
            reference_node=target_reference,
        )
    raise UnsupportedModelError(
        f"linear_mna cannot produce observable quantity {observable.quantity!r}"
    )


def _source_name(graph: CircuitGraph, request: SimulationRequest, quantity: str) -> str:
    for excitation in request.excitations:
        if excitation.target_kind == "component":
            return excitation.target_id
    preferred_kind = "current_source" if quantity == "transimpedance" else "voltage_source"
    for component in graph.components:
        if component.model.kind == preferred_kind:
            return component.reference
    raise UnsupportedModelError(f"graph has no {preferred_kind} for {quantity}")


def _port_nodes(graph: CircuitGraph, port_id: str) -> tuple[str, str]:
    try:
        port = next(item for item in graph.ports if item.port_id == port_id)
    except StopIteration as exc:
        raise UnsupportedModelError(f"graph has no port {port_id!r}") from exc
    reference = next(
        (item.net_id for item in port.terminals if item.role == "reference"),
        None,
    )
    positive = next(
        (item.net_id for item in port.terminals if item.role != "reference"),
        None,
    )
    if positive is None or reference is None:
        raise UnsupportedModelError(f"port {port_id!r} is not a two-terminal electrical port")
    return positive, reference


def _power_input_values(request: SimulationRequest) -> tuple[float, ...]:
    if request.analysis.sweeps:
        sweep = request.analysis.sweeps[0]
        if sweep.variable in {"input_voltage", "input_voltage_v", "source_voltage"}:
            return sweep.materialize()
    for excitation in request.excitations:
        if excitation.kind == "dc_voltage":
            value = excitation.parameters.get("value")
            if value is not None:
                return (float(value),)
    value = request.conditions.variables.get("input_voltage_v")
    if value is not None:
        return (float(value),)
    raise ValueError("power simulation requires a DC input voltage excitation")


_POWER_UNITS = {
    "input_voltage_v": "V",
    "output_voltage_v": "V",
    "input_current_a": "A",
    "output_current_a": "A",
    "output_power_w": "W",
    "efficiency": "1",
    "predicted_ripple_mv": "mV",
    "inductor_ripple_a": "A",
}


def _power_outputs(request, input_values, operating_points):
    observables = request.requested_observables
    names = tuple(
        dict.fromkeys(
            (*tuple(item.observable_id for item in observables), *tuple(_POWER_UNITS))
        )
    )
    if len(input_values) == 1:
        values = operating_points[0].as_dict()
        return (
            {
                name: Quantity(values[name], _POWER_UNITS.get(name, ""))
                for name in names
                if name in values
            },
            {},
        )
    axis = ResultAxis("input_voltage", "V", tuple(input_values))
    waveforms = {}
    for name in names:
        values = tuple(point.as_dict().get(name) for point in operating_points)
        if all(value is not None for value in values):
            waveforms[name] = Waveform(
                observable_id=name,
                unit=_POWER_UNITS.get(name, ""),
                axes=(axis,),
                values=values,
            )
    return {}, waveforms


def _linear_plan_cache_key(
    graph: CircuitGraph,
    capabilities: BackendCapabilities,
) -> tuple[str, str, str, tuple[tuple[str, str], ...]]:
    model_checksum = tuple(
        sorted(
            (component.model.model_id, component.model.version)
            for component in graph.components
        )
    )
    return (
        graph.topology_hash,
        capabilities.backend_id,
        capabilities.backend_version,
        model_checksum,
    )


def _model_manifest(graph: CircuitGraph) -> tuple[ModelRef, ...]:
    unique: dict[tuple[str, str], ModelRef] = {}
    for component in graph.components:
        unique[(component.model.model_id, component.model.version)] = component.model
    return tuple(unique[key] for key in sorted(unique))


class _FixedLinearTemplate:
    params: tuple[Any, ...] = ()

    def __init__(self, circuit: LinearCircuit, analysis: AnalysisRequest) -> None:
        self._circuit = circuit
        self.output_node = analysis.output_node
        self.source_name = analysis.source_name

    def to_circuit(self, values: dict[str, Any]) -> LinearCircuit:
        return self._circuit
