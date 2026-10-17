"""Graph-backed nonlinear DC and small-signal AC using the common contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from ..devices import (
    NMOS,
    PMOS,
    ConvergenceError,
    Diode,
    DiodeParameters,
    NMOSParameters,
    PMOSParameters,
)
from ..graph import CircuitGraph, ModelRef, graph_to_linear_circuit, primitive_model_ref
from ..graph.primitives import NONLINEAR_PARAMETER_UNITS, parameter_unit
from ..mna import CompiledLinearMNA
from ..nonlinear import NonlinearMNA
from ..units import convert_scalar
from .backends import BackendCapabilities, UnsupportedModelError, _model_manifest
from .contracts import (
    Diagnostic,
    Quantity,
    ResultAxis,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    Waveform,
)


@dataclass(frozen=True)
class NonlinearCompiledModel:
    graph: CircuitGraph
    model_manifest: tuple[ModelRef, ...]


class NonlinearMNASimulatorBackend:
    """Nominal device equations, not switching, thermal or vendor-model SPICE.

    Fixed source values are DC biases. AC sources are zero unless an explicit
    ac_voltage/ac_current excitation supplies magnitude and phase_deg. DC and
    AC excitations can address the same component without overwriting each other.
    """

    capabilities = BackendCapabilities(
        backend_id="nonlinear_mna", backend_version="1.0.0",
        analysis_kinds=frozenset({"dc_operating_point", "small_signal_ac"}),
        model_kinds=frozenset({"R", "C", "voltage_source", "current_source", "vcvs", "diode", "nmos_channel", "pmos_channel"}),
        fidelities=frozenset({"nonlinear_device"}),
    )

    def compile(self, graph: CircuitGraph) -> NonlinearCompiledModel:
        graph.require_valid()
        if len(graph.domains) != 1:
            raise UnsupportedModelError("nonlinear_mna requires one electrical domain")
        references = [net.net_id for net in graph.nets if net.is_reference]
        if len(references) != 1 or graph.domains[0].reference_net_id != references[0]:
            raise UnsupportedModelError("nonlinear_mna requires one declared reference net")
        for component in graph.components:
            kind = component.model.kind
            if kind not in self.capabilities.model_kinds:
                raise UnsupportedModelError(f"nonlinear_mna does not support model {kind!r}")
            if component.model != primitive_model_ref(kind):
                raise UnsupportedModelError(
                    f"nonlinear_mna requires the builtin model identity/version/pins for {kind!r}"
                )
        return NonlinearCompiledModel(graph, _model_manifest(graph))

    def simulate(self, compiled: NonlinearCompiledModel, request: SimulationRequest) -> SimulationResult:
        started = perf_counter()

        def result(status, **payload):
            return SimulationResult(
                request_id=request.request_id, request_hash=request.request_hash,
                graph_hash=compiled.graph.graph_hash,
                backend_id=self.capabilities.backend_id, backend_version=self.capabilities.backend_version,
                model_manifest=compiled.model_manifest, status=status,
                runtime_s=perf_counter() - started, fidelity="nonlinear_device", **payload,
            )

        if request.graph_id != compiled.graph.graph_id:
            return result(SimulationStatus.FAILED, diagnostics=(Diagnostic(
                "graph_mismatch", "error", "request graph does not match compiled nonlinear model"),))
        try:
            self._validate_request(compiled.graph, request)
            circuit, devices, nodes = _materialize(compiled.graph, request)
            dc, ac = _source_circuits(compiled.graph, circuit, request)
            # A node can be connected only to nonlinear devices. Append those
            # real terminals without adding fictitious resistors to the graph.
            plan = CompiledLinearMNA.compile(dc).with_internal_nodes(
                sorted({node for device in devices for node, _ in device.terminals()} - {"0"})
            )
            options = request.analysis.options
            deadline = started + request.timeout_s if request.timeout_s is not None else None
            simulator = NonlinearMNA(dc, devices, compiled=plan,
                                     max_iterations=options.get("max_iterations", 200),
                                     tolerance=options.get("tolerance", 1e-9))
            bias = simulator.operating_point(deadline=deadline)
            scalars, waveforms = {}, {}
            if request.analysis.kind == "dc_operating_point":
                solution = bias
                frequencies = ()
            else:
                sweep = request.analysis.sweeps[0]
                frequencies = tuple(convert_scalar(v, sweep.unit, "Hz") for v in sweep.materialize())
                if any(v <= 0 for v in frequencies):
                    raise ValueError("AC frequency must be positive")
                solution = NonlinearMNA(ac, devices, compiled=plan).solve_ac(
                    np.asarray(frequencies), operating_point=bias
                )
            for observable in request.requested_observables:
                values, unit = _observe(compiled.graph, nodes, solution, ac, observable, bool(frequencies))
                values = values * convert_scalar(1.0, unit, observable.unit)
                if frequencies:
                    waveforms[observable.observable_id] = Waveform(
                        observable.observable_id, observable.unit,
                        (ResultAxis("frequency", "Hz", frequencies),),
                        tuple(complex(value) for value in values),
                    )
                else:
                    scalars[observable.observable_id] = Quantity(float(values), observable.unit)
            if deadline is not None and perf_counter() > deadline:
                raise TimeoutError("nonlinear simulation exceeded its time budget")
        except UnsupportedModelError as exc:
            return result(SimulationStatus.UNSUPPORTED, diagnostics=(Diagnostic("unsupported_request", "error", str(exc)),))
        except TimeoutError as exc:
            return result(SimulationStatus.TIMEOUT, diagnostics=(Diagnostic("timeout", "error", str(exc)),))
        except ConvergenceError as exc:
            return result(SimulationStatus.FAILED, diagnostics=(Diagnostic("nonconverged", "error", str(exc)),))
        except np.linalg.LinAlgError as exc:
            return result(SimulationStatus.FAILED, diagnostics=(Diagnostic("singular", "error", str(exc)),))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return result(SimulationStatus.FAILED, diagnostics=(Diagnostic("invalid_request", "error", str(exc)),))
        diagnostics = ()
        channel = [type(device).__name__ for device in devices
                   if isinstance(device, (NMOS, PMOS))]
        if channel:
            diagnostics = (Diagnostic("channel_only_model", "info",
                f"{'/'.join(sorted(set(channel)))} is a 27 C channel model without bulk, "
                "junctions, intrinsic capacitance or thermal effects."),)
        return result(SimulationStatus.PASSED, scalars=scalars, waveforms=waveforms,
                      diagnostics=diagnostics, statistics={
                          "operating_point": bias.as_dict(), "frequency_points": len(frequencies),
                          "node_mapping": nodes,
                      })

    def _validate_request(self, graph: CircuitGraph, request: SimulationRequest) -> None:
        if request.analysis.kind not in self.capabilities.analysis_kinds:
            raise UnsupportedModelError(f"unsupported nonlinear analysis {request.analysis.kind!r}")
        if request.fidelity not in {"unspecified", "nonlinear_device"}:
            raise UnsupportedModelError(f"cannot provide requested fidelity {request.fidelity!r}")
        if request.conditions.corner != "typical" or request.conditions.variables:
            raise UnsupportedModelError("nonlinear_mna does not model corners or additional operating variables")
        channel_kinds = {"nmos_channel", "pmos_channel"} & {c.model.kind for c in graph.components}
        if channel_kinds and request.conditions.temperature_c != 27.0:
            raise UnsupportedModelError(
                f"{'/'.join(sorted(channel_kinds))} is fixed at 27 C; temperature scaling is not implemented")
        if request.conditions.temperature_c <= -273.15:
            raise ValueError("temperature must be above absolute zero")
        sweeps = request.analysis.sweeps
        if request.analysis.kind == "dc_operating_point" and sweeps:
            raise UnsupportedModelError("dc_operating_point does not accept sweeps")
        if request.analysis.kind == "small_signal_ac" and (len(sweeps) != 1 or sweeps[0].variable != "frequency"):
            raise UnsupportedModelError("small_signal_ac requires exactly one frequency sweep")
        options = request.analysis.options
        if set(options) - {"max_iterations", "tolerance"}:
            raise UnsupportedModelError("unsupported nonlinear solver options")
        iterations, tolerance = options.get("max_iterations", 200), options.get("tolerance", 1e-9)
        if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not request.requested_observables:
            raise UnsupportedModelError("nonlinear request must declare observables")


def _materialize(graph, request):
    nodes = {net.net_id: "0" if net.is_reference else f"n{i}" for i, net in enumerate(graph.nets)}
    bound = {p.variable_id for c in graph.components for p in c.parameters if p.variable_id is not None}
    if set(request.parameter_values) != bound:
        raise ValueError(f"parameter values must match graph bindings; missing={sorted(bound - set(request.parameter_values))}, "
                         f"unknown={sorted(set(request.parameter_values) - bound)}")
    linear, devices = [], []
    for component in graph.components:
        kind = component.model.kind
        allowed = NONLINEAR_PARAMETER_UNITS.get(kind, {"gain": "1"} if kind == "vcvs" else {"value": parameter_unit(kind, "value")})
        parameters = {}
        for binding in component.parameters:
            if binding.name not in allowed:
                raise UnsupportedModelError(f"unsupported parameter {component.reference}.{binding.name}")
            if binding.expression is not None:
                raise UnsupportedModelError("nonlinear backend does not evaluate parameter expressions")
            value = binding.value if binding.variable_id is None else request.parameter_values[binding.variable_id]
            numeric = complex(value)
            if numeric.imag != 0:
                raise ValueError("nonlinear model parameters and DC source values must be real")
            parameters[binding.name] = convert_scalar(numeric.real, binding.unit, allowed[binding.name])
        terminals = {c.terminal: nodes[c.net_id] for c in component.connections}
        if kind == "diode":
            devices.append(Diode(component.reference, terminals["anode"], terminals["cathode"],
                                 DiodeParameters(**parameters, temperature_c=request.conditions.temperature_c)))
        elif kind == "nmos_channel":
            devices.append(NMOS(component.reference, terminals["drain"], terminals["gate"], terminals["source"],
                                NMOSParameters(**parameters)))
        elif kind == "pmos_channel":
            devices.append(PMOS(component.reference, terminals["drain"], terminals["gate"], terminals["source"],
                                PMOSParameters(**parameters)))
        else:
            linear.append(replace(component, parameters=tuple(
                replace(binding, value=parameters[binding.name], unit=allowed[binding.name], variable_id=None)
                for binding in component.parameters
            )))
    linear_graph = replace(graph, components=tuple(linear), variables=(),
                           nets=tuple(replace(net, name=nodes[net.net_id]) for net in graph.nets))
    return graph_to_linear_circuit(linear_graph), tuple(devices), nodes


def _component(graph, identifier):
    matches = [c for c in graph.components if identifier in {c.instance_id, c.reference}]
    if len(matches) != 1:
        raise UnsupportedModelError(f"component {identifier!r} is missing or ambiguous")
    return matches[0]


def _source_circuits(graph, circuit, request):
    dc_values, ac_values = {}, {}
    for excitation in request.excitations:
        if excitation.target_kind != "component":
            raise UnsupportedModelError("nonlinear source excitations must target a component")
        component = _component(graph, excitation.target_id)
        source_kind = component.model.kind
        suffix = {"voltage_source": "voltage", "current_source": "current"}.get(source_kind)
        if suffix is None or excitation.kind not in {f"ac_{suffix}", f"dc_{suffix}"}:
            raise UnsupportedModelError("excitation type does not match an independent source")
        is_ac = excitation.kind.startswith("ac_")
        if is_ac and request.analysis.kind != "small_signal_ac":
            raise UnsupportedModelError("AC excitation is invalid for DC operating-point analysis")
        allowed = {"magnitude", "phase_deg"} if is_ac else {"value"}
        if set(excitation.parameters) - allowed:
            raise UnsupportedModelError("unsupported source excitation parameter")
        values = ac_values if is_ac else dc_values
        if component.reference in values:
            raise ValueError("duplicate excitation for the same source and analysis")
        if is_ac:
            magnitude = float(excitation.parameters.get("magnitude", 1.0))
            if magnitude < 0:
                raise ValueError("AC magnitude must not be negative")
            value = magnitude * np.exp(1j * np.deg2rad(float(excitation.parameters.get("phase_deg", 0.0))))
        else:
            value = float(excitation.parameters["value"])
        values[component.reference] = value
    def build(values, ac):
        return replace(circuit, **{attribute: tuple(
            replace(source, value=values.get(source.name, 0.0 if ac else source.value))
            for source in getattr(circuit, attribute)
        ) for attribute in ("voltage_sources", "current_sources")})
    return build(dc_values, False), build(ac_values, True)


def _port_nets(graph, port_id):
    matches = [port for port in graph.ports if port.port_id == port_id]
    if len(matches) != 1 or len(matches[0].terminals) != 2:
        raise UnsupportedModelError("voltage observation requires a two-terminal port")
    terminals = matches[0].terminals
    refs = [terminal for terminal in terminals if terminal.role == "reference"]
    if len(refs) != 1:
        raise UnsupportedModelError("voltage port requires a reference terminal")
    return next(t.net_id for t in terminals if t.role != "reference"), refs[0].net_id


def _voltage(solution, nodes, positive, negative):
    for net in (positive, negative):
        if net not in nodes:
            raise UnsupportedModelError("voltage observation references an unknown net")
        if nodes[net] != "0" and nodes[net] not in solution.node_index:
            raise UnsupportedModelError("voltage observation references an unconnected net")
    return solution.node_voltage(nodes[positive]) - solution.node_voltage(nodes[negative])


def _observe(graph, nodes, solution, ac, observable, is_ac):
    if observable.representation == "tensor" or (is_ac and observable.representation == "real"):
        raise UnsupportedModelError("nonlinear backend returns scalar DC or complex AC observations")
    quantity = observable.quantity
    if quantity in {"current", "branch_current"}:
        if observable.target_kind != "component":
            raise UnsupportedModelError("branch current requires a component target")
        component = _component(graph, observable.target_id)
        if component.model.kind not in {"voltage_source", "vcvs"}:
            raise UnsupportedModelError("current observation requires a voltage-source branch (use a sense source)")
        return solution.branch_current(component.reference), "A"
    if quantity not in {"voltage", "voltage_gain", "transimpedance"}:
        raise UnsupportedModelError(f"unsupported nonlinear observable {quantity!r}")
    if observable.target_kind == "port":
        positive, negative = _port_nets(graph, observable.target_id)
    elif observable.target_kind == "net":
        positive = observable.target_id
        negative = observable.reference_id or next(n.net_id for n in graph.nets if n.is_reference)
    else:
        raise UnsupportedModelError("voltage observation target must be a net or port")
    voltage = _voltage(solution, nodes, positive, negative)
    if quantity == "voltage":
        if observable.target_kind == "port" and observable.reference_id is not None:
            raise UnsupportedModelError("a port voltage uses its own declared reference terminal")
        return voltage, "V"
    if not is_ac:
        raise UnsupportedModelError("gain and transimpedance require small_signal_ac")
    sources = (*ac.voltage_sources, *ac.current_sources)
    active = [source for source in sources if abs(source.value) > 0]
    if len(active) != 1:
        raise UnsupportedModelError("ratio observations require exactly one nonzero AC excitation")
    source = active[0]
    if observable.source_id is not None and _component(graph, observable.source_id).reference != source.name:
        raise UnsupportedModelError("ratio source_id does not match the AC excitation")
    expected = ac.voltage_sources if quantity == "voltage_gain" else ac.current_sources
    if source not in expected:
        raise UnsupportedModelError("ratio observable requires a compatible source type")
    denominator = source.value
    if observable.target_kind == "port" and observable.reference_id is not None:
        if quantity != "voltage_gain":
            raise UnsupportedModelError("transimpedance uses source_id, not a voltage reference port")
        denominator = _voltage(solution, nodes, *_port_nets(graph, observable.reference_id))
    if np.any(np.abs(denominator) == 0):
        raise ValueError("ratio reference has zero AC amplitude")
    return voltage / denominator, "1" if quantity == "voltage_gain" else "ohm"
