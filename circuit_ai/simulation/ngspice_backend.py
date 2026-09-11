"""External ngspice backend for unified simulation requests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np

from ..graph import (
    CircuitGraph,
    ModelRef,
    graph_parameter_defaults,
    graph_to_linear_circuit,
)
from ..mna import LinearCircuit
from ..spice import parse_ngspice_ascii_raw
from .backends import BackendCapabilities, UnsupportedModelError
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
class NgspiceCompiledModel:
    graph: CircuitGraph
    circuit: LinearCircuit
    model_manifest: tuple[ModelRef, ...]
    topology_template: "NgspiceTopologyTemplate | None" = None


@dataclass(frozen=True)
class NgspiceTopologyTemplate:
    """Reusable netlist cards whose topology is independent of values."""

    element_cards: tuple[tuple[str, str, str, str], ...]
    vcvs_cards: tuple[tuple[str, str, str, str, str, str], ...]

    def render(self, circuit: LinearCircuit) -> list[str]:
        values = {item.name: item.value for item in circuit.elements}
        values.update({item.name: item.gain for item in circuit.controlled_voltage_sources})
        lines: list[str] = []
        for name, n1, n2, token in self.element_cards:
            lines.append(f".param {token}={_spice_number(values[name])}")
            lines.append(f"{name} {n1} {n2} {{{token}}}")
        for name, n_plus, n_minus, control_plus, control_minus, token in self.vcvs_cards:
            lines.append(f".param {token}={_spice_number(values[name])}")
            lines.append(
                f"{name} {n_plus} {n_minus} {control_plus} {control_minus} {{{token}}}"
            )
        return lines


class NgspiceSimulatorBackend:
    capabilities = BackendCapabilities(
        backend_id="ngspice",
        backend_version="unknown",
        analysis_kinds=frozenset(
            {"small_signal_ac", "dc_operating_point", "dc_transfer", "transient"}
        ),
        model_kinds=frozenset(
            {"R", "C", "L", "voltage_source", "current_source", "vcvs"}
        ),
        fidelities=frozenset({"spice"}),
    )

    def __init__(
        self,
        executable: str = "ngspice",
        *,
        extra_args: tuple[str, ...] = (),
        timeout_s: float = 30.0,
        backend_version: str = "unknown",
    ) -> None:
        self.executable = executable
        self.extra_args = extra_args
        self.timeout_s = timeout_s
        self.backend_version = backend_version
        self._compiled: dict[str, NgspiceCompiledModel] = {}
        self._topology_templates: dict[
            tuple[str, str, tuple[tuple[str, str], ...]], NgspiceTopologyTemplate
        ] = {}
        self._execution_lock = Lock()

    def available(self) -> tuple[bool, str]:
        available = Path(self.executable).is_file() or shutil.which(self.executable) is not None
        if available:
            return True, f"ngspice executable is available: {self.executable}"
        return False, f"ngspice executable is unavailable: {self.executable}"

    def compile(self, graph: CircuitGraph) -> NgspiceCompiledModel:
        graph.require_valid()
        cached = self._compiled.get(graph.graph_hash)
        if cached is not None:
            return cached
        model_kinds = {component.model.kind for component in graph.components}
        unsupported = model_kinds - self.capabilities.model_kinds
        if unsupported:
            raise UnsupportedModelError(
                f"ngspice linear adapter does not support models {sorted(unsupported)}"
            )
        circuit = graph_to_linear_circuit(graph, graph_parameter_defaults(graph))
        template_key = (
            graph.topology_hash,
            self.backend_version,
            tuple(
                sorted(
                    (component.model.model_id, component.model.version)
                    for component in graph.components
                )
            ),
        )
        topology_template = self._topology_templates.get(template_key)
        if topology_template is None:
            topology_template = _build_topology_template(circuit)
            self._topology_templates[template_key] = topology_template
        compiled = NgspiceCompiledModel(
            graph=graph,
            circuit=circuit,
            model_manifest=_model_manifest(graph),
            topology_template=topology_template,
        )
        self._compiled[graph.graph_hash] = compiled
        return compiled

    def simulate(
        self,
        compiled: NgspiceCompiledModel,
        request: SimulationRequest,
    ) -> SimulationResult:
        started = perf_counter()
        available, reason = self.available()
        if not available:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNAVAILABLE,
                "backend_unavailable",
                reason,
                started,
            )
        if request.graph_id != compiled.graph.graph_id:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.FAILED,
                "graph_mismatch",
                "request graph does not match compiled ngspice model",
                started,
            )
        if request.analysis.kind not in self.capabilities.analysis_kinds:
            return self._error_result(
                compiled,
                request,
                SimulationStatus.UNSUPPORTED,
                "unsupported_analysis",
                f"ngspice adapter does not support analysis {request.analysis.kind!r}",
                started,
            )

        with tempfile.TemporaryDirectory(prefix="circuit_ai_ngspice_") as directory:
            work_dir = Path(directory)
            raw_path = work_dir / "result.raw"
            netlist_path = work_dir / "simulation.cir"
            try:
                netlist_path.write_text(
                    build_ngspice_netlist(compiled, request, raw_path.name),
                    encoding="utf-8",
                )
            except (UnsupportedModelError, ValueError) as exc:
                code = (
                    "invalid_parameters"
                    if "parameter" in str(exc).casefold()
                    else "unsupported_request"
                )
                return self._error_result(
                    compiled,
                    request,
                    SimulationStatus.FAILED
                    if code == "invalid_parameters"
                    else SimulationStatus.UNSUPPORTED,
                    code,
                    str(exc),
                    started,
                )
            timeout = request.timeout_s or self.timeout_s
            queue_started = perf_counter()
            acquired = self._execution_lock.acquire(timeout=timeout)
            if not acquired:
                return self._error_result(
                    compiled,
                    request,
                    SimulationStatus.TIMEOUT,
                    "spice_queue_timeout",
                    f"ngspice queue wait exceeded the {timeout:g} s execution budget",
                    started,
                )
            try:
                queue_wait_s = perf_counter() - queue_started
                remaining_timeout = max(1e-9, timeout - queue_wait_s)
                try:
                    completed = subprocess.run(
                        [self.executable, *self.extra_args, "-b", str(netlist_path)],
                        cwd=work_dir,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=remaining_timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return self._error_result(
                        compiled,
                        request,
                        SimulationStatus.TIMEOUT,
                        "timeout",
                        f"ngspice exceeded the {timeout:g} s execution budget",
                        started,
                    )
            finally:
                self._execution_lock.release()
            output = (completed.stderr or completed.stdout or "").strip()
            self._capture_version(completed.stdout + "\n" + completed.stderr)
            if completed.returncode != 0:
                code = _ngspice_failure_code(output)
                return self._error_result(
                    compiled,
                    request,
                    SimulationStatus.FAILED,
                    code,
                    output[:2000] or f"ngspice exited with code {completed.returncode}",
                    started,
                )
            if not raw_path.is_file():
                return self._error_result(
                    compiled,
                    request,
                    SimulationStatus.FAILED,
                    "output_missing",
                    "ngspice completed without producing the requested raw file",
                    started,
                )
            try:
                raw = parse_ngspice_ascii_raw(
                    raw_path.read_text(encoding="utf-8", errors="replace")
                )
                scalars, waveforms = _extract_outputs(compiled.graph, request, raw)
            except (KeyError, ValueError) as exc:
                return self._error_result(
                    compiled,
                    request,
                    SimulationStatus.FAILED,
                    "parse_failed",
                    str(exc),
                    started,
                )
        return SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=compiled.graph.graph_hash,
            backend_id=self.capabilities.backend_id,
            backend_version=self.backend_version,
            model_manifest=compiled.model_manifest,
            status=SimulationStatus.PASSED,
            scalars=scalars,
            waveforms=waveforms,
            runtime_s=perf_counter() - started,
            fidelity="spice",
            statistics={
                "process_returncode": 0,
                "compiled_cache_entries": len(self._compiled),
                "topology_template_cache_entries": len(self._topology_templates),
                "queue_wait_s": queue_wait_s,
            },
        )

    def _capture_version(self, output: str) -> None:
        if self.backend_version != "unknown":
            return
        match = re.search(r"ngspice[- ](?:revision )?([0-9][\w.-]*)", output, re.IGNORECASE)
        if match:
            self.backend_version = match.group(1)

    def _error_result(
        self,
        compiled: NgspiceCompiledModel,
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
            backend_version=self.backend_version,
            model_manifest=compiled.model_manifest,
            status=status,
            diagnostics=(Diagnostic(code, "error", message),),
            runtime_s=perf_counter() - started,
            fidelity="spice",
        )


def build_ngspice_netlist(
    compiled: NgspiceCompiledModel,
    request: SimulationRequest,
    raw_filename: str,
) -> str:
    circuit = graph_to_linear_circuit(compiled.graph, request.parameter_values)
    lines = [f"* CircuitGraph {compiled.graph.graph_id} via unified ngspice backend"]
    if compiled.topology_template is not None:
        lines.extend(compiled.topology_template.render(circuit))
    else:
        for element in circuit.elements:
            lines.append(
                f"{element.name} {element.n1} {element.n2} {_spice_number(element.value)}"
            )
    for source in circuit.voltage_sources:
        lines.append(_source_card(source, request, voltage=True))
    for source in circuit.current_sources:
        lines.append(_source_card(source, request, voltage=False))
    if compiled.topology_template is None:
        for source in circuit.controlled_voltage_sources:
            lines.append(
                f"{source.name} {source.n_plus} {source.n_minus} "
                f"{source.control_plus} {source.control_minus} {_spice_number(source.gain)}"
            )
    command = _analysis_command(compiled.graph, request)
    lines.extend(
        [
            ".option filetype=ascii",
            ".control",
            "set filetype=ascii",
            command,
            f"write {raw_filename} all",
            "quit",
            ".endc",
            ".end",
        ]
    )
    return "\n".join(lines) + "\n"


def _source_card(source: Any, request: SimulationRequest, *, voltage: bool) -> str:
    excitation = _source_excitation(source, request, voltage=voltage)
    prefix = f"{source.name} {source.n_plus} {source.n_minus}"
    base_value = float(complex(source.value).real)
    if excitation is None:
        return f"{prefix} DC {_spice_number(base_value)}"
    parameters = excitation.parameters
    if excitation.kind in {"ac_voltage", "ac_current"}:
        magnitude = float(parameters.get("magnitude", 1.0))
        phase = float(parameters.get("phase_deg", 0.0))
        return (
            f"{prefix} DC {_spice_number(base_value)} "
            f"AC {_spice_number(magnitude)} {_spice_number(phase)}"
        )
    if excitation.kind in {"dc_voltage", "dc_current"}:
        return f"{prefix} DC {_spice_number(parameters.get('value', base_value))}"
    if excitation.kind == "sine":
        offset = float(parameters.get("offset_v", parameters.get("offset", 0.0)))
        amplitude = float(parameters.get("amplitude_v", parameters.get("amplitude", 1.0)))
        frequency = float(parameters.get("frequency_hz", parameters.get("frequency", 1.0)))
        phase = float(parameters.get("phase_deg", 0.0))
        return (
            f"{prefix} SIN({_spice_number(offset)} {_spice_number(amplitude)} "
            f"{_spice_number(frequency)} 0 0 {_spice_number(phase)})"
        )
    raise UnsupportedModelError(f"ngspice source waveform {excitation.kind!r} is unsupported")


def _source_excitation(source: Any, request: SimulationRequest, *, voltage: bool):
    expected = {"ac_voltage", "dc_voltage", "sine"} if voltage else {"ac_current", "dc_current"}
    for excitation in request.excitations:
        if excitation.kind not in expected:
            continue
        if excitation.target_kind == "component" and excitation.target_id == source.name:
            return excitation
        if excitation.target_kind == "port":
            return excitation
    return None


def _analysis_command(graph: CircuitGraph, request: SimulationRequest) -> str:
    kind = request.analysis.kind
    if kind == "small_signal_ac":
        sweep = _required_sweep(request, "frequency")
        values = sweep.materialize()
        if len(values) < 2 or any(value <= 0.0 for value in values):
            raise UnsupportedModelError("ngspice AC sweep requires at least two positive points")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise UnsupportedModelError("ngspice AC sweep must be strictly increasing")
        step = values[1] - values[0]
        tolerance = max(abs(step) * 1e-9, 1e-15)
        if all(abs((right - left) - step) <= tolerance for left, right in zip(values, values[1:])):
            return f"ac lin {len(values)} {_spice_number(values[0])} {_spice_number(values[-1])}"
        if request.analysis.options.get("allow_native_sweep_approximation") is True:
            decades = math.log10(values[-1] / values[0])
            if decades <= 0.0:
                raise UnsupportedModelError("ngspice AC sweep must be increasing")
            points_per_decade = max(1, round((len(values) - 1) / decades))
            return (
                f"ac dec {points_per_decade} {_spice_number(values[0])} "
                f"{_spice_number(values[-1])}"
            )
        raise UnsupportedModelError(
            "ngspice AC adapter cannot represent explicit non-linear frequency points exactly"
        )
    if kind == "dc_operating_point" or (kind == "dc_transfer" and not request.analysis.sweeps):
        return "op"
    if kind == "dc_transfer":
        sweep = request.analysis.sweeps[0]
        values = sweep.materialize()
        if len(values) < 2:
            raise UnsupportedModelError("ngspice DC transfer requires at least two points")
        step = values[1] - values[0]
        if any(abs((right - left) - step) > max(abs(step) * 1e-9, 1e-15) for left, right in zip(values, values[1:])):
            raise UnsupportedModelError("ngspice DC adapter requires a linear sweep")
        source = _swept_source_name(graph, request)
        return f"dc {source} {_spice_number(values[0])} {_spice_number(values[-1])} {_spice_number(step)}"
    if kind == "transient":
        sweep = _required_sweep(request, "time")
        values = sweep.materialize()
        if len(values) < 2:
            raise UnsupportedModelError("ngspice transient analysis requires at least two times")
        step = (values[-1] - values[0]) / (len(values) - 1)
        return f"tran {_spice_number(step)} {_spice_number(values[-1])} {_spice_number(values[0])}"
    raise UnsupportedModelError(f"unsupported ngspice analysis {kind!r}")


def _required_sweep(request: SimulationRequest, variable: str):
    sweep = next((item for item in request.analysis.sweeps if item.variable == variable), None)
    if sweep is None:
        raise UnsupportedModelError(f"analysis requires a {variable} sweep")
    return sweep


def _swept_source_name(graph: CircuitGraph, request: SimulationRequest) -> str:
    for excitation in request.excitations:
        if excitation.target_kind == "component":
            return excitation.target_id
    source = next(
        (component.reference for component in graph.components if component.model.kind == "voltage_source"),
        None,
    )
    if source is None:
        raise UnsupportedModelError("DC transfer requires an independent voltage source")
    return source


def _extract_outputs(graph: CircuitGraph, request: SimulationRequest, raw):
    kind = request.analysis.kind
    if kind == "small_signal_ac":
        axis_name, axis_unit, axis_values = "frequency", "Hz", tuple(float(item.real) for item in raw["frequency"])
    elif kind == "transient":
        axis_name, axis_unit, axis_values = "time", "s", tuple(float(item.real) for item in raw["time"])
    elif kind == "dc_transfer" and request.analysis.sweeps:
        key = next((name for name in raw if name not in {"time", "frequency"} and not name.startswith("v(") and not name.startswith("i(")), None)
        axis_values = request.analysis.sweeps[0].materialize() if key is None else tuple(float(item.real) for item in raw[key])
        axis_name, axis_unit = request.analysis.sweeps[0].variable, request.analysis.sweeps[0].unit
    else:
        scalars = {}
        for observable in request.requested_observables:
            values = _observable_values(graph, request, raw, observable)
            scalars[observable.observable_id] = Quantity(values[-1], observable.unit)
        return scalars, {}

    axis = ResultAxis(axis_name, axis_unit, axis_values)
    waveforms = {}
    for observable in request.requested_observables:
        values = _observable_values(graph, request, raw, observable)
        waveforms[observable.observable_id] = Waveform(
            observable_id=observable.observable_id,
            unit=observable.unit,
            axes=(axis,),
            values=tuple(values),
            attributes={"external_backend": "ngspice"},
        )
    return {}, waveforms


def _observable_values(graph, request, raw, observable):
    positive, reference = _port_nodes(graph, observable.target_id)
    voltage = _raw_voltage(raw, positive) - _raw_voltage(raw, reference)
    if observable.quantity == "voltage_gain":
        return voltage / _excitation_phasor(request, voltage_source=True)
    if observable.quantity == "transimpedance":
        return voltage / _excitation_phasor(request, voltage_source=False)
    if observable.quantity in {"voltage", "port_voltage"}:
        return voltage
    raise UnsupportedModelError(
        f"ngspice output adapter does not support observable {observable.quantity!r}"
    )


def _raw_voltage(raw, node: str):
    if node in {"0", "gnd", "GND"}:
        length = len(next(iter(raw.values())))
        return np.zeros(length, dtype=np.complex128)
    for key in (f"v({node})".casefold(), node.casefold()):
        if key in raw:
            return raw[key]
    raise KeyError(f"ngspice raw output has no voltage for node {node!r}")


def _port_nodes(graph: CircuitGraph, port_id: str) -> tuple[str, str]:
    port = next((item for item in graph.ports if item.port_id == port_id), None)
    if port is None:
        raise UnsupportedModelError(f"graph has no port {port_id!r}")
    positive = next((item.net_id for item in port.terminals if item.role != "reference"), None)
    reference = next((item.net_id for item in port.terminals if item.role == "reference"), None)
    if positive is None or reference is None:
        raise UnsupportedModelError(f"port {port_id!r} is not a two-terminal electrical port")
    return positive, reference


def _excitation_phasor(request: SimulationRequest, *, voltage_source: bool) -> complex:
    kinds = {"ac_voltage"} if voltage_source else {"ac_current"}
    excitation = next((item for item in request.excitations if item.kind in kinds), None)
    if excitation is None:
        raise UnsupportedModelError("AC ratio observable has no compatible excitation")
    magnitude = float(excitation.parameters.get("magnitude", 1.0))
    phase = math.radians(float(excitation.parameters.get("phase_deg", 0.0)))
    return magnitude * complex(math.cos(phase), math.sin(phase))


def _ngspice_failure_code(output: str) -> str:
    text = output.casefold()
    if "singular matrix" in text:
        return "singular"
    if any(marker in text for marker in ("no convergence", "failed to converge", "timestep too small")):
        return "nonconverged"
    return "execution_failed"


def _spice_number(value: Any) -> str:
    number = complex(value)
    if abs(number.imag) > 1e-15:
        raise UnsupportedModelError("ngspice card requires a real scalar value")
    if not math.isfinite(number.real):
        raise UnsupportedModelError("ngspice card value must be finite")
    return f"{number.real:.12g}"


def _build_topology_template(circuit: LinearCircuit) -> NgspiceTopologyTemplate:
    element_cards = tuple(
        (
            element.name,
            element.n1,
            element.n2,
            f"p_{index}_{element.name}",
        )
        for index, element in enumerate(circuit.elements)
    )
    vcvs_offset = len(circuit.elements)
    vcvs_cards = tuple(
        (
            source.name,
            source.n_plus,
            source.n_minus,
            source.control_plus,
            source.control_minus,
            f"p_{vcvs_offset + index}_{source.name}",
        )
        for index, source in enumerate(circuit.controlled_voltage_sources)
    )
    return NgspiceTopologyTemplate(element_cards, vcvs_cards)


def _model_manifest(graph: CircuitGraph) -> tuple[ModelRef, ...]:
    unique = {
        (component.model.model_id, component.model.version): component.model
        for component in graph.components
    }
    return tuple(unique[key] for key in sorted(unique))
