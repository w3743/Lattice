"""Channel-only NMOS: independent KCL roots, derivatives and live SPICE checks."""

from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest
from scipy.optimize import brentq

from circuit_ai.devices import ConvergenceError, NMOS, NMOSParameters, NonlinearStamp
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.nonlinear import NonlinearMNA
from circuit_ai.spice_discovery import find_executable
from circuit_ai.graph import (ComponentInstance, ParameterBinding, ParameterVariable,
                              TerminalConnection, linear_circuit_to_graph, primitive_model_ref)
from circuit_ai.simulation import (AnalysisSpec, Excitation, NonlinearMNASimulatorBackend,
    ObservableSpec, OperatingConditions, SimulationExecutor, SimulationRequest,
    SimulationResult, SimulationResultCache, SimulationStatus, SweepSpec)


def clamped(vd=5.0, vg=3.5, vs=0.0):
    return LinearCircuit(elements=(), voltage_sources=(
        VoltageSource("VD", "d", "0", complex(vd)),
        VoltageSource("VG", "g", "0", complex(vg)),
        VoltageSource("VS", "s", "0", complex(vs)),
    ))


def loaded(vg=3.5, supply=5.0, resistance=1000.0):
    return LinearCircuit(
        elements=(LinearElement("RD", "R", "vdd", "d", resistance),),
        voltage_sources=(VoltageSource("VD", "vdd", "0", complex(supply)),
                         VoltageSource("VG", "g", "0", complex(vg))),
    )


def test_clamped_branch_current_obeys_kcl_and_preserves_gate_isolation():
    # Regression for the earlier 13.375 mA result: forgetting J*v on the RHS
    # gives exactly that magnitude instead of 3.4375 mA.
    mos = NMOS("M1", "d", "g", "s", NMOSParameters(channel_length_modulation_per_v=0.02))
    point = NonlinearMNA(clamped(), [mos]).operating_point()
    assert point.branch_current("VD") == pytest.approx(-0.0034375, abs=1e-10)
    assert point.branch_current("VS") == pytest.approx(0.0034375, abs=1e-10)
    assert point.branch_current("VG") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("vg", [0.0, 0.8, 1.0, 1.5, 2.5, 3.5, 5.0])
def test_resistive_load_matches_independent_scalar_root(vg):
    # Independent square-law load-line calculation, no device/stamping helpers.
    def drain_current(vd):
        overdrive = max(vg - 1.0, 0.0)
        return 0.001 * (overdrive * min(vd, overdrive) - min(vd, overdrive)**2 / 2)

    expected = brentq(lambda vd: (5.0 - vd) / 1000.0 - drain_current(vd), 0.0, 5.0)
    point = NonlinearMNA(loaded(vg), [NMOS("M1", "d", "g", "0")]).operating_point()
    assert point.node_voltage("d") == pytest.approx(expected, abs=1e-8)
    assert point.branch_current("VD") == pytest.approx(-drain_current(expected), abs=1e-11)
    if vg == 3.5:
        assert point.node_voltage("d") == pytest.approx(2.0, abs=1e-8)


@pytest.mark.parametrize("vd,vg,vs", [(5, 3.5, 0), (0.7, 3, 0.2), (5, 0, 0),
                                      (-2, 1.5, 0), (0.2, 3, 0.7), (2, 3, 2)])
def test_jacobian_matches_finite_difference_and_conserves_charge(vd, vg, vs):
    mos = NMOS("M1", "d", "g", "s", NMOSParameters(channel_length_modulation_per_v=0.02))
    voltages = dict(drain=vd, gate=vg, source=vs)
    stamp = mos.stamp(voltages)
    assert sum(stamp.terminal_currents.values()) == pytest.approx(0.0, abs=1e-15)
    for col in voltages:
        high, low = dict(voltages), dict(voltages)
        high[col] += 1e-6
        low[col] -= 1e-6
        for row in voltages:
            derivative = (mos.stamp(high).terminal_currents[row]
                          - mos.stamp(low).terminal_currents[row]) / 2e-6
            assert stamp.jacobian.get((row, col), 0.0) == pytest.approx(derivative, abs=2e-9)
        assert sum(stamp.jacobian.get((row, col), 0.0) for row in voltages) == pytest.approx(0.0)


def test_source_follower_and_tied_gate_drain():
    circuit = LinearCircuit(
        elements=(LinearElement("RS", "R", "s", "0", 1000.0),),
        voltage_sources=(VoltageSource("VD", "d", "0", 5.0),
                         VoltageSource("VG", "g", "0", 3.0)),
    )
    point = NonlinearMNA(circuit, [NMOS("M1", "d", "g", "s")]).operating_point()
    expected = brentq(lambda vs: vs / 1000 - 0.0005 * (2 - vs)**2, 0, 2)
    assert point.node_voltage("s") == pytest.approx(expected, abs=1e-8)
    diode_connected = NonlinearMNA(loaded(), [NMOS("M1", "d", "d", "0")]).operating_point()
    expected = brentq(lambda vd: (5 - vd) / 1000 - 0.0005 * (vd - 1)**2, 1, 5)
    assert diode_connected.node_voltage("d") == pytest.approx(expected, abs=1e-8)


def test_ac_transconductance_matches_dc_perturbation():
    mos = NMOS("M1", "d", "g", "0", NMOSParameters(channel_length_modulation_per_v=0.02))
    dc = loaded(vg=2.5)
    bias = NonlinearMNA(dc, [mos]).operating_point()
    # Same graph/index maps; independent sources now carry AC amplitudes.
    ac = replace(dc, voltage_sources=(VoltageSource("VD", "vdd", "0", 0.0),
                                     VoltageSource("VG", "g", "0", 1.0)))
    response = NonlinearMNA(ac, [mos]).solve_ac(np.array([1.0, 1e3]), operating_point=bias)
    eps = 1e-5
    high = NonlinearMNA(loaded(2.5 + eps), [mos]).operating_point().node_voltage("d")
    low = NonlinearMNA(loaded(2.5 - eps), [mos]).operating_point().node_voltage("d")
    assert response.node_voltage("d") == pytest.approx((high - low) / (2 * eps), abs=1e-7)
    assert response.branch_current("VG") == pytest.approx([0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("kwargs", [dict(width_m=0), dict(length_m=-1),
    dict(transconductance_a_per_v2=0), dict(threshold_voltage_v=float("nan")),
    dict(channel_length_modulation_per_v=-0.1), dict(width_m=float("inf"))])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        NMOSParameters(**kwargs)


def test_invalid_multi_terminal_stamp_rejected():
    class Broken(NMOS):
        def stamp(self, voltages):
            return NonlinearStamp(0.0, 0.0, {("drain", "gate"): float("nan")},
                                  {"drain": 0.0, "source": 0.0, "gate": 0.0})
    with pytest.raises(ConvergenceError, match="non-finite"):
        NonlinearMNA(clamped(), [Broken("M1", "d", "g", "s")]).operating_point()


def _ngspice(deck, files):
    executable = find_executable()
    if executable is None:
        pytest.skip("ngspice unavailable; independent analytic tests still run")
    # Some Windows builds do not support non-ASCII paths. Keep this temporary
    # external-simulator input separate from pytest's repository-local basetemp.
    with tempfile.TemporaryDirectory(prefix="lattice_mos_") as directory:
        root = Path(directory)
        netlist = root / "model.cir"
        netlist.write_text(deck, encoding="ascii")
        result = subprocess.run(
            [str(executable), "-n", "-b", "-o", str(root / "run.log"), str(netlist)],
            cwd=root, capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log = (root / "run.log").read_text(errors="replace")
        assert result.returncode == 0, log
        assert "error:" not in log.lower(), log
        for file in files:
            assert (root / file).is_file(), log
        return {file: np.loadtxt(root / file) for file in files}


@pytest.mark.parametrize("lam", [0.0, 0.02])
def test_channel_dc_sweep_against_live_ngspice(lam):
    gates = [0.0, 0.8, 1.0, 1.5, 2.5, 3.5]
    commands = "\n".join(
        f"alter VG = {vg + 0.7}\ndc VD -1.3 5.7 0.25\nwrdata sweep_{i}.dat v(d) i(VSENSE)"
        for i, vg in enumerate(gates)
    )
    files = [f"sweep_{i}.dat" for i in range(len(gates))]
    reference = _ngspice(f"""NMOS channel reference, nominal 27 C
VD drive 0 5
VSENSE drive d 0
VG g 0 3.5
VS s 0 0.7
VB b 0 -5
M1 d g s b MOD W=10u L=1u
.model MOD NMOS (LEVEL=1 VTO=1 KP=1e-4 LAMBDA={lam} GAMMA=0 PHI=0.6)
.options temp=27 tnom=27 reltol=1e-9 abstol=1e-14 vntol=1e-10
.control
set numdgt=15
{commands}
quit
.endc
.end
""", files)
    mos = NMOS("M1", "d", "g", "s", NMOSParameters(channel_length_modulation_per_v=lam))
    max_error = 0.0
    for i, vg in enumerate(gates):
        rows = reference[files[i]]
        assert rows.shape == (29, 4)
        for _, vd, _, measured in rows:
            point = NonlinearMNA(clamped(vd, vg + 0.7, 0.7), [mos]).operating_point()
            simulated = -point.branch_current("VD")
            max_error = max(max_error, abs(simulated - measured))
            assert simulated == pytest.approx(measured, rel=1e-6, abs=2e-10)
            assert point.branch_current("VS") == pytest.approx(simulated, abs=1e-12)
            assert point.branch_current("VG") == pytest.approx(0.0, abs=1e-12)
    print(f"MOS1 DC lambda={lam}: {len(gates) * 29} points, max current error={max_error:.3e} A")


def test_loaded_bias_and_ac_against_live_ngspice():
    data = _ngspice("""NMOS common-source amplifier, channel only
VD vdd 0 DC 5 AC 0
VG g 0 DC 2.5 AC 1
RD vdd d 1k
CLOAD d 0 1n
M1 d g 0 0 MOD W=10u L=1u
.model MOD NMOS (LEVEL=1 VTO=1 KP=1e-4 LAMBDA=0.02 GAMMA=0 PHI=0.6)
.options temp=27 tnom=27 reltol=1e-9 abstol=1e-14 vntol=1e-10
.control
set numdgt=15
op
wrdata bias.dat v(d) i(VD)
ac dec 2 1 1e6
wrdata ac.dat v(d) i(VD)
quit
.endc
.end
""", ["bias.dat", "ac.dat"])
    mos = NMOS("M1", "d", "g", "0", NMOSParameters(channel_length_modulation_per_v=0.02))
    circuit = loaded(vg=2.5)
    circuit = replace(circuit, elements=circuit.elements + (LinearElement("CL", "C", "d", "0", 1e-9),))
    bias = NonlinearMNA(circuit, [mos]).operating_point()
    assert bias.node_voltage("d") == pytest.approx(data["bias.dat"][1], abs=1e-7)
    assert bias.branch_current("VD") == pytest.approx(data["bias.dat"][3], abs=2e-10)
    ac_circuit = replace(circuit, voltage_sources=(VoltageSource("VD", "vdd", "0", 0.0),
                                                 VoltageSource("VG", "g", "0", 1.0)))
    reference = data["ac.dat"]
    assert reference.shape == (13, 6)
    ac = NonlinearMNA(ac_circuit, [mos]).solve_ac(reference[:, 0], operating_point=bias)
    expected = reference[:, 1] + 1j * reference[:, 2]
    assert ac.node_voltage("d") == pytest.approx(expected, abs=1e-7)
    assert ac.branch_current("VD") == pytest.approx(reference[:, 4] + 1j * reference[:, 5], abs=2e-10)
    print(f"MOS1 AC: 13 points, max voltage error={max(abs(ac.node_voltage('d') - expected)):.3e} V")
    # Exercise the actual graph/backend path against the same external run too.
    graph = nonlinear_graph()
    capacitor = ComponentInstance("load_cap", "CL", primitive_model_ref("C"),
        (TerminalConnection("p", "d"), TerminalConnection("n", "0")),
        (ParameterBinding("value", "nF", 1.0),))
    graph = replace(graph, components=graph.components + (capacitor,))
    request = replace(backend_request(graph, ac=True),
        analysis=AnalysisSpec("small_signal_ac", (SweepSpec("frequency", "Hz", values=tuple(reference[:, 0])),)),
        excitations=(Excitation("input", "ac_voltage", "component", "VG", {"magnitude": 1.0}),),
        requested_observables=(ObservableSpec("out", "voltage", "V", "port", "output"),))
    unified = execute_graph(graph, request)
    assert unified.status == SimulationStatus.PASSED, unified.diagnostics
    assert unified.waveforms["out"].values == pytest.approx(expected, abs=1e-7)


# Unified graph -> resolver -> executor -> evidence integration.


def nonlinear_graph(*, variable=False):
    graph = linear_circuit_to_graph(loaded(2.5), name="common_source", input_node="g", output_node="d")
    mos = ComponentInstance("mos", "M1", primitive_model_ref("nmos_channel"),
        (TerminalConnection("drain", "d"), TerminalConnection("gate", "g"), TerminalConnection("source", "0")),
        (ParameterBinding("width_m", "um", variable_id="width") if variable else ParameterBinding("width_m", "um", 10.0),
         ParameterBinding("channel_length_modulation_per_v", "1/V", 0.02)))
    return replace(graph,
        components=tuple(replace(c, instance_id="gate_bias") if c.reference == "VG" else c for c in graph.components) + (mos,),
        variables=(ParameterVariable("width", "continuous", "um", lower=1, upper=20),) if variable else (),
        family="arbitrary_name", preferred_solver_ids=("nonlinear_mna",))


def backend_request(graph, *, ac=False, **kwargs):
    return SimulationRequest(request_id="unified_nonlinear", graph_id=graph.graph_id,
        analysis=AnalysisSpec("small_signal_ac", (SweepSpec("frequency", "kHz", values=(1.0, 10.0)),)) if ac else AnalysisSpec("dc_operating_point"),
        conditions=OperatingConditions(temperature_c=27.0), fidelity="nonlinear_device",
        excitations=(Excitation("signal", "ac_voltage", "component", "gate_bias", {"magnitude": 2.0, "phase_deg": 60.0}),) if ac else (),
        requested_observables=(ObservableSpec("out", "voltage", "mV", "port", "output"),), **kwargs)


def execute_graph(graph, request):
    return SimulationExecutor(cache=SimulationResultCache()).execute(NonlinearMNASimulatorBackend(), graph, [request])[0]


def test_registry_and_executor_produce_roundtrippable_nonlinear_evidence():
    from circuit_ai.capabilities import CapabilityRequest, CapabilityResolver, CapabilityRole, build_default_registry
    graph = nonlinear_graph()
    resolution = CapabilityResolver(build_default_registry()).resolve(CapabilityRequest(
        role=CapabilityRole.SIMULATION_BACKEND, model_kinds=frozenset(c.model.kind for c in graph.components),
        analysis_kinds=frozenset({"dc_operating_point"}), solver_id="nonlinear_mna", family="arbitrary_name"))
    assert resolution.selected.target.capability_id == "simulation.nonlinear_mna"
    request = backend_request(graph)
    result = SimulationExecutor(cache=SimulationResultCache()).execute(resolution.selected.implementation, graph, [request])[0]
    assert result.status == SimulationStatus.PASSED, result.diagnostics
    # Independent load line for saturation: (5-v)/1000 = .001125*(1+.02*v).
    assert result.scalars["out"].value == pytest.approx((5 - 1.125) / 1.0225 * 1000, abs=1e-6)
    assert result.graph_hash == graph.graph_hash and result.request_hash == request.request_hash
    assert any(m.kind == "nmos_channel" for m in result.model_manifest)
    assert result.fidelity == "nonlinear_device"
    assert result.statistics["operating_point"]["residual"] < 1e-8
    assert SimulationResult.from_dict(result.as_dict()).evidence_hash == result.evidence_hash


def test_backend_ac_keeps_bias_and_phasor_separate_and_extracts_gain():
    graph = nonlinear_graph()
    request = backend_request(graph, ac=True)
    request = replace(request, requested_observables=request.requested_observables + (
        ObservableSpec("gain", "voltage_gain", "1", "port", "output", source_id="gate_bias"),
        ObservableSpec("supply_current", "current", "A", "component", "VD"),))
    result = execute_graph(graph, request)
    assert result.status == SimulationStatus.PASSED, result.diagnostics
    bias = (5 - 1.125) / 1.0225
    gain = -0.0015 * (1 + 0.02 * bias) / 0.0010225
    phasor = 2 * np.exp(1j * np.pi / 3)
    assert result.waveforms["gain"].values == pytest.approx([gain, gain], abs=1e-9)
    assert result.waveforms["out"].values == pytest.approx([gain * phasor * 1000] * 2, abs=1e-6)
    assert result.waveforms["supply_current"].values == pytest.approx([gain * phasor / 1000] * 2, abs=1e-9)
    assert result.waveforms["out"].axes[0].values == (1000, 10000)
    # DC override and AC amplitude on the same component must coexist.
    shifted = execute_graph(graph, replace(request, excitations=request.excitations + (
        Excitation("bias", "dc_voltage", "component", "gate_bias", {"value": 3.5}),)))
    assert shifted.status == SimulationStatus.PASSED, shifted.diagnostics
    assert shifted.waveforms["gain"].values != result.waveforms["gain"].values


def test_bound_device_parameters_change_results_without_stale_cache():
    graph = nonlinear_graph(variable=True)
    executor = SimulationExecutor(cache=SimulationResultCache(), max_workers=2)
    requests = [backend_request(graph, parameter_values={"width": width}) for width in (5.0, 10.0)]
    results = executor.execute(NonlinearMNASimulatorBackend(), graph, requests)
    assert all(r.status == SimulationStatus.PASSED for r in results)
    assert results[0].scalars["out"].value > results[1].scalars["out"].value
    assert results[0].request_hash != results[1].request_hash
    missing = execute_graph(graph, backend_request(graph))
    assert missing.status == SimulationStatus.FAILED
    assert "missing" in missing.diagnostics[0].message


@pytest.mark.parametrize("change", [
    {"analysis": AnalysisSpec("transient")},
    {"conditions": OperatingConditions(temperature_c=85)},
    {"conditions": OperatingConditions(temperature_c=27, corner="fast")},
    {"fidelity": "spice"},
    {"requested_observables": (ObservableSpec("noise", "noise", "V", "port", "output"),)},
    {"excitations": (Excitation("bad", "dc_current", "component", "VG", {"value": 1}),)},
])
def test_unsupported_nonlinear_requirements_never_pass(change):
    graph = nonlinear_graph()
    result = execute_graph(graph, replace(backend_request(graph), **change))
    assert result.status == SimulationStatus.UNSUPPORTED
    assert not result.scalars and not result.waveforms


def test_unknown_model_version_does_not_silently_use_builtin_equations():
    graph = nonlinear_graph()
    mos = graph.components[-1]
    graph = replace(graph, components=graph.components[:-1] + (replace(mos, model=replace(mos.model, version="2.0.0")),))
    result = execute_graph(graph, backend_request(graph))
    assert result.status == SimulationStatus.UNSUPPORTED
    assert result.diagnostics[0].code == "unsupported_model"


def test_nonconvergence_and_deadline_are_structured_failures():
    graph = nonlinear_graph()
    request = backend_request(graph)
    result = execute_graph(graph, replace(request, analysis=AnalysisSpec("dc_operating_point", options={"max_iterations": 1})))
    assert result.status == SimulationStatus.FAILED
    assert result.diagnostics[0].code == "nonconverged"
    timeout = execute_graph(graph, replace(request, timeout_s=1e-12))
    assert timeout.status == SimulationStatus.TIMEOUT
    assert timeout.diagnostics[0].code == "timeout"


def test_current_mirror_without_a_linear_dc_path_converges():
    from circuit_ai.mna import CurrentSource
    circuit = LinearCircuit(elements=(), voltage_sources=(VoltageSource("VD", "d", "0", 5.0),),
                            current_sources=(CurrentSource("Iref", "0", "x", 0.001),))
    graph = linear_circuit_to_graph(circuit, name="mirror", input_node="x", output_node="d")
    devices = tuple(ComponentInstance(name, name, primitive_model_ref("nmos_channel"), (
        TerminalConnection("drain", drain), TerminalConnection("gate", "x"), TerminalConnection("source", "0")))
        for name, drain in (("M1", "x"), ("M2", "d")))
    graph = replace(graph, components=graph.components + devices)
    request = replace(backend_request(graph), requested_observables=(
        ObservableSpec("current", "current", "mA", "component", "VD"),
        ObservableSpec("gate", "voltage", "V", "net", "x"),))
    result = execute_graph(graph, request)
    assert result.status == SimulationStatus.PASSED, result.diagnostics
    assert result.scalars["current"].value == pytest.approx(-1.0, abs=1e-7)
    assert result.scalars["gate"].value == pytest.approx(1 + np.sqrt(2), abs=1e-8)


def test_unconnected_voltage_is_not_fabricated_as_zero():
    from circuit_ai.graph import Net
    graph = nonlinear_graph()
    graph = replace(graph, nets=graph.nets + (Net("unused", "unused", "electrical"),))
    request = replace(backend_request(graph), requested_observables=(ObservableSpec("missing", "voltage", "V", "net", "unused"),))
    result = execute_graph(graph, request)
    assert result.status == SimulationStatus.UNSUPPORTED


def test_singular_final_nmos_network_cannot_pass_via_regularization():
    from circuit_ai.mna import CurrentSource
    circuit = LinearCircuit(elements=(), voltage_sources=(VoltageSource("VG", "g", "0", 0.0),),
                            current_sources=(CurrentSource("anchor", "d", "0", 0.0),))
    with pytest.raises(ConvergenceError, match="did not converge"):
        NonlinearMNA(circuit, [NMOS("M1", "d", "g", "0")], max_iterations=5).operating_point()


def test_graph_unit_mismatch_and_unknown_source_fail_explicitly():
    graph = nonlinear_graph()
    mos = graph.components[-1]
    wrong = replace(mos, parameters=(ParameterBinding("width_m", "V", 10.0),))
    malformed = replace(graph, components=graph.components[:-1] + (wrong,))
    assert execute_graph(malformed, backend_request(malformed)).status == SimulationStatus.FAILED
    request = replace(backend_request(graph), excitations=(Excitation("missing", "dc_voltage", "component", "absent", {"value": 1}),))
    assert execute_graph(graph, request).status == SimulationStatus.UNSUPPORTED


def test_gain_uses_the_declared_input_port_and_rejects_missing_reference():
    graph = nonlinear_graph()
    request = replace(backend_request(graph, ac=True), requested_observables=(
        ObservableSpec("gain", "voltage_gain", "1", "port", "output", reference_id="input"),))
    normal = execute_graph(graph, request)
    assert normal.status == SimulationStatus.PASSED
    # Selecting the output itself as reference must produce unity, not silently
    # retain the independent-source denominator.
    unity = execute_graph(graph, replace(request, requested_observables=(
        replace(request.requested_observables[0], reference_id="output"),)))
    assert unity.status == SimulationStatus.PASSED
    assert unity.waveforms["gain"].values == pytest.approx([1.0, 1.0])
    missing = execute_graph(graph, replace(request, requested_observables=(
        replace(request.requested_observables[0], reference_id="absent"),)))
    assert missing.status == SimulationStatus.UNSUPPORTED
