from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pytest

from circuit_ai.analysis import AnalysisRequest, LinearACAnalyzer
from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.power import BoostParameters, solve_ideal_boost_dc
from circuit_ai.simulation import (
    AnalysisSpec,
    AnalyticPowerSimulatorBackend,
    Excitation,
    LinearMNASimulatorBackend,
    ObservableSpec,
    SimulationRequest,
    SimulationExecutor,
    SimulationResultCache,
    SimulationResult,
    SimulationStatus,
    SweepSpec,
    TorchMNASimulatorBackend,
    UnsupportedModelError,
)
from circuit_ai.topology_grammar import PowerTopologyGrammar


POWER_MODELS = frozenset(
    {"R", "C", "L", "ideal_switch", "ideal_diode", "ideal_transformer"}
)


def _linear_fixture():
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", 1000.0),
            LinearElement("C1", "C", "out", "0", 100e-9),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )
    return circuit, linear_circuit_to_graph(circuit, name="linear_backend_graph")


def _linear_request(graph, *, kind: str = "small_signal_ac") -> SimulationRequest:
    return SimulationRequest(
        request_id=f"linear_{kind}",
        analysis=AnalysisSpec(
            kind=kind,
            sweeps=(SweepSpec("frequency", "Hz", start=10.0, stop=1e6, points=25, scale="log"),),
        ),
        graph_id=graph.graph_id,
        excitations=(
            Excitation("vin", "ac_voltage", "component", "Vin", {"magnitude": 1.0}),
        ),
        requested_observables=(
            ObservableSpec(
                "gain",
                "voltage_gain",
                "1",
                "port",
                "output",
                reference_id="input",
                representation="complex",
            ),
        ),
        fidelity="linear_frequency_domain",
    )


def _boost_fixture():
    spec = {
        "name": "boost_backend",
        "ports": [
            {
                "name": "input",
                "role": "input",
                "terminals": [
                    {"name": "in", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
            {
                "name": "output",
                "role": "output",
                "terminals": [
                    {"name": "out", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
        ],
        "analyses": [
            {"kind": "dc_transfer", "source_port": "input", "output_port": "output"}
        ],
        "targets": [
            {
                "target_kind": "dc",
                "input_voltage_v": 5.0,
                "output_voltage_v": 10.0,
                "output_current_a": 10.0,
            }
        ],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_diode"],
            "max_component_count": 8,
        },
    }
    ir = pbdl_to_ir(spec)
    candidate = next(
        item for item in PowerTopologyGrammar().search(ir).candidates if item.family == "dc_boost"
    )
    return ir, candidate.graph


def _boost_backend() -> AnalyticPowerSimulatorBackend:
    return AnalyticPowerSimulatorBackend(
        backend_id="ideal_boost_averaged",
        solver_id="ideal_boost_averaged",
        parameter_type=BoostParameters,
        dc_solver=solve_ideal_boost_dc,
        model_kinds=POWER_MODELS,
    )


def _boost_request(graph, *, analysis_kind: str = "dc_transfer", missing: bool = False):
    parameters = {
        "duty_cycle": 0.5,
        "inductance_h": 100e-6,
        "capacitance_f": 470e-6,
        "switching_frequency_hz": 100e3,
        "load_ohm": 1.0,
    }
    if missing:
        parameters.pop("load_ohm")
    return SimulationRequest(
        request_id=f"boost_{analysis_kind}",
        analysis=AnalysisSpec(kind=analysis_kind),
        graph_id=graph.graph_id,
        parameter_values=parameters,
        excitations=(
            Excitation("input_dc", "dc_voltage", "port", "input", {"value": 5.0, "unit": "V"}),
        ),
        requested_observables=(
            ObservableSpec("output_voltage_v", "voltage", "V", "port", "output"),
            ObservableSpec("output_current_a", "current", "A", "port", "output"),
            ObservableSpec("efficiency", "efficiency", "1", "design", "boost_backend"),
        ),
        fidelity="ideal_averaged",
    )


def test_linear_backend_matches_legacy_analyzer_and_reuses_compilation() -> None:
    circuit, graph = _linear_fixture()
    request = _linear_request(graph)
    backend = LinearMNASimulatorBackend()

    compiled = backend.compile(graph)
    assert backend.compile(graph) is compiled
    result = backend.simulate(compiled, request)

    frequencies = np.asarray(request.analysis.sweeps[0].materialize())
    expected = LinearACAnalyzer().analyze(
        circuit,
        AnalysisRequest.voltage_transfer(),
        frequencies,
    ).values
    actual = np.asarray(result.waveforms["gain"].values)
    assert result.status is SimulationStatus.PASSED
    assert result.graph_hash == graph.graph_hash
    assert result.request_hash == request.request_hash
    assert np.max(np.abs(actual - expected)) < 1e-12
    assert result.statistics["matrix_solves"] == len(frequencies)


def test_linear_backend_reports_unsupported_analysis_separately() -> None:
    _, graph = _linear_fixture()
    request = _linear_request(graph, kind="transient")
    backend = LinearMNASimulatorBackend()

    result = backend.simulate(backend.compile(graph), request)

    assert result.status is SimulationStatus.UNSUPPORTED
    assert result.diagnostics[0].code == "unsupported_analysis"


def test_linear_backend_reports_singular_matrix_as_failed() -> None:
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", 1000.0),
            LinearElement("Rfloat", "R", "floating_a", "floating_b", 1000.0),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )
    graph = linear_circuit_to_graph(circuit, name="singular_graph")
    backend = LinearMNASimulatorBackend()

    result = backend.simulate(backend.compile(graph), _linear_request(graph))

    assert result.status is SimulationStatus.FAILED
    assert result.diagnostics[0].code == "singular"


def test_torch_backend_matches_numeric_mna_and_reuses_compilation() -> None:
    pytest.importorskip("torch")
    _, graph = _linear_fixture()
    request = _linear_request(graph)
    numeric = LinearMNASimulatorBackend()
    torch_backend = TorchMNASimulatorBackend()

    compiled = torch_backend.compile(graph)
    assert torch_backend.compile(graph) is compiled
    torch_result = torch_backend.simulate(compiled, request)
    numeric_result = numeric.simulate(numeric.compile(graph), request)

    assert torch_result.status is SimulationStatus.PASSED
    torch_values = np.asarray(torch_result.waveforms["gain"].values)
    numeric_values = np.asarray(numeric_result.waveforms["gain"].values)
    assert np.max(np.abs(torch_values - numeric_values)) < 1e-12
    assert torch_result.statistics["frequency_points"] == 25
    assert torch_result.statistics["batched_matrix_solves"] == 1


def test_analytic_boost_backend_matches_verified_equation() -> None:
    _, graph = _boost_fixture()
    backend = _boost_backend()
    request = _boost_request(graph)

    compiled = backend.compile(graph)
    assert backend.compile(graph) is compiled
    result = backend.simulate(compiled, request)

    expected = solve_ideal_boost_dc(
        5.0,
        BoostParameters(0.5, 100e-6, 470e-6, 100e3, 1.0),
    )
    assert result.status is SimulationStatus.PASSED
    assert result.scalars["output_voltage_v"].value == expected.output_voltage_v
    assert result.scalars["output_current_a"].value == expected.output_current_a
    assert result.scalars["efficiency"].value == expected.efficiency
    assert result.statistics["equation_evaluations"] == 1


def test_analytic_power_backend_emits_dc_sweep_waveforms() -> None:
    _, graph = _boost_fixture()
    backend = _boost_backend()
    base = _boost_request(graph)
    request = SimulationRequest(
        request_id="boost_sweep",
        analysis=AnalysisSpec(
            kind="dc_transfer",
            sweeps=(SweepSpec("input_voltage", "V", values=(4.0, 5.0, 6.0)),),
        ),
        graph_id=base.graph_id,
        parameter_values=base.parameter_values,
        excitations=base.excitations,
        requested_observables=base.requested_observables,
        fidelity=base.fidelity,
    )

    result = backend.simulate(backend.compile(graph), request)

    assert result.status is SimulationStatus.PASSED
    assert result.scalars == {}
    assert result.waveforms["output_voltage_v"].values == (8.0, 10.0, 12.0)
    assert result.statistics["equation_evaluations"] == 3


def test_analytic_power_backend_distinguishes_missing_parameters_and_unsupported_analysis() -> None:
    _, graph = _boost_fixture()
    backend = _boost_backend()
    compiled = backend.compile(graph)

    missing = backend.simulate(compiled, _boost_request(graph, missing=True))
    unsupported = backend.simulate(
        compiled,
        _boost_request(graph, analysis_kind="small_signal_ac"),
    )

    assert missing.status is SimulationStatus.FAILED
    assert missing.diagnostics[0].code == "invalid_parameters"
    assert unsupported.status is SimulationStatus.UNSUPPORTED
    assert unsupported.diagnostics[0].code == "unsupported_analysis"


def test_executor_maps_availability_compile_and_backend_failures() -> None:
    _, graph = _linear_fixture()
    request = _linear_request(graph)

    class UnavailableBackend:
        capabilities = LinearMNASimulatorBackend.capabilities

        def available(self):
            return False, "dependency missing"

        def compile(self, graph):
            raise AssertionError("compile must not run")

    class UnsupportedBackend:
        capabilities = LinearMNASimulatorBackend.capabilities

        def compile(self, graph):
            raise UnsupportedModelError("model x is unsupported")

    class CrashingBackend:
        capabilities = LinearMNASimulatorBackend.capabilities

        def compile(self, graph):
            return graph

        def simulate(self, compiled, request):
            raise RuntimeError("unexpected numerical failure")

    executor = SimulationExecutor()
    unavailable = executor.execute(UnavailableBackend(), graph, (request,))[0]
    unsupported = executor.execute(UnsupportedBackend(), graph, (request,))[0]
    crashed = executor.execute(CrashingBackend(), graph, (request,))[0]

    assert unavailable.status is SimulationStatus.UNAVAILABLE
    assert unavailable.diagnostics[0].code == "backend_unavailable"
    assert unsupported.status is SimulationStatus.UNSUPPORTED
    assert unsupported.diagnostics[0].code == "unsupported_model"
    assert crashed.status is SimulationStatus.FAILED
    assert crashed.diagnostics[0].code == "backend_exception"


def test_executor_rejects_mismatched_evidence_and_marks_posthoc_timeout() -> None:
    _, graph = _linear_fixture()
    request = replace(_linear_request(graph), timeout_s=0.001)

    class SlowBackend:
        capabilities = LinearMNASimulatorBackend.capabilities

        def compile(self, graph):
            return graph

        def simulate(self, compiled, current_request):
            time.sleep(0.01)
            return SimulationResult(
                request_id=current_request.request_id,
                request_hash=current_request.request_hash,
                graph_hash=compiled.graph_hash,
                backend_id="slow",
                backend_version="1",
                model_manifest=(),
                status=SimulationStatus.PASSED,
                fidelity=current_request.fidelity,
            )

    class MismatchedBackend(SlowBackend):
        def simulate(self, compiled, current_request):
            return SimulationResult(
                request_id="different_request",
                request_hash="0" * 64,
                graph_hash=compiled.graph_hash,
                backend_id="mismatched",
                backend_version="1",
                model_manifest=(),
                status=SimulationStatus.PASSED,
                fidelity=current_request.fidelity,
            )

    timeout = SimulationExecutor().execute(SlowBackend(), graph, (request,))[0]
    mismatched = SimulationExecutor().execute(
        MismatchedBackend(),
        graph,
        (replace(request, timeout_s=1.0),),
    )[0]

    assert timeout.status is SimulationStatus.TIMEOUT
    assert timeout.diagnostics[0].code == "timeout"
    assert mismatched.status is SimulationStatus.FAILED
    assert mismatched.diagnostics[0].code == "result_contract_mismatch"


def test_executor_semantic_cache_rebinds_request_identity_and_deduplicates() -> None:
    _, graph = _linear_fixture()
    first = _linear_request(graph)
    second = replace(
        first,
        request_id="same_physics_new_request",
        timeout_s=5.0,
        metadata={"trace_id": "observability-only"},
    )
    assert first.request_hash != second.request_hash
    assert first.semantic_hash == second.semantic_hash

    class CountingBackend:
        capabilities = LinearMNASimulatorBackend.capabilities

        def __init__(self) -> None:
            self.delegate = LinearMNASimulatorBackend()
            self.calls = 0

        def compile(self, graph):
            return self.delegate.compile(graph)

        def simulate(self, compiled, request):
            self.calls += 1
            return self.delegate.simulate(compiled, request)

    backend = CountingBackend()
    results = SimulationExecutor(cache=SimulationResultCache()).execute(
        backend,
        graph,
        (first, second),
    )

    assert backend.calls == 1
    assert results[0].request_id == first.request_id
    assert results[1].request_id == second.request_id
    assert results[1].statistics["cache_hit"] is True


def test_executor_budget_counts_unique_physical_requests() -> None:
    _, graph = _linear_fixture()
    first = _linear_request(graph)
    second = replace(
        first,
        request_id="different_sweep",
        analysis=AnalysisSpec(
            kind="small_signal_ac",
            sweeps=(SweepSpec("frequency", "Hz", values=(10.0, 100.0)),),
        ),
    )

    results = SimulationExecutor(
        cache=SimulationResultCache(),
        max_evaluations=1,
    ).execute(LinearMNASimulatorBackend(), graph, (first, second))

    assert results[0].status is SimulationStatus.PASSED
    assert results[1].status is SimulationStatus.FAILED
    assert results[1].diagnostics[0].code == "evaluation_budget_exhausted"
