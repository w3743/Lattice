from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st
import pytest

from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.simulation import (
    AnalysisSpec,
    Diagnostic,
    Excitation,
    ObservableSpec,
    OperatingConditions,
    Quantity,
    ResultAxis,
    SimulationContractError,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    SweepSpec,
    Waveform,
)


FINITE = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


def _graph():
    return linear_circuit_to_graph(
        LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "out", 1000.0),
                LinearElement("C1", "C", "out", "0", 1e-9),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        ),
        name="simulation_contract_graph",
    )


def _request() -> SimulationRequest:
    graph = _graph()
    return SimulationRequest(
        request_id="ac_transfer_1",
        analysis=AnalysisSpec(
            kind="small_signal_ac",
            sweeps=(
                SweepSpec(
                    variable="frequency",
                    unit="Hz",
                    start=10.0,
                    stop=1e6,
                    points=6,
                    scale="log",
                ),
            ),
            options={"linearize_at": "dc_operating_point"},
        ),
        graph_id=graph.graph_id,
        parameter_values={"R1.value": 1000.0, "C1.value": 1e-9},
        excitations=(
            Excitation(
                excitation_id="vin_ac",
                kind="ac_voltage",
                target_kind="component",
                target_id="Vin",
                parameters={"magnitude": 1.0, "phase_deg": 0.0},
            ),
        ),
        conditions=OperatingConditions(temperature_c=27.0, variables={"supply_v": 5.0}),
        requested_observables=(
            ObservableSpec(
                observable_id="vout_over_vin",
                quantity="voltage_gain",
                unit="1",
                target_kind="port",
                target_id="output",
                reference_id="input",
                source_id="vin_ac",
                representation="complex",
            ),
        ),
        fidelity="reduced",
        timeout_s=5.0,
        random_seed=17,
    )


def _result(status: SimulationStatus = SimulationStatus.PASSED) -> SimulationResult:
    graph = _graph()
    request = _request()
    frequencies = request.analysis.sweeps[0].materialize()
    diagnostics = ()
    if status is not SimulationStatus.PASSED:
        diagnostics = (
            Diagnostic(
                code=f"backend_{status.value}",
                severity="error",
                message=f"backend returned {status.value}",
            ),
        )
    return SimulationResult(
        request_id=request.request_id,
        request_hash=request.request_hash,
        graph_hash=graph.graph_hash,
        backend_id="linear_mna",
        backend_version="1.0.0",
        model_manifest=tuple(component.model for component in graph.components),
        status=status,
        scalars={"dc_output_voltage": Quantity(4.95, "V", uncertainty=0.01)},
        waveforms={
            "vout_over_vin": Waveform(
                observable_id="vout_over_vin",
                unit="1",
                axes=(ResultAxis("frequency", "Hz", frequencies),),
                values=tuple(complex(1.0 / (1.0 + index), -0.1 * index) for index in range(6)),
            )
        },
        diagnostics=diagnostics,
        runtime_s=0.012,
        fidelity="reduced",
        statistics={"matrix_solves": 6, "cache_hit": False},
    )


def test_request_round_trip_and_hash_are_stable() -> None:
    request = _request()

    wire = json.loads(json.dumps(request.as_dict(), allow_nan=False))
    restored = SimulationRequest.from_dict(wire)

    assert restored.as_dict() == request.as_dict()
    assert restored.request_hash == request.request_hash
    assert restored.analysis.sweeps[0].materialize()[0] == 10.0
    assert restored.analysis.sweeps[0].materialize()[-1] == pytest.approx(1e6)


def test_request_hash_detects_tampering() -> None:
    wire = _request().as_dict()
    wire["parameter_values"]["R1.value"] = 2000.0

    with pytest.raises(SimulationContractError, match="request_hash mismatch"):
        SimulationRequest.from_dict(wire)


def test_result_round_trip_preserves_complex_waveforms_and_evidence_hash() -> None:
    result = _result()

    wire = json.loads(json.dumps(result.as_dict(), allow_nan=False))
    restored = SimulationResult.from_dict(wire)

    assert restored.as_dict() == result.as_dict()
    assert restored.succeeded
    assert restored.waveforms["vout_over_vin"].values[2] == result.waveforms[
        "vout_over_vin"
    ].values[2]
    assert restored.evidence_hash == result.evidence_hash


def test_result_evidence_hash_detects_tampering() -> None:
    wire = _result().as_dict()
    wire["scalars"]["dc_output_voltage"]["value"] = 9.9

    with pytest.raises(SimulationContractError, match="evidence_hash mismatch"):
        SimulationResult.from_dict(wire)


@pytest.mark.parametrize(
    "status",
    (
        SimulationStatus.FAILED,
        SimulationStatus.UNSUPPORTED,
        SimulationStatus.UNAVAILABLE,
        SimulationStatus.TIMEOUT,
    ),
)
def test_non_passing_backend_statuses_are_never_success(status: SimulationStatus) -> None:
    result = _result(status)

    assert not result.succeeded
    assert result.status is status


def test_passed_result_rejects_error_diagnostic() -> None:
    request = _request()
    graph = _graph()

    with pytest.raises(SimulationContractError, match="cannot contain error diagnostics"):
        SimulationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            graph_hash=graph.graph_hash,
            backend_id="linear_mna",
            backend_version="1.0.0",
            model_manifest=(),
            status=SimulationStatus.PASSED,
            diagnostics=(Diagnostic("singular", "error", "matrix is singular"),),
        )


def test_waveform_shape_must_match_all_axes() -> None:
    with pytest.raises(SimulationContractError, match="value count"):
        Waveform(
            observable_id="s_parameters",
            unit="1",
            axes=(
                ResultAxis("frequency", "Hz", (1e6, 2e6)),
                ResultAxis("port", "1", ("p1", "p2")),
            ),
            shape=(2, 2),
            values=(1 + 0j, 0.1 + 0j, 0.1 + 0j),
        )


@settings(max_examples=50, deadline=None)
@given(real_values=st.lists(FINITE, min_size=1, max_size=24))
def test_generated_complex_waveforms_are_json_round_trip_safe(real_values: list[float]) -> None:
    graph = _graph()
    request = _request()
    values = tuple(complex(value, -value / 3.0) for value in real_values)
    result = SimulationResult(
        request_id=request.request_id,
        request_hash=request.request_hash,
        graph_hash=graph.graph_hash,
        backend_id="property_backend",
        backend_version="1.0.0",
        model_manifest=(),
        status=SimulationStatus.PASSED,
        waveforms={
            "generated": Waveform(
                observable_id="generated",
                unit="1",
                axes=(ResultAxis("sample", "1", tuple(range(len(values)))),),
                values=values,
            )
        },
        fidelity="analytic",
    )

    restored = SimulationResult.from_dict(
        json.loads(json.dumps(result.as_dict(), allow_nan=False))
    )

    assert restored.waveforms["generated"].values == values
    assert restored.evidence_hash == result.evidence_hash
