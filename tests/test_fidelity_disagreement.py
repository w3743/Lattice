"""Multi-fidelity backend disagreement (plan §18.2 多保真 disagreement 测试, §7.3 规则 4).

Requirement (``计划:652-653`` rule 4): "后端间差异超过门槛时，候选标记为
model_disagreement 并进入诊断队列" -- when the same candidate evaluated at two fidelity
levels disagrees by more than the configured threshold, the low-fidelity conclusion must
not be accepted silently: the candidate is marked ``model_disagreement`` and enters the
diagnostics queue.

Status: **not implemented**.  ``grep -rn "model_disagreement" circuit_ai/`` returns no hit
(the only disagreement notion in the tree is the surrogate ensemble spread at
``circuit_ai/surrogate.py:314``).  Per the task rules this needs new result fields/enum
values, so it is reported rather than patched.  The behavioural test below is therefore
kept skipped; see ``outputs/_baseline_report.md`` §E4 for the un-skip conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.simulation import (
    AnalysisSpec,
    Excitation,
    LinearMNASimulatorBackend,
    ObservableSpec,
    SimulationRequest,
    SimulationStatus,
    SweepSpec,
    TorchMNASimulatorBackend,
)


DISAGREEMENT_THRESHOLD_DB = 1.0
DISAGREEMENT_CODE = "model_disagreement"


def _graph(*, capacitance_f: float):
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", 1000.0),
            LinearElement("C1", "C", "out", "0", capacitance_f),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )
    return linear_circuit_to_graph(circuit, name="two_fidelity_graph")


def _request(graph) -> SimulationRequest:
    return SimulationRequest(
        request_id="two_fidelity_ac",
        analysis=AnalysisSpec(
            kind="small_signal_ac",
            sweeps=(SweepSpec("frequency", "Hz", start=10.0, stop=1e5, points=25, scale="log"),),
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


def _magnitude_db(result) -> np.ndarray:
    values = np.asarray(result.waveforms["gain"].values)
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-300))


def test_low_and_high_fidelity_internal_backends_agree_for_identical_candidates() -> None:
    """Positive control: with identical parameters both internal backends agree."""
    pytest.importorskip("torch")
    graph = _graph(capacitance_f=100e-9)
    request = _request(graph)

    low = LinearMNASimulatorBackend().simulate(LinearMNASimulatorBackend().compile(graph), request)
    high = TorchMNASimulatorBackend().simulate(TorchMNASimulatorBackend().compile(graph), request)

    assert low.status is SimulationStatus.PASSED
    assert high.status is SimulationStatus.PASSED
    delta_db = float(np.max(np.abs(_magnitude_db(low) - _magnitude_db(high))))
    assert delta_db < DISAGREEMENT_THRESHOLD_DB
    assert not low.diagnostics and not high.diagnostics


@pytest.mark.skip(
    reason=(
        "requires model_disagreement implementation (plan §7.3 rule 4): no diagnostic code "
        "or result field exists yet; see outputs/_baseline_report.md §E4"
    )
)
def test_backend_disagreement_beyond_threshold_is_marked_model_disagreement() -> None:
    pytest.importorskip("torch")
    low_graph = _graph(capacitance_f=100e-9)
    high_graph = _graph(capacitance_f=10e-9)  # same candidate, different fidelity model

    low = LinearMNASimulatorBackend().simulate(
        LinearMNASimulatorBackend().compile(low_graph), _request(low_graph)
    )
    high = TorchMNASimulatorBackend().simulate(
        TorchMNASimulatorBackend().compile(high_graph), _request(high_graph)
    )

    delta_db = float(np.max(np.abs(_magnitude_db(low) - _magnitude_db(high))))
    assert delta_db > DISAGREEMENT_THRESHOLD_DB, "harness sanity: the two fidelities must disagree"

    # Intended contract: the multi-fidelity acceptance step records a diagnostic with the
    # stable code ``model_disagreement`` instead of accepting the low-fidelity result.
    codes = {item.code for item in (*low.diagnostics, *high.diagnostics)}
    assert DISAGREEMENT_CODE in codes
