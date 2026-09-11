"""Linear/torch MNA backend compiled and plan caches.

Regression guard for a real defect: ``CircuitGraph.graph_hash`` and
``topology_hash`` are *structural* fingerprints that deliberately ignore
component instance ids and reference designators (see ``graph/canonical.py``).
``CompiledLinearMNA``, however, validates its plan against a signature that
*does* include element names.  Keying the backend caches on the structural hash
alone therefore let two different circuits share one plan and one compiled
model, and the second circuit failed with ``graph_mismatch`` or
``analysis_failed`` depending on which one was compiled first.
"""

from __future__ import annotations

from dataclasses import replace

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
)


def _graph(ref_prefix: str):
    """Same structure, different component names.

    The structural hashes of the two prefixes are equal by design; the
    compiled MNA signature is not.
    """

    circuit = LinearCircuit(
        elements=(
            LinearElement(f"{ref_prefix}1", "R", "in", "out", 1000.0),
            LinearElement(f"{ref_prefix}2", "C", "out", "0", 100e-9),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )
    return linear_circuit_to_graph(circuit, name=f"graph_{ref_prefix}")


def _request(graph) -> SimulationRequest:
    return SimulationRequest(
        request_id=f"{graph.graph_id}__ac",
        analysis=AnalysisSpec(
            kind="small_signal_ac",
            sweeps=(SweepSpec("frequency", "Hz", start=10.0, stop=1e5, points=9, scale="log"),),
        ),
        graph_id=graph.graph_id,
        excitations=(
            Excitation("vin", "ac_voltage", "component", "Vin", {"magnitude": 1.0}),
        ),
        requested_observables=(
            ObservableSpec(
                "voltage_transfer",
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


def test_structural_hashes_are_equal_for_renamed_components() -> None:
    """Documents the premise: the hashes cannot distinguish these graphs."""

    first = _graph("R")
    second = _graph("X")
    assert first.topology_hash == second.topology_hash
    assert first.graph_hash == second.graph_hash
    assert first.document_hash != second.document_hash


def test_renamed_graphs_do_not_share_a_compiled_model_or_plan() -> None:
    """Both circuits must simulate on one shared backend instance.

    A synthesizer reuses one backend across candidates, so this is exactly the
    real failure mode rather than a synthetic one.
    """

    backend = LinearMNASimulatorBackend()
    first = _graph("R")
    second = _graph("X")

    first_result = backend.simulate(backend.compile(first), _request(first))
    second_result = backend.simulate(backend.compile(second), _request(second))

    assert first_result.status is SimulationStatus.PASSED
    assert second_result.status is SimulationStatus.PASSED, (
        f"second graph must not inherit the first graph's plan: "
        f"{[item.code for item in second_result.diagnostics]}"
    )
    assert not second_result.diagnostics

    # The evidence must describe the graph that was actually simulated.
    assert first_result.graph_hash == first.graph_hash
    assert second_result.graph_hash == second.graph_hash

    # Both must agree numerically: same structure, same values.
    first_values = first_result.waveforms["voltage_transfer"].values
    second_values = second_result.waveforms["voltage_transfer"].values
    assert first_values == second_values


def test_compiled_model_is_reused_for_the_identical_graph() -> None:
    """The fix must not destroy the cache it guards."""

    backend = LinearMNASimulatorBackend()
    graph = _graph("R")
    first = backend.compile(graph)
    second = backend.compile(graph)
    assert first is second, "an identical graph must still hit the cache"
    assert first.mna_plan is second.mna_plan
