"""Multi-fidelity backend disagreement (plan §18.2, §7.3 rule 4).

Requirement (``计划:658``): "后端间差异超过门槛时，候选标记为 model_disagreement
并进入诊断队列" -- when two backends disagree on the same candidate by more than
the configured threshold, the low-fidelity conclusion must not be accepted
silently: the candidate is marked ``model_disagreement`` and enters the
diagnostics queue.

The comparison is inherently an *external* step: a single backend cannot know
another backend's result, so no single ``SimulationResult`` can carry the
verdict on its own.  ``compare_fidelity_results`` is the entry point; these
tests drive that, not the backends in isolation.  (An earlier revision of this
file asserted the code appeared in the two backends' own diagnostics, which no
correct implementation could satisfy.)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.optimization import (
    MODEL_DISAGREEMENT_CODE,
    FidelityDisagreementError,
    compare_fidelity_results,
    disagreement_from_options,
    graphs_are_comparable,
)
from circuit_ai.simulation import (
    AnalysisSpec,
    Excitation,
    LinearMNASimulatorBackend,
    ObservableSpec,
    Quantity,
    SimulationRequest,
    SimulationStatus,
    SweepSpec,
    TorchMNASimulatorBackend,
)

DISAGREEMENT_THRESHOLD_DB = 1.0
DISAGREEMENT_CODE = MODEL_DISAGREEMENT_CODE


def _quantity(value: float) -> Quantity:
    return Quantity(value=value, unit="V")


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


def _simulate(backend_class, graph, *, fidelity: str | None = None):
    backend = backend_class()
    request = _request(graph)
    if fidelity is not None:
        request = replace(request, fidelity=fidelity)
    return backend.simulate(backend.compile(graph), request)


def _rescaled_magnitude(result, *, factor_db: float):
    """Return *result* with every observable scaled by a constant in dB.

    Used to build a backend that genuinely disagrees without pretending that a
    different circuit is "the same candidate": the graph, the request and the
    parameters are untouched, only the reported magnitude moves.
    """

    scale = 10.0 ** (factor_db / 20.0)
    waveforms = {
        name: replace(waveform, values=tuple(np.asarray(waveform.values) * scale))
        for name, waveform in result.waveforms.items()
    }
    return replace(result, waveforms=waveforms)


def test_low_and_high_fidelity_internal_backends_agree_for_identical_candidates() -> None:
    """Positive control: with identical parameters both internal backends agree."""
    pytest.importorskip("torch")
    graph = _graph(capacitance_f=100e-9)

    low = _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain")
    high = _simulate(TorchMNASimulatorBackend, graph, fidelity="differentiable")

    assert low.status is SimulationStatus.PASSED
    assert high.status is SimulationStatus.PASSED
    delta_db = float(np.max(np.abs(_magnitude_db(low) - _magnitude_db(high))))
    assert delta_db < DISAGREEMENT_THRESHOLD_DB
    assert not low.diagnostics and not high.diagnostics

    comparison = compare_fidelity_results(low, high, threshold=DISAGREEMENT_THRESHOLD_DB)
    assert comparison.compared_anything
    assert not comparison.disagreed
    assert comparison.max_deviation < DISAGREEMENT_THRESHOLD_DB
    assert comparison.as_diagnostic() is None


def test_backend_disagreement_beyond_threshold_is_marked_model_disagreement() -> None:
    """Beyond the threshold the verdict is recorded with the stable code."""
    pytest.importorskip("torch")
    graph = _graph(capacitance_f=100e-9)

    low = _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain")
    honest = _simulate(TorchMNASimulatorBackend, graph, fidelity="differentiable")
    # A truth backend whose reported magnitude is off by 6 dB on the same
    # candidate: same graph, same parameters, different model conclusion.
    high = _rescaled_magnitude(honest, factor_db=6.0)

    delta_db = float(np.max(np.abs(_magnitude_db(low) - _magnitude_db(high))))
    assert delta_db > DISAGREEMENT_THRESHOLD_DB, "harness sanity: the two fidelities must disagree"

    comparison = compare_fidelity_results(low, high, threshold=DISAGREEMENT_THRESHOLD_DB)
    assert comparison.disagreed
    assert comparison.max_deviation > DISAGREEMENT_THRESHOLD_DB

    diagnostic = comparison.as_diagnostic()
    assert diagnostic is not None
    assert diagnostic.code == DISAGREEMENT_CODE
    # Disagreement qualifies the high-fidelity conclusion; it is not proof the
    # candidate is wrong, and a PASSED result may not carry error diagnostics.
    assert diagnostic.severity == "warning"


def test_disagreement_diagnostic_enters_the_diagnostics_queue_once() -> None:
    """The diagnostic is attached to the high-fidelity result, exactly once."""
    pytest.importorskip("torch")
    from circuit_ai.optimization import attach_disagreement_diagnostics

    graph = _graph(capacitance_f=100e-9)
    low = _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain")
    high = _rescaled_magnitude(
        _simulate(TorchMNASimulatorBackend, graph, fidelity="differentiable"),
        factor_db=6.0,
    )
    comparison = compare_fidelity_results(low, high, threshold=DISAGREEMENT_THRESHOLD_DB)

    results = attach_disagreement_diagnostics((low, high), (comparison,))
    codes_low = [item.code for item in results[0].diagnostics]
    codes_high = [item.code for item in results[1].diagnostics]
    assert DISAGREEMENT_CODE not in codes_low, "the low-fidelity result is not the qualified one"
    assert codes_high == [DISAGREEMENT_CODE]
    assert results[1].status is SimulationStatus.PASSED

    # Re-attaching must not duplicate the entry.
    again = attach_disagreement_diagnostics(results, (comparison,))
    assert [item.code for item in again[1].diagnostics] == [DISAGREEMENT_CODE]


def test_agreement_never_produces_a_disagreement_diagnostic() -> None:
    """The negative control at the same code path: no breach, no diagnostic."""
    pytest.importorskip("torch")
    from circuit_ai.optimization import attach_disagreement_diagnostics

    graph = _graph(capacitance_f=100e-9)
    low = _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain")
    high = _simulate(TorchMNASimulatorBackend, graph, fidelity="differentiable")
    comparison = compare_fidelity_results(low, high, threshold=DISAGREEMENT_THRESHOLD_DB)

    assert not comparison.disagreed
    results = attach_disagreement_diagnostics((low, high), (comparison,))
    assert not results[0].diagnostics
    assert not results[1].diagnostics


def test_threshold_boundary_is_inclusive_of_agreement() -> None:
    """Deviation equal to the threshold is not a breach; anything above is.

    The comparison uses a strict ``>``, so the boundary is pinned on the
    deviation the harness actually measures.  Asserting on a nominal 1.0 dB
    would be at the mercy of ``log10`` rounding in the rescale helper.
    """
    pytest.importorskip("torch")
    graph = _graph(capacitance_f=100e-9)
    low = _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain")
    honest = _simulate(TorchMNASimulatorBackend, graph, fidelity="differentiable")

    probe = compare_fidelity_results(
        low, _rescaled_magnitude(honest, factor_db=4.0), threshold=DISAGREEMENT_THRESHOLD_DB
    )
    measured = probe.max_deviation
    assert measured > DISAGREEMENT_THRESHOLD_DB
    assert probe.disagreed

    high_same = _rescaled_magnitude(honest, factor_db=4.0)
    at_exact = compare_fidelity_results(low, high_same, threshold=measured)
    assert at_exact.max_deviation == measured
    assert not at_exact.disagreed, "deviation equal to the threshold is not a breach"

    below = compare_fidelity_results(low, high_same, threshold=measured * 0.999)
    assert below.disagreed, "the same deviation must breach a stricter threshold"

    above = compare_fidelity_results(
        low, _rescaled_magnitude(honest, factor_db=4.01), threshold=measured
    )
    assert above.disagreed


def test_agreement_holds_up_to_and_including_the_threshold() -> None:
    """Exact-threshold agreement, without depending on an analytic rescale."""

    low = replace(
        _simulate(
            LinearMNASimulatorBackend, _graph(capacitance_f=100e-9),
            fidelity="linear_frequency_domain",
        ),
        waveforms={},
        scalars={"vout": _quantity(1.0)},
    )
    high = replace(
        _simulate(
            LinearMNASimulatorBackend, _graph(capacitance_f=100e-9),
            fidelity="linear_frequency_domain",
        ),
        waveforms={},
        scalars={"vout": _quantity(1.0 + 0.005)},
    )

    # Real-valued samples are compared as a fraction of the larger magnitude,
    # so 1.0 vs 1.005 is 0.005/1.005, not 0.005.
    expected = 0.005 / 1.005
    exact = compare_fidelity_results(low, high, threshold=expected)
    assert exact.max_deviation == pytest.approx(expected, abs=1e-15)
    assert not exact.disagreed

    stricter = compare_fidelity_results(low, high, threshold=expected * 0.999)
    assert stricter.disagreed


def test_empty_comparison_is_not_reported_as_agreement() -> None:
    """Nothing comparable must not be silently counted as agreement."""
    graph = _graph(capacitance_f=100e-9)
    low = _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain")
    # Same magnitudes, different observable set -> nothing to compare.
    high = replace(
        _simulate(LinearMNASimulatorBackend, graph, fidelity="linear_frequency_domain"),
        waveforms={},
    )
    comparison = compare_fidelity_results(low, high, threshold=DISAGREEMENT_THRESHOLD_DB)

    assert not comparison.compared_anything
    assert not comparison.disagreed
    assert comparison.as_diagnostic() is None
    assert comparison.skipped_observables


def test_different_graphs_are_not_treated_as_comparable() -> None:
    """A disagreement verdict needs results for one candidate, not two."""
    graph_a = _graph(capacitance_f=100e-9)
    graph_b = _graph(capacitance_f=10e-9)
    low = _simulate(LinearMNASimulatorBackend, graph_a, fidelity="linear_frequency_domain")
    high = _simulate(LinearMNASimulatorBackend, graph_b, fidelity="linear_frequency_domain")

    assert not graphs_are_comparable(low, high)


def test_power_truth_pairing_uses_the_declared_request_correspondence() -> None:
    """The power path pairs truth with planning by request id, not position.

    ``_optional_power_truth`` derives each truth request id from its planning
    request, so pairing is by that declared link.  A truth result that did not
    pass carries no comparable numbers and must be skipped, not compared.
    """
    from circuit_ai.pipeline import _pair_truth_with_planning

    graph = _graph(capacitance_f=100e-9)
    base = _simulate(LinearMNASimulatorBackend, graph, fidelity="ideal_averaged")

    planning_requests = tuple(
        replace(_request(graph), request_id=f"req-{index}") for index in (1, 2)
    )
    planning_results = tuple(
        replace(base, request_id=req.request_id) for req in planning_requests
    )
    truth_requests = tuple(
        replace(req, request_id=f"{req.request_id}__spice_truth", fidelity="spice")
        for req in planning_requests
    )
    truth_results = tuple(
        replace(
            base,
            request_id=req.request_id,
            fidelity="spice",
            # req-1 passed, req-2 did not.
            status=SimulationStatus.PASSED if index == 0 else SimulationStatus.UNAVAILABLE,
        )
        for index, req in enumerate(truth_requests)
    )

    # Reverse the truth results: position must not decide the pairing.  The
    # passing result belongs to req-1, and it must be paired with req-1's
    # planning result whichever slot it arrives in.
    pairs = _pair_truth_with_planning(
        planning_requests,
        planning_results,
        truth_requests,
        (truth_results[1], truth_results[0]),
    )
    assert len(pairs) == 1, "the non-passing truth result must be skipped"
    paired = {(planning.request_id, truth.request_id) for planning, truth in pairs}
    assert paired == {("req-1", "req-1__spice_truth")}
    assert pairs[0][1].status is SimulationStatus.PASSED


def test_power_truth_pairing_skips_results_without_a_planning_counterpart() -> None:
    """An orphan truth result is skipped rather than compared against a guess."""
    from circuit_ai.pipeline import _pair_truth_with_planning

    graph = _graph(capacitance_f=100e-9)
    base = _simulate(LinearMNASimulatorBackend, graph, fidelity="ideal_averaged")
    planning_requests = (replace(_request(graph), request_id="req-1"),)
    planning_results = (replace(base, request_id="req-1"),)
    orphan_request = replace(
        _request(graph), request_id="not-a-planning-request__spice_truth", fidelity="spice"
    )
    orphan_result = replace(base, request_id=orphan_request.request_id, fidelity="spice")

    pairs = _pair_truth_with_planning(
        planning_requests, planning_results, (orphan_request,), (orphan_result,)
    )
    assert pairs == ()


def test_policy_options_default_to_disabled_and_reject_bad_thresholds() -> None:
    """The policy is opt-in, and a malformed threshold fails loudly."""
    """The policy is opt-in, and a malformed threshold fails loudly."""
    assert disagreement_from_options(None) == (False, 1.0)
    assert disagreement_from_options({})[0] is False
    assert disagreement_from_options({"enabled": True, "threshold": 0.5}) == (True, 0.5)

    with pytest.raises(FidelityDisagreementError):
        disagreement_from_options({"enabled": True, "threshold": "not-a-number"})
    with pytest.raises(FidelityDisagreementError):
        disagreement_from_options({"enabled": True, "threshold": -1.0})


def test_ac_synthesizer_records_two_backends_and_marks_disagreement(monkeypatch) -> None:
    """Wiring: the AC truth gate also runs the internal model and compares.

    ngspice is not installed here, so the external truth backend is made to
    delegate to the internal model and then have its reported magnitude moved
    by 6 dB.  The two fidelity results are then genuine simulation results for
    the same candidate, and the comparison path runs for real.
    """

    pytest.importorskip("torch")
    from circuit_ai.simulation import NgspiceSimulatorBackend
    from circuit_ai.spec import SynthesisSpec
    from circuit_ai.synthesis import CircuitSynthesizer

    internal = LinearMNASimulatorBackend()

    def fake_available(self):
        return True, "delegating test backend"

    def fake_compile(self, graph):
        return graph

    def fake_simulate(self, compiled, request):
        """Honor the request's own observables, then disagree by 6 dB."""

        honest = internal.simulate(internal.compile(compiled), request)
        return _rescaled_magnitude(honest, factor_db=6.0)

    monkeypatch.setattr(NgspiceSimulatorBackend, "available", fake_available)
    monkeypatch.setattr(NgspiceSimulatorBackend, "compile", fake_compile)
    monkeypatch.setattr(NgspiceSimulatorBackend, "simulate", fake_simulate)

    spec = SynthesisSpec.from_dict(
        {
            "name": "disagreement_wiring",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {
                "points": 16,
                "max_iterations": 1,
                "top_k": 1,
                "spice_verification": {"enabled": True, "executable": "delegating_truth"},
                "fidelity_disagreement": {"enabled": True, "threshold": 1.0},
            },
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.fidelity_comparison is not None
    comparison = result.fidelity_comparison
    assert comparison.compared_anything, "the two results must actually be comparable"
    assert comparison.disagreed
    assert comparison.max_deviation == pytest.approx(6.0, abs=1e-6)
    assert comparison.low_fidelity == "linear_frequency_domain"
    assert comparison.high_fidelity == "spice"

    # Both fidelities are kept as evidence, and the verdict lands on the
    # high-fidelity result.
    fidelities = {item.fidelity for item in result.simulation_results}
    assert {"spice", "linear_frequency_domain"} <= fidelities
    spice_result = next(item for item in result.simulation_results if item.fidelity == "spice")
    assert [item.code for item in spice_result.diagnostics] == [DISAGREEMENT_CODE]
    assert spice_result.status is SimulationStatus.PASSED

    payload = result.as_dict()
    assert payload["fidelity_comparison"]["disagreed"] is True
    assert payload["fidelity_comparison"]["threshold"] == 1.0
