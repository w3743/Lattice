from __future__ import annotations

from dataclasses import replace

from circuit_ai.constraints import ConstraintReport, FeasibilityStatus
from circuit_ai.synthesis import CircuitSynthesizer
from circuit_ai.spec import SynthesisSpec
from circuit_ai.simulation import SimulationStatus


def _staged_spec() -> SynthesisSpec:
    return SynthesisSpec.from_dict(
        {
            "name": "truth_rerank",
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
                "points": 24,
                "max_iterations": 2,
                "top_k": 2,
                "seed": 3,
                "graph_search": {
                    "enabled": True,
                    "include_fallback": False,
                    "max_candidates": 4,
                    "internal_nodes": 1,
                },
                "fidelity": {
                    "enabled": True,
                    "screen_points": 6,
                    "screen_iterations": 1,
                    "promotion_fraction": 0.8,
                },
            },
        }
    )


def test_staged_truth_evidence_reorders_promoted_candidates(monkeypatch) -> None:
    synthesizer = CircuitSynthesizer()
    attached_names: list[str] = []

    def fake_truth(result, *, spec=None):
        attached_names.append(result.template.name)
        # Later candidates receive a strictly better truth objective.  This
        # intentionally disagrees with the full-optimization ordering.
        objective = -float(len(attached_names))
        report = ConstraintReport(
            evaluations=(),
            feasibility=FeasibilityStatus.VERIFIED_FEASIBLE,
            metric_values={},
            objective_values={"truth.objective": objective},
        )
        return replace(result, constraint_report=report, simulation_results=())

    monkeypatch.setattr(synthesizer, "_attach_simulation_evidence", fake_truth)
    results = synthesizer.synthesize(_staged_spec())

    assert len(attached_names) == 4
    assert [item.template.name for item in results] == attached_names[-2:][::-1]
    truth_ids = results[0].fidelity_schedule.selections[-1].candidate_ids
    assert truth_ids[0].endswith(results[0].template.name)
    assert truth_ids[1].endswith(results[1].template.name)


def test_requested_spice_truth_without_executable_is_not_verified() -> None:
    raw = {
        "name": "spice_gate",
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
            "spice_verification": {
                "enabled": True,
                "executable": "definitely_missing_ngspice_for_test",
            },
        },
    }
    result = CircuitSynthesizer().synthesize(SynthesisSpec.from_dict(raw))[0]
    assert result.simulation_results[0].status is SimulationStatus.UNAVAILABLE
    assert result.simulation_results[0].fidelity == "spice"
    assert result.constraint_report is not None
    assert not result.constraint_report.passed
