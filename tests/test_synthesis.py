from __future__ import annotations

import numpy as np
import pytest

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.optimizers import ParameterOptimizationResult, parse_differential_evolution_options
from circuit_ai.simulation import SimulationStatus


def test_lowpass_synthesis_finds_two_component_rc() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_lowpass",
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
            "optimization": {"points": 80, "max_iterations": 35, "top_k": 1, "seed": 3},
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.template.name == "rc_lowpass"
    assert result.metrics.rmse_db < 0.05
    assert result.metrics.pareto_rank == 0
    assert result.metrics.objectives["rmse_db"] == result.metrics.rmse_db
    assert result.metrics.optimization_run is not None
    assert result.metrics.optimization_run.truth_evaluated
    assert result.metrics.optimization_run.problem_id == "ac.rc_lowpass"
    payload = result.as_dict()
    assert payload["metrics"]["pareto_rank"] == 0
    assert payload["metrics"]["optimization_run"]["truth_evaluated"] is True
    assert payload["circuit_graph"]["schema"] == "circuit_ai.circuit_graph"
    assert payload["graph_hash"] == result.circuit_graph().graph_hash
    assert payload["topology_hash"] == result.circuit_graph().topology_hash
    assert result.simulation_results[0].status is SimulationStatus.PASSED
    verified = np.asarray(
        result.simulation_results[0].waveforms["voltage_transfer"].values
    )
    assert np.max(np.abs(verified - result.response)) < 1e-10
    assert payload["simulation_requests"][0]["request_hash"]
    assert payload["simulation_results"][0]["evidence_hash"]
    assert result.fidelity_schedule is not None
    assert result.fidelity_schedule.selected_candidate_ids == ("rank-0001:rc_lowpass",)
    assert payload["fidelity_schedule"]["selections"][0]["fidelity"] == "optimization_model"
    assert payload["fidelity_schedule"]["selections"][1]["fidelity"] == "linear_frequency_domain"
    assert (
        payload["simulation_capabilities"][0]["selected"]["capability_id"]
        == "simulation.linear_mna.ac"
    )
    assert result.constraint_report is not None
    assert result.constraint_report.passed
    assert result.constraint_report.metric_values["ac.rmse_db"].value == pytest.approx(
        result.metrics.rmse_db
    )
    assert result.constraint_report.metric_values[
        "ac.max_abs_error_db"
    ].value == pytest.approx(result.metrics.max_abs_db)
    assert result.constraint_report.objective_values[
        "objective.ac.rmse_db"
    ] == pytest.approx(result.metrics.rmse_db)
    assert payload["constraint_report"]["schema"] == "circuit_ai.constraint_report"
    assert result.template.component_count == 2
    assert "R1 in out" in result.netlist()


def test_highpass_gain_needs_opamp_candidate() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_gain_highpass",
            "ports": 2,
            "behavior": {
                "kind": "highpass",
                "cutoff_hz": 5000,
                "gain": 4.0,
                "order": 1,
                "frequency_range_hz": [50, 500000],
            },
            "library": {
                "allowed": ["R", "C", "opamp"],
                "parameter_ranges": {"R": [100, 2000000], "C": [1e-11, 1e-5]},
            },
            "optimization": {"points": 80, "max_iterations": 35, "top_k": 2, "seed": 4},
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.template.name == "gain_rc_highpass"
    assert result.metrics.rmse_db < 0.15
    assert "Eop out 0" in result.netlist()


def test_library_filters_incompatible_templates() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_bandpass",
            "ports": 2,
            "behavior": {
                "kind": "bandpass",
                "center_hz": 10000,
                "q": 5.0,
                "gain": 1.0,
                "frequency_range_hz": [100, 1000000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [1, 100000], "C": [1e-10, 1e-3]},
            },
            "optimization": {"points": 40, "max_iterations": 5, "top_k": 1},
        }
    )

    try:
        CircuitSynthesizer().synthesize(spec)
    except ValueError as exc:
        assert "no compatible" in str(exc)
    else:
        raise AssertionError("expected no compatible templates without inductors")


def test_synthesizer_accepts_custom_parameter_optimizer() -> None:
    class FixedMidpointOptimizer:
        def optimize(self, template, spec, target, score_response):
            parameters = {
                param.name: float(np.sqrt(lo * hi))
                for param, (lo, hi) in zip(template.params, template.parameter_bounds(spec.library))
            }
            response = template.response(parameters, target.frequencies_hz)
            assert isinstance(score_response(response), float)
            return ParameterOptimizationResult(
                parameters=parameters,
                response=response,
                success=True,
                message="fixed midpoint optimizer",
                evaluations=1,
            )

    spec = SynthesisSpec.from_dict(
        {
            "name": "test_custom_optimizer",
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
            "optimization": {"points": 24, "max_components": 2, "top_k": 1},
        }
    )

    result = CircuitSynthesizer(optimizer=FixedMidpointOptimizer()).synthesize(spec)[0]

    assert result.metrics.optimizer_success
    assert result.metrics.optimizer_message == "fixed midpoint optimizer"


def test_spec_accepts_optimizer_configuration() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_optimizer_config",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000},
            "library": {"allowed": ["R", "C"]},
            "optimization": {
                "optimizer": {
                    "differential_evolution": {
                        "popsize": 18,
                        "tol": 1e-5,
                        "polish": False,
                    }
                }
            },
        }
    )

    assert spec.optimization.optimizer["differential_evolution"]["popsize"] == 18
    assert spec.optimization.optimizer["differential_evolution"]["polish"] is False


def test_differential_evolution_options_are_validated() -> None:
    default_spec = SynthesisSpec.from_dict(
        {
            "name": "test_optimizer_defaults",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000},
            "library": {"allowed": ["R", "C"]},
        }
    )
    assert parse_differential_evolution_options(default_spec).popsize == 15

    bad_spec = SynthesisSpec.from_dict(
        {
            "name": "test_bad_optimizer_config",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000},
            "library": {"allowed": ["R", "C"]},
            "optimization": {"optimizer": {"differential_evolution": {"popsize": "large"}}},
        }
    )
    with pytest.raises(ValueError, match="popsize"):
        parse_differential_evolution_options(bad_spec)
