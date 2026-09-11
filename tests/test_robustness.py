from __future__ import annotations

import numpy as np

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.robustness import analyze_robustness
from circuit_ai.targets import target_from_behavior
from circuit_ai.templates import RCLowPass


def test_robustness_metrics_are_reported_when_enabled() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_robust_lowpass",
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
                "points": 64,
                "max_iterations": 25,
                "top_k": 1,
                "seed": 8,
                "robustness": {
                    "enabled": True,
                    "tolerance_fraction": 0.05,
                    "samples": 24,
                    "acceptance_rmse_db": 1.0,
                    "acceptance_max_abs_db": 6.0,
                },
            },
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]
    robust = result.metrics.robustness

    assert robust is not None
    assert robust.samples == 24
    assert robust.tolerance_fraction == 0.05
    assert 0.0 <= robust.yield_fraction <= 1.0
    assert robust.finite_fraction == 1.0
    assert result.as_dict()["metrics"]["robustness"]["samples"] == 24


def test_robustness_sobol_sampling_is_reproducible() -> None:
    target = target_from_behavior(
        {"kind": "lowpass", "cutoff_hz": 1000.0},
        np.logspace(1, 5, 20),
    )
    options = {
        "enabled": True,
        "tolerance_fraction": 0.05,
        "samples": 16,
        "batch_size": 4,
        "distribution": "sobol",
        "confidence_level": 0.95,
    }
    first = analyze_robustness(
        RCLowPass(),
        {"R": 10_000.0, "C": 1.5915494309e-8},
        target,
        options,
        seed=17,
    )
    second = analyze_robustness(
        RCLowPass(),
        {"R": 10_000.0, "C": 1.5915494309e-8},
        target,
        options,
        seed=17,
    )

    assert first is not None and second is not None
    assert first.as_dict() == second.as_dict()
    assert first.samples == 16
    assert first.yield_confidence_interval is not None


def test_robustness_reports_pvt_corner_evidence_and_combined_worst_case() -> None:
    target = target_from_behavior(
        {"kind": "lowpass", "cutoff_hz": 1000.0},
        np.logspace(1, 5, 20),
    )
    robust = analyze_robustness(
        RCLowPass(),
        {"R": 10_000.0, "C": 1.5915494309e-8},
        target,
        {
            "enabled": True,
            "tolerance_fraction": 0.01,
            "samples": 8,
            "corners": [
                {"name": "slow", "multipliers": {"R": 1.1, "C": 1.1}},
                {"name": "fast", "multipliers": {"R": 0.9, "C": 0.9}},
            ],
        },
        seed=19,
    )
    assert robust is not None
    assert [item["name"] for item in robust.corner_results] == ["slow", "fast"]
    assert robust.total_evaluations == robust.samples + 2
    assert robust.corner_yield_fraction is not None
    assert robust.rmse_db_worst >= max(
        float(item["rmse_db"]) for item in robust.corner_results
    )
