from __future__ import annotations

import pytest

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.feasibility import FeasibilityError, analyze_feasibility


def test_lowpass_feasibility_passes_and_is_reported() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "feasible_lowpass",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000, "gain": 1.0},
            "library": {"allowed": ["R", "C"]},
            "optimization": {"points": 24, "max_iterations": 8, "top_k": 1},
        }
    )

    report = analyze_feasibility(spec)
    result = CircuitSynthesizer().synthesize(spec)[0]

    assert report.passed
    assert result.feasibility is not None
    assert result.as_dict()["feasibility"]["passed"]


def test_unstable_zpk_is_blocked_by_default() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "unstable_zpk",
            "ports": 2,
            "behavior": {
                "kind": "zpk",
                "zeros_rad_s": [],
                "poles_rad_s": ["1000+0j"],
                "gain": 1.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {"allowed": ["R", "C", "L"]},
            "optimization": {"points": 24, "max_iterations": 3, "top_k": 1},
        }
    )

    report = analyze_feasibility(spec)
    assert not report.passed
    assert any(issue.code == "unstable_poles" for issue in report.errors)
    with pytest.raises(FeasibilityError):
        CircuitSynthesizer().synthesize(spec)


def test_improper_zpk_can_be_warned_without_enforcement() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "improper_zpk",
            "ports": 2,
            "behavior": {
                "kind": "zpk",
                "zeros_rad_s": ["0+0j", "-10+0j"],
                "poles_rad_s": ["-1000+0j"],
                "gain": 1.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {"allowed": ["R", "C", "L"]},
            "optimization": {
                "points": 24,
                "max_iterations": 2,
                "top_k": 1,
                "feasibility": {"enabled": True, "enforce": False},
            },
        }
    )

    report = analyze_feasibility(spec)
    assert not report.passed
    assert not report.enforce


def test_samples_feasibility_warns_but_passes_numeric_data() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "sampled",
            "ports": 2,
            "behavior": {
                "kind": "samples",
                "frequency_hz": [10, 100, 1000],
                "magnitude_db": [0, -3, -20],
            },
            "library": {"allowed": ["R", "C"]},
            "optimization": {"points": 24, "max_iterations": 2, "top_k": 1},
        }
    )

    report = analyze_feasibility(spec)

    assert report.passed
    assert any(issue.code == "sample_feasibility_limited" for issue in report.warnings)
