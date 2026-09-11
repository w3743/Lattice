"""Falsification sweep: every declared requirement must be able to fail.

Rounds 9 and 10 each found a requirement that was compiled or claimed but never
actually evaluated -- an AC target tolerance that was never compiled, and an
efficiency limit checked against the lossless model rather than the design. Both
were invisible from the passing side, because a gate that never says no looks
exactly like a gate that always says yes.

So this file drives each requirement axis to a value that cannot be met and
asserts the run refuses. A constraint that cannot fail is not a constraint.

Two axes are deliberately *not* asserted to refuse, and the reasons are recorded
where they appear:

* an enormous output current is a legitimate design request, not a violation --
  the ideal model will happily deliver 5 V at 1 MA, and nothing the spec declared
  is breached;
* an extremely tight tolerance can be met by a first-order response, because a
  single pole tracks a single pole exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_ai.pbdl_runner import run_pbdl_file
from circuit_ai.pipeline import design_from_pbdl
from circuit_ai.spec import SynthesisSpec
from circuit_ai.synthesis import CircuitSynthesizer

TEMPLATE = Path("端口描述语言/examples/lowpass_1khz.json")


def _ac_spec(behavior: dict, *, allowed=("R", "C", "L"), iterations: int = 20):
    return SynthesisSpec.from_dict(
        {
            "name": "falsify",
            "ports": 2,
            "behavior": {"frequency_range_hz": [10, 100000], **behavior},
            "library": {
                "allowed": list(allowed),
                "parameter_ranges": {"R": [100, 1e6], "C": [1e-10, 1e-4], "L": [1e-6, 1.0]},
            },
            "optimization": {"points": 48, "max_iterations": iterations, "top_k": 1, "seed": 3},
        }
    )


def _dc_spec(target_overrides: dict, constraints_overrides: dict | None = None, optimization=None):
    target = {
        "target_kind": "dc",
        "input_voltage_v": 12,
        "output_voltage_v": 5,
        "output_current_a": 3,
        "tolerance": {"relative": 0.02},
    }
    target.update(target_overrides)
    constraints = {
        "element_types": ["R", "C", "L", "ideal_switch", "ideal_diode"],
        "max_component_count": 4,
        "parameter_ranges": {"R": [100.0, 1e6], "C": [1e-6, 5e-3], "L": [1e-6, 1e-3]},
    }
    constraints.update(constraints_overrides or {})
    return {
        "name": "falsify_dc",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        ],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [target],
        "constraints": constraints,
        "optimization": optimization or {"max_iterations": 8, "seed": 3},
    }


# ---------------------------------------------------------------------------
# AC: the path that had the hole
# ---------------------------------------------------------------------------


def test_ac_tolerance_is_enforced_when_it_cannot_be_met() -> None:
    """A 4th-order Butterworth is unreachable with one pole, so 1e-6 dB must fail.

    Before this was fixed the legacy AC path emitted only OBJECTIVE constraints,
    so constraint_report.passed was True at 32.4 dB of RMSE regardless of what
    the spec declared.
    """

    result = CircuitSynthesizer().synthesize(
        _ac_spec(
            {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "order": 4,
                "response": "butterworth",
                "gain": 1.0,
                "tolerance": {"absolute": 1e-6, "relative": None},
            }
        )
    )[0]

    assert result.metrics.rmse_db > 1.0, "the harness must actually miss the target"
    assert result.constraint_report is not None
    assert not result.constraint_report.passed
    assert any(
        item.severity.value == "hard" and not item.passed
        for item in result.constraint_report.evaluations
    )


def test_ac_tolerance_is_enforced_for_a_moderate_value_too() -> None:
    """The gate must not be special-cased to absurd limits."""

    result = CircuitSynthesizer().synthesize(
        _ac_spec(
            {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "order": 4,
                "response": "butterworth",
                "gain": 1.0,
                "tolerance": {"absolute": 0.5, "relative": None},
            }
        )
    )[0]
    assert not result.constraint_report.passed


def test_ac_without_a_declared_tolerance_is_unchanged() -> None:
    """Absent a tolerance nothing is claimed, so the previous behaviour stands."""

    result = CircuitSynthesizer().synthesize(
        _ac_spec(
            {"kind": "lowpass", "cutoff_hz": 1000, "order": 4, "response": "butterworth", "gain": 1.0}
        )
    )[0]
    assert result.constraint_report is not None
    assert result.constraint_report.passed
    assert all(
        item.severity.value == "objective" for item in result.constraint_report.evaluations
    )


def test_a_met_tolerance_still_passes() -> None:
    """Guards against the gate becoming unconditional."""

    result = CircuitSynthesizer().synthesize(
        _ac_spec(
            {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "order": 1,
                "response": "butterworth",
                "gain": 1.0,
                "tolerance": {"absolute": 0.5, "relative": None},
            }
        )
    )[0]
    assert result.metrics.rmse_db < 0.5
    assert result.constraint_report.passed


def test_ac_component_budget_is_enforced() -> None:
    spec = _ac_spec({"kind": "lowpass", "cutoff_hz": 1000, "order": 2})
    with pytest.raises(ValueError, match="no compatible circuit templates"):
        CircuitSynthesizer().synthesize(
            SynthesisSpec(
                name=spec.name,
                ports=spec.ports,
                analysis=spec.analysis,
                behavior=spec.behavior,
                library=spec.library,
                optimization=type(spec.optimization)(
                    **{**spec.optimization.__dict__, "max_components": 1}
                ),
            )
        )


def test_an_element_outside_the_library_is_rejected() -> None:
    with pytest.raises(ValueError, match="required elements must also be allowed"):
        SynthesisSpec.from_dict(
            {
                "name": "bad_library",
                "ports": 2,
                "behavior": {"kind": "lowpass", "cutoff_hz": 1000, "order": 1, "frequency_range_hz": [10, 1e5]},
                "library": {"allowed": ["C"], "required": ["R"], "parameter_ranges": {"C": [1e-10, 1e-4]}},
                "optimization": {"points": 24, "max_iterations": 4, "top_k": 1, "seed": 3},
            }
        )


# ---------------------------------------------------------------------------
# DC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,overrides,optimization",
    [
        ("efficiency above unity", {"efficiency": 1.5}, None),
        (
            "efficiency above what losses allow",
            {"efficiency": 0.99},
            {"max_iterations": 8, "seed": 3, "loss_model": {"enabled": True}},
        ),
        ("ripple below what the parts allow", {"ripple_mv": 0.001}, None),
        ("output voltage far beyond the input", {"output_voltage_v": 1e6}, None),
    ],
)
def test_impossible_dc_requirements_are_refused(label, overrides, optimization) -> None:
    try:
        result = design_from_pbdl(_dc_spec(overrides, optimization=optimization))
    except Exception:  # noqa: BLE001 - a refusal is the expected outcome
        return
    assert not result.validation.passed, f"{label} was accepted"


def test_a_too_small_component_budget_is_refused() -> None:
    try:
        design_from_pbdl(
            _dc_spec({}, {"max_component_count": 1}, optimization={"max_iterations": 4, "seed": 3})
        )
    except Exception:  # noqa: BLE001 - a refusal is the expected outcome
        return
    pytest.fail("a one-component budget must not yield a design")


def test_an_enormous_output_current_is_a_request_not_a_violation() -> None:
    """Recorded rather than asserted away: this is *not* a defect.

    Nothing the spec declares is breached -- the ideal model really does deliver
    5 V at 1 MA -- so refusing it would be the bug. It is kept here so a future
    reader does not mistake the acceptance for a gap.
    """

    result = design_from_pbdl(_dc_spec({"output_current_a": 1e6}))
    point = result.optimization.operating_point
    assert point.output_current_a == pytest.approx(1e6, rel=1e-3)
    assert point.output_voltage_v == pytest.approx(5.0, rel=1e-3)
    assert result.validation.passed


# ---------------------------------------------------------------------------
# PBDL runner
# ---------------------------------------------------------------------------


def _pbdl_probe(tmp_path: Path, label: str, **mutations):
    data = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    data["name"] = label
    data["targets"][0].update(mutations)
    path = tmp_path / f"{label}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return run_pbdl_file(path, tmp_path / f"{label}_run", top_k=1)


@pytest.mark.parametrize("tolerance", [0.5, 1e-6])
def test_a_pbdl_order_mismatch_is_refused(tmp_path, tolerance) -> None:
    report = _pbdl_probe(
        tmp_path, f"tight_{tolerance}", order=4, tolerance={"relative": None, "absolute": tolerance}
    )
    assert not report.succeeded


def test_a_tight_tolerance_is_met_when_the_topology_can_meet_it(tmp_path) -> None:
    """A single pole tracks a single pole exactly, so 1e-6 dB is achievable.

    This is the non-defect that a naive sweep would report as a hole.
    """

    report = _pbdl_probe(
        tmp_path, "order1_tight", order=1, tolerance={"relative": None, "absolute": 1e-6}
    )
    assert report.succeeded
    assert report.stages[0].best_rmse_db < 1e-6
