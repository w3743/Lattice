"""Load-step transient response.

A load-step budget -- "the rail must not sag more than X mV when the load steps
from A to B" -- was inexpressible before this: the only transient model was
startup.  These tests cover the charge-balance model, its budget verdict, its
refusal to imply a stability claim, and the fact that the budget steers the
optimizer rather than only judging the result.
"""

from __future__ import annotations

import json

import pytest

from circuit_ai.load_step import (
    LoadStepRequirement,
    evaluate_load_step,
    load_step_from_mapping,
)


def _requirement(**overrides) -> LoadStepRequirement:
    payload = {
        "from_fraction": 0.1,
        "to_fraction": 1.0,
        "max_undershoot_mv": 50.0,
        "loop_response_s": 50e-6,
    }
    payload.update(overrides)
    return LoadStepRequirement(**payload)


def _evaluate(capacitance_f: float, requirement: LoadStepRequirement | None = None):
    return evaluate_load_step(
        requirement or _requirement(),
        nominal_output_current_a=2.0,
        output_voltage_v=5.0,
        capacitance_f=capacitance_f,
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def test_excursion_matches_the_charge_balance_relation() -> None:
    """dI * tau / (2C) is the sizing relation the model must reproduce."""

    result = _evaluate(470e-6)
    expected_mv = 1.8 * 50e-6 / (2.0 * 470e-6) * 1000.0
    assert result.excursion_mv == pytest.approx(expected_mv, rel=0.01)
    # The numeric trajectory and the closed form must agree.
    assert result.extensions["charge_balance_excursion_mv"] == pytest.approx(
        result.excursion_mv, rel=0.01
    )


def test_more_capacitance_means_less_excursion() -> None:
    excursions = [_evaluate(c).excursion_mv for c in (100e-6, 470e-6, 1000e-6, 2200e-6)]
    assert excursions == sorted(excursions, reverse=True)


def test_a_faster_loop_means_less_excursion() -> None:
    fast = _evaluate(470e-6, _requirement(loop_response_s=10e-6))
    slow = _evaluate(470e-6, _requirement(loop_response_s=200e-6))
    assert fast.excursion_mv < slow.excursion_mv


def test_the_budget_verdict_tracks_the_excursion() -> None:
    tight = _evaluate(470e-6)          # ~96 mV against a 50 mV budget
    comfortable = _evaluate(2000e-6)   # ~22 mV
    assert not tight.passed
    assert comfortable.passed


def test_capacitance_for_budget_is_the_actionable_number() -> None:
    """It must be the value at which the excursion exactly meets the budget.

    The closed form is exact; the reported excursion comes from an Euler
    trajectory, which lands about 0.15 % low at this grid.  The tolerance
    reflects the integrator rather than pretending to precision it does not have.
    """

    result = _evaluate(470e-6)
    needed = result.capacitance_for_budget_f
    assert needed is not None
    at_budget = _evaluate(needed)
    assert at_budget.excursion_mv == pytest.approx(50.0, rel=5e-3)
    assert at_budget.excursion_mv <= 50.0


def test_a_load_release_is_judged_against_the_overshoot_budget() -> None:
    release = _requirement(
        from_fraction=1.0, to_fraction=0.1, max_undershoot_mv=None, max_overshoot_mv=50.0
    )
    assert release.is_release
    result = _evaluate(470e-6, release)
    # Same magnitude of charge, opposite sign, and it is the overshoot budget
    # that decides.
    assert result.excursion_mv == pytest.approx(96.0, rel=0.02)
    assert not result.passed
    assert _evaluate(2000e-6, release).passed


def test_a_recovery_budget_is_checked_independently() -> None:
    result = _evaluate(470e-6, _requirement(max_undershoot_mv=None, max_recovery_us=100.0))
    assert result.recovery_us > 100.0
    assert not result.passed
    # The excursion still exists but is not budgeted, so only recovery decides.
    assert result.excursion_mv > 0.0


def test_the_model_refuses_to_imply_a_stability_claim() -> None:
    """This is a feasibility model, and the record has to say so."""

    result = _evaluate(470e-6)
    assert result.justifies_stability_claim is False
    payload = result.as_dict()
    assert payload["justifies_stability_claim"] is False
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"from_fraction": 1.0, "to_fraction": 1.0, "max_undershoot_mv": 50.0},
        {"from_fraction": -0.1, "to_fraction": 1.0, "max_undershoot_mv": 50.0},
        {"from_fraction": 0.1, "to_fraction": 1.0},
        {"from_fraction": 0.1, "to_fraction": 1.0, "max_undershoot_mv": 50.0, "loop_response_s": 0.0},
        {"from_fraction": 0.1, "to_fraction": 1.0, "max_undershoot_mv": -5.0},
    ],
)
def test_requirements_reject_nonsense(payload) -> None:
    with pytest.raises(ValueError):
        LoadStepRequirement(**payload)


def test_evaluation_rejects_degenerate_designs() -> None:
    requirement = _requirement()
    with pytest.raises(ValueError):
        evaluate_load_step(requirement, nominal_output_current_a=0.0, output_voltage_v=5.0, capacitance_f=1e-4)
    with pytest.raises(ValueError):
        evaluate_load_step(requirement, nominal_output_current_a=2.0, output_voltage_v=0.0, capacitance_f=1e-4)
    with pytest.raises(ValueError):
        evaluate_load_step(requirement, nominal_output_current_a=2.0, output_voltage_v=5.0, capacitance_f=0.0)


# ---------------------------------------------------------------------------
# Parsing, both container and bare block
# ---------------------------------------------------------------------------


def test_a_spec_without_a_budget_declares_none() -> None:
    assert load_step_from_mapping({}) is None
    assert load_step_from_mapping({"name": "x"}) is None


def test_both_the_container_and_the_bare_block_parse() -> None:
    """A container lookup on a block silently finds nothing, so cover both."""

    block = {"from_fraction": 0.2, "to_fraction": 0.8, "max_undershoot_mv": 40.0}
    from_container = load_step_from_mapping({"load_step": block})
    from_block = load_step_from_mapping(block)
    assert from_container is not None and from_block is not None
    assert from_container.from_fraction == from_block.from_fraction == 0.2
    assert from_container.max_undershoot_mv == from_block.max_undershoot_mv == 40.0


# ---------------------------------------------------------------------------
# The budget steers the design
# ---------------------------------------------------------------------------


def _spec(*, with_budget: bool, c_max: float = 5e-3) -> dict:
    spec = {
        "name": "load_step_design",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "out_0", "quantity": "ground"}]},
        ],
        "relations": [{"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [{"target_kind": "dc", "input_voltage_v": 36, "output_voltage_v": 5, "output_current_a": 2}],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_transformer", "ideal_diode"],
            "max_component_count": 5,
            "parameter_ranges": {
                "R": [100.0, 1e6], "C": [1e-6, c_max], "L": [1e-9, 1.0],
                "turns_ratio": [0.05, 2.0],
            },
        },
        "optimization": {"max_iterations": 12, "seed": 5, "loss_model": {"enabled": True}},
    }
    if with_budget:
        spec["load_step"] = {
            "from_fraction": 0.1,
            "to_fraction": 1.0,
            "max_undershoot_mv": 50.0,
            "loop_response_s": 50e-6,
        }
    return spec


def test_a_declared_budget_changes_the_selected_capacitance() -> None:
    """The point of charging the breach: the optimizer has to react to it."""

    from circuit_ai.pipeline import design_from_pbdl

    without = design_from_pbdl(_spec(with_budget=False))
    with_budget = design_from_pbdl(_spec(with_budget=True))

    assert without.optimization.load_step is None
    assert with_budget.optimization.load_step is not None

    c_without = without.optimization.parameters.capacitance_f
    c_with = with_budget.optimization.parameters.capacitance_f
    assert c_with > c_without, (c_with, c_without)

    # And the excursion lands on the budget it was charged against.  A squared
    # penalty stops just short of the boundary, so the verdict may still read
    # failed; what must hold is that the excursion is at the budget, not far
    # inside or far outside it.
    excursion = with_budget.optimization.load_step.excursion_mv
    assert excursion == pytest.approx(50.0, rel=0.05)


def test_the_budget_reaches_the_report_artifact(tmp_path) -> None:
    from circuit_ai.pipeline import design_from_pbdl

    design_from_pbdl(_spec(with_budget=True), output_dir=tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["load_step"]["passed"] in {True, False}
    assert report["load_step"]["excursion_mv"] > 0.0
    assert report["load_step"]["justifies_stability_claim"] is False
    assert report["load_step"]["capacitance_for_budget_f"] is not None
    json.dumps(report)


def test_no_budget_means_no_penalty_and_no_claim() -> None:
    from circuit_ai.pipeline import design_from_pbdl

    result = design_from_pbdl(_spec(with_budget=False))
    assert result.optimization.load_step is None
