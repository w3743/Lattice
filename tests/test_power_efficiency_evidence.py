"""Efficiency evidence must come from the same model the design was sized with.

The analytic power backend binds a family's *lossless* solver, so a loss-aware
design used to report simulation evidence of ``efficiency == 1.0`` while the
optimizer's own operating point said 0.9215. The efficiency metric the validation
gate reads came from the backend, so an efficiency requirement was checked
against a number the design does not achieve: asking for 0.99 passed, because the
gate saw 1.0.

The loss coefficients now travel in the request's conditions, which is where they
have to live for a result to be reproducible from its own request.

These tests pin both halves: the loss-aware request produces loss-aware evidence
and can fail an efficiency requirement, and a request without loss coefficients
is byte-identical to before.
"""

from __future__ import annotations

import json

import pytest

from circuit_ai.pipeline import design_from_pbdl
from circuit_ai.power_request import (
    POWER_LOSS_CONDITIONS_KEY,
    loss_parameters_from_conditions,
)
from circuit_ai.simulation import OperatingConditions


def _spec(*, efficiency: float | None = None, loss: bool = False) -> dict:
    target = {
        "target_kind": "dc",
        "input_voltage_v": 12,
        "output_voltage_v": 5,
        "output_current_a": 3,
        "ripple_mv": 50.0,
        "tolerance": {"relative": 0.02},
    }
    if efficiency is not None:
        target["efficiency"] = efficiency
    optimization: dict = {"max_iterations": 8, "seed": 3}
    if loss:
        optimization["loss_model"] = {"enabled": True}
    return {
        "name": "efficiency_evidence",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        ],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [target],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_diode"],
            "max_component_count": 4,
            "parameter_ranges": {"R": [100.0, 1e6], "C": [1e-6, 5e-3], "L": [1e-6, 1e-3]},
        },
        "optimization": optimization,
    }


def _efficiency_check(result):
    checks = [item for item in result.validation.evaluations if "efficiency" in item.constraint_id]
    assert checks, "an efficiency target must produce an efficiency constraint"
    return checks[0]


def _efficiency_checks(result):
    return [item for item in result.validation.evaluations if "efficiency" in item.constraint_id]


# ---------------------------------------------------------------------------
# The key and its reader
# ---------------------------------------------------------------------------


def test_conditions_without_the_key_mean_no_loss_model() -> None:
    assert loss_parameters_from_conditions(OperatingConditions()) is None
    assert loss_parameters_from_conditions(None) is None


def test_conditions_carrying_the_key_resolve_to_parameters() -> None:
    conditions = OperatingConditions(
        variables={POWER_LOSS_CONDITIONS_KEY: {"switch_resistance_ohm": 0.02}}
    )
    resolved = loss_parameters_from_conditions(conditions)
    assert resolved is not None
    assert resolved.switch_resistance_ohm == 0.02


def test_a_malformed_key_is_ignored_rather_than_guessed() -> None:
    for bad in ("not-a-mapping", 3, None, [1, 2]):
        conditions = OperatingConditions(variables={POWER_LOSS_CONDITIONS_KEY: bad})
        assert loss_parameters_from_conditions(conditions) is None


# ---------------------------------------------------------------------------
# The defect: evidence and design must agree
# ---------------------------------------------------------------------------


def test_loss_aware_evidence_matches_the_loss_aware_operating_point() -> None:
    """The two numbers used to disagree: 1.0 in evidence, 0.9215 in the design.

    No efficiency target is declared here, so this checks the evidence the
    simulation itself carries rather than a constraint evaluation.
    """

    result = design_from_pbdl(_spec(loss=True))
    reported = result.optimization.operating_point.efficiency
    assert reported < 1.0, "the loss model must actually cost something"

    efficiencies = [
        item.scalars["efficiency"].value
        for item in result.simulation_results
        if "efficiency" in item.scalars
    ]
    assert efficiencies, "the power simulation must publish an efficiency"
    for value in efficiencies:
        assert value == pytest.approx(reported, rel=1e-9), (
            "simulation evidence must agree with the design it describes"
        )


def test_an_unreachable_efficiency_requirement_now_fails() -> None:
    """0.99 is not achieved at 0.9215, and the gate must say so."""

    result = design_from_pbdl(_spec(efficiency=0.99, loss=True))
    assert not result.validation.passed
    assert not _efficiency_check(result).passed


def test_a_reachable_efficiency_requirement_still_passes() -> None:
    result = design_from_pbdl(_spec(efficiency=0.8, loss=True))
    assert _efficiency_check(result).passed


def test_without_the_loss_model_the_ideal_answer_is_unchanged() -> None:
    """The default path must stay exactly as it was: no loss, no claim of loss."""

    result = design_from_pbdl(_spec(efficiency=0.8, loss=False))
    check = _efficiency_check(result)
    assert check.actual == pytest.approx(1.0)
    assert check.passed
    assert result.optimization.operating_point.efficiency == 1.0


def test_a_request_carries_the_loss_model_only_when_the_spec_asks() -> None:
    """Existing specs must keep their request hash, cache key and artifacts."""

    from circuit_ai.ir import pbdl_to_ir
    from circuit_ai.simulation_tasks import simulation_tasks_from_ir

    without = simulation_tasks_from_ir(pbdl_to_ir(_spec(loss=False)))
    assert POWER_LOSS_CONDITIONS_KEY not in without[0].conditions

    with_loss = simulation_tasks_from_ir(pbdl_to_ir(_spec(loss=True)))
    assert POWER_LOSS_CONDITIONS_KEY in with_loss[0].conditions


def test_the_loss_key_reaches_the_request_the_backend_sees() -> None:
    """The chain the fix depends on: spec -> task condition -> request variable."""

    from circuit_ai.ir import pbdl_to_ir
    from circuit_ai.simulation_tasks import simulation_tasks_from_ir

    ir = pbdl_to_ir(_spec(loss=True))
    task = simulation_tasks_from_ir(ir)[0]
    assert task.conditions[POWER_LOSS_CONDITIONS_KEY]["enabled"] is True

    request = task.to_requests(
        _graph_from(ir),
        parameter_values={},
        fidelity="ideal_averaged",
    )[0]
    resolved = loss_parameters_from_conditions(request.conditions)
    assert resolved is not None


def _graph_from(ir):
    """A minimal stage graph for the propagation test."""

    from circuit_ai.topology_grammar import PowerTopologyGrammar

    search = PowerTopologyGrammar().search(ir)
    assert search.candidates
    graph = search.candidates[0].graph
    assert graph is not None
    return graph


def test_the_loss_key_changes_the_cache_key_but_only_when_present() -> None:
    """Loss-aware and lossless results must not collide in the executor cache."""

    from circuit_ai.ir import pbdl_to_ir
    from circuit_ai.simulation_tasks import simulation_tasks_from_ir

    graph = _graph_from(pbdl_to_ir(_spec(loss=False)))
    plain = simulation_tasks_from_ir(pbdl_to_ir(_spec(loss=False)))[0]
    with_loss = simulation_tasks_from_ir(pbdl_to_ir(_spec(loss=True)))[0]

    plain_request = plain.to_requests(graph, parameter_values={}, fidelity="ideal_averaged")[0]
    loss_request = with_loss.to_requests(graph, parameter_values={}, fidelity="ideal_averaged")[0]
    assert plain_request.semantic_hash != loss_request.semantic_hash


def test_the_report_records_the_efficiency_it_was_judged_on(tmp_path) -> None:
    design_from_pbdl(_spec(efficiency=0.99, loss=True), output_dir=tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["validation"]["passed"] is False
    checks = [
        item for item in report["validation"]["evaluations"] if "efficiency" in item["constraint_id"]
    ]
    assert checks and checks[0]["actual"] == pytest.approx(
        report["operating_point"]["efficiency"], rel=1e-9
    )
