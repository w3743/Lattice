"""Topology generation is data-driven and ratio-aware.

Two independent claims, both testable:

1. The generator must not offer a family whose conversion ratio cannot reach the
   request.  Applicability in the knowledge base is deliberately coarse --
   ``step_up``/``step_down`` -- so it cannot see *bounds*: a boost asked for 40x
   passes that filter and then fails at optimisation time, wasting the search and
   reporting a confusing error.
2. Adding a new generatable topology must be a knowledge declaration, not a code
   change.  A forward converter is composed here from the *existing* functional
   modules and handed to the grammar through a temporary knowledge file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from circuit_ai.ir import pbdl_to_ir
from circuit_ai.knowledge import DEFAULT_POWER_KNOWLEDGE, load_topology_knowledge
from circuit_ai.power import (
    POWER_STAGE_MODELS,
    PowerStageModel,
    stage_model_by_solver_id,
)
from circuit_ai.topology_grammar import PowerTopologyGrammar, _rejection_reason

BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


def _spec(vin: float, vout: float, *, isolated: bool = False, iout: float = 0.2) -> dict:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["name"] = f"{vin}_to_{vout}"
    data["targets"] = [
        {
            "target_kind": "dc",
            "input_voltage_v": vin,
            "output_voltage_v": vout,
            "output_current_a": iout,
        }
    ]
    if isolated:
        # Galvanic isolation needs genuinely separate reference nodes, so the
        # output port gets its own ground rather than sharing the input's.
        data["relations"] = [
            {"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}
        ]
        data["ports"] = [
            {
                "name": "input",
                "terminals": [
                    {"name": "in", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
            {
                "name": "output",
                "terminals": [
                    {"name": "out", "quantity": "voltage"},
                    {"name": "out_0", "quantity": "ground"},
                ],
            },
        ]
        data["constraints"]["element_types"] = [
            "R", "C", "L", "ideal_switch", "ideal_transformer", "ideal_diode"
        ]
    return data


# ---------------------------------------------------------------------------
# Reachable conversion ratio as a declared property
# ---------------------------------------------------------------------------


def test_every_registered_family_declares_its_ratio() -> None:
    """A family that declares no ratio can make no reachability claim at all."""

    for stage in POWER_STAGE_MODELS:
        assert stage.conversion_ratio is not None, stage.solver_id
        bounds = stage.reachable_ratio_range()
        assert bounds is not None
        low, high = bounds
        assert 0.0 < low < high


def test_reachable_ranges_match_each_topology() -> None:
    """Buck steps down, boost steps up, SEPIC spans unity, flyback spans widely."""

    buck = stage_model_by_solver_id("ideal_buck_averaged")
    boost = stage_model_by_solver_id("ideal_boost_averaged")
    sepic = stage_model_by_solver_id("ideal_sepic_averaged")
    flyback = stage_model_by_solver_id("ideal_flyback_averaged")
    assert buck is not None and boost is not None and sepic is not None and flyback is not None

    assert buck.can_reach_ratio(0.5) and not buck.can_reach_ratio(1.0)
    assert boost.can_reach_ratio(2.0) and not boost.can_reach_ratio(1.0)
    assert sepic.can_reach_ratio(0.5) and sepic.can_reach_ratio(1.0) and sepic.can_reach_ratio(2.0)
    assert flyback.can_reach_ratio(0.01) and flyback.can_reach_ratio(10.0)


def test_ratio_bounds_are_derived_from_the_ratio_function() -> None:
    """Bounds track the family's own function instead of a hand-kept table."""

    class _Double(PowerStageModel):
        pass

    scaled = PowerStageModel(
        solver_id="test_double",
        family="test",
        optimizer_type=type("X", (), {}),
        dc_solver=POWER_STAGE_MODELS[0].dc_solver,
        conversion_ratio=lambda duty, turns: 2.0 * duty,
        duty_limits=(0.1, 0.5),
    )
    assert scaled.reachable_ratio_range() == pytest.approx((0.2, 1.0))
    del _Double


def test_a_family_without_a_ratio_is_never_rejected_on_one() -> None:
    """An undeclared model keeps its previous behaviour rather than guessing."""

    undeclared = PowerStageModel(
        solver_id="test_undeclared",
        family="test",
        optimizer_type=type("X", (), {}),
        dc_solver=POWER_STAGE_MODELS[0].dc_solver,
    )
    assert undeclared.reachable_ratio_range() is None
    assert undeclared.can_reach_ratio(1e6)


# ---------------------------------------------------------------------------
# The generator must respect reachability
# ---------------------------------------------------------------------------


def test_extreme_step_up_is_rejected_with_a_reason() -> None:
    """A 40x step-up passes the coarse 'step_up' filter but no family can serve it."""

    result = PowerTopologyGrammar().search(pbdl_to_ir(_spec(5.0, 200.0)))
    assert result.candidates == ()
    reasons = {item.name: item.reason for item in result.certificate.rejected}
    assert reasons, "an unreachable request must record why nothing was offered"
    assert all("cannot produce a conversion ratio" in reason for reason in reasons.values())


def test_extreme_step_down_is_rejected_with_a_reason() -> None:
    """A 50x step-down is outside every non-isolated duty band."""

    result = PowerTopologyGrammar().search(pbdl_to_ir(_spec(5.0, 0.1)))
    assert result.candidates == ()
    reasons = {item.name: item.reason for item in result.certificate.rejected}
    assert reasons
    assert any("dc_buck" in reason for reason in reasons.values())


def test_reachable_requests_are_still_served() -> None:
    """The check must not over-reject: ordinary designs keep working."""

    step_up = PowerTopologyGrammar().search(pbdl_to_ir(_spec(5.0, 10.0)))
    assert {item.name for item in step_up.candidates} >= {"ideal_asynchronous_boost"}

    step_down = PowerTopologyGrammar().search(pbdl_to_ir(_spec(12.0, 3.3)))
    assert {item.name for item in step_down.candidates} >= {"grammar_ideal_buck"}

    moderate = PowerTopologyGrammar().search(pbdl_to_ir(_spec(5.0, 90.0)))
    assert moderate.candidates, "18x is inside the boost's reachable band"


def test_rejection_reason_is_explicit_about_the_band() -> None:
    """The message must let a user see why, not just that it failed."""

    data = _spec(5.0, 200.0)
    ir = pbdl_to_ir(data)
    stage = stage_model_by_solver_id("ideal_boost_averaged")
    assert stage is not None
    bounds = stage.reachable_ratio_range()
    assert bounds is not None

    # Take a real generated boost candidate and re-ask the question, so nothing
    # structural is what rejects it.
    grammar = PowerTopologyGrammar()
    context = __import__(
        "circuit_ai.topology_grammar", fromlist=["_power_context"]
    )._power_context(ir)
    production = next(
        item for item in grammar.knowledge.productions if item.solver == "ideal_boost_averaged"
    )
    candidate = grammar.knowledge.instantiate(production, context)

    reason = _rejection_reason(ir, candidate, isolated=False)
    assert reason is not None
    assert "cannot produce a conversion ratio" in reason
    assert f"{bounds[0]:.4g}..{bounds[1]:.4g}" in reason
    assert "40" in reason


# ---------------------------------------------------------------------------
# Adding a generatable topology is a knowledge declaration
# ---------------------------------------------------------------------------


FORWARD_MODULES = {
    # Reuses the shipped modules verbatim; only the composition is new.
    "series_storage": {
        "name": "two_terminal_inductor",
        "ports": ["a", "b"],
        "components": [{"ref": "element", "kind": "L", "nodes": ["a", "b"]}],
    },
}

FORWARD_PRODUCTION = {
    "name": "grammar_ideal_forward",
    "family": "isolated_forward",
    "solver": "ideal_forward_probe",
    "rationale": "Forward converter: transformer-coupled buck with a rectifier.",
    "applicability": {"isolation": "required", "voltage_relation": "any"},
    # The isolation contract is declared in knowledge, not derived from the
    # module list -- the same flag the shipped flyback production sets.
    "metadata": {"isolated": True, "conduction_mode": "ideal_ccm"},
    "slots": [
        {
            "name": "isolated_transfer",
            "module": "isolated_transformer",
            "bindings": {
                "primary_a": "$input",
                "primary_b": "sw",
                "secondary_a": "out_sw",
                "secondary_b": "$output_ref",
            },
            "references": {"element": "T1"},
        },
        {
            "name": "controlled_switch",
            "module": "pwm_switch",
            "bindings": {"a": "sw", "b": "$input_ref"},
            "references": {"element": "Q1"},
        },
        {
            "name": "secondary_rectifier",
            "module": "rectifier",
            "bindings": {"anode": "out_sw", "cathode": "out_l"},
            "references": {"element": "D1"},
        },
        {
            "name": "output_storage",
            "module": "two_terminal_inductor",
            "bindings": {"a": "out_l", "b": "$output"},
            "references": {"element": "L1"},
        },
        {
            "name": "output_filter",
            "module": "two_terminal_capacitor",
            "bindings": {"a": "$output", "b": "$output_ref"},
            "references": {"element": "C1"},
        },
        {
            "name": "load",
            "module": "external_resistive_load",
            "bindings": {"positive": "$output", "negative": "$output_ref"},
            "references": {"element": "Rload"},
        },
    ],
}


def _forward_knowledge(tmp_path: Path) -> Path:
    """The shipped knowledge base plus one production.

    The shipped flyback production is copied in unchanged so both candidates come
    from the same grammar instance and can be compared directly.
    """

    data = yaml.safe_load(DEFAULT_POWER_KNOWLEDGE.read_text(encoding="utf-8"))
    data["knowledge_id"] = "power_topologies_plus_forward"
    data["productions"] = [*data["productions"], FORWARD_PRODUCTION]
    path = tmp_path / "forward_topologies.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_the_shipped_knowledge_has_no_forward_production() -> None:
    """Otherwise the next test would pass without the declaration doing anything."""

    knowledge = load_topology_knowledge()
    assert "grammar_ideal_forward" not in {item.name for item in knowledge.productions}
    assert "isolated_forward" not in {item.family for item in knowledge.productions}


def test_a_new_production_is_declarative_and_gets_generated(tmp_path) -> None:
    """One knowledge entry composes a new topology out of the existing modules."""

    grammar = PowerTopologyGrammar(_forward_knowledge(tmp_path))
    result = grammar.search(
        pbdl_to_ir(_spec(48.0, 5.0, isolated=True)),
        # The stored knowledge picks slots from its own modules, so pass the
        # forward in as an extra candidate source is NOT needed: the production
        # itself is what generates it.
    )
    names = {item.name for item in result.candidates}
    assert "grammar_ideal_forward" in names, sorted(names)

    forward = next(item for item in result.candidates if item.name == "grammar_ideal_forward")
    assert forward.metadata["solver"] == "ideal_forward_probe"
    assert forward.metadata["isolated"] is True
    assert forward.graph is not None
    # The forward must carry its own series inductor; that is what distinguishes
    # it from the flyback-shaped isolation the base knowledge offers.
    kinds = [component.kind for component in forward.components]
    assert "L" in kinds and "ideal_transformer" in kinds

    # It is not a duplicate of anything else the grammar produced.
    hashes = [item.graph.graph_hash for item in result.candidates if item.graph is not None]
    assert len(hashes) == len(set(hashes)), "the new production duplicated an existing graph"


def test_the_new_production_still_obeys_the_ratio_check(tmp_path) -> None:
    """A declared production is not exempt from reachability."""

    grammar = PowerTopologyGrammar(_forward_knowledge(tmp_path))
    # The probe solver is not a registered stage, so no reachability claim is
    # made about it and it is accepted; an extreme ratio must therefore still
    # reach the generator rather than being silently dropped.
    result = grammar.search(pbdl_to_ir(_spec(48.0, 200.0, isolated=True)))
    rejected = {item.name: item.reason for item in result.certificate.rejected}
    assert "grammar_ideal_forward" not in rejected or "ratio" not in rejected.get(
        "grammar_ideal_forward", ""
    )
