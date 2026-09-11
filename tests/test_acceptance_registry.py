"""The acceptance-domain registry.

Three acceptance domains exist -- the operating envelope, the frequency-band mask
and the load-step budget -- and they are structurally the same thing: declare an
acceptable region, evaluate a chosen design pointwise across it, attribute the
worst case, and refuse to imply more than is proved.

Before this registry each one was wired at four near-identical result sites,
because the four power result classes are copies of one another.  Adding a fourth
domain meant four more edits that could silently disagree.  These tests pin the
registry, the guard rails that stop a domain being silently dropped, and the fact
that the shipping domains are reachable through it.
"""

from __future__ import annotations

import json

import pytest

from circuit_ai import acceptance
from circuit_ai.acceptance import (
    ACCEPTANCE_DOMAIN_NAMES,
    AcceptanceDomain,
    DesignContext,
    acceptance_fields,
    apply_acceptance_domains,
    domain_names,
    domain_results,
    register_domain,
)


@pytest.fixture(autouse=True)
def _restore_registry():
    """The registry is module-global, so snapshot and restore around each test."""

    saved = dict(acceptance._REGISTRY)
    yield
    acceptance._REGISTRY.clear()
    acceptance._REGISTRY.update(saved)


class _StubIR:
    """The minimum an IR must expose for a domain to be evaluated on it."""

    operating_envelope = None

    def __init__(self) -> None:
        self.metadata: dict = {}


def _context() -> DesignContext:
    return DesignContext(
        ir=_StubIR(),
        design=None,
        nominal_input_voltage_v=36.0,
        nominal_output_voltage_v=5.0,
        nominal_output_current_a=2.0,
        design_rail_v=36.0,
    )


class _Record:
    def __init__(self, value: float) -> None:
        self.value = value

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value}


# ---------------------------------------------------------------------------
# The registry contract
# ---------------------------------------------------------------------------


def test_shipping_domains_are_registered_under_result_fields() -> None:
    from circuit_ai import power

    assert domain_names() == ("envelope_analysis", "load_step")
    for name in domain_names():
        assert name in ACCEPTANCE_DOMAIN_NAMES
    # Every registered name must actually exist on the result record, which is
    # what makes the splat at the call site safe.
    fields = {item.name for item in power.PowerStageResult.__dataclass_fields__.values()} if hasattr(
        power, "PowerStageResult"
    ) else {item.name for item in power.BoostOptimizationResult.__dataclass_fields__.values()}
    assert set(domain_names()) <= fields


def test_a_domain_may_only_register_under_a_real_result_field() -> None:
    """A silently dropped domain is worse than a loud error."""

    with pytest.raises(ValueError, match="no field on the power result"):
        register_domain(AcceptanceDomain(name="not_a_field", evaluate=lambda context: None))


def test_a_domain_needs_a_name_and_a_callable() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        AcceptanceDomain(name="  ", evaluate=lambda context: None)
    with pytest.raises(TypeError, match="callable"):
        AcceptanceDomain(name="envelope_analysis", evaluate=None)  # type: ignore[arg-type]


def test_registering_by_an_existing_name_replaces_it() -> None:
    acceptance._REGISTRY.clear()
    first = register_domain(
        AcceptanceDomain(name="load_step", evaluate=lambda context: _Record(1.0))
    )
    second = register_domain(
        AcceptanceDomain(name="load_step", evaluate=lambda context: _Record(2.0))
    )
    assert first is not second
    assert domain_names().count("load_step") == 1
    produced = domain_results(_context())
    assert produced["load_step"].value == 2.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_a_domain_returning_none_makes_no_claim() -> None:
    """Absent must stay distinct from explicitly None."""

    acceptance._REGISTRY.clear()
    register_domain(AcceptanceDomain(name="load_step", evaluate=lambda context: None))
    assert domain_results(_context()) == {}
    assert acceptance_fields(_context(), result_type=object) == {}


def test_every_registered_domain_is_evaluated() -> None:
    acceptance._REGISTRY.clear()
    register_domain(AcceptanceDomain(name="envelope_analysis", evaluate=lambda c: _Record(1.0)))
    register_domain(AcceptanceDomain(name="load_step", evaluate=lambda c: _Record(2.0)))
    produced = domain_results(_context())
    assert set(produced) == {"envelope_analysis", "load_step"}


def test_the_context_carries_what_domains_need() -> None:
    acceptance._REGISTRY.clear()
    seen: dict[str, object] = {}

    def spy(context: DesignContext):
        seen["vout"] = context.nominal_output_voltage_v
        seen["iout"] = context.nominal_output_current_a
        seen["rail"] = context.design_rail_v
        seen["power"] = context.nominal_output_power_w

    register_domain(AcceptanceDomain(name="load_step", evaluate=spy))
    domain_results(_context())
    assert seen == {"vout": 5.0, "iout": 2.0, "rail": 36.0, "power": 10.0}


def test_a_domain_may_not_produce_a_field_the_result_lacks() -> None:
    """The check that turns a silent drop into an error."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Tiny:
        # Deliberately no ``load_step`` field.
        other: object = None

    acceptance._REGISTRY.clear()
    register_domain(AcceptanceDomain(name="load_step", evaluate=lambda c: _Record(1.0)))
    with pytest.raises(ValueError, match="does not have"):
        acceptance_fields(_context(), result_type=_Tiny)


def test_apply_writes_the_records_onto_a_frozen_result() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Result:
        load_step: object = None

    acceptance._REGISTRY.clear()
    register_domain(AcceptanceDomain(name="load_step", evaluate=lambda c: _Record(7.0)))
    updated = apply_acceptance_domains(_Result(), _context())
    assert updated.load_step is not None
    assert updated.load_step.value == 7.0
    # The original must be untouched: these are frozen records.
    assert _Result().load_step is None


def test_apply_returns_the_same_object_when_nothing_is_declared() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Result:
        load_step: object = None

    acceptance._REGISTRY.clear()
    register_domain(AcceptanceDomain(name="load_step", evaluate=lambda c: None))
    original = _Result()
    assert apply_acceptance_domains(original, _context()) is original


# ---------------------------------------------------------------------------
# End to end: the shipping domains still reach the artifact
# ---------------------------------------------------------------------------


def _enveloped_with_budget() -> dict:
    return {
        "name": "acceptance_registry",
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
                "R": [100.0, 1e6], "C": [1e-6, 5e-3], "L": [1e-9, 1.0],
                "turns_ratio": [0.05, 2.0],
            },
        },
        "operating_envelope": {
            "min_input_voltage_v": 24.0, "max_input_voltage_v": 48.0,
            "min_load_fraction": 0.1, "max_load_fraction": 1.0,
        },
        "load_step": {
            "from_fraction": 0.1, "to_fraction": 1.0,
            "max_undershoot_mv": 50.0, "loop_response_s": 50e-6,
        },
        "optimization": {"max_iterations": 6, "seed": 5, "loss_model": {"enabled": True}},
    }


def test_both_domains_reach_a_real_design() -> None:
    from circuit_ai.pipeline import design_from_pbdl

    result = design_from_pbdl(_enveloped_with_budget())
    assert result.optimization.envelope_analysis is not None
    assert result.optimization.load_step is not None


def test_an_undeclared_domain_stays_none_on_a_real_design() -> None:
    """The four call sites pass through the registry, so absence must survive it."""

    from circuit_ai.pipeline import design_from_pbdl

    spec = _enveloped_with_budget()
    spec.pop("load_step")
    spec.pop("operating_envelope")
    result = design_from_pbdl(spec)
    assert result.optimization.envelope_analysis is None
    assert result.optimization.load_step is None


def test_domain_records_serialise_through_the_report(tmp_path) -> None:
    from circuit_ai.pipeline import design_from_pbdl

    design_from_pbdl(_enveloped_with_budget(), output_dir=tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["worst_case"] is not None
    assert report["load_step"] is not None
    json.dumps(report)
