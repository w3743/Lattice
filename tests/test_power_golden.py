"""Golden behaviour lock for the four ideal power families (plan §18.2 / §1527).

Scope and limits -- read before changing tolerances:

* This is a **characterization lock**: it records the *current* observable behaviour of
  Buck / Boost / SEPIC / Flyback synthesis (``测试集/golden/power_families.json``) so that a
  refactor which silently changes it is detected.  It is **not** a correctness proof: a
  snapshot can only be trusted as far as the code that produced it.
* The SEPIC entry documents current behaviour, not a desired outcome: the ``dc_sepic_*``
  spec currently resolves to the **boost** family/solver although the knowledge base
  declares ``grammar_ideal_sepic`` (see the ``notes`` field of the snapshot).  The lock
  intentionally pins this so any change to production selection is surfaced.
* The snapshot itself records the environment (python/numpy/scipy) and the seed source;
  it was verified repeatable by running every family twice.
* No case needs ngspice; the averaged ideal models are closed-form.  If a family cannot be
  executed the snapshot marks it ``unavailable`` and this test fails loudly rather than
  pretending a value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_ai.pbdl_runner import run_pbdl_file


GOLDEN_PATH = Path(__file__).resolve().parent.parent / "测试集" / "golden" / "power_families.json"
FAMILIES = ("buck", "boost", "sepic", "flyback")
OPERATING_POINT_TOLERANCE = {
    "input_voltage_v": "voltage_v",
    "output_voltage_v": "voltage_v",
    "output_current_a": "current_a",
    "input_current_a": "current_a",
    "output_power_w": "power_w",
    "efficiency": "efficiency",
    "predicted_ripple_mv": "ripple_mv",
}


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _close(actual: float, expected: float, tolerance: dict) -> bool:
    if tolerance == "exact":
        return actual == expected
    return abs(actual - expected) <= max(
        float(tolerance["absolute"]),
        float(tolerance["relative"]) * abs(expected),
    )


@pytest.fixture(scope="module")
def synthesised(tmp_path_factory) -> dict:
    """Run every golden family once and return the parsed stage reports."""
    golden = _golden()
    root = tmp_path_factory.mktemp("power_golden")
    results: dict[str, dict] = {}
    for family in FAMILIES:
        entry = golden["families"][family]
        report = run_pbdl_file(Path(entry["case_file"]), root / family, top_k=1)
        stage = report.stages[0]
        assert report.succeeded, f"{family}: PBDL run failed ({stage.error})"
        payload = json.loads(
            (root / family / "stage_1_dc_transfer" / "report.json").read_text(encoding="utf-8")
        )
        results[family] = {"stage": stage, "report": payload}
    return results


def test_golden_snapshot_declares_environment_and_seed() -> None:
    golden = _golden()

    assert golden["schema"] == "circuit_ai.power_family_golden"
    assert golden["environment"]["python"] and golden["environment"]["numpy"]
    assert golden["determinism"]["verified_repeatable"] is True
    for family in FAMILIES:
        entry = golden["families"][family]
        assert entry["status"] == "available", f"{family} is marked {entry['status']}"
        assert isinstance(entry["seed"], int), f"{family} must pin a seed"


@pytest.mark.parametrize("family", FAMILIES)
def test_selected_topology_matches_golden(family, synthesised) -> None:
    expected = _golden()["families"][family]["selected_topology"]
    actual = synthesised[family]

    assert actual["stage"].best_template == expected["template"]
    assert actual["report"]["topology"]["family"] == expected["family"]
    assert actual["report"]["topology"]["solver_id"] == expected["solver_id"]
    assert actual["report"]["topology"]["graph_hash"] == expected["graph_hash"]


@pytest.mark.parametrize("family", FAMILIES)
def test_operating_point_matches_golden(family, synthesised) -> None:
    golden = _golden()
    tolerances = golden["tolerances"]
    expected = golden["families"][family]
    actual = synthesised[family]["report"]["operating_point"]

    for name, tolerance_key in OPERATING_POINT_TOLERANCE.items():
        assert _close(
            actual[name], expected["operating_point"][name], tolerances[tolerance_key]
        ), f"{family}.{name}: {actual[name]!r} != {expected['operating_point'][name]!r}"


@pytest.mark.parametrize("family", FAMILIES)
def test_resource_and_validation_match_golden(family, synthesised) -> None:
    golden = _golden()
    tolerances = golden["tolerances"]
    expected = golden["families"][family]
    report = synthesised[family]["report"]

    assert report["validation"]["passed"] is True
    assert report["validation"]["feasibility"] == expected["validation"]["feasibility"]
    assert report["validation"]["unavailable_hard_count"] == 0
    assert _close(report["validation"]["cost_cny"], expected["cost_cny"], tolerances["cost_cny"])
    assert _close(report["validation"]["area_mm2"], expected["area_mm2"], tolerances["area_mm2"])
    actual_count = report["validation"]["metrics"]["resource.component_count"]["value"]
    assert actual_count == expected["component_count"]


@pytest.mark.parametrize("family", FAMILIES)
def test_output_error_stays_inside_declared_spec_tolerance(family, synthesised) -> None:
    """Requirement-level check: the tuned design must meet the spec's own tolerance."""
    entry = synthesised[family]["report"]
    target = _golden()["families"][family]["spec_target"]
    assert entry["validation"]["passed"] is True
    assert entry["validation"]["hard_violation"] == 0.0

    error = abs(entry["operating_point"]["output_voltage_v"] - target["output_voltage_v"])
    error /= target["output_voltage_v"]
    assert error <= target["relative_tolerance"]


@pytest.mark.parametrize("family", FAMILIES)
def test_low_fidelity_result_is_truth_evaluated(family, synthesised) -> None:
    """Plan §1663: any surrogate/gradient/analytic result is re-evaluated before acceptance."""
    run = synthesised[family]["report"]["candidates"][0]["optimization_run"]
    expected = _golden()["families"][family]["optimization_run"]

    assert run["truth_evaluated"] is True
    assert run["truth_evaluations"] == expected["truth_evaluations"]
    assert run["solver_id"] == expected["solver_id"]
    assert run["evaluations"] > 0


def test_isolated_family_keeps_symmetric_isolation_domains(synthesised) -> None:
    report = synthesised["flyback"]["report"]
    graph = report["topology"]["graph"]
    domains = {item["domain_id"]: item for item in graph["domains"]}

    assert set(domains) == {"input_domain", "output_domain"}
    assert domains["input_domain"]["isolated_from"] == ["output_domain"]
    assert domains["output_domain"]["isolated_from"] == ["input_domain"]
    assert domains["input_domain"]["reference_net_id"] != domains["output_domain"]["reference_net_id"]

    spanning = [
        component["reference"]
        for component in graph["components"]
        if len(
            {
                net["domain_id"]
                for connection in component["connections"]
                for net in graph["nets"]
                if net["net_id"] == connection["net_id"]
            }
        )
        > 1
    ]
    assert spanning == ["T1"]


@pytest.mark.parametrize("family", ("buck", "boost", "sepic"))
def test_non_isolated_families_have_a_single_domain(family, synthesised) -> None:
    domains = synthesised[family]["report"]["topology"]["graph"]["domains"]

    assert len(domains) == 1
    assert not domains[0].get("isolated_from")
