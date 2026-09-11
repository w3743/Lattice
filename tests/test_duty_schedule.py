"""Duty scheduling: separating feasibility from regulation.

A design stores one fixed duty cycle, so it cannot hold its output across an
input range -- the envelope analysis reports exactly that. But the *topology*
may still be able to: if the duty each corner needs lies inside the usable band,
a controller that schedules duty against the input could hold the rail, and the
requirement is not asking for something impossible.

Those are two different claims and they are reported separately:

* ``regulates_output`` -- what the fixed-duty build actually achieves;
* ``duty_schedule.all_reachable`` -- whether the conversion is possible
  everywhere, plus the control authority and headroom it would take.

Neither is a stability claim. Whether a loop can be compensated needs an
analysis this project does not have, and the record says so in a field rather
than leaving it to the reader's optimism.
"""

from __future__ import annotations

import json

import pytest

from circuit_ai.operating_envelope import (
    DutySchedule,
    OperatingCorner,
    OperatingEnvelope,
    enumerate_corners,
    worst_case_summary,
)


def isolated_power_spec(
    *,
    name: str,
    vin: float,
    vout: float,
    iout: float,
    envelope: dict | None = None,
) -> dict:
    """An isolated DC requirement with its own output reference."""

    spec = {
        "name": name,
        "ports": [
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
        ],
        "relations": [
            {"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}
        ],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [
            {
                "target_kind": "dc",
                "input_voltage_v": vin,
                "output_voltage_v": vout,
                "output_current_a": iout,
            }
        ],
        "constraints": {
            "element_types": [
                "R", "C", "L", "ideal_switch", "ideal_transformer", "ideal_diode"
            ],
            "max_component_count": 5,
            "parameter_ranges": {
                "R": [100.0, 1_000_000.0],
                "C": [1e-12, 1e-4],
                "L": [1e-9, 1.0],
                "turns_ratio": [0.05, 2.0],
            },
        },
    }
    if envelope is not None:
        spec["operating_envelope"] = envelope
    return spec


def _schedule(**overrides) -> DutySchedule:
    entries = tuple(
        {
            "corner_id": f"corner_{index}",
            "input_voltage_v": vin,
            "load_fraction": 1.0,
            "required_duty": duty,
            "inside_usable_band": 0.05 <= duty <= 0.95,
            "headroom_to_upper": 0.95 - duty,
            "headroom_to_lower": duty - 0.05,
        }
        for index, (vin, duty) in enumerate(overrides.pop("pairs", ((24.0, 0.81), (36.0, 0.74))))
    )
    return DutySchedule(
        entries=entries,
        usable_band=overrides.pop("usable_band", (0.05, 0.95)),
        control_effort=(
            min(e["required_duty"] for e in entries),
            max(e["required_duty"] for e in entries),
        ),
        **overrides,
    )


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


def test_a_schedule_inside_the_band_is_reachable() -> None:
    schedule = _schedule()
    assert schedule.all_reachable
    assert schedule.spread == pytest.approx(0.07)
    # Duties are 0.81 and 0.74, so the *high* duty corner is the binding one:
    # it has 0.14 to the ceiling against 0.69 to the floor.
    assert schedule.limiting_corner == "corner_0"
    assert schedule.as_dict()["limiting_corner"] == "corner_0"


def test_a_schedule_outside_the_band_is_not_reachable() -> None:
    schedule = _schedule(pairs=((24.0, 0.81), (12.0, 0.99)))
    assert not schedule.all_reachable


def test_an_empty_schedule_makes_no_claim() -> None:
    """No corners means nothing was checked, which is not the same as reachable."""

    schedule = DutySchedule(entries=(), usable_band=(0.05, 0.95), control_effort=(0.0, 0.0))
    assert not schedule.all_reachable
    assert schedule.limiting_corner is None


def test_the_record_refuses_to_imply_a_stability_claim() -> None:
    """Feasibility is not stability, and the field has to be changed to say so."""

    schedule = _schedule()
    assert schedule.justifies_stability_claim is False
    payload = schedule.as_dict()
    assert payload["justifies_stability_claim"] is False
    json.dumps(payload)


def test_limiting_corner_is_the_one_with_least_headroom() -> None:
    # corner_0 needs 0.90 (0.05 to the ceiling); corner_1 needs 0.10 (0.05 to the
    # floor).  A tie is broken deterministically by the minimum.
    schedule = _schedule(pairs=((48.0, 0.90), (12.0, 0.10)))
    assert schedule.limiting_corner in {"corner_0", "corner_1"}
    assert schedule.spread == pytest.approx(0.80)


def test_summary_reports_the_schedule_only_when_one_was_computed() -> None:
    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0, min_input_voltage_v=24.0)

    def evaluate(corner: OperatingCorner) -> dict[str, float]:
        return {
            "efficiency": 0.9,
            "loss_w": 1.0,
            "input_current_a": 1.0,
            "duty": 0.5,
            "predicted_ripple_mv": 1.0,
            "output_voltage_v": 5.0,
        }

    without = worst_case_summary(enumerate_corners(envelope), evaluate)
    assert without.duty_schedule is None
    assert without.as_dict()["duty_schedule"] is None

    with_schedule = worst_case_summary(
        enumerate_corners(envelope), evaluate, duty_schedule=_schedule()
    )
    assert with_schedule.duty_schedule is not None
    assert with_schedule.as_dict()["duty_schedule"]["all_reachable"] is True


# ---------------------------------------------------------------------------
# End to end on a real isolated requirement
# ---------------------------------------------------------------------------


def _design(**optimization_overrides):
    from circuit_ai.pipeline import design_from_pbdl

    spec = isolated_power_spec(
        name="duty_schedule_probe",
        vin=36.0,
        vout=5.0,
        iout=2.0,
        envelope={
            "min_input_voltage_v": 24.0,
            "max_input_voltage_v": 48.0,
            "min_load_fraction": 0.1,
            "max_load_fraction": 1.0,
        },
    )
    spec["optimization"] = {
        "max_iterations": 6,
        "seed": 5,
        "loss_model": {"enabled": True},
        **optimization_overrides,
    }
    return design_from_pbdl(spec)


def test_fixed_duty_cannot_regulate_but_the_topology_is_reachable() -> None:
    """The distinction this module exists to make."""

    result = _design()
    summary = result.optimization.envelope_analysis
    assert summary is not None

    # What the build achieves: one fixed duty, so the rail moves with the input.
    assert summary.regulates_output is False
    assert summary.output_voltage_spread_v > 0.0

    # What the topology could do: every corner is inside the usable duty band.
    schedule = summary.duty_schedule
    assert schedule is not None
    assert schedule.all_reachable
    assert schedule.usable_band == (0.05, 0.95)
    assert 0.0 < schedule.spread < 1.0
    assert schedule.limiting_corner is not None


def test_required_duty_is_highest_at_the_lowest_rail() -> None:
    """Physics check: a step-down conversion needs more duty as input falls."""

    summary = _design().optimization.envelope_analysis
    assert summary is not None and summary.duty_schedule is not None
    by_rail = {
        dict(entry)["input_voltage_v"]: float(dict(entry)["required_duty"])
        for entry in summary.duty_schedule.entries
    }
    assert by_rail[24.0] > by_rail[36.0] > by_rail[48.0]


def test_the_schedule_reaches_the_report_artifact(tmp_path) -> None:
    from circuit_ai.pipeline import design_from_pbdl

    spec = isolated_power_spec(
        name="duty_schedule_artifact",
        vin=36.0,
        vout=5.0,
        iout=2.0,
        envelope={"min_input_voltage_v": 24.0, "max_input_voltage_v": 48.0},
    )
    spec["optimization"] = {"max_iterations": 4, "seed": 5, "loss_model": {"enabled": True}}
    design_from_pbdl(spec, output_dir=tmp_path)

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    schedule = report["worst_case"]["duty_schedule"]
    assert schedule is not None
    assert schedule["all_reachable"] is True
    assert schedule["justifies_stability_claim"] is False
    assert schedule["limiting_corner"] is not None
    json.dumps(report)


def test_no_envelope_means_no_schedule() -> None:
    """Nothing is claimed when nothing was checked."""

    from circuit_ai.pipeline import design_from_pbdl

    spec = isolated_power_spec(name="no_envelope", vin=36.0, vout=5.0, iout=2.0)
    spec["optimization"] = {"max_iterations": 4, "seed": 5}
    result = design_from_pbdl(spec)
    assert result.optimization.envelope_analysis is None
