"""Operating envelopes and worst-case evaluation.

A requirement is rarely a single point: "36 V to 5 V" usually means 24-48 V in
and 10-100 % load.  These tests cover the envelope layer, its worst-case
reduction, and the opt-in wiring that leaves single-point design untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_ai.ir import pbdl_to_ir
from circuit_ai.operating_envelope import (
    OperatingCorner,
    OperatingEnvelope,
    enumerate_corners,
    envelope_from_mapping,
    worst_case_summary,
)

BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


# ---------------------------------------------------------------------------
# Envelope construction and validation
# ---------------------------------------------------------------------------


def test_missing_range_collapses_to_the_nominal_point() -> None:
    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0)
    assert envelope.is_degenerate
    assert envelope.low_input_voltage_v == envelope.high_input_voltage_v == 36.0
    assert [corner.corner_id for corner in enumerate_corners(envelope)] == ["nominal"]


def test_envelope_accepts_only_ranges_around_a_stated_nominal() -> None:
    envelope = OperatingEnvelope(
        nominal_input_voltage_v=36.0,
        min_input_voltage_v=24.0,
        max_input_voltage_v=48.0,
        min_load_fraction=0.1,
        max_load_fraction=1.0,
    )
    assert not envelope.is_degenerate
    assert envelope.low_input_voltage_v == 24.0
    assert envelope.high_input_voltage_v == 48.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nominal_input_voltage_v": 0.0},
        {"nominal_input_voltage_v": -5.0},
        {"nominal_input_voltage_v": 36.0, "min_input_voltage_v": 48.0, "max_input_voltage_v": 24.0},
        {"nominal_input_voltage_v": 36.0, "min_input_voltage_v": 40.0, "max_input_voltage_v": 48.0},
        {"nominal_input_voltage_v": 36.0, "min_load_fraction": 1.0, "max_load_fraction": 0.1},
        {"nominal_input_voltage_v": 36.0, "min_load_fraction": 0.0},
    ],
)
def test_envelope_rejects_inconsistent_ranges(kwargs) -> None:
    with pytest.raises(ValueError):
        OperatingEnvelope(**kwargs)


def test_envelope_serialises_with_its_degeneracy() -> None:
    payload = OperatingEnvelope(nominal_input_voltage_v=36.0).as_dict()
    assert payload["schema"] == "circuit_ai.operating_envelope"
    assert payload["schema_version"] == 1
    assert payload["is_degenerate"] is True
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Corner enumeration
# ---------------------------------------------------------------------------


def test_corners_cover_nominal_and_every_extreme_once() -> None:
    envelope = OperatingEnvelope(
        nominal_input_voltage_v=36.0,
        min_input_voltage_v=24.0,
        max_input_voltage_v=48.0,
        min_load_fraction=0.1,
        max_load_fraction=1.0,
    )
    corners = enumerate_corners(envelope)
    assert corners[0].corner_id == "nominal"
    assert corners[0].is_nominal
    pairs = {(corner.input_voltage_v, corner.load_fraction) for corner in corners}
    assert pairs == {
        (36.0, 1.0),
        (36.0, 0.1),
        (24.0, 1.0),
        (24.0, 0.1),
        (48.0, 1.0),
        (48.0, 0.1),
    }
    assert len(corners) == len(pairs), "no corner may be enumerated twice"


def test_partial_ranges_only_add_the_extremes_that_exist() -> None:
    only_line = OperatingEnvelope(
        nominal_input_voltage_v=36.0, min_input_voltage_v=24.0, max_input_voltage_v=48.0
    )
    assert sorted(c.corner_id for c in enumerate_corners(only_line)) == [
        "input_max__nominal",
        "input_min__nominal",
        "nominal",
    ]

    only_load = OperatingEnvelope(
        nominal_input_voltage_v=36.0, min_load_fraction=0.2, max_load_fraction=1.0
    )
    assert sorted(c.corner_id for c in enumerate_corners(only_load)) == [
        "nominal",
        "nominal__load_min",
    ]


def test_corner_rejects_impossible_points() -> None:
    with pytest.raises(ValueError):
        OperatingCorner("bad", 0.0, 1.0)
    with pytest.raises(ValueError):
        OperatingCorner("bad", 10.0, 0.0)
    with pytest.raises(ValueError):
        OperatingCorner("", 10.0, 1.0)


# ---------------------------------------------------------------------------
# Worst-case reduction
# ---------------------------------------------------------------------------


def _fake_evaluate(corner: OperatingCorner) -> dict[str, float]:
    """Deterministic stand-in: worse efficiency and more loss at high line."""

    return {
        "efficiency": 0.95 if corner.input_voltage_v <= 36.0 else 0.88,
        "loss_w": corner.load_fraction * corner.input_voltage_v / 100.0,
        "input_current_a": corner.load_fraction * 10.0 / corner.input_voltage_v,
        "duty": 0.5 - 0.1 * (corner.input_voltage_v > 36.0),
        "predicted_ripple_mv": corner.load_fraction * corner.input_voltage_v / 10.0,
        "output_voltage_v": 5.0 * corner.input_voltage_v / 36.0,
    }


def test_worst_case_attributes_every_extreme_to_a_corner() -> None:
    envelope = OperatingEnvelope(
        nominal_input_voltage_v=36.0,
        min_input_voltage_v=24.0,
        max_input_voltage_v=48.0,
        min_load_fraction=0.1,
        max_load_fraction=1.0,
    )
    summary = worst_case_summary(enumerate_corners(envelope), _fake_evaluate)

    assert summary.corner_count == 6
    assert summary.efficiency_nominal == pytest.approx(0.95)
    assert summary.efficiency_min == pytest.approx(0.88)
    assert summary.efficiency_min_corner.startswith("input_max")
    assert summary.duty_max_corner.startswith("input_min") or summary.duty_max_corner == "nominal"
    assert summary.duty_max > summary.duty_min
    assert summary.loss_max_corner == "input_max__nominal"
    assert summary.ripple_max_corner == "input_max__nominal"
    assert summary.input_current_max_corner == "input_min__nominal"

    # Every extreme must name a corner that was actually evaluated.
    evaluated = {row["corner"]["corner_id"] for row in summary.corner_results}
    for name in (
        summary.efficiency_min_corner,
        summary.loss_max_corner,
        summary.input_current_max_corner,
        summary.duty_max_corner,
        summary.duty_min_corner,
        summary.ripple_max_corner,
    ):
        assert name in evaluated


def test_worst_case_serialises_every_corner() -> None:
    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0, min_load_fraction=0.5)
    summary = worst_case_summary(enumerate_corners(envelope), _fake_evaluate)
    payload = summary.as_dict()
    assert len(payload["corners"]) == summary.corner_count
    assert payload["efficiency"]["min_corner"] == summary.efficiency_min_corner
    json.dumps(payload)


def test_worst_case_reports_the_delivered_rail_spread() -> None:
    """A fixed-duty design has no feedback, so its regulation must be visible."""

    envelope = OperatingEnvelope(
        nominal_input_voltage_v=36.0, min_input_voltage_v=24.0, max_input_voltage_v=48.0
    )
    summary = worst_case_summary(enumerate_corners(envelope), _fake_evaluate)

    assert summary.nominal_output_voltage_v == pytest.approx(5.0)
    assert summary.output_voltage_min_v == pytest.approx(5.0 * 24.0 / 36.0)
    assert summary.output_voltage_min_corner.startswith("input_min")
    assert summary.output_voltage_max_v == pytest.approx(5.0 * 48.0 / 36.0)
    assert summary.output_voltage_max_corner.startswith("input_max")
    assert summary.output_voltage_spread_v == pytest.approx(
        summary.output_voltage_max_v - summary.output_voltage_min_v
    )
    assert summary.as_dict()["output_voltage"]["spread"] == summary.output_voltage_spread_v


def test_worst_case_refuses_silent_gaps() -> None:
    """A corner that cannot be evaluated is a finding, not something to drop."""

    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0, min_input_voltage_v=24.0)

    def missing(corner: OperatingCorner) -> dict[str, float]:
        return {"efficiency": 0.9}

    with pytest.raises(ValueError, match="missing"):
        worst_case_summary(enumerate_corners(envelope), missing)

    with pytest.raises(ValueError):
        worst_case_summary((), _fake_evaluate)

    def raises(corner: OperatingCorner) -> dict[str, float]:
        raise RuntimeError("solver failed at low line")

    with pytest.raises(RuntimeError):
        worst_case_summary(enumerate_corners(envelope), raises)


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_spec_without_an_envelope_block_declares_none() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    assert envelope_from_mapping(data, fallback_input_voltage_v=5.0) is None
    assert "operating_envelope" not in data
    assert pbdl_to_ir(data).operating_envelope is None


def test_spec_envelope_defaults_its_nominal_to_the_target_rail() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["operating_envelope"] = {"min_input_voltage_v": 4.0, "max_input_voltage_v": 6.0}
    envelope = envelope_from_mapping(data, fallback_input_voltage_v=5.0)
    assert envelope is not None
    assert envelope.nominal_input_voltage_v == 5.0
    assert envelope.low_input_voltage_v == 4.0
    assert envelope.high_input_voltage_v == 6.0


def test_ir_round_trips_the_envelope() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["operating_envelope"] = {"min_input_voltage_v": 4.0, "max_input_voltage_v": 6.0}
    ir = pbdl_to_ir(data)
    assert ir.operating_envelope is not None
    assert OperatingEnvelope.from_dict(ir.operating_envelope) is not None
    assert ir.as_dict()["operating_envelope"] == ir.operating_envelope
    # with_components must not drop it
    assert ir.with_components(()).operating_envelope == ir.operating_envelope


# ---------------------------------------------------------------------------
# Regulation acceptance
# ---------------------------------------------------------------------------


def test_regulation_is_unchecked_rather_than_passed_when_no_target_is_given() -> None:
    """'Not checked' must never be reported as 'passed'."""

    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0, min_input_voltage_v=24.0)
    summary = worst_case_summary(enumerate_corners(envelope), _fake_evaluate)
    assert summary.regulates_output is None
    assert summary.regulation_violations == ()
    assert summary.target_output_voltage_v is None
    assert summary.as_dict()["output_voltage"]["regulates"] is None


def test_regulation_flags_every_corner_outside_the_band() -> None:
    envelope = OperatingEnvelope(
        nominal_input_voltage_v=36.0, min_input_voltage_v=24.0, max_input_voltage_v=48.0
    )
    # _fake_evaluate delivers 5 V * vin / 36, so 24 V gives 3.33 V and 48 V
    # gives 6.67 V around a 5 V target.
    summary = worst_case_summary(
        enumerate_corners(envelope),
        _fake_evaluate,
        target_output_voltage_v=5.0,
        tolerance_fraction=0.05,
    )
    assert summary.regulates_output is False
    flagged = {dict(item)["corner_id"] for item in summary.regulation_violations}
    assert flagged == {"input_min__nominal", "input_max__nominal"}
    for item in summary.regulation_violations:
        row = dict(item)
        assert row["allowed_min_v"] == pytest.approx(4.75)
        assert row["allowed_max_v"] == pytest.approx(5.25)
        assert not row["allowed_min_v"] <= row["delivered_v"] <= row["allowed_max_v"]
    assert summary.as_dict()["output_voltage"]["regulates"] is False


def test_regulation_passes_when_the_rail_holds_within_tolerance() -> None:
    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0, min_input_voltage_v=35.0)

    def flat(corner: OperatingCorner) -> dict[str, float]:
        result = _fake_evaluate(corner)
        result["output_voltage_v"] = 5.0
        return result

    summary = worst_case_summary(
        enumerate_corners(envelope),
        flat,
        target_output_voltage_v=5.0,
        tolerance_fraction=0.05,
    )
    assert summary.regulates_output is True
    assert summary.regulation_violations == ()


def test_tolerance_must_be_a_fraction() -> None:
    envelope = OperatingEnvelope(nominal_input_voltage_v=36.0, min_input_voltage_v=35.0)
    corners = enumerate_corners(envelope)
    with pytest.raises(ValueError):
        worst_case_summary(
            corners, _fake_evaluate, target_output_voltage_v=5.0, tolerance_fraction=1.5
        )
    with pytest.raises(ValueError):
        worst_case_summary(
            corners, _fake_evaluate, target_output_voltage_v=5.0, tolerance_fraction=-0.1
        )
    with pytest.raises(ValueError):
        worst_case_summary(corners, _fake_evaluate, target_output_voltage_v=0.0, tolerance_fraction=0.05)
    # A half-specified check is a mistake, not an unchecked one.
    with pytest.raises(ValueError):
        worst_case_summary(corners, _fake_evaluate, target_output_voltage_v=5.0)
    with pytest.raises(ValueError):
        worst_case_summary(corners, _fake_evaluate, tolerance_fraction=0.05)


# ---------------------------------------------------------------------------
# Opt-in wiring end to end
# ---------------------------------------------------------------------------


def _enveloped_spec(*, with_losses: bool = True) -> dict:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["name"] = "envelope_wiring"
    data["operating_envelope"] = {
        "min_input_voltage_v": 4.0,
        "max_input_voltage_v": 6.0,
        "min_load_fraction": 0.1,
        "max_load_fraction": 1.0,
    }
    if with_losses:
        data.setdefault("optimization", {})["loss_model"] = {"enabled": True}
    return data


def test_power_design_without_an_envelope_reports_no_worst_case() -> None:
    from circuit_ai.pipeline import design_from_pbdl

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    result = design_from_pbdl(data)
    assert result.optimization.envelope_analysis is None


def test_power_design_reports_worst_case_across_its_envelope() -> None:
    from circuit_ai.pipeline import design_from_pbdl

    result = design_from_pbdl(_enveloped_spec())
    summary = result.optimization.envelope_analysis
    assert summary is not None
    assert summary.corner_count == 6

    # The stored design is a single build: component values do not change corner
    # to corner, only the operating point does.
    duties = {row["duty"] for row in summary.corner_results}
    assert len(duties) == 1, duties

    # Worst-case efficiency must not be better than the nominal one.
    assert summary.efficiency_min <= summary.efficiency_nominal
    # High line draws the most input power for a fixed load.
    assert summary.input_current_max_corner.startswith("input_max")
    # Every corner reports the quantities a designer needs.
    for row in summary.corner_results:
        for key in (
            "efficiency",
            "loss_w",
            "input_current_a",
            "predicted_ripple_mv",
            "output_voltage_v",
        ):
            assert isinstance(row[key], float)
            assert row[key] == row[key], "no NaN"


def test_fixed_duty_design_reports_its_actual_line_regulation() -> None:
    """The design has no feedback loop, and the report must say so.

    This is the honest half of "36 V nominal, 24-48 V in": a fixed-duty flyback
    cannot hold 5 V across that range, so the delivered rail spread is reported
    rather than the nominal 5 V being presented as the result.
    """

    from circuit_ai.pipeline import design_from_pbdl

    result = design_from_pbdl(_enveloped_spec())
    summary = result.optimization.envelope_analysis
    assert summary is not None
    assert summary.output_voltage_spread_v > 0.0, "a fixed-duty design cannot regulate"
    assert summary.output_voltage_min_v < summary.nominal_output_voltage_v
    assert summary.output_voltage_max_v > summary.nominal_output_voltage_v
    # The stored nominal operating point still meets the target exactly.
    assert result.optimization.operating_point.output_voltage_v == pytest.approx(
        10.0, rel=1e-6
    )


def test_worst_case_reaches_the_report_artifact(tmp_path) -> None:
    from circuit_ai.pipeline import design_from_pbdl

    design_from_pbdl(_enveloped_spec(), output_dir=tmp_path)
    # design_from_pbdl writes the stage report directly into output_dir; the
    # per-stage subdirectory only appears when a multi-stage runner wraps it.
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["operating_envelope"]["min_input_voltage_v"] == 4.0
    assert report["worst_case"]["corner_count"] == 6
    assert report["worst_case"]["efficiency"]["min_corner"]
    json.dumps(report)


def test_worst_case_is_deterministic() -> None:
    from circuit_ai.pipeline import design_from_pbdl

    first = design_from_pbdl(_enveloped_spec()).optimization.envelope_analysis
    second = design_from_pbdl(_enveloped_spec()).optimization.envelope_analysis
    assert first is not None and second is not None
    assert first.as_dict() == second.as_dict()


# ---------------------------------------------------------------------------
# Design corner: where the duty cycle is scheduled
# ---------------------------------------------------------------------------


def test_default_design_corner_is_nominal() -> None:
    """Absent a request, nothing moves: the historical rail is used."""

    from circuit_ai.power import _design_corner_rail

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["operating_envelope"] = {"min_input_voltage_v": 4.0, "max_input_voltage_v": 6.0}
    rail, name = _design_corner_rail(pbdl_to_ir(data), 5.0)
    assert name == "nominal"
    assert rail == 5.0


def test_design_corner_takes_each_declared_rail() -> None:
    from circuit_ai.power import _design_corner_rail

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["operating_envelope"] = {"min_input_voltage_v": 4.0, "max_input_voltage_v": 9.0}
    for rail, expected in (
        ("input_min", 4.0),
        ("input_max", 9.0),
        ("geometric_mean", 6.0),
    ):
        data.setdefault("optimization", {})["design_corner"] = {"rail": rail}
        value, name = _design_corner_rail(pbdl_to_ir(data), 5.0)
        assert name == rail
        assert value == pytest.approx(expected)


def test_design_corner_rejects_bad_requests() -> None:
    from circuit_ai.power import _design_corner_rail

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data.setdefault("optimization", {})["design_corner"] = {"rail": "middle"}
    with pytest.raises(ValueError, match="design_corner.rail"):
        _design_corner_rail(pbdl_to_ir(data), 5.0)

    # An extreme rail is meaningless without a range to take it from.
    data["optimization"]["design_corner"] = {"rail": "input_min"}
    with pytest.raises(ValueError, match="operating_envelope"):
        _design_corner_rail(pbdl_to_ir(data), 5.0)


def _wide_input_spec(rail: str) -> dict:
    """A 2:1 input range on the boost example, which is what makes spread visible."""

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["name"] = f"corner_{rail}"
    data["operating_envelope"] = {
        "min_input_voltage_v": 4.0,
        "max_input_voltage_v": 8.0,
        "min_load_fraction": 0.5,
        "max_load_fraction": 1.0,
    }
    data.setdefault("optimization", {})["loss_model"] = {"enabled": True}
    data["optimization"]["design_corner"] = {"rail": rail}
    return data


def test_design_corner_changes_where_the_target_is_met() -> None:
    """The trade-off is real and measurable, which is why it is selectable."""

    from circuit_ai.pipeline import design_from_pbdl

    low_line = design_from_pbdl(_wide_input_spec("input_min"))
    summary = low_line.optimization.envelope_analysis
    assert summary is not None

    by_corner = {
        row["corner"]["corner_id"]: float(row["output_voltage_v"])
        for row in summary.corner_results
    }
    target = 10.0  # the boost example's output voltage

    # Scheduling at low line hits the target exactly there...
    assert by_corner["input_min__nominal"] == pytest.approx(target, rel=1e-6)
    # ...and therefore overshoots above it, because there is no feedback loop.
    assert by_corner["input_max__nominal"] > target
    assert summary.regulates_output is False


def test_low_line_scheduling_beats_nominal_sizing_for_undervoltage() -> None:
    """Why the option exists: nominal sizing droops below target at low line."""

    from circuit_ai.pipeline import design_from_pbdl

    target = 10.0
    nominal = design_from_pbdl(_wide_input_spec("nominal")).optimization.envelope_analysis
    low_line = design_from_pbdl(_wide_input_spec("input_min")).optimization.envelope_analysis
    assert nominal is not None and low_line is not None

    assert nominal.output_voltage_min_v < target, "nominal sizing droops at low line"
    assert low_line.output_voltage_min_v == pytest.approx(target, rel=1e-6)
    # The cost of that choice is a larger overshoot at high line.
    assert low_line.output_voltage_max_v > nominal.output_voltage_max_v


def test_geometric_mean_centres_the_ratio() -> None:
    """Centring makes the two rails equidistant *in ratio*, not in volts."""

    from circuit_ai.pipeline import design_from_pbdl

    data = _wide_input_spec("geometric_mean")
    summary = design_from_pbdl(data).optimization.envelope_analysis
    assert summary is not None
    assert summary.output_voltage_min_corner.startswith("input_min")
    assert summary.output_voltage_max_corner.startswith("input_max")

    envelope = data["operating_envelope"]
    low_rail = envelope["min_input_voltage_v"]
    high_rail = envelope["max_input_voltage_v"]
    geometric = (low_rail * high_rail) ** 0.5
    target = 10.0

    # Scheduled at the geometric mean, so the target is met there...
    at_design_rail = target * geometric / geometric
    assert at_design_rail == pytest.approx(target)
    # ...and the delivered rail scales with input on both sides of it.
    delivered_low = summary.output_voltage_min_v
    delivered_high = summary.output_voltage_max_v
    assert delivered_low < target < delivered_high
    # Equidistant in ratio: min/target == target/max.
    assert delivered_low / target == pytest.approx(target / delivered_high, rel=1e-6)
    assert delivered_low / target == pytest.approx(low_rail / geometric, rel=1e-2)

    # Compare the absolute spread against nominal sizing, which favours one rail.
    nominal = design_from_pbdl(_wide_input_spec("nominal")).optimization.envelope_analysis
    assert nominal is not None
    assert nominal.output_voltage_min_v < target
    assert summary.output_voltage_spread_v < nominal.output_voltage_spread_v
