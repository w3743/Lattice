"""Frequency-band acceptance masks for AC designs.

The DC power path has an operating envelope and a worst-case reduction; the AC
path only had ``rmse_db``/``max_abs_db`` against an analytic target, which
answers "how close is the shape on average" but never "does the passband hold
its ripple and does the stopband reach its attenuation". A design can track a
target closely on average and still break its stopband floor, so these tests
pin the band-level verdict and its separation from the shape score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from circuit_ai.frequency_mask import (
    BandSpec,
    FilterMask,
    build_mask,
    evaluate_filter_mask,
    mask_from_mapping,
)
from circuit_ai.spec import SynthesisSpec
from circuit_ai.synthesis import CircuitSynthesizer

BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


def _first_order_lowpass(frequencies: np.ndarray, cutoff_hz: float, order: int = 1):
    return np.abs(1.0 / (1.0 + 1j * frequencies / cutoff_hz) ** order)


def _mask() -> FilterMask:
    """Two bands with a 40x transition.

    Boundaries chosen so the arithmetic is checkable by hand: a first-order
    corner at 1 kHz sits at -0.969 dB at the 500 Hz passband edge (inside
    +-1 dB) and -26.03 dB at the 20 kHz stopband edge (just inside -26 dB).
    """

    return build_mask(
        "lp_1k",
        [
            {"band_id": "pass", "kind": "passband", "upper_hz": 500.0, "min_db": -1.0, "max_db": 1.0},
            {"band_id": "stop", "kind": "stopband", "lower_hz": 20000.0, "upper_hz": 1e6, "max_db": -26.0},
        ],
    )


# ---------------------------------------------------------------------------
# Mask structure
# ---------------------------------------------------------------------------


def test_transitions_are_derived_not_restated() -> None:
    mask = _mask()
    assert [band.band_id for band in mask.bands] == ["pass", "stop"]
    regions = mask.transitions
    assert len(regions) == 1
    assert regions[0].from_band == "pass"
    assert regions[0].to_band == "stop"
    assert regions[0].width_ratio == pytest.approx(40.0)


def test_bands_are_sorted_and_must_not_overlap() -> None:
    unsorted = build_mask(
        "m",
        [
            {"band_id": "stop", "kind": "stopband", "lower_hz": 2000.0, "upper_hz": 1e5, "max_db": -40.0},
            {"band_id": "pass", "kind": "passband", "upper_hz": 1000.0, "min_db": -1.0},
        ],
    )
    assert [band.band_id for band in unsorted.bands] == ["pass", "stop"]

    with pytest.raises(ValueError, match="overlap"):
        build_mask(
            "m",
            [
                {"band_id": "a", "kind": "passband", "upper_hz": 2000.0, "min_db": -1.0},
                {"band_id": "b", "kind": "stopband", "lower_hz": 1000.0, "upper_hz": 1e5, "max_db": -40.0},
            ],
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"band_id": "", "kind": "passband", "upper_hz": 100.0, "min_db": -1.0},
        {"band_id": "b", "kind": "not-a-kind", "upper_hz": 100.0, "min_db": -1.0},
        {"band_id": "b", "kind": "passband", "upper_hz": 100.0},
        {"band_id": "b", "kind": "passband", "upper_hz": 0.0, "min_db": -1.0},
        {"band_id": "b", "kind": "passband", "lower_hz": 100.0, "upper_hz": 100.0, "min_db": -1.0},
        {"band_id": "b", "kind": "passband", "upper_hz": 100.0, "min_db": 1.0, "max_db": -1.0},
    ],
)
def test_band_specs_reject_nonsense(spec) -> None:
    with pytest.raises(ValueError):
        BandSpec(**spec)


def test_mask_rejects_duplicate_band_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_mask(
            "m",
            [
                {"band_id": "same", "kind": "passband", "upper_hz": 100.0, "min_db": -1.0},
                {"band_id": "same", "kind": "stopband", "lower_hz": 200.0, "upper_hz": 1e4, "max_db": -40.0},
            ],
        )


def test_band_lookup_covers_only_declared_bands() -> None:
    mask = _mask()
    assert mask.band_for(100.0).band_id == "pass"
    assert mask.band_for(50000.0).band_id == "stop"
    assert mask.band_for(5000.0) is None, "the transition region requires nothing"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_a_compliant_design_passes_every_band() -> None:
    frequencies = np.logspace(1, 6, 400)
    # Corner at 1 kHz: about -0.97 dB at 500 Hz and -26.03 dB at 20 kHz, both
    # just inside their limits, so the margins are small but positive.  The
    # analysis grid does not land exactly on the band edges, so the assertions
    # are on sign and attribution rather than on a brittle third decimal.
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 1000.0))
    report = evaluate_filter_mask(_mask(), frequencies, gains)
    assert report.passed
    assert report.failed_bands == ()
    assert report.worst_margin_db is not None and report.worst_margin_db >= 0.0
    margins = {band.band_id: band.margin_db for band in report.bands}
    assert 0.0 < margins["pass"] < 0.5, "tight but compliant"
    assert 0.0 < margins["stop"] < 0.5, "tight but compliant"


def test_a_design_that_misses_its_stopband_is_rejected() -> None:
    """The case an average-error metric cannot see."""

    frequencies = np.logspace(1, 6, 400)
    # A 10 kHz corner is about -6 dB at 20 kHz: the passband still holds, the
    # stopband floor does not.
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 10000.0))
    report = evaluate_filter_mask(_mask(), frequencies, gains)
    assert not report.passed
    assert report.failed_bands == ("stop",)
    assert report.worst_band == "stop"
    assert report.worst_margin_db is not None and report.worst_margin_db < -10.0

    stop = next(band for band in report.bands if band.band_id == "stop")
    assert stop.violated_limit == "max_db"
    # The violation is real: the worst sample sits above the -26 dB ceiling.
    assert stop.worst_gain_db is not None and stop.worst_gain_db > -26.0
    # And the passband is untouched, so exactly one band failed.
    passed = next(band for band in report.bands if band.band_id == "pass")
    assert passed.passed


def test_a_design_that_misses_its_passband_floor_is_rejected() -> None:
    """The other direction: too much droop inside the passband."""

    frequencies = np.logspace(1, 6, 400)
    # A 100 Hz corner is -14.15 dB at the 500 Hz passband edge, while the
    # stopband floor is comfortably met.
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 100.0))
    report = evaluate_filter_mask(_mask(), frequencies, gains)
    assert not report.passed
    assert report.failed_bands == ("pass",)
    passed = next(band for band in report.bands if band.band_id == "stop")
    assert passed.passed, "the stopband is fine; the passband is what broke"
    failing = next(band for band in report.bands if band.band_id == "pass")
    assert failing.violated_limit == "min_db"
    # The droop is worst at the top of the passband, which is where a low-pass
    # corner bites first.
    assert failing.worst_frequency_hz == pytest.approx(500.0, rel=0.05)
    assert failing.worst_gain_db is not None and failing.worst_gain_db < -1.0


def test_margin_is_signed_and_attributed_to_the_deciding_frequency() -> None:
    frequencies = np.logspace(1, 6, 400)
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 1000.0))
    report = evaluate_filter_mask(_mask(), frequencies, gains)
    for band in report.bands:
        assert band.margin_db is not None
        assert band.sample_count > 0
        assert band.passed == (band.margin_db >= 0.0)
        assert band.worst_frequency_hz is not None
        assert band.min_observed_db <= band.worst_gain_db <= band.max_observed_db


def test_an_empty_band_does_not_pass_by_default() -> None:
    """Absence of evidence is not evidence of compliance."""

    frequencies = np.logspace(1, 3, 100)
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 100.0))

    strict = build_mask(
        "m",
        [
            {
                "band_id": "far",
                "kind": "stopband",
                "lower_hz": 1e9,
                "upper_hz": 2e9,
                "max_db": -60.0,
                "allow_empty": False,
            }
        ],
    )
    report = evaluate_filter_mask(strict, frequencies, gains)
    assert not report.passed
    assert "no analysed sample" in report.bands[0].note
    assert report.bands[0].margin_db is None

    lenient = build_mask(
        "m",
        [{"band_id": "far", "kind": "stopband", "lower_hz": 1e9, "upper_hz": 2e9, "max_db": -60.0}],
    )
    assert evaluate_filter_mask(lenient, frequencies, gains).passed


def test_evaluation_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="match"):
        evaluate_filter_mask(_mask(), np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_filter_mask(_mask(), np.array([]), np.array([]))


def test_report_serialises() -> None:
    frequencies = np.logspace(1, 6, 200)
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 10000.0))
    payload = evaluate_filter_mask(_mask(), frequencies, gains).as_dict()
    assert payload["schema"] == "circuit_ai.filter_mask"
    assert payload["passed"] is False
    assert payload["failed_bands"] == ["stop"]
    assert payload["transitions"][0]["from_band"] == "pass"
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Spec parsing and wiring
# ---------------------------------------------------------------------------


def test_spec_without_a_mask_declares_none() -> None:
    assert mask_from_mapping({}) is None
    assert mask_from_mapping({"kind": "lowpass"}) is None


def test_spec_mask_is_parsed_from_the_behavior_block() -> None:
    mask = mask_from_mapping(
        {"filter_mask": {"mask_id": "m", "bands": [{"band_id": "p", "kind": "passband", "upper_hz": 100.0, "min_db": -1.0}]}}
    )
    assert mask is not None and mask.mask_id == "m"


def test_spec_mask_rejects_an_empty_band_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mask_from_mapping({"filter_mask": {"mask_id": "m", "bands": []}})


def _masked_spec() -> dict:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    return {
        "name": "masked_probe",
        "ports": 2,
        "behavior": {
            "kind": "lowpass",
            "cutoff_hz": 1000,
            "gain": 1.0,
            "order": 1,
            "frequency_range_hz": [10, 100000],
            "filter_mask": {
                "mask_id": "lp_1k",
                "bands": [
                    {"band_id": "pass", "kind": "passband", "upper_hz": 500.0, "min_db": -1.0, "max_db": 1.0},
                    {"band_id": "stop", "kind": "stopband", "lower_hz": 30000.0, "upper_hz": 100000.0, "max_db": -20.0},
                ],
            },
        },
        "library": {"allowed": ["R", "C"], "parameter_ranges": {"R": [100, 1e6], "C": [1e-10, 1e-4]}},
        "optimization": {"points": 40, "max_iterations": 4, "top_k": 1, "seed": 5},
        "relation": data.get("relation"),
    }


def test_synthesis_reports_no_mask_when_none_is_declared() -> None:
    """A spec that states only an analytic target claims no band verdict."""

    spec = SynthesisSpec.from_dict(
        {
            "name": "unmasked",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000, "order": 1, "frequency_range_hz": [10, 100000]},
            "library": {"allowed": ["R", "C"], "parameter_ranges": {"R": [100, 1e6], "C": [1e-10, 1e-4]}},
            "optimization": {"points": 24, "max_iterations": 2, "top_k": 1, "seed": 3},
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]
    assert result.filter_mask_report is None
    assert result.as_dict()["filter_mask"] is None


def test_synthesis_evaluates_the_declared_mask_on_the_measured_response() -> None:
    payload = _masked_spec()
    payload.pop("relation", None)
    spec = SynthesisSpec.from_dict(payload)
    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.filter_mask_report is not None
    report = result.filter_mask_report
    assert {band.band_id for band in report.bands} == {"pass", "stop"}
    # Every band must have been evaluated against real samples.
    for band in report.bands:
        assert band.sample_count > 0
        assert band.worst_frequency_hz is not None

    entry = result.as_dict()["filter_mask"]
    assert entry is not None
    assert entry["passed"] == report.passed
    json.dumps(entry)


def test_band_verdict_is_reported_separately_from_the_score() -> None:
    """The verdict is a pass/fail fact; the score is a number to minimise.

    These must not be conflated: the report has to name *which* band decided it,
    at which frequency, against which limit.
    """

    frequencies = np.logspace(1, 6, 400)
    gains = 20 * np.log10(_first_order_lowpass(frequencies, 10000.0))
    report = evaluate_filter_mask(_mask(), frequencies, gains)
    assert not report.passed
    failing = next(band for band in report.bands if not band.passed)
    assert failing.band_id == report.worst_band
    assert failing.worst_frequency_hz is not None
    assert failing.violated_limit in {"min_db", "max_db"}


def test_an_unsatisfiable_band_floor_is_charged_into_the_score() -> None:
    """The mask guides the search; it does not only judge the result."""

    payload = _masked_spec()
    payload.pop("relation", None)
    # No first-order R/C reaches -80 dB at 30 kHz, so every candidate must carry
    # a band penalty and the stopband must stay failed.
    payload["behavior"]["filter_mask"]["bands"][1]["max_db"] = -80.0
    spec = SynthesisSpec.from_dict(payload)
    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.filter_mask_report is not None
    assert not result.filter_mask_report.passed
    assert "stop" in result.filter_mask_report.failed_bands
    assert result.metrics.band_mask_penalty > 0.0

    # The charge is the sum of measured shortfalls, so it is attributable rather
    # than a flat constant.  With an impossible stopband floor the optimizer
    # trades the passband away chasing it, so more than one band may be short --
    # which is exactly the pressure the mask is supposed to exert.
    shortfall = sum(
        -float(band.margin_db)
        for band in result.filter_mask_report.bands
        if band.margin_db is not None and band.margin_db < 0.0
    )
    assert shortfall > 0.0
    assert result.metrics.band_mask_penalty == pytest.approx(shortfall, rel=1e-6)


def test_a_satisfiable_mask_carries_no_penalty() -> None:
    """Guards against the charge being unconditional."""

    payload = _masked_spec()
    payload.pop("relation", None)
    # -6 dB at 30 kHz is easily reached by a first-order corner near 1 kHz.
    payload["behavior"]["filter_mask"]["bands"][1]["max_db"] = -6.0
    spec = SynthesisSpec.from_dict(payload)
    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.metrics.band_mask_penalty == 0.0
    assert result.filter_mask_report is not None
    assert result.filter_mask_report.passed


def test_an_unmasked_design_is_charged_nothing() -> None:
    """No mask means no penalty, so existing metrics are untouched."""

    spec = SynthesisSpec.from_dict(
        {
            "name": "unmasked_penalty",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000, "order": 1, "frequency_range_hz": [10, 100000]},
            "library": {"allowed": ["R", "C"], "parameter_ranges": {"R": [100, 1e6], "C": [1e-10, 1e-4]}},
            "optimization": {"points": 24, "max_iterations": 2, "top_k": 1, "seed": 3},
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]
    assert result.metrics.band_mask_penalty == 0.0
    assert result.filter_mask_report is None
