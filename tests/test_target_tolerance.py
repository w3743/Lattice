"""A stated tolerance must be compiled into a hard acceptance limit.

The frequency-target compiler used to emit only a soft MINIMIZE, so *any* design
satisfied the acceptance gate: a 4th-order Butterworth request served by a
first-order R/C was reported ``verified_feasible`` at 32.9 dB RMSE against a
declared 0.5 dB tolerance. A tolerance that is never evaluated is not a limit.

These tests pin the compiled limit, its conversion from a relative tolerance, and
the end-to-end consequence that an over-budget design stops being accepted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from circuit_ai.constraints.compiler import (
    _positive_finite,
    _target_tolerance_limit,
    compile_constraint_program,
)
from circuit_ai.ir import pbdl_to_ir

TEMPLATE = Path("端口描述语言/examples/lowpass_1khz.json")


def _spec_with_tolerance(**tolerance) -> dict:
    import json

    data = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    data["targets"][0]["tolerance"] = tolerance
    return data


# ---------------------------------------------------------------------------
# The compiled limit
# ---------------------------------------------------------------------------


def test_absolute_tolerance_becomes_the_limit() -> None:
    limit = _target_tolerance_limit({"tolerance": {"absolute": 0.5}}, (0.0, -3.0))
    assert limit is not None
    assert limit.value == pytest.approx(0.5)
    assert "absolute" in limit.basis


def test_relative_tolerance_is_converted_to_db() -> None:
    limit = _target_tolerance_limit({"tolerance": {"relative": 0.05}}, (0.0, -3.0))
    assert limit is not None
    # A symmetric +-5 % band is 20*log10(1.05) dB, independent of filter gain.
    assert limit.value == pytest.approx(20.0 * 0.0211893, rel=1e-4)
    assert "relative" in limit.basis


def test_relative_wins_when_both_are_stated() -> None:
    """A relative band scales with the requirement; an absolute one does not."""

    limit = _target_tolerance_limit(
        {"tolerance": {"absolute": 99.0, "relative": 0.1}}, (0.0,)
    )
    assert limit is not None
    assert limit.value == pytest.approx(20.0 * 0.0413927, rel=1e-4)


@pytest.mark.parametrize(
    "tolerance",
    [
        {},
        {"absolute": None, "relative": None},
        {"absolute": 0.0},
        {"absolute": -1.0},
        {"relative": 0.0},
        {"relative": 1.5},
        {"absolute": True},
        {"absolute": "0.5"},
    ],
)
def test_absent_or_unusable_tolerances_produce_no_limit(tolerance) -> None:
    assert _target_tolerance_limit({"tolerance": tolerance}, (0.0, -3.0)) is None


def test_no_tolerance_block_produces_no_limit() -> None:
    assert _target_tolerance_limit({}, (0.0,)) is None
    assert _target_tolerance_limit({"tolerance": None}, (0.0,)) is None


def test_a_relative_tolerance_needs_a_target_trace() -> None:
    """Without the target there is nothing to be relative to."""

    assert _target_tolerance_limit({"tolerance": {"relative": 0.05}}, ()) is None


def test_positive_finite_rejects_booleans_and_non_finite() -> None:
    assert _positive_finite(True) is None
    assert _positive_finite(False) is None
    assert _positive_finite(float("inf")) is None
    assert _positive_finite(float("nan")) is None
    assert _positive_finite(0.0) is None
    assert _positive_finite(2) == 2.0


# ---------------------------------------------------------------------------
# The compiled programme
# ---------------------------------------------------------------------------


def test_a_declared_tolerance_reaches_the_compiled_programme() -> None:
    programme = compile_constraint_program(pbdl_to_ir(_spec_with_tolerance(absolute=0.5)))
    hard = [
        item
        for item in programme.constraints
        if "within_tolerance" in item.constraint_id
    ]
    assert hard, [item.constraint_id for item in programme.constraints]
    limit = hard[0]
    assert limit.maximum == pytest.approx(0.5)
    assert limit.severity.value == "hard"
    # It must share the metric the soft objective uses, so the two cannot drift:
    # the same measurement has to serve both the objective and the limit.
    soft = next(
        item for item in programme.constraints if item.constraint_id.endswith("minimize_rmse_db")
    )
    assert limit.metric.metric_id == soft.metric.metric_id


def test_no_tolerance_means_no_hard_limit() -> None:
    programme = compile_constraint_program(pbdl_to_ir(_spec_with_tolerance()))
    assert not [
        item for item in programme.constraints if "within_tolerance" in item.constraint_id
    ]
    # The soft objective is still there, so nothing else changed.
    assert [
        item for item in programme.constraints if item.constraint_id.endswith("minimize_rmse_db")
    ]


def test_relative_tolerance_reaches_the_programme_as_db() -> None:
    programme = compile_constraint_program(pbdl_to_ir(_spec_with_tolerance(relative=0.05)))
    limit = next(
        item for item in programme.constraints if "within_tolerance" in item.constraint_id
    )
    assert limit.maximum == pytest.approx(0.4238, abs=1e-3)


# ---------------------------------------------------------------------------
# End to end: an over-budget design stops being accepted
# ---------------------------------------------------------------------------


def _run(order: int):
    import json

    from circuit_ai.pbdl_runner import run_pbdl_file

    data = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    data["name"] = f"tolerance_order{order}"
    data["targets"][0]["order"] = order
    data["targets"][0]["tolerance"] = {"relative": None, "absolute": 0.5}
    root = Path("outputs/_r9_tolerance_test") / f"order{order}"
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return run_pbdl_file(spec_path, root / "run", top_k=1)


def test_a_met_tolerance_still_succeeds() -> None:
    """Order 1 is reachable with R/C, so the limit must not break a good design."""

    report = _run(1)
    assert report.succeeded


def test_an_unreachable_order_is_no_longer_accepted() -> None:
    """Order 4 needs a 4th-order response; a first-order R/C is 32 dB out.

    Before the tolerance was compiled this reported success.
    """

    report = _run(4)
    assert not report.succeeded, "a 32 dB error must not pass a 0.5 dB tolerance"
    stage = report.stages[0]
    message = str(stage.message or "")
    assert message, "a refusal has to say why"
