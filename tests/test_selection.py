from __future__ import annotations

import inspect

from circuit_ai.constraints import ConstraintReport, FeasibilityStatus
from circuit_ai.selection import rank_candidates
import circuit_ai.pipeline as pipeline_module


def _report(status: FeasibilityStatus) -> ConstraintReport:
    return ConstraintReport((), status, {})


def test_feasibility_precedes_any_objective_advantage() -> None:
    reports = (
        _report(FeasibilityStatus.KNOWN_INFEASIBLE),
        _report(FeasibilityStatus.VERIFIED_FEASIBLE),
        _report(FeasibilityStatus.INDETERMINATE),
    )
    decisions = rank_candidates(
        reports,
        ({"cost": 0.0}, {"cost": 1_000_000.0}, {"cost": -1_000_000.0}),
        (0.0, 1_000_000.0, -1_000_000.0),
    )
    ordered = sorted(decisions, key=lambda item: item.sort_key)
    assert [item.index for item in ordered] == [1, 0, 2]


def test_generalized_pareto_keeps_tradeoffs_and_dominates_strictly_worse_point() -> None:
    reports = tuple(_report(FeasibilityStatus.VERIFIED_FEASIBLE) for _ in range(3))
    decisions = rank_candidates(
        reports,
        (
            {"cost": 1.0, "area": 10.0},
            {"cost": 10.0, "area": 1.0},
            {"cost": 11.0, "area": 11.0},
        ),
        (0.0, 0.0, 0.0),
    )
    assert decisions[0].pareto_rank == 0
    assert decisions[1].pareto_rank == 0
    assert decisions[2].pareto_rank == 1


def test_candidate_ranking_is_deterministic_for_identical_inputs() -> None:
    reports = tuple(_report(FeasibilityStatus.VERIFIED_FEASIBLE) for _ in range(3))
    args = (reports, ({"x": 1.0}, {"x": 1.0}, {"x": 1.0}), (3.0, 1.0, 2.0))
    first = rank_candidates(*args)
    second = rank_candidates(*args)
    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]
    assert [item.index for item in sorted(first, key=lambda item: item.sort_key)] == [1, 2, 0]


def test_power_pipeline_contains_no_fixed_invalid_penalty() -> None:
    source = inspect.getsource(pipeline_module)
    assert "1_000_000" not in source
    assert "invalid_penalty" not in source
    assert "_power_selection_score" not in source
