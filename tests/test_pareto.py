from __future__ import annotations

from circuit_ai.pareto import dominates_objectives, pareto_points


def test_pareto_points_rank_dominated_candidates() -> None:
    points = pareto_points(
        [
            {"rmse_db": 0.1, "max_abs_db": 1.0, "component_count": 4},
            {"rmse_db": 0.2, "max_abs_db": 1.2, "component_count": 4},
            {"rmse_db": 0.3, "max_abs_db": 2.0, "component_count": 2},
        ]
    )

    assert points[0].rank == 0
    assert points[1].rank > points[0].rank
    assert points[2].rank == 0


def test_dominates_objectives_requires_at_least_one_strict_improvement() -> None:
    left = {"rmse_db": 1.0, "component_count": 2}
    equal = {"rmse_db": 1.0, "component_count": 2}
    worse = {"rmse_db": 1.0, "component_count": 3}

    assert not dominates_objectives(left, equal)
    assert dominates_objectives(left, worse)
