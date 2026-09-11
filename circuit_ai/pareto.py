from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OBJECTIVE_NAMES = (
    "rmse_db",
    "max_abs_db",
    "phase_rmse_deg",
    "component_count",
    "estimated_cost",
    "estimated_area_mm2",
    "topology_risk",
    "optimizer_failure",
    "robustness_yield_loss",
    "robustness_p95_rmse_db",
)


@dataclass(frozen=True)
class ParetoPoint:
    index: int
    objectives: dict[str, float]
    rank: int
    crowding_distance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            "objectives": self.objectives,
        }


def metrics_to_objectives(metrics) -> dict[str, float]:
    robustness = metrics.robustness
    return {
        "rmse_db": _finite(metrics.rmse_db),
        "max_abs_db": _finite(metrics.max_abs_db),
        "phase_rmse_deg": _finite(metrics.phase_rmse_deg or 0.0),
        "component_count": _finite(metrics.component_count),
        "estimated_cost": _finite(getattr(metrics, "estimated_cost", 0.0)),
        "estimated_area_mm2": _finite(getattr(metrics, "estimated_area_mm2", 0.0)),
        "topology_risk": _finite(getattr(metrics, "topology_risk", 0.0)),
        "optimizer_failure": 0.0 if metrics.optimizer_success else 1.0,
        "robustness_yield_loss": _finite(1.0 - robustness.yield_fraction if robustness else 0.0),
        "robustness_p95_rmse_db": _finite(robustness.rmse_db_p95 if robustness else 0.0),
    }


def pareto_points(objectives: list[dict[str, float]]) -> list[ParetoPoint]:
    if not objectives:
        return []

    normalized = [_normalize_objectives(item) for item in objectives]
    dominates: list[set[int]] = [set() for _ in normalized]
    dominated_count = [0 for _ in normalized]
    fronts: list[list[int]] = [[]]

    for i, left in enumerate(normalized):
        for j, right in enumerate(normalized):
            if i == j:
                continue
            if dominates_objectives(left, right):
                dominates[i].add(j)
            elif dominates_objectives(right, left):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    rank_by_index: dict[int, int] = {}
    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            rank_by_index[i] = current
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current += 1

    crowding: dict[int, float] = {}
    for front in fronts:
        if front:
            crowding.update(_crowding_distance(front, normalized))

    return [
        ParetoPoint(
            index=index,
            objectives=normalized[index],
            rank=rank_by_index.get(index, len(fronts)),
            crowding_distance=crowding.get(index, 0.0),
        )
        for index in range(len(normalized))
    ]


def dominates_objectives(left: dict[str, float], right: dict[str, float]) -> bool:
    names = sorted(set(left) | set(right))
    left_values = [_finite(left.get(name, 0.0)) for name in names]
    right_values = [_finite(right.get(name, 0.0)) for name in names]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _crowding_distance(front: list[int], objectives: list[dict[str, float]]) -> dict[int, float]:
    if len(front) == 1:
        return {front[0]: float("inf")}
    if len(front) == 2:
        return {front[0]: float("inf"), front[1]: float("inf")}

    distances = {index: 0.0 for index in front}
    for name in OBJECTIVE_NAMES:
        ordered = sorted(front, key=lambda index: objectives[index][name])
        lo = objectives[ordered[0]][name]
        hi = objectives[ordered[-1]][name]
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        if hi <= lo:
            continue
        for pos in range(1, len(ordered) - 1):
            if np.isinf(distances[ordered[pos]]):
                continue
            before = objectives[ordered[pos - 1]][name]
            after = objectives[ordered[pos + 1]][name]
            distances[ordered[pos]] += (after - before) / (hi - lo)
    return distances


def _normalize_objectives(objectives: dict[str, float]) -> dict[str, float]:
    return {name: _finite(objectives.get(name, 0.0)) for name in OBJECTIVE_NAMES}


def _finite(value: float | int | None) -> float:
    if value is None:
        return 0.0
    parsed = float(value)
    if not np.isfinite(parsed):
        return 1e12
    return parsed
