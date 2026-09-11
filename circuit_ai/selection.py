"""Feasibility-first, Pareto-aware candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .constraints import ConstraintReport


@dataclass(frozen=True)
class CandidateDecision:
    index: int
    feasibility_rank: int
    feasibility: str
    unavailable_hard_count: int
    hard_violation: float
    pareto_rank: int
    crowding_distance: float
    soft_penalty: float
    tie_breaker: float
    objectives: Mapping[str, float]

    @property
    def sort_key(self) -> tuple[float, ...]:
        crowding = self.crowding_distance
        crowding_key = float("-inf") if math.isinf(crowding) else -crowding
        return (
            float(self.feasibility_rank),
            float(self.unavailable_hard_count),
            float(self.hard_violation),
            float(self.pareto_rank),
            crowding_key,
            float(self.soft_penalty),
            float(self.tie_breaker),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "feasibility": self.feasibility,
            "feasibility_rank": self.feasibility_rank,
            "unavailable_hard_count": self.unavailable_hard_count,
            "hard_violation": self.hard_violation,
            "pareto_rank": self.pareto_rank,
            "crowding_distance": (
                "infinity" if math.isinf(self.crowding_distance) else self.crowding_distance
            ),
            "soft_penalty": self.soft_penalty,
            "tie_breaker": self.tie_breaker,
            "objectives": dict(sorted(self.objectives.items())),
        }


def rank_candidates(
    reports: Sequence[ConstraintReport],
    objectives: Sequence[Mapping[str, float]],
    tie_breakers: Sequence[float],
) -> tuple[CandidateDecision, ...]:
    if not (len(reports) == len(objectives) == len(tie_breakers)):
        raise ValueError("candidate reports, objectives, and tie breakers must have equal length")
    normalized = tuple(_normalize_objectives(item) for item in objectives)
    ranks: dict[int, int] = {}
    crowding: dict[int, float] = {}
    tiers = sorted({report.feasibility_rank for report in reports})
    for tier in tiers:
        indices = [index for index, report in enumerate(reports) if report.feasibility_rank == tier]
        tier_ranks, tier_crowding = _pareto(indices, normalized)
        ranks.update(tier_ranks)
        crowding.update(tier_crowding)
    return tuple(
        CandidateDecision(
            index=index,
            feasibility_rank=report.feasibility_rank,
            feasibility=report.feasibility.value,
            unavailable_hard_count=report.unavailable_hard_count,
            hard_violation=report.hard_violation,
            pareto_rank=ranks.get(index, 0),
            crowding_distance=crowding.get(index, 0.0),
            soft_penalty=report.soft_penalty,
            tie_breaker=_finite(tie_breakers[index]),
            objectives=normalized[index],
        )
        for index, report in enumerate(reports)
    )


def _pareto(
    indices: list[int],
    objectives: tuple[dict[str, float], ...],
) -> tuple[dict[int, int], dict[int, float]]:
    if not indices:
        return {}, {}
    dominates: dict[int, set[int]] = {index: set() for index in indices}
    dominated_count = {index: 0 for index in indices}
    fronts: list[list[int]] = [[]]
    for left in indices:
        for right in indices:
            if left == right:
                continue
            if _dominates(objectives[left], objectives[right]):
                dominates[left].add(right)
            elif _dominates(objectives[right], objectives[left]):
                dominated_count[left] += 1
        if dominated_count[left] == 0:
            fronts[0].append(left)
    rank: dict[int, int] = {}
    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        next_front: list[int] = []
        for left in sorted(fronts[cursor]):
            rank[left] = cursor
            for right in sorted(dominates[left]):
                dominated_count[right] -= 1
                if dominated_count[right] == 0:
                    next_front.append(right)
        if next_front:
            fronts.append(sorted(set(next_front)))
        cursor += 1
    crowding: dict[int, float] = {}
    for front in fronts:
        if front:
            crowding.update(_crowding(front, objectives))
    return rank, crowding


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    names = sorted(set(left) | set(right))
    pairs = [(left.get(name, 0.0), right.get(name, 0.0)) for name in names]
    return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)


def _crowding(front: list[int], objectives: tuple[dict[str, float], ...]) -> dict[int, float]:
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    distances = {index: 0.0 for index in front}
    names = sorted({name for index in front for name in objectives[index]})
    for name in names:
        ordered = sorted(front, key=lambda index: (objectives[index].get(name, 0.0), index))
        lower = objectives[ordered[0]].get(name, 0.0)
        upper = objectives[ordered[-1]].get(name, 0.0)
        if upper <= lower:
            continue
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if math.isinf(distances[index]):
                continue
            before = objectives[ordered[position - 1]].get(name, 0.0)
            after = objectives[ordered[position + 1]].get(name, 0.0)
            distances[index] += (after - before) / (upper - lower)
    return distances


def _normalize_objectives(values: Mapping[str, float]) -> dict[str, float]:
    return {str(name): _finite(value) for name, value in sorted(values.items())}


def _finite(value: float) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 1e12
