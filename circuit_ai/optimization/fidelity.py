"""Topology-agnostic selection records for staged simulation fidelity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


FIDELITY_SCHEDULE_SCHEMA = "circuit_ai.fidelity_schedule"
FIDELITY_SCHEDULE_VERSION = 1


class FidelityRole(str, Enum):
    SCREEN = "screen"
    TRUTH = "truth"


@dataclass(frozen=True)
class FidelityStage:
    """One ordered selection gate in a multi-fidelity execution plan."""

    stage_id: str
    fidelity: str
    role: FidelityRole
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.stage_id.strip() or not self.fidelity.strip():
            raise ValueError("fidelity stage id and fidelity must not be empty")
        role = self.role if isinstance(self.role, FidelityRole) else FidelityRole(self.role)
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("fidelity stage max_candidates must be positive")
        object.__setattr__(self, "role", role)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "fidelity": self.fidelity,
            "role": self.role.value,
            "max_candidates": self.max_candidates,
        }


@dataclass(frozen=True)
class FidelitySchedule:
    """A versioned policy that promotes ranked candidates across fidelities."""

    schedule_id: str
    stages: tuple[FidelityStage, ...]
    schema: str = FIDELITY_SCHEDULE_SCHEMA
    schema_version: int = FIDELITY_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.schema != FIDELITY_SCHEDULE_SCHEMA or self.schema_version != FIDELITY_SCHEDULE_VERSION:
            raise ValueError("unsupported fidelity schedule schema")
        if not self.schedule_id.strip() or not self.stages:
            raise ValueError("fidelity schedule id and at least one stage are required")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("fidelity stage ids must be unique")
        if self.stages[-1].role is not FidelityRole.TRUTH:
            raise ValueError("the final fidelity stage must be a truth stage")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "stages": [stage.as_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class FidelityStageSelection:
    stage_id: str
    fidelity: str
    role: FidelityRole
    candidate_ids: tuple[str, ...]
    input_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "fidelity": self.fidelity,
            "role": self.role.value,
            "candidate_ids": list(self.candidate_ids),
            "input_count": self.input_count,
            "selected_count": len(self.candidate_ids),
        }


@dataclass(frozen=True)
class FidelityScheduleRun:
    schedule: FidelitySchedule
    selections: tuple[FidelityStageSelection, ...]

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return self.selections[-1].candidate_ids

    def as_dict(self) -> dict[str, object]:
        return {
            **self.schedule.as_dict(),
            "selections": [selection.as_dict() for selection in self.selections],
            "selected_candidate_ids": list(self.selected_candidate_ids),
        }


class FidelityScheduler:
    """Apply a declared schedule to candidates already sorted by low-cost rank."""

    def schedule(
        self,
        policy: FidelitySchedule,
        candidate_ids: tuple[str, ...],
        *,
        stage_candidate_ids: Mapping[str, tuple[str, ...]] | None = None,
    ) -> FidelityScheduleRun:
        if len(set(candidate_ids)) != len(candidate_ids) or any(not item.strip() for item in candidate_ids):
            raise ValueError("fidelity scheduler requires unique, non-empty candidate ids")
        selected = candidate_ids
        selections: list[FidelityStageSelection] = []
        for stage in policy.stages:
            input_count = len(selected)
            requested = (stage_candidate_ids or {}).get(stage.stage_id)
            if requested is not None:
                if len(set(requested)) != len(requested) or any(
                    candidate not in selected for candidate in requested
                ):
                    raise ValueError(
                        f"fidelity stage {stage.stage_id!r} contains an invalid promotion set"
                    )
                selected = tuple(requested)
            elif stage.max_candidates is not None:
                selected = selected[: stage.max_candidates]
            selections.append(
                FidelityStageSelection(
                    stage_id=stage.stage_id,
                    fidelity=stage.fidelity,
                    role=stage.role,
                    candidate_ids=selected,
                    input_count=input_count,
                )
            )
        return FidelityScheduleRun(policy, tuple(selections))


def default_ac_fidelity_schedule(top_k: int) -> FidelitySchedule:
    """Record the existing AC flow: optimize all, then truth-check the Top-K."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    return FidelitySchedule(
        schedule_id="ac.default.v1",
        stages=(
            FidelityStage("parameter_optimization", "optimization_model", FidelityRole.SCREEN),
            FidelityStage("linear_mna_truth", "linear_frequency_domain", FidelityRole.TRUTH, top_k),
        ),
    )
