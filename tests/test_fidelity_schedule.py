from __future__ import annotations

import pytest

from circuit_ai.optimization import (
    FidelityRole,
    FidelitySchedule,
    FidelityScheduler,
    FidelityStage,
    default_ac_fidelity_schedule,
)


def test_scheduler_preserves_rank_order_and_limits_truth_candidates() -> None:
    schedule = default_ac_fidelity_schedule(top_k=2)

    run = FidelityScheduler().schedule(schedule, ("candidate_a", "candidate_b", "candidate_c"))

    assert run.selections[0].role is FidelityRole.SCREEN
    assert run.selections[0].candidate_ids == ("candidate_a", "candidate_b", "candidate_c")
    assert run.selections[1].role is FidelityRole.TRUTH
    assert run.selections[1].candidate_ids == ("candidate_a", "candidate_b")
    assert run.selected_candidate_ids == ("candidate_a", "candidate_b")
    assert run.as_dict()["selections"][1]["fidelity"] == "linear_frequency_domain"


def test_schedule_requires_unique_candidates_and_a_final_truth_stage() -> None:
    with pytest.raises(ValueError, match="final fidelity stage"):
        FidelitySchedule(
            "invalid",
            (FidelityStage("screen", "analytic", FidelityRole.SCREEN),),
        )

    schedule = default_ac_fidelity_schedule(top_k=1)
    with pytest.raises(ValueError, match="unique"):
        FidelityScheduler().schedule(schedule, ("candidate", "candidate"))
