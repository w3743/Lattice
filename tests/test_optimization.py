from __future__ import annotations

import pytest

from circuit_ai.optimization import (
    DifferentialEvolutionConfig,
    DifferentialEvolutionDriver,
    EvaluationBudget,
    OptimizationContractError,
    OptimizationProblem,
    OptimizationStatus,
    OptimizationVariable,
    VariableKind,
    VariableScale,
)


def test_mixed_variable_encoding_round_trip_keeps_physical_values() -> None:
    problem = OptimizationProblem(
        problem_id="mixed",
        variables=(
            OptimizationVariable(
                "resistance",
                VariableKind.CONTINUOUS,
                "ohm",
                100.0,
                1_000_000.0,
                scale=VariableScale.LOG10,
            ),
            OptimizationVariable("sections", VariableKind.INTEGER, "1", 1, 5),
            OptimizationVariable(
                "family",
                VariableKind.CATEGORICAL,
                choices=("RC", "RLC", "active"),
            ),
            OptimizationVariable(
                "area",
                VariableKind.DERIVED,
                "mm2",
                expression="sum(component.area)",
            ),
        ),
        objective_ids=("error",),
    )

    encoded = problem.encode(
        {"resistance": 10_000.0, "sections": 3, "family": "RLC"}
    )
    decoded = problem.decode(
        encoded,
        derived_resolver=lambda variable, values: values["sections"] * 2.5,
    )

    assert encoded == pytest.approx((4.0, 3.0, 1.0))
    assert decoded == {
        "resistance": pytest.approx(10_000.0),
        "sections": 3,
        "family": "RLC",
        "area": 7.5,
    }
    assert problem.integrality == (False, True, True)
    assert problem.as_dict()["variables"][0]["unit"] == "ohm"


def test_invalid_log_and_integer_domains_fail_at_contract_boundary() -> None:
    with pytest.raises(OptimizationContractError, match="positive bounds"):
        OptimizationVariable(
            "gain",
            VariableKind.CONTINUOUS,
            "1",
            -1.0,
            10.0,
            scale=VariableScale.LOG10,
        )
    with pytest.raises(OptimizationContractError, match="no integral value"):
        OptimizationVariable("turns", VariableKind.INTEGER, "1", 0.1, 0.9)


def test_driver_handles_integer_and_categorical_coordinates() -> None:
    problem = OptimizationProblem(
        "mixed_driver",
        (
            OptimizationVariable("x", VariableKind.CONTINUOUS, "V", 0.1, 10.0),
            OptimizationVariable("n", VariableKind.INTEGER, "1", 1, 5),
            OptimizationVariable("mode", VariableKind.CATEGORICAL, choices=("a", "b")),
        ),
        ("loss",),
    )

    result = DifferentialEvolutionDriver().solve(
        problem,
        lambda values: (values["x"] - 2.0) ** 2
        + (values["n"] - 3) ** 2
        + (0.0 if values["mode"] == "b" else 4.0),
        config=DifferentialEvolutionConfig(
            max_iterations=25,
            popsize=6,
            polish=False,
            seed=13,
        ),
    )

    assert result.status is OptimizationStatus.COMPLETED
    assert result.values["x"] == pytest.approx(2.0, abs=0.05)
    assert result.values["n"] == 3
    assert result.values["mode"] == "b"
    assert result.evaluations > 0


def test_driver_stops_at_exact_evaluation_budget_and_keeps_best() -> None:
    problem = OptimizationProblem(
        "budgeted",
        (OptimizationVariable("x", VariableKind.CONTINUOUS, "1", -5.0, 5.0),),
        ("loss",),
    )
    result = DifferentialEvolutionDriver().solve(
        problem,
        lambda values: (values["x"] - 1.25) ** 2,
        config=DifferentialEvolutionConfig(
            max_iterations=100,
            popsize=5,
            polish=False,
            seed=4,
            budget=EvaluationBudget(max_evaluations=5),
        ),
    )

    assert result.status is OptimizationStatus.BUDGET_EXHAUSTED
    assert result.evaluations == 5
    assert -5.0 <= result.values["x"] <= 5.0
    assert result.objective_values["loss"] >= 0.0


def test_truth_re_evaluation_is_distinct_from_search_objective() -> None:
    problem = OptimizationProblem(
        "truth_gate",
        (OptimizationVariable("x", VariableKind.CONTINUOUS, "1", 0.0, 10.0),),
        ("loss",),
    )
    result = DifferentialEvolutionDriver().solve(
        problem,
        lambda values: (values["x"] - 2.0) ** 2,
        truth_evaluator=lambda values: (values["x"] - 7.0) ** 2,
        config=DifferentialEvolutionConfig(
            max_iterations=12,
            popsize=5,
            polish=False,
            seed=8,
        ),
    )

    assert result.truth_evaluated
    assert result.truth_evaluations == 1
    assert result.objective_values["loss"] == result.truth_objective
    assert result.search_objective != result.truth_objective
