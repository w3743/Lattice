"""Generic SciPy differential-evolution driver over OptimizationProblem."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

import numpy as np
import scipy
from scipy.optimize import differential_evolution

from .contracts import (
    EvaluationBudget,
    OptimizationProblem,
    OptimizationRunResult,
    OptimizationStatus,
)


ObjectiveEvaluator = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class DifferentialEvolutionConfig:
    max_iterations: int = 70
    popsize: int = 15
    tol: float = 1e-7
    polish: bool = True
    updating: str = "immediate"
    workers: int = 1
    seed: int = 7
    cache: bool = True
    budget: EvaluationBudget = EvaluationBudget()

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.popsize < 1:
            raise ValueError("popsize must be positive")
        if not math.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if self.updating not in {"immediate", "deferred"}:
            raise ValueError("updating must be 'immediate' or 'deferred'")
        if self.workers == 0:
            raise ValueError("workers must not be zero")


class _BudgetStop(RuntimeError):
    pass


class DifferentialEvolutionDriver:
    solver_id = "scipy.differential_evolution"

    def solve(
        self,
        problem: OptimizationProblem,
        objective: ObjectiveEvaluator,
        *,
        config: DifferentialEvolutionConfig | None = None,
        truth_evaluator: ObjectiveEvaluator | None = None,
    ) -> OptimizationRunResult:
        config = config or DifferentialEvolutionConfig()
        if len(problem.objective_ids) != 1:
            raise ValueError("SciPy differential evolution currently requires exactly one objective")
        started = time.perf_counter()
        evaluations = 0
        cache_hits = 0
        cache: dict[tuple[float, ...], float] = {}
        best_coordinate: tuple[float, ...] | None = None
        best_search = float("inf")

        def elapsed() -> float:
            return time.perf_counter() - started

        def evaluate(coordinates) -> float:
            nonlocal evaluations, cache_hits, best_coordinate, best_search
            key = tuple(float(item) for item in np.asarray(coordinates, dtype=float))
            if config.cache and key in cache:
                cache_hits += 1
                return cache[key]
            if config.budget.max_evaluations is not None and evaluations >= config.budget.max_evaluations:
                raise _BudgetStop("evaluation budget exhausted")
            if config.budget.max_time_s is not None and elapsed() >= config.budget.max_time_s:
                raise _BudgetStop("time budget exhausted")
            value = float(objective(problem.decode(key)))
            if not math.isfinite(value):
                raise ValueError("optimization objective returned a non-finite value")
            evaluations += 1
            if config.cache:
                cache[key] = value
            if value < best_search:
                best_coordinate, best_search = key, value
            return value

        status = OptimizationStatus.COMPLETED
        success = True
        message = "problem has no free decision variables"
        coordinate = problem.initial_vector
        search_objective: float | None = None

        if problem.decision_variables:
            kwargs: dict[str, Any] = {
                "maxiter": config.max_iterations,
                "popsize": config.popsize,
                "tol": config.tol,
                "polish": config.polish,
                "rng": np.random.default_rng(config.seed),
                "updating": config.updating,
                "workers": config.workers,
                "integrality": np.asarray(problem.integrality, dtype=bool),
            }
            if problem.has_explicit_initial:
                kwargs["x0"] = np.asarray(problem.initial_vector, dtype=float)
            try:
                raw = differential_evolution(evaluate, problem.encoded_bounds, **kwargs)
                coordinate = tuple(float(item) for item in raw.x)
                search_objective = float(raw.fun)
                success = bool(raw.success)
                message = str(raw.message)
            except _BudgetStop as exc:
                if best_coordinate is None:
                    raise RuntimeError("optimization budget ended before one evaluation completed") from exc
                coordinate = best_coordinate
                search_objective = best_search
                status = OptimizationStatus.BUDGET_EXHAUSTED
                success = False
                message = str(exc)
        else:
            search_objective = evaluate(coordinate)

        values = problem.decode(coordinate)
        truth_objective: float | None = None
        truth_evaluations = 0
        if truth_evaluator is not None:
            truth_objective = float(truth_evaluator(values))
            truth_evaluations = 1
            if not math.isfinite(truth_objective):
                status = OptimizationStatus.FAILED
                success = False
                message = "truth evaluator returned a non-finite value"
        final_objective = truth_objective if truth_objective is not None else search_objective
        assert final_objective is not None
        return OptimizationRunResult(
            problem_id=problem.problem_id,
            solver_id=self.solver_id,
            solver_version=scipy.__version__,
            status=status,
            success=success,
            values=values,
            objective_values={problem.objective_ids[0]: final_objective},
            search_objective=search_objective,
            truth_objective=truth_objective,
            evaluations=evaluations,
            truth_evaluations=truth_evaluations,
            cache_hits=cache_hits,
            duration_s=elapsed(),
            message=message,
            budget=config.budget,
        )
