from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .optimization import (
    DifferentialEvolutionConfig,
    DifferentialEvolutionDriver,
    EvaluationBudget,
    OptimizationProblem,
    OptimizationRunResult,
    OptimizationVariable,
    VariableKind,
    VariableScale,
)
from .spec import SynthesisSpec
from .targets import TargetResponse
from .templates import CircuitTemplate


ScoreFunction = Callable[[np.ndarray], float]
ResponseEvaluator = Callable[[dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class ParameterOptimizationResult:
    parameters: dict[str, float]
    response: np.ndarray
    success: bool
    message: str
    evaluations: int | None = None
    optimization_run: OptimizationRunResult | None = None


@dataclass(frozen=True)
class DifferentialEvolutionOptions:
    popsize: int = 15
    tol: float = 1e-7
    polish: bool = True
    updating: str = "immediate"
    workers: int = 1
    max_evaluations: int | None = None
    max_time_s: float | None = None
    cache: bool = True


class ParameterOptimizer(Protocol):
    def optimize(
        self,
        template: CircuitTemplate,
        spec: SynthesisSpec,
        target: TargetResponse,
        score_response: ScoreFunction,
    ) -> ParameterOptimizationResult:
        ...


@dataclass(frozen=True)
class DifferentialEvolutionParameterOptimizer:
    """Default global optimizer for continuous component values.

    Topology selection stays outside this class.  This component only searches
    positive-valued template parameters in log10 space and asks the synthesis
    scorer to judge each simulated response.
    """

    def optimize(
        self,
        template: CircuitTemplate,
        spec: SynthesisSpec,
        target: TargetResponse,
        score_response: ScoreFunction,
    ) -> ParameterOptimizationResult:
        return self._optimize(template, spec, target, score_response, response_evaluator=None)

    def optimize_with_response_evaluator(
        self,
        template: CircuitTemplate,
        spec: SynthesisSpec,
        target: TargetResponse,
        score_response: ScoreFunction,
        response_evaluator: ResponseEvaluator,
    ) -> ParameterOptimizationResult:
        return self._optimize(template, spec, target, score_response, response_evaluator=response_evaluator)

    def _optimize(
        self,
        template: CircuitTemplate,
        spec: SynthesisSpec,
        target: TargetResponse,
        score_response: ScoreFunction,
        response_evaluator: ResponseEvaluator | None,
    ) -> ParameterOptimizationResult:
        problem = _template_optimization_problem(template, spec)

        def objective(values) -> float:
            parameters = {str(key): float(value) for key, value in values.items()}
            response = (
                response_evaluator(parameters)
                if response_evaluator is not None
                else template.analyze(parameters, target.frequencies_hz, target.analysis)
            )
            return score_response(response)

        def truth_objective(values) -> float:
            parameters = {str(key): float(value) for key, value in values.items()}
            return score_response(
                template.analyze(parameters, target.frequencies_hz, target.analysis)
            )

        options = parse_differential_evolution_options(spec)
        run = DifferentialEvolutionDriver().solve(
            problem,
            objective,
            config=DifferentialEvolutionConfig(
                max_iterations=spec.optimization.max_iterations,
                popsize=options.popsize,
                tol=options.tol,
                polish=options.polish,
                seed=spec.optimization.seed,
                updating=options.updating,
                workers=options.workers,
                cache=options.cache,
                budget=EvaluationBudget(options.max_evaluations, options.max_time_s),
            ),
            truth_evaluator=truth_objective,
        )
        parameters = {
            str(key): float(value)
            for key, value in run.values.items()
        }
        response = template.analyze(
            parameters,
            target.frequencies_hz,
            target.analysis,
        )
        return ParameterOptimizationResult(
            parameters=parameters,
            response=response,
            success=run.success,
            message=run.message,
            evaluations=run.evaluations,
            optimization_run=run,
        )


def parse_differential_evolution_options(spec: SynthesisSpec) -> DifferentialEvolutionOptions:
    options = dict(spec.optimization.optimizer)
    nested = options.get("differential_evolution")
    if isinstance(nested, dict):
        options.update(nested)

    popsize = _require_int(options, "popsize", 15, minimum=1)
    tol = _require_float(options, "tol", 1e-7, minimum=0.0)
    polish = _require_bool(options, "polish", True)
    updating = _require_str(options, "updating", "immediate")
    if updating not in {"immediate", "deferred"}:
        raise ValueError("optimizer.differential_evolution.updating must be 'immediate' or 'deferred'")
    workers = _require_int(options, "workers", 1)
    if workers == 0:
        raise ValueError("optimizer.differential_evolution.workers must not be 0")
    max_evaluations = _optional_int(options, "max_evaluations", minimum=1)
    max_time_s = _optional_float(options, "max_time_s", minimum=0.0, exclusive=True)
    cache = _require_bool(options, "cache", True)
    return DifferentialEvolutionOptions(
        popsize=popsize,
        tol=tol,
        polish=polish,
        updating=updating,
        workers=workers,
        max_evaluations=max_evaluations,
        max_time_s=max_time_s,
        cache=cache,
    )


def _template_optimization_problem(
    template: CircuitTemplate,
    spec: SynthesisSpec,
) -> OptimizationProblem:
    units = {"R": "ohm", "C": "F", "L": "H"}
    variables = tuple(
        OptimizationVariable(
            variable_id=parameter.name,
            kind=VariableKind.CONTINUOUS,
            unit=units.get(parameter.element_type, "1"),
            lower=lower,
            upper=upper,
            scale=VariableScale.LOG10 if lower > 0.0 else VariableScale.LINEAR,
            source_path=f"template.{template.name}.{parameter.name}",
            attributes={"element_type": parameter.element_type},
        )
        for parameter, (lower, upper) in zip(
            template.params,
            template.parameter_bounds(spec.library),
        )
    )
    return OptimizationProblem(
        problem_id=f"ac.{template.name}",
        variables=variables,
        objective_ids=("synthesis.score",),
        metadata={"template": template.name, "analysis": target_analysis_name(spec)},
    )


def target_analysis_name(spec: SynthesisSpec) -> str:
    return spec.analysis.kind


def _require_int(options: dict[str, object], key: str, default: int, minimum: int | None = None) -> int:
    value = options.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"optimizer.differential_evolution.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"optimizer.differential_evolution.{key} must be >= {minimum}")
    return value


def _require_float(
    options: dict[str, object],
    key: str,
    default: float,
    minimum: float | None = None,
) -> float:
    value = options.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"optimizer.differential_evolution.{key} must be a number")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"optimizer.differential_evolution.{key} must be >= {minimum}")
    return parsed


def _require_bool(options: dict[str, object], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"optimizer.differential_evolution.{key} must be a boolean")
    return value


def _require_str(options: dict[str, object], key: str, default: str) -> str:
    value = options.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"optimizer.differential_evolution.{key} must be a string")
    return value


def _optional_int(
    options: dict[str, object],
    key: str,
    *,
    minimum: int,
) -> int | None:
    value = options.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"optimizer.differential_evolution.{key} must be an integer >= {minimum}"
        )
    return value


def _optional_float(
    options: dict[str, object],
    key: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float | None:
    value = options.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"optimizer.differential_evolution.{key} must be a number")
    parsed = float(value)
    invalid = parsed <= minimum if exclusive else parsed < minimum
    if not np.isfinite(parsed) or invalid:
        comparison = ">" if exclusive else ">="
        raise ValueError(
            f"optimizer.differential_evolution.{key} must be finite and {comparison} {minimum}"
        )
    return parsed
