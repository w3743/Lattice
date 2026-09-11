"""Application service that compiles, extracts, and evaluates constraints."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..metrics import MetricContext, MetricEngine
from .compiler import compile_constraint_program
from .contracts import ConstraintReport
from .evaluator import ConstraintEvaluator


class DeclarativeConstraintValidator:
    """Topology-neutral validator used by synthesis capability registrations."""

    def __init__(
        self,
        metric_engine: MetricEngine | None = None,
        evaluator: ConstraintEvaluator | None = None,
    ) -> None:
        self.metric_engine = metric_engine or MetricEngine()
        self.evaluator = evaluator or ConstraintEvaluator()

    def validate(
        self,
        ir: Any,
        candidate: Any,
        simulation_result: Any,
        *,
        parameter_values: Any | None = None,
    ) -> ConstraintReport:
        graph = getattr(candidate, "graph", None)
        if graph is None:
            raise ValueError("declarative constraint validation requires a CircuitGraph candidate")
        results = tuple(simulation_result) if isinstance(simulation_result, (tuple, list)) else (simulation_result,)
        program = compile_constraint_program(ir, graph)
        metrics = self.metric_engine.extract(
            program.metrics,
            MetricContext(
                graph=graph,
                ir=ir,
                simulation_results=results,
                parameter_values=parameter_values,
            ),
        )
        report = self.evaluator.evaluate(program.constraints, metrics)
        if program.diagnostics:
            report = replace(report, diagnostics=(*report.diagnostics, *program.diagnostics))
        return report
