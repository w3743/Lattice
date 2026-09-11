"""Unified, solver-independent design problem compilation.

``DesignProblem`` is the application-facing contract between PBDL and the
existing topology/optimization backends.  It deliberately contains no solver
implementation: AC and DC adapters consume the same compiled tasks, metrics,
constraints, and optimization metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .constraints import ConstraintProgram, compile_constraint_program
from .ir import UnifiedIR, pbdl_to_ir
from .optimization import (
    OptimizationProblem,
    OptimizationVariable,
    VariableKind,
    VariableScale,
)
from .simulation import SimulationRequest, simulation_requests_for_tasks
from .simulation_tasks import SimulationTask, simulation_tasks_from_ir


DESIGN_PROBLEM_SCHEMA = "circuit_ai.design_problem"
DESIGN_PROBLEM_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DesignProblem:
    """Compiled PBDL intent shared by all design backends."""

    problem_id: str
    ir: UnifiedIR
    source_data: Mapping[str, Any]
    tasks: tuple[SimulationTask, ...]
    constraint_program: ConstraintProgram
    optimization_problem: OptimizationProblem
    parent_problem_id: str | None = None
    stage_index: int | None = None
    schema: str = DESIGN_PROBLEM_SCHEMA
    schema_version: int = DESIGN_PROBLEM_SCHEMA_VERSION

    @property
    def metrics(self):
        """Metric specifications compiled from all analyses and constraints."""

        return self.constraint_program.metrics

    @property
    def constraints(self):
        """Constraint specifications compiled from the public PBDL contract."""

        return self.constraint_program.constraints

    @property
    def simulation_tasks(self) -> tuple[SimulationTask, ...]:
        """Compatibility/readability alias for the request templates."""

        return self.tasks

    @property
    def requests(self) -> tuple[SimulationTask, ...]:
        """Request templates; concrete requests require a candidate graph."""

        return self.tasks

    @property
    def intent_kind(self) -> str:
        return self.ir.intent_kind

    def requests_for_graph(
        self,
        graph,
        *,
        parameter_values: Mapping[str, Any] | None = None,
        fidelity: str = "unspecified",
    ) -> tuple[SimulationRequest, ...]:
        """Materialize all compiled simulation requests for ``graph``."""

        return simulation_requests_for_tasks(
            self.tasks,
            graph,
            parameter_values=parameter_values,
            fidelity=fidelity,
        )

    def stage(self, index: int) -> "DesignProblem":
        """Return a single-analysis view without losing the parent constraints."""

        if index < 0 or index >= len(self.ir.analyses):
            raise IndexError(f"analysis index {index} is out of range")
        compiler = DesignProblemCompiler()
        data = dict(self.source_data)
        data["functions"] = []
        data["analyses"] = [dict(self.source_data["analyses"][index])]
        targets = list(self.source_data.get("targets", ()))
        data["targets"] = [dict(targets[index])] if index < len(targets) else []
        return compiler.compile(
            data,
            problem_id=f"{self.problem_id}.stage_{index + 1}",
            parent_problem_id=self.problem_id,
            stage_index=index,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "parent_problem_id": self.parent_problem_id,
            "stage_index": self.stage_index,
            "intent_kind": self.intent_kind,
            "ir": self.ir.as_dict(),
            "source": dict(self.source_data),
            "tasks": [task.as_dict() for task in self.tasks],
            "metrics": [metric.as_dict() for metric in self.metrics],
            "constraints": [constraint.as_dict() for constraint in self.constraints],
            "constraint_diagnostics": list(self.constraint_program.diagnostics),
            "optimization_problem": self.optimization_problem.as_dict(),
        }


class DesignProblemCompiler:
    """Compile canonical PBDL into the unified execution contract."""

    def compile(
        self,
        source: Mapping[str, Any] | UnifiedIR,
        *,
        problem_id: str | None = None,
        parent_problem_id: str | None = None,
        stage_index: int | None = None,
    ) -> DesignProblem:
        if isinstance(source, UnifiedIR):
            ir = source
            source_data = _canonical_source(_ir_to_source_data(ir))
        else:
            source_data = _canonical_source(dict(source))
            ir = pbdl_to_ir(source_data)
        program = compile_constraint_program(ir)
        tasks = simulation_tasks_from_ir(ir)
        optimization_problem = _compile_optimization_problem(ir, program)
        resolved_id = problem_id or f"{ir.name}.design"
        return DesignProblem(
            problem_id=resolved_id,
            ir=ir,
            source_data=source_data,
            tasks=tasks,
            constraint_program=program,
            optimization_problem=optimization_problem,
            parent_problem_id=parent_problem_id,
            stage_index=stage_index,
        )


def compile_design_problem(
    source: Mapping[str, Any] | UnifiedIR,
    *,
    problem_id: str | None = None,
) -> DesignProblem:
    """Functional entry point for callers that do not need a compiler object."""

    return DesignProblemCompiler().compile(source, problem_id=problem_id)


def _canonical_source(data: dict[str, Any]) -> dict[str, Any]:
    from .pbdl_boundary import canonicalize_pbdl_dict

    return canonicalize_pbdl_dict(data)


def _ir_to_source_data(ir: UnifiedIR) -> dict[str, Any]:
    """Best-effort source projection for callers that already own a UnifiedIR."""

    ports = []
    for port in ir.ports:
        terminals = [
            {"name": name, "quantity": "ground" if name == "0" else "voltage"}
            for name in port.terminals
        ]
        ports.append(
            {
                "name": port.name,
                "terminals": terminals,
                "role": port.role,
                "domain": port.domain,
                "variables": list(port.variables),
                "variable_constraints": list(port.constraints),
                "excitation": port.excitation,
            }
        )
    return {
        "name": ir.name,
        "description": ir.description,
        "ports": ports,
        "relations": [dict(item) for item in ir.relations],
        "functions": [],
        "analyses": [dict(item) for item in ir.analyses],
        "targets": [dict(item) for item in ir.targets],
        "constraints": dict(ir.constraints),
        "operating_point": dict(ir.operating_point),
        "optimization": dict(ir.optimization),
    }


def _compile_optimization_problem(
    ir: UnifiedIR,
    program: ConstraintProgram,
) -> OptimizationProblem:
    raw_variables = ir.optimization.get("variables", ()) or ()
    if isinstance(raw_variables, Mapping):
        raw_variables = [
            {"variable_id": key, **dict(value)}
            for key, value in raw_variables.items()
            if isinstance(value, Mapping)
        ]
    variables = tuple(_optimization_variable(item) for item in raw_variables)
    objective_ids = tuple(
        item.constraint_id
        for item in program.constraints
        if item.severity.value == "objective"
    ) or ("design.feasibility",)
    return OptimizationProblem(
        problem_id=f"{ir.name}.optimization",
        variables=variables,
        objective_ids=objective_ids,
        constraint_ids=tuple(item.constraint_id for item in program.constraints),
        metadata={
            "source": "PBDL",
            "analysis_count": len(ir.analyses),
            "target_count": len(ir.targets),
            "parameter_ranges": dict(ir.constraints.get("parameter_ranges", {})),
        },
    )


def _optimization_variable(raw: Mapping[str, Any]) -> OptimizationVariable:
    kind = VariableKind(str(raw.get("kind", raw.get("value_type", "continuous"))))
    scale = VariableScale(str(raw.get("scale", "linear")))
    lower = raw.get("lower", raw.get("minimum"))
    upper = raw.get("upper", raw.get("maximum"))
    choices = tuple(raw.get("choices", ()))
    return OptimizationVariable(
        variable_id=str(raw.get("variable_id", raw.get("id", ""))),
        kind=kind,
        unit=str(raw.get("unit", "1")),
        lower=float(lower) if lower is not None else None,
        upper=float(upper) if upper is not None else None,
        choices=choices,
        initial=raw.get("initial"),
        scale=scale,
        expression=(str(raw["expression"]) if raw.get("expression") is not None else None),
        source_path=str(raw.get("source_path", "optimization.variables")),
        attributes=dict(raw.get("attributes", {})),
    )


__all__ = [
    "DESIGN_PROBLEM_SCHEMA",
    "DESIGN_PROBLEM_SCHEMA_VERSION",
    "DesignProblem",
    "DesignProblemCompiler",
    "compile_design_problem",
]
