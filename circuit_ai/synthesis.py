from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import json
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from .capabilities import (
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResolver,
    CapabilityRole,
    build_default_registry,
)
from .constraints import (
    ConstraintEvaluator,
    FeasibilityStatus,
    ConstraintOperator,
    ConstraintReport,
    ConstraintSeverity,
    ConstraintSpec,
)
from .catalog import topology_record_by_name
from .differentiable import DifferentiableRefinementResult, refine_graph_parameters
from .discretization import DiscretizationResult, discretize_template_candidates
from .feasibility import FeasibilityReport, enforce_feasibility
from .formatting import db20
from .optimizers import DifferentialEvolutionParameterOptimizer, ParameterOptimizer
from .optimization import (
    FidelityRole,
    FidelitySchedule,
    FidelityScheduleRun,
    FidelityScheduler,
    FidelityStage,
    OptimizationRunResult,
    default_ac_fidelity_schedule,
)
from .pareto import metrics_to_objectives, pareto_points
from .metrics import MetricContext, MetricEngine, MetricSpec
from .proposers import HeuristicProposer, TopologyProposer
from .robustness import RobustnessMetrics, analyze_robustness, robustness_penalty
from .spec import SynthesisSpec
from .targets import TargetResponse, frequency_grid, target_from_behavior
from .templates import CircuitTemplate
from .simulation import (
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    SimulationExecutor,
    legacy_ac_simulation_request,
)

if TYPE_CHECKING:
    from .graph import CircuitGraph


@dataclass(frozen=True)
class SynthesisMetrics:
    score: float
    rmse_db: float
    max_abs_db: float
    phase_rmse_deg: float | None
    component_count: int
    optimizer_success: bool
    optimizer_message: str
    estimated_cost: float = 0.0
    estimated_area_mm2: float = 0.0
    topology_risk: float = 1.0
    robustness: RobustnessMetrics | None = None
    differentiable: DifferentiableRefinementResult | None = None
    objectives: dict[str, float] = field(default_factory=dict)
    pareto_rank: int | None = None
    pareto_crowding: float | None = None
    selection_score: float | None = None
    surrogate: dict[str, Any] | None = None
    optimization_run: OptimizationRunResult | None = None


@dataclass(frozen=True)
class SynthesisResult:
    template: CircuitTemplate
    parameters: dict[str, float]
    metrics: SynthesisMetrics
    target: TargetResponse
    response: np.ndarray
    feasibility: FeasibilityReport | None = None
    selected_real_components: tuple[str, ...] = ()
    model_bindings: tuple[dict[str, Any], ...] = ()
    simulation_requests: tuple[SimulationRequest, ...] = ()
    simulation_results: tuple[SimulationResult, ...] = ()
    simulation_capabilities: tuple[dict[str, Any], ...] = ()
    constraint_report: ConstraintReport | None = None
    fidelity_schedule: FidelityScheduleRun | None = None
    discretization: DiscretizationResult | None = None

    def netlist(self, title: str | None = None) -> str:
        return self.template.netlist(self.parameters, title=title or self.template.name)

    def circuit_graph(self) -> "CircuitGraph":
        graph = self.template.to_graph(self.parameters)
        if self.discretization is not None:
            from .graph import Rating

            components = list(graph.components)
            assigned: set[str] = set()
            for selection in self.discretization.selections:
                if selection.part is None or selection.selected_value is None:
                    continue
                prefix = selection.variable_id.split(".", 1)[0]
                index = next(
                    (
                        index
                        for index, component in enumerate(components)
                        if component.instance_id == prefix
                        or component.reference == prefix
                    ),
                    None,
                )
                if index is None:
                    index = next(
                        (
                            index
                            for index, component in enumerate(components)
                            if index not in assigned
                            and component.model.kind.casefold() == selection.element_type.casefold()
                        ),
                        None,
                    )
                if index is None:
                    continue
                assigned.add(index)
                component = components[index]
                part = dict(selection.part)
                attributes = {
                    **dict(component.attributes),
                    "selected_part_id": selection.part_id,
                    "selected_part": part,
                    "selected_value": selection.selected_value,
                }
                ratings = list(component.ratings)
                raw_ratings = dict(part.get("metadata", {})).get("ratings", ())
                if isinstance(raw_ratings, Mapping):
                    raw_ratings = [
                        {"quantity": quantity, **dict(value)}
                        for quantity, value in raw_ratings.items()
                        if isinstance(value, Mapping)
                    ]
                for raw_rating in raw_ratings or ():
                    if not isinstance(raw_rating, Mapping):
                        continue
                    rating = Rating.from_dict(raw_rating)
                    if rating not in ratings:
                        ratings.append(rating)
                components[index] = replace(
                    component,
                    attributes=attributes,
                    ratings=tuple(ratings),
                )
            graph = replace(graph, components=tuple(components))
            graph = replace(
                graph,
                extensions={
                    **dict(graph.extensions),
                    "discretization": self.discretization.as_dict(),
                },
            )
        return graph

    def as_dict(self) -> dict[str, Any]:
        graph = self.circuit_graph()
        structure = None
        if hasattr(self.template, "structural_report"):
            structure = self.template.structural_report().as_dict()
        topology = topology_record_by_name(self.template.name)
        return {
            "template": self.template.name,
            "description": self.template.description,
            "topology": topology.as_dict() if topology is not None else None,
            "analysis": self.target.analysis.as_dict(),
            "parameters": self.parameters,
            "circuit_graph": graph.as_dict(),
            "document_hash": graph.document_hash,
            "graph_hash": graph.graph_hash,
            "topology_hash": graph.topology_hash,
            "component_inventory": self.template.component_inventory(),
            "feasibility": self.feasibility.as_dict() if self.feasibility is not None else None,
            "structure": structure,
            "metrics": {
                "score": self.metrics.score,
                "rmse_db": self.metrics.rmse_db,
                "max_abs_db": self.metrics.max_abs_db,
                "phase_rmse_deg": self.metrics.phase_rmse_deg,
                "component_count": self.metrics.component_count,
                "estimated_cost": self.metrics.estimated_cost,
                "estimated_area_mm2": self.metrics.estimated_area_mm2,
                "topology_risk": self.metrics.topology_risk,
                "objectives": self.metrics.objectives,
                "pareto_rank": self.metrics.pareto_rank,
                "pareto_crowding": _json_float(self.metrics.pareto_crowding),
                "selection_score": self.metrics.selection_score,
                "surrogate": self.metrics.surrogate,
                "optimizer_success": self.metrics.optimizer_success,
                "optimizer_message": self.metrics.optimizer_message,
                "optimization_run": (
                    self.metrics.optimization_run.as_dict()
                    if self.metrics.optimization_run is not None
                    else None
                ),
                "robustness": (
                    self.metrics.robustness.as_dict() if self.metrics.robustness is not None else None
                ),
                "differentiable": (
                    self.metrics.differentiable.as_dict()
                    if self.metrics.differentiable is not None
                    else None
                ),
            },
            "netlist": self.netlist(),
            "selected_real_components": list(self.selected_real_components),
            "model_bindings": list(self.model_bindings),
            "simulation_requests": [item.as_dict() for item in self.simulation_requests],
            "simulation_results": [item.as_dict() for item in self.simulation_results],
            "simulation_capabilities": list(self.simulation_capabilities),
            "constraint_report": (
                self.constraint_report.as_dict() if self.constraint_report is not None else None
            ),
            "discretization": (
                self.discretization.as_dict() if self.discretization is not None else None
            ),
            "fidelity_schedule": (
                self.fidelity_schedule.as_dict() if self.fidelity_schedule is not None else None
            ),
        }


class CircuitSynthesizer:
    def __init__(
        self,
        proposer: TopologyProposer | None = None,
        optimizer: ParameterOptimizer | None = None,
        capability_registry: CapabilityRegistry | None = None,
    ):
        self.proposer = proposer
        self.optimizer = optimizer or DifferentialEvolutionParameterOptimizer()
        self.capability_registry = capability_registry or build_default_registry()

    def synthesize(self, spec: SynthesisSpec) -> list[SynthesisResult]:
        feasibility = enforce_feasibility(spec)
        freqs = frequency_grid(
            spec.optimization.f_min_hz,
            spec.optimization.f_max_hz,
            spec.optimization.points,
        )
        target = target_from_behavior(spec.behavior, freqs, spec.analysis)
        proposer = self.proposer or _default_proposer(spec)
        candidates = proposer.propose(spec)
        if not candidates:
            raise ValueError("no compatible circuit templates for this behavior and library")

        surrogate_bundle = _load_surrogate_bundle(spec.optimization.surrogate)
        if spec.optimization.fidelity.get("enabled", False) and len(candidates) > 1:
            results, schedule = self._staged_optimize(
                candidates,
                spec,
                target,
                feasibility,
                surrogate_bundle,
            )
            return [
                replace(result, fidelity_schedule=schedule)
                for result in results
            ]
        else:
            results = [
                self._optimize_template(template, spec, target, feasibility, surrogate_bundle)
                for template in candidates
            ]
            results = annotate_pareto(results)
            results.sort(key=_selection_key)
            candidate_ids = tuple(
                f"rank-{index:04d}:{result.template.name}"
                for index, result in enumerate(results, start=1)
            )
            policy = default_ac_fidelity_schedule(spec.optimization.top_k)
            screen_run = FidelityScheduler().schedule(policy, candidate_ids)
            result_by_id = dict(zip(candidate_ids, results))
            truth_results = [
                self._attach_simulation_evidence(
                    result_by_id[candidate_id],
                    spec=spec,
                )
                for candidate_id in screen_run.selected_candidate_ids
            ]
            truth_results.sort(key=_truth_selection_key)
            id_by_name = {
                result.template.name: candidate_id
                for candidate_id, result in zip(candidate_ids, results)
            }
            truth_ids = tuple(
                id_by_name[result.template.name]
                for result in truth_results[: spec.optimization.top_k]
            )
            schedule = FidelityScheduler().schedule(
                policy,
                candidate_ids,
                stage_candidate_ids={"linear_mna_truth": truth_ids},
            )
            selected_ids = set(schedule.selected_candidate_ids)
            return [
                replace(result, fidelity_schedule=schedule)
                for result in truth_results
                if id_by_name[result.template.name] in selected_ids
            ]

    def _staged_optimize(
        self,
        candidates: list[CircuitTemplate],
        spec: SynthesisSpec,
        target: TargetResponse,
        feasibility: FeasibilityReport,
        surrogate_bundle: dict[str, Any] | None,
    ) -> tuple[list[SynthesisResult], FidelityScheduleRun]:
        """Run coarse screening, promotion, and final truth selection."""

        options = dict(spec.optimization.fidelity)
        screen_points = max(
            2,
            min(
                target.frequencies_hz.size,
                int(options.get("screen_points", min(24, target.frequencies_hz.size))),
            ),
        )
        screen_iterations = max(
            0,
            min(
                spec.optimization.max_iterations,
                int(options.get("screen_iterations", min(8, spec.optimization.max_iterations))),
            ),
        )
        promotion_fraction = float(options.get("promotion_fraction", 0.4))
        if not 0.0 < promotion_fraction <= 1.0:
            raise ValueError("optimization.fidelity.promotion_fraction must be in (0, 1]")
        promotion_count = max(
            spec.optimization.top_k,
            min(len(candidates), math.ceil(len(candidates) * promotion_fraction)),
        )

        indices = np.linspace(0, target.frequencies_hz.size - 1, screen_points, dtype=int)
        screen_target = replace(
            target,
            frequencies_hz=target.frequencies_hz[indices],
            values=target.values[indices],
        )
        screen_spec = replace(
            spec,
            optimization=replace(
                spec.optimization,
                max_iterations=screen_iterations,
                optimizer=_without_polish(spec.optimization.optimizer),
            ),
        )
        screen_results = [
            self._optimize_template(
                template,
                screen_spec,
                screen_target,
                feasibility,
                surrogate_bundle,
                postprocess=False,
            )
            for template in candidates
        ]
        screen_results = annotate_pareto(screen_results)
        screen_results.sort(key=_selection_key)
        screen_ids = tuple(
            f"candidate:{result.template.name}" for result in screen_results
        )
        template_by_screen_id = {
            candidate_id: result.template
            for candidate_id, result in zip(screen_ids, screen_results)
        }
        promoted_templates = [template_by_screen_id[item] for item in screen_ids[:promotion_count]]
        full_results = [
            self._optimize_template(
                template,
                spec,
                target,
                feasibility,
                surrogate_bundle,
            )
            for template in promoted_templates
        ]
        full_results = annotate_pareto(full_results)
        full_results.sort(key=_selection_key)
        full_ids = tuple(f"candidate:{result.template.name}" for result in full_results)
        candidate_id_by_name = {
            result.template.name: candidate_id
            for candidate_id, result in zip(full_ids, full_results)
        }

        # Truth is an execution gate, not just an output annotation.  Re-run
        # the promoted candidates through the selected backend, then reorder
        # them from the actual evidence before constructing the final stage.
        truth_results = [
            self._attach_simulation_evidence(result, spec=spec)
            for result in full_results
        ]
        truth_results.sort(key=_truth_selection_key)
        truth_ids = tuple(
            candidate_id_by_name[result.template.name]
            for result in truth_results[: spec.optimization.top_k]
        )
        schedule = FidelityScheduler().schedule(
            FidelitySchedule(
                "ac.staged.v1",
                (
                    FidelityStage(
                        "coarse_parameter_screen",
                        "optimization_model_coarse",
                        FidelityRole.SCREEN,
                    ),
                    FidelityStage(
                        "full_parameter_optimization",
                        "optimization_model",
                        FidelityRole.SCREEN,
                    ),
                    FidelityStage(
                        "linear_mna_truth",
                        "linear_frequency_domain",
                        FidelityRole.TRUTH,
                        spec.optimization.top_k,
                    ),
                ),
            ),
            screen_ids,
            stage_candidate_ids={
                "full_parameter_optimization": tuple(
                    screen_id
                    for result in full_results
                    for screen_id, screened in zip(screen_ids, screen_results)
                    if screened.template.name == result.template.name
                ),
                "linear_mna_truth": tuple(
                    candidate_id_by_name[result.template.name]
                    for result in truth_results[: spec.optimization.top_k]
                ),
            },
        )
        # Return full-fidelity result objects keyed to the truth-stage order.
        final_ids = set(result.template.name for result in truth_results[: spec.optimization.top_k])
        return (
            [result for result in truth_results if result.template.name in final_ids],
            schedule,
        )

    def _attach_simulation_evidence(
        self,
        result: SynthesisResult,
        *,
        spec: SynthesisSpec | None = None,
    ) -> SynthesisResult:
        graph = result.circuit_graph()
        request = legacy_ac_simulation_request(
            graph,
            result.target.analysis,
            result.target.frequencies_hz,
            request_id=f"{result.template.name}__final_ac_verification",
        )
        spice_options = dict(spec.optimization.spice_verification) if spec is not None else {}
        use_spice = bool(spice_options.get("enabled", False))
        if use_spice:
            from .simulation import NgspiceSimulatorBackend

            backend = NgspiceSimulatorBackend(
                executable=str(spice_options.get("executable", "ngspice")),
                timeout_s=float(spice_options.get("timeout_s", 30.0)),
                backend_version=str(spice_options.get("backend_version", "unknown")),
            )
            request = replace(
                request,
                fidelity="spice",
                metadata={
                    **dict(request.metadata),
                    "truth_backend": "ngspice",
                },
            )
            available, availability_reason = backend.available()
            capability = {
                "capability_id": "simulation.ngspice",
                "backend_id": "ngspice",
                "executable": backend.executable,
                "available": available,
                "availability_reason": availability_reason,
            }
        else:
            resolution = CapabilityResolver(self.capability_registry).resolve(
                CapabilityRequest(
                    role=CapabilityRole.SIMULATION_BACKEND,
                    solver_id="linear_mna",
                    analysis_kinds=frozenset({request.analysis.kind}),
                    model_kinds=frozenset(
                        component.model.kind for component in graph.components
                    ),
                )
            )
            backend = resolution.selected.implementation
            capability = resolution.as_dict()
        simulation_result = SimulationExecutor().execute(backend, graph, (request,))[0]
        constraint_report = _ac_constraint_report(result, graph, request, simulation_result)
        return replace(
            result,
            simulation_requests=(request,),
            simulation_results=(simulation_result,),
            simulation_capabilities=(capability,),
            constraint_report=constraint_report,
        )

    def _optimize_template(
        self,
        template: CircuitTemplate,
        spec: SynthesisSpec,
        target: TargetResponse,
        feasibility: FeasibilityReport,
        surrogate_bundle: dict[str, Any] | None = None,
        *,
        postprocess: bool = True,
    ) -> SynthesisResult:
        actual_bounds = template.parameter_bounds(spec.library)

        score_response = lambda response: self._score_response(
            template,
            spec,
            target,
            response,
            True,
            "running",
        ).score
        surrogate_details: dict[str, Any] | None = None
        surrogate_optimizer = getattr(self.optimizer, "optimize_with_response_evaluator", None)
        if (
            surrogate_bundle is not None
            and template.name in surrogate_bundle.get("models", {})
            and callable(surrogate_optimizer)
        ):
            from .surrogate import predict_response

            trust_threshold = float(
                spec.optimization.surrogate.get(
                    "trust_uncertainty_db",
                    surrogate_bundle.get("trust_uncertainty_db", 1.0),
                )
            )
            surrogate_rejected = False
            surrogate_evaluations = 0

            def surrogate_response(parameters: dict[str, float]) -> np.ndarray:
                nonlocal surrogate_rejected, surrogate_evaluations
                surrogate_evaluations += 1
                prediction = predict_response(
                    surrogate_bundle,
                    template.name,
                    parameters,
                    target.frequencies_hz,
                    trust_uncertainty_db=trust_threshold,
                )
                if not prediction.trusted:
                    surrogate_rejected = True
                    return np.full(len(target.frequencies_hz), np.nan + 1j * np.nan, dtype=np.complex128)
                return prediction.response

            opt = surrogate_optimizer(
                template,
                spec,
                target,
                score_response,
                surrogate_response,
            )
            if surrogate_rejected or not np.all(np.isfinite(opt.response)):
                opt = self.optimizer.optimize(template, spec, target, score_response)
                surrogate_details = {
                    "used": True,
                    "fallback_to_truth": True,
                    "trust_uncertainty_db": trust_threshold,
                    "rejected_evaluations": surrogate_evaluations,
                }
            else:
                surrogate_details = {
                    "used": True,
                    "fallback_to_truth": False,
                    "trust_uncertainty_db": trust_threshold,
                    "rejected_evaluations": surrogate_evaluations,
                }
        else:
            opt = self.optimizer.optimize(template, spec, target, score_response)
        params = opt.parameters
        response = template.analyze(params, target.frequencies_hz, target.analysis)
        metrics = self._score_response(
            template,
            spec,
            target,
            response,
            bool(opt.success),
            str(opt.message),
        )
        metrics = replace(metrics, optimization_run=getattr(opt, "optimization_run", None))
        discretization_result: DiscretizationResult | None = None
        if postprocess:
            refined_params, diff_result = refine_graph_parameters(
                template,
                params,
                actual_bounds,
                target,
                spec.optimization.differentiable,
            )
            if diff_result is not None:
                refined_response = template.analyze(refined_params, target.frequencies_hz, target.analysis)
                refined_metrics = self._score_response(
                    template,
                    spec,
                    target,
                    refined_response,
                    bool(opt.success),
                    str(opt.message),
                )
                if refined_metrics.score <= metrics.score:
                    params = refined_params
                    response = refined_response
                    metrics = refined_metrics
                metrics = replace(metrics, differentiable=diff_result)

            discrete_candidates = discretize_template_candidates(
                template,
                params,
                actual_bounds,
                spec.optimization.discretization,
                real_components=spec.library.real_components,
            )
            if discrete_candidates:
                feasible_candidates = [
                    candidate for candidate in discrete_candidates if candidate.feasible
                ]
                if feasible_candidates:
                    scored_discrete: list[tuple[float, DiscretizationResult, np.ndarray, SynthesisMetrics]] = []
                    for candidate in feasible_candidates:
                        candidate_response = template.analyze(
                            dict(candidate.parameter_values),
                            target.frequencies_hz,
                            target.analysis,
                        )
                        candidate_metrics = self._score_response(
                            template,
                            spec,
                            target,
                            candidate_response,
                            bool(opt.success),
                            "discrete component evaluation",
                        )
                        scored_discrete.append(
                            (
                                candidate_metrics.score,
                                candidate,
                                candidate_response,
                                candidate_metrics,
                            )
                        )
                    _, discretization_result, response, discrete_metrics = min(
                        scored_discrete,
                        key=lambda item: (
                            item[0],
                            item[1].source,
                            tuple(sorted(item[1].parameter_values.items())),
                        ),
                    )
                    params = dict(discretization_result.parameter_values)
                    metrics = replace(
                        discrete_metrics,
                        optimization_run=metrics.optimization_run,
                        differentiable=metrics.differentiable,
                    )
                else:
                    discretization_result = discrete_candidates[0]
                    metrics = replace(
                        metrics,
                        score=1e9 + template.component_count,
                        optimizer_success=False,
                        optimizer_message="discrete component selection failed: "
                        + "; ".join(discretization_result.diagnostics),
                    )

            robust = analyze_robustness(
                template,
                params,
                target,
                spec.optimization.robustness,
                seed=spec.optimization.seed + 10_000,
            )
            if robust is not None:
                metrics = replace(
                    metrics,
                    score=metrics.score + robustness_penalty(robust, spec.optimization.robustness),
                    robustness=robust,
                )
        if surrogate_details is not None:
            metrics = replace(metrics, surrogate=surrogate_details)
        return SynthesisResult(
            template, params, metrics, target, response,
            feasibility=feasibility,
            selected_real_components=spec.library.real_components,
            model_bindings=spec.library.model_bindings,
            discretization=discretization_result,
        )

    def _score_response(
        self,
        template: CircuitTemplate,
        spec: SynthesisSpec,
        target: TargetResponse,
        response: np.ndarray,
        optimizer_success: bool,
        optimizer_message: str,
    ) -> SynthesisMetrics:
        estimated_cost, estimated_area_mm2 = _estimate_resources(template, spec)
        topology_risk = _topology_risk(template)
        if not np.all(np.isfinite(response)):
            return SynthesisMetrics(
                score=1e9 + template.component_count,
                rmse_db=1e9,
                max_abs_db=1e9,
                phase_rmse_deg=None,
                component_count=template.component_count,
                optimizer_success=False,
                optimizer_message="non-finite circuit response",
                estimated_cost=estimated_cost,
                estimated_area_mm2=estimated_area_mm2,
                topology_risk=topology_risk,
                robustness=None,
                differentiable=None,
            )

        target_db = db20(target.values)
        response_db = db20(response)
        db_error = response_db - target_db
        rmse_db = float(np.sqrt(np.mean(db_error**2)))
        max_abs_db = float(np.max(np.abs(db_error)))

        phase_rmse_deg = None
        phase_penalty = 0.0
        if target.has_phase:
            phase_error = np.angle(response / np.maximum(np.abs(response), 1e-18)) - np.angle(
                target.values / np.maximum(np.abs(target.values), 1e-18)
            )
            phase_error = np.angle(np.exp(1j * phase_error))
            phase_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
            phase_penalty = phase_rmse_deg / 90.0

        weights = {
            "rmse_db": 1.0,
            "max_abs_db": 0.05,
            "phase": 0.2,
            "component_count": 0.03,
            "cost": 0.0,
            "area_mm2": 0.0,
            "topology_risk": 0.0,
        }
        weights.update(spec.optimization.weights)

        score = (
            weights["rmse_db"] * rmse_db
            + weights["max_abs_db"] * max_abs_db
            + weights["phase"] * phase_penalty
            + weights["component_count"] * template.component_count
            + weights["cost"] * estimated_cost
            + weights["area_mm2"] * estimated_area_mm2
            + weights["topology_risk"] * topology_risk
        )

        return SynthesisMetrics(
            score=float(score),
            rmse_db=rmse_db,
            max_abs_db=max_abs_db,
            phase_rmse_deg=phase_rmse_deg,
            component_count=template.component_count,
            optimizer_success=optimizer_success,
            optimizer_message=optimizer_message,
            estimated_cost=estimated_cost,
            estimated_area_mm2=estimated_area_mm2,
            topology_risk=topology_risk,
            robustness=None,
            differentiable=None,
        )


def write_results(
    results: list[SynthesisResult],
    output_dir: str | Path,
    spice_verifications: list[Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schematic_files: list[str] = []
    kicad_schematic_files: list[str] = []
    best_bound_netlist: str | None = None
    export_records: list[dict[str, Any]] = []
    if results:
        from .schematic import render_result_svg, render_results_index_html
        from .kicad import materialize_model_bindings, render_result_kicad_schematic, render_kicad_project
        from .spice import apply_model_bindings
        from .graph_exports import export_circuit_graph

        for index, result in enumerate(results, start=1):
            filename = f"candidate_{index}.svg"
            graph = result.circuit_graph()
            request = result.simulation_requests[0] if result.simulation_requests else None
            materialized_bindings = materialize_model_bindings(result.model_bindings, output_dir)
            try:
                exports = export_circuit_graph(
                    graph,
                    result.parameters,
                    formats=("svg", "kicad", "spice"),
                    spice_request=request,
                    model_bindings=materialized_bindings,
                    title=result.template.description or result.template.name,
                )
                svg = exports.svg or render_result_svg(result)
                kicad = exports.kicad_schematic
                spice = exports.spice_netlist
                diagnostics = list(exports.diagnostics)
            except (ValueError, TypeError, KeyError) as exc:
                # Preserve legacy output for a non-linear/partially bound
                # template, but make the graph-export gap explicit.
                svg = render_result_svg(result)
                kicad = None
                spice = None
                diagnostics = [f"unified graph export failed: {type(exc).__name__}: {exc}"]
            (output_dir / filename).write_text(svg, encoding="utf-8")
            schematic_files.append(filename)
            kicad_filename = f"candidate_{index}.kicad_sch"
            if kicad is None:
                result_for_kicad = replace(result, model_bindings=materialized_bindings)
                kicad = render_result_kicad_schematic(
                    result_for_kicad,
                    project_name=f"candidate_{index}",
                )
            if index == 1:
                best_bound_netlist = apply_model_bindings(
                    spice or result.netlist("best_candidate"),
                    materialized_bindings,
                )
            (output_dir / kicad_filename).write_text(
                kicad,
                encoding="utf-8",
            )
            kicad_schematic_files.append(kicad_filename)
            export_records.append(
                {
                    "candidate": index,
                    "graph_hash": graph.graph_hash,
                    "topology_hash": graph.topology_hash,
                    "formats": ["svg", "kicad", "spice"],
                    "diagnostics": diagnostics,
                    "spice_request_id": request.request_id if request is not None else None,
                }
            )
        (output_dir / "best.svg").write_text((output_dir / schematic_files[0]).read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "best.kicad_sch").write_text(
            (output_dir / kicad_schematic_files[0]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (output_dir / "best.kicad_pro").write_text(render_kicad_project("best"), encoding="utf-8")
        (output_dir / "schematics.html").write_text(
            render_results_index_html(results, schematic_files),
            encoding="utf-8",
        )

    report = []
    for index, result in enumerate(results):
        entry = result.as_dict()
        if index < len(schematic_files):
            entry["schematic"] = schematic_files[index]
            if index == 0:
                entry["best_schematic"] = "best.svg"
        if index < len(kicad_schematic_files):
            entry["kicad_schematic"] = kicad_schematic_files[index]
            if index == 0:
                entry["best_kicad_schematic"] = "best.kicad_sch"
                entry["best_kicad_project"] = "best.kicad_pro"
        if spice_verifications is not None and index < len(spice_verifications):
            verification = spice_verifications[index]
            entry["spice_verification"] = (
                verification.as_dict() if hasattr(verification, "as_dict") else verification
            )
        report.append(entry)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "export_manifest.json").write_text(
        json.dumps(export_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if results:
        (output_dir / "best.spice").write_text(
            best_bound_netlist or results[0].netlist("best_candidate"),
            encoding="utf-8",
        )


def _ac_constraint_report(
    result: SynthesisResult,
    graph: "CircuitGraph",
    request: SimulationRequest,
    simulation_result: SimulationResult,
) -> ConstraintReport:
    """Evaluate final AC evidence through the shared metric/constraint pipeline."""
    if not request.requested_observables:
        raise ValueError("AC verification request must contain an observable")

    source = request.requested_observables[0].observable_id
    target_magnitude_db = tuple(
        float(value)
        for value in 20.0
        * np.log10(np.maximum(np.abs(result.target.values), 1e-300))
    )
    metric_specs = [
        MetricSpec(
            metric_id="ac.rmse_db",
            provider_id="frequency_response",
            source=source,
            quantity="magnitude_error",
            unit="dB",
            reduction="rmse_db",
            analysis_kind=request.analysis.kind,
            selectors={"target_magnitude_db": target_magnitude_db},
        ),
        MetricSpec(
            metric_id="ac.max_abs_error_db",
            provider_id="frequency_response",
            source=source,
            quantity="magnitude_error",
            unit="dB",
            reduction="max_abs_error_db",
            analysis_kind=request.analysis.kind,
            selectors={"target_magnitude_db": target_magnitude_db},
        ),
    ]
    if result.target.has_phase:
        target_phase_deg = tuple(
            float(value)
            for value in np.degrees(np.unwrap(np.angle(result.target.values)))
        )
        metric_specs.append(
            MetricSpec(
                metric_id="ac.phase_rmse_deg",
                provider_id="frequency_response",
                source=source,
                quantity="phase_error",
                unit="deg",
                reduction="phase_rmse_deg",
                analysis_kind=request.analysis.kind,
                selectors={"target_phase_deg": target_phase_deg},
            )
        )

    constraints = tuple(
        ConstraintSpec(
            constraint_id=f"objective.{metric.metric_id}",
            metric=metric,
            operator=ConstraintOperator.MINIMIZE,
            severity=ConstraintSeverity.OBJECTIVE,
            unit=metric.unit,
            source_path="synthesis.final_ac_verification",
        )
        for metric in metric_specs
    )
    metric_values = MetricEngine().extract(
        metric_specs,
        MetricContext(
            graph=graph,
            simulation_requests=(request,),
            simulation_results=(simulation_result,),
            parameter_values=result.parameters,
        ),
    )
    report = ConstraintEvaluator().evaluate(constraints, metric_values)
    if simulation_result.status is not SimulationStatus.PASSED:
        report = replace(
            report,
            feasibility=FeasibilityStatus.EXECUTION_FAILED,
            diagnostics=(
                *report.diagnostics,
                "truth simulation did not complete successfully: "
                + simulation_result.status.value,
            ),
        )
    return report


def annotate_pareto(results: list[SynthesisResult]) -> list[SynthesisResult]:
    points = pareto_points([metrics_to_objectives(result.metrics) for result in results])
    point_by_index = {point.index: point for point in points}
    annotated: list[SynthesisResult] = []
    for index, result in enumerate(results):
        point = point_by_index[index]
        selection_score = _selection_score(result.metrics.score, point.rank, point.crowding_distance)
        annotated.append(
            replace(
                result,
                metrics=replace(
                    result.metrics,
                    objectives=point.objectives,
                    pareto_rank=point.rank,
                    pareto_crowding=point.crowding_distance,
                    selection_score=selection_score,
                ),
            )
        )
    return annotated


def _load_surrogate_bundle(options: dict[str, Any]) -> dict[str, Any] | None:
    if not options or not bool(options.get("enabled", True)):
        return None
    model_path = options.get("model_path")
    if not model_path:
        raise ValueError("optimization.surrogate.model_path is required when surrogate is enabled")
    from .surrogate import load_response_surrogate

    bundle = load_response_surrogate(model_path)
    if bundle.get("kind") != "validated_response_surrogate":
        raise ValueError("surrogate model has an unsupported kind")
    return bundle


def _default_proposer(spec: SynthesisSpec) -> TopologyProposer:
    """Compose catalog and bounded graph generation from spec policy."""
    heuristic = HeuristicProposer()
    options = spec.optimization.graph_search
    native_requested = bool(options.get("native_enabled", False)) or str(
        options.get("mode", "")
    ).casefold() == "native"
    if not bool(options.get("enabled", False)) and not native_requested:
        return heuristic
    from .topology import GraphSearchProposer

    if native_requested:
        from .topology import NativeGraphSearchProposer

        return NativeGraphSearchProposer(
            fallback=heuristic,
            max_candidates=max(1, int(options.get("max_candidates", 8))),
            include_fallback=bool(options.get("include_fallback", True)),
            beam_width=max(1, int(options.get("beam_width", 24))),
        )

    return GraphSearchProposer(
        fallback=heuristic,
        max_candidates=max(1, int(options.get("max_candidates", 8))),
        internal_nodes=max(0, int(options.get("internal_nodes", 1))),
        include_fallback=bool(options.get("include_fallback", True)),
    )


def _selection_score(score: float, pareto_rank: int, crowding_distance: float) -> float:
    crowding_bonus = 0.0 if np.isinf(crowding_distance) else 1e-6 / (1.0 + max(crowding_distance, 0.0))
    return float(pareto_rank + crowding_bonus + 1e-9 * score)


def _without_polish(options: dict[str, Any]) -> dict[str, Any]:
    """Copy optimizer options with local polish disabled for screening."""

    copied = dict(options)
    nested = copied.get("differential_evolution")
    if isinstance(nested, dict):
        nested_copy = dict(nested)
        nested_copy["polish"] = False
        copied["differential_evolution"] = nested_copy
    else:
        copied["polish"] = False
    return copied


def _selection_key(result: SynthesisResult) -> tuple[float, float, float, int, str]:
    metrics = result.metrics
    rank = metrics.pareto_rank if metrics.pareto_rank is not None else 1_000_000
    crowding = metrics.pareto_crowding if metrics.pareto_crowding is not None else 0.0
    crowding_key = -1e12 if np.isinf(crowding) else -crowding
    return (rank, metrics.score, crowding_key, metrics.component_count, result.template.name)


def _truth_selection_key(result: SynthesisResult) -> tuple[int, float, float, int, str]:
    """Order promoted candidates from executed truth evidence.

    The status tier is deliberately lexicographic and cannot be offset by a
    weighted objective: verified feasibility wins first, then known
    infeasibility, missing/uncertain evidence, and execution failure.
    """

    simulation_failed = any(
        item.status is not SimulationStatus.PASSED
        for item in result.simulation_results
    )
    report = result.constraint_report
    if simulation_failed:
        tier = 3
    elif report is None:
        tier = 2
    else:
        tier = {
            FeasibilityStatus.VERIFIED_FEASIBLE: 0,
            FeasibilityStatus.KNOWN_INFEASIBLE: 1,
            FeasibilityStatus.INDETERMINATE: 2,
            FeasibilityStatus.EXECUTION_FAILED: 3,
        }.get(report.feasibility, 2)
    objective_penalty = (
        float(sum(report.objective_values.values()))
        if report is not None and report.objective_values
        else float(result.metrics.score)
    )
    return (
        tier,
        objective_penalty,
        float(result.metrics.score),
        result.metrics.component_count,
        result.template.name,
    )


def _json_float(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _estimate_resources(template: CircuitTemplate, spec: SynthesisSpec) -> tuple[float, float]:
    cost = 0.0
    area = 0.0
    for element_type, count in template.component_inventory().items():
        cost += spec.library.cost_for(element_type) * count
        area += spec.library.area_for(element_type) * count
    return float(cost), float(area)


def _topology_risk(template: CircuitTemplate) -> float:
    record = topology_record_by_name(template.name)
    if record is not None:
        return float(record.topology_risk)
    if template.name.startswith("graph_"):
        return 0.65
    return 1.0
