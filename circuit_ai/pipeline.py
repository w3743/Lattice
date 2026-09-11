"""End-to-end PBDL design pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .capabilities import (
    CapabilityGap,
    CapabilityPlanError,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResolution,
    CapabilityResolutionError,
    CapabilityResolver,
    CapabilityRole,
    build_default_registry,
)
from .constraints import ConstraintReport, FeasibilityStatus
from .evidence import (
    EpisodeCandidate,
    candidate_accepted,
    episode_candidate,
    manifest_hash,
    spec_hash_for,
    write_episode_artifact,
)
from .experts import ExpertTopologySelector, TopologyCandidate, UnsupportedTopology
from .ir import UnifiedIR, pbdl_to_ir
from .optimization import (
    FidelityDisagreement,
    attach_disagreement_diagnostics,
    compare_fidelity_results,
    disagreement_from_options,
)
from .selection import CandidateDecision, rank_candidates
from .simulation import (
    NgspiceSimulatorBackend,
    SimulationExecutor,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    simulation_requests_for_tasks,
)
from .simulation_tasks import (
    SimulationTask,
    SimulationTaskEvaluation,
    evaluate_simulation_task,
    simulation_tasks_from_ir,
)
from .topology_grammar import PowerTopologyGrammar, TopologySearchCertificate


@dataclass(frozen=True)
class PowerDesignResult:
    ir: UnifiedIR
    optimization: Any
    validation: ConstraintReport
    kicad_schematic: Path | None = None
    kicad_project: Path | None = None
    graph_svg: Path | None = None
    graph_kicad_schematic: Path | None = None
    graph_export_manifest: Path | None = None
    evaluations: tuple[PowerCandidateEvaluation, ...] = ()
    search_certificate: TopologySearchCertificate | None = None
    simulation_tasks: tuple[SimulationTask, ...] = ()
    simulation_evaluations: tuple[SimulationTaskEvaluation, ...] = ()
    simulation_requests: tuple[SimulationRequest, ...] = ()
    simulation_results: tuple[SimulationResult, ...] = ()
    capability_resolutions: tuple[CapabilityResolution, ...] = ()
    capability_gaps: tuple[CapabilityGap, ...] = ()
    fidelity_comparisons: tuple[FidelityDisagreement, ...] = ()

    @property
    def succeeded(self) -> bool:
        return (
            self.validation.passed
            and all(item.passed for item in self.simulation_evaluations)
            and all(item.succeeded for item in self.simulation_results)
        )

    @property
    def model_disagreements(self) -> tuple[FidelityDisagreement, ...]:
        """Comparisons whose backends breached the agreement threshold."""

        return tuple(item for item in self.fidelity_comparisons if item.disagreed)


@dataclass(frozen=True)
class PowerCandidateEvaluation:
    candidate: TopologyCandidate
    optimization: Any
    validation: ConstraintReport
    task_evaluations: tuple[SimulationTaskEvaluation, ...]
    simulation_requests: tuple[SimulationRequest, ...]
    simulation_results: tuple[SimulationResult, ...]
    selection_score: float
    capability_resolutions: tuple[CapabilityResolution, ...]
    selection_rank: CandidateDecision | None = None
    fidelity_comparisons: tuple[FidelityDisagreement, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        measurements = _simulation_measurements(self.simulation_results)
        return {
            "candidate": self.optimization.candidate.as_dict(),
            "parameters": self.optimization.parameters.as_dict(),
            "operating_point": measurements,
            "validation": self.validation.as_dict(),
            "simulation_requests": [item.as_dict() for item in self.simulation_requests],
            "simulation_results": [item.as_dict() for item in self.simulation_results],
            "simulation_task_evaluations": [item.as_dict() for item in self.task_evaluations],
            "fidelity_comparisons": [item.as_dict() for item in self.fidelity_comparisons],
            "objective": self.optimization.objective,
            "optimization_run": (
                self.optimization.optimization_run.as_dict()
                if self.optimization.optimization_run is not None
                else None
            ),
            "selection_score": self.selection_score,
            "selection_rank": self.selection_rank.as_dict() if self.selection_rank is not None else None,
            "optimizer_message": self.optimization.optimizer_message,
            "capabilities": [item.as_dict() for item in self.capability_resolutions],
        }


def design_from_pbdl(
    data: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    capability_registry: CapabilityRegistry | None = None,
    episode_spec_hash: str | None = None,
) -> PowerDesignResult:
    ir = pbdl_to_ir(data)
    simulation_tasks = simulation_tasks_from_ir(ir)
    registry = capability_registry or build_default_registry()
    resolver = CapabilityResolver(registry)
    catalog_candidates: tuple[TopologyCandidate, ...] = ()
    try:
        catalog_candidates = (ExpertTopologySelector().select(ir),)
    except UnsupportedTopology:
        pass
    search = PowerTopologyGrammar().search(ir, additional_candidates=catalog_candidates)
    if not search.candidates:
        reasons = "; ".join(item.reason for item in search.certificate.rejected)
        raise UnsupportedTopology(
            "no executable topology exists inside the declared search bounds"
            + (f": {reasons}" if reasons else "")
        )

    evaluations: list[PowerCandidateEvaluation] = []
    capability_gaps: list[CapabilityGap] = []
    for candidate in search.candidates:
        planning_requests = simulation_requests_for_tasks(
            simulation_tasks,
            candidate.graph,
            fidelity="ideal_averaged",
        )
        try:
            optimizer_resolution = resolver.resolve(
                _capability_request(
                    candidate,
                    CapabilityRole.PARAMETER_OPTIMIZER,
                    planning_requests,
                )
            )
            validator_resolution = resolver.resolve(
                _capability_request(
                    candidate,
                    CapabilityRole.CONSTRAINT_VALIDATOR,
                    planning_requests,
                )
            )
            simulator_resolution = resolver.resolve(
                _capability_request(
                    candidate,
                    CapabilityRole.SIMULATION_BACKEND,
                    planning_requests,
                )
            )
        except CapabilityResolutionError as exc:
            capability_gaps.append(exc.gap)
            continue
        optimizer = optimizer_resolution.selected.implementation
        candidate_optimization = optimizer.optimize(ir, candidate)
        optimized_graph = candidate_optimization.candidate.graph
        request_groups = tuple(
            task.to_requests(
                optimized_graph,
                parameter_values=candidate_optimization.parameters.as_dict(),
                fidelity=simulator_resolution.selected.target.fidelity,
            )
            for task in simulation_tasks
        )
        simulation_requests = tuple(
            request for group in request_groups for request in group
        )
        simulator = simulator_resolution.selected.implementation
        simulation_results = SimulationExecutor().execute(
            simulator,
            optimized_graph,
            simulation_requests,
        )
        validator = validator_resolution.selected.implementation
        candidate_validation = validator.validate(
            ir,
            candidate_optimization.candidate,
            simulation_results,
        )
        ideal_simulation_results = simulation_results
        high_fidelity_requests, high_fidelity_results = _optional_power_truth(
            ir,
            optimized_graph,
            simulation_requests,
            simulation_executor=SimulationExecutor(),
        )
        if high_fidelity_results:
            # Plan §7.3 rule 4: the averaged model and the SPICE gate are two
            # backends on one candidate, so their difference is the
            # model_disagreement signal.  Pair them before the planning results
            # are concatenated with the truth results.  The comparison record is
            # always kept; the policy only decides whether a breach becomes a
            # diagnostic.
            truth_pairs = _pair_truth_with_planning(
                simulation_requests,
                simulation_results,
                high_fidelity_requests,
                high_fidelity_results,
            )
            disagreement_enabled, disagreement_threshold = disagreement_from_options(
                dict(ir.optimization.get("fidelity_disagreement", {}) or {})
            )
            fidelity_comparisons = tuple(
                compare_fidelity_results(planning, truth, threshold=disagreement_threshold)
                for planning, truth in truth_pairs
            )
            simulation_requests = simulation_requests + high_fidelity_requests
            simulation_results = simulation_results + high_fidelity_results
            if disagreement_enabled and fidelity_comparisons:
                simulation_results = tuple(
                    attach_disagreement_diagnostics(simulation_results, fidelity_comparisons)
                )
            if any(item.status is not SimulationStatus.PASSED for item in high_fidelity_results):
                candidate_validation = replace(
                    candidate_validation,
                    feasibility=FeasibilityStatus.EXECUTION_FAILED,
                    diagnostics=(
                        *candidate_validation.diagnostics,
                        "requested high-fidelity power verification did not pass",
                    ),
                )
        else:
            fidelity_comparisons = ()
        task_evaluations = _evaluate_task_request_groups(
            simulation_tasks,
            request_groups,
            ideal_simulation_results,
        )
        score = _power_tie_breaker(
            candidate_optimization,
            candidate_validation,
            ir,
        )
        evaluations.append(
            PowerCandidateEvaluation(
                candidate=candidate,
                optimization=candidate_optimization,
                validation=candidate_validation,
                task_evaluations=task_evaluations,
                simulation_requests=simulation_requests,
                simulation_results=simulation_results,
                selection_score=score,
                capability_resolutions=(
                    optimizer_resolution,
                    simulator_resolution,
                    validator_resolution,
                ),
                fidelity_comparisons=fidelity_comparisons,
            )
        )
    if not evaluations:
        if output_dir is not None:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "ir.json", ir.as_dict())
            _write_json(output / "topology_search.json", search.certificate.as_dict())
            _write_json(output / "capability_manifest.json", registry.manifest())
            _write_json(
                output / "capability_gaps.json",
                {"gaps": [gap.as_dict() for gap in capability_gaps]},
            )
        raise CapabilityPlanError(capability_gaps)
    evaluations = _rank_power_evaluations(evaluations, ir)
    evaluations.sort(
        key=lambda item: (
            item.selection_rank.sort_key if item.selection_rank is not None else (float("inf"),),
            item.candidate.name,
        )
    )
    selected = evaluations[0]
    optimization = selected.optimization
    validation = selected.validation
    selected_resolutions = list(selected.capability_resolutions)
    schematic = None
    project = None
    graph_svg = None
    graph_kicad_schematic = None
    graph_export_manifest = None
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        renderer_resolution = resolver.resolve(
            _capability_request(
                selected.candidate,
                CapabilityRole.SCHEMATIC_RENDERER,
                selected.simulation_requests,
                export_format="svg",
            )
        )
        exporter_resolution = resolver.resolve(
            _capability_request(
                selected.candidate,
                CapabilityRole.PROJECT_EXPORTER,
                selected.simulation_requests,
                export_format="kicad",
            )
        )
        selected_resolutions.extend((renderer_resolution, exporter_resolution))
        _write_json(output / "ir.json", ir.as_dict())
        _write_json(output / "topology.json", optimization.candidate.as_dict())
        _write_json(output / "parameters.json", optimization.parameters.as_dict())
        _write_json(output / "dc_solution.json", _simulation_measurements(selected.simulation_results))
        _write_json(output / "transient.json", optimization.transient.as_dict())
        _write_json(output / "validation.json", validation.as_dict())
        _write_json(
            output / "simulation_tasks.json",
            {
                "tasks": [task.as_dict() for task in simulation_tasks],
                "requests": [item.as_dict() for item in selected.simulation_requests],
                "results": [item.as_dict() for item in selected.simulation_results],
                "selected_evaluations": [item.as_dict() for item in selected.task_evaluations],
            },
        )
        _write_json(output / "topology_search.json", search.certificate.as_dict())
        _write_json(output / "capability_manifest.json", registry.manifest())
        _write_json(
            output / "capability_gaps.json",
            {"gaps": [gap.as_dict() for gap in capability_gaps]},
        )
        renderer = renderer_resolution.selected.implementation
        (output / "best.svg").write_text(renderer.render(optimization), encoding="utf-8")
        _write_json(
            output / "report.json",
            {
                "design": "dc_power",
                "topology": optimization.candidate.as_dict(),
                "parameters": optimization.parameters.as_dict(),
                "operating_point": _simulation_measurements(selected.simulation_results),
                "validation": validation.as_dict(),
                "simulation_tasks": [task.as_dict() for task in simulation_tasks],
                "simulation_requests": [item.as_dict() for item in selected.simulation_requests],
                "simulation_results": [item.as_dict() for item in selected.simulation_results],
                "simulation_task_evaluations": [
                    item.as_dict() for item in selected.task_evaluations
                ],
                "optimizer_message": optimization.optimizer_message,
                "objective": optimization.objective,
                "optimization_run": (
                    optimization.optimization_run.as_dict()
                    if optimization.optimization_run is not None
                    else None
                ),
                "selection_score": selected.selection_score,
                "topology_search": search.certificate.as_dict(),
                "fidelity_comparisons": [
                    item.as_dict() for item in selected.fidelity_comparisons
                ],
                "operating_envelope": (
                    dict(ir.operating_envelope)
                    if ir.operating_envelope is not None
                    else None
                ),
                "worst_case": (
                    optimization.envelope_analysis.as_dict()
                    if getattr(optimization, "envelope_analysis", None) is not None
                    else None
                ),
                "load_step": (
                    optimization.load_step.as_dict()
                    if getattr(optimization, "load_step", None) is not None
                    else None
                ),
                "candidates": [item.as_dict() for item in evaluations],
                "capabilities": {
                    "selected": [item.as_dict() for item in selected_resolutions],
                    "gaps": [gap.as_dict() for gap in capability_gaps],
                },
            },
        )
        _write_replay(output / "replay.jsonl", ir, evaluations)
        write_episode_artifact(
            output / "episode.jsonl",
            _episode_candidates(evaluations),
            spec_hash=episode_spec_hash or spec_hash_for(data),
            environment_manifest_hash=manifest_hash(registry.manifest()),
            certificate=_native_search_certificate(search.certificate),
            budget=_search_budget(ir),
            random_seed=_configured_seed(ir),
        )
        exporter = exporter_resolution.selected.implementation
        schematic, project = exporter.export(optimization, output, project_name="best")
        from .graph_exports import export_circuit_graph

        graph = selected.optimization.candidate.graph
        graph_request = selected.simulation_requests[0] if selected.simulation_requests else None
        try:
            graph_bundle = export_circuit_graph(
                graph,
                selected.optimization.parameters.as_dict(),
                formats=("svg", "kicad", "spice"),
                spice_request=graph_request,
                title=selected.candidate.name,
            )
            if graph_bundle.svg is not None:
                graph_svg = output / "circuit_graph.svg"
                graph_svg.write_text(graph_bundle.svg, encoding="utf-8")
            if graph_bundle.kicad_schematic is not None:
                graph_kicad_schematic = output / "circuit_graph.kicad_sch"
                graph_kicad_schematic.write_text(
                    graph_bundle.kicad_schematic,
                    encoding="utf-8",
                )
            graph_export_manifest = output / "circuit_graph_export.json"
            _write_json(graph_export_manifest, graph_bundle.as_dict())
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            report["circuit_graph_export"] = graph_bundle.as_dict()
            _write_json(output / "report.json", report)
        except (ValueError, TypeError, KeyError) as exc:
            graph_export_manifest = output / "circuit_graph_export.json"
            _write_json(
                graph_export_manifest,
                {
                    "graph_hash": getattr(graph, "graph_hash", None),
                    "topology_hash": getattr(graph, "topology_hash", None),
                    "diagnostics": [
                        f"unified graph export failed: {type(exc).__name__}: {exc}"
                    ],
                },
            )
    return PowerDesignResult(
        ir=ir,
        optimization=optimization,
        validation=validation,
        kicad_schematic=schematic,
        kicad_project=project,
        graph_svg=graph_svg,
        graph_kicad_schematic=graph_kicad_schematic,
        graph_export_manifest=graph_export_manifest,
        evaluations=tuple(evaluations),
        search_certificate=search.certificate,
        simulation_tasks=simulation_tasks,
        simulation_evaluations=selected.task_evaluations,
        simulation_requests=selected.simulation_requests,
        simulation_results=selected.simulation_results,
        capability_resolutions=tuple(selected_resolutions),
        capability_gaps=tuple(capability_gaps),
        fidelity_comparisons=tuple(selected.fidelity_comparisons),
    )


def _capability_request(
    candidate: TopologyCandidate,
    role: CapabilityRole,
    simulation_requests: tuple[SimulationRequest, ...],
    *,
    export_format: str | None = None,
) -> CapabilityRequest:
    return CapabilityRequest(
        role=role,
        family=candidate.family,
        solver_id=str(candidate.solver_id or candidate.metadata.get("solver") or "") or None,
        analysis_kinds=frozenset(request.analysis.kind for request in simulation_requests),
        model_kinds=frozenset(component.kind for component in candidate.components),
        export_format=export_format,
    )


def _pair_truth_with_planning(
    planning_requests: tuple[SimulationRequest, ...],
    planning_results: tuple[SimulationResult, ...],
    truth_requests: tuple[SimulationRequest, ...],
    truth_results: tuple[SimulationResult, ...],
) -> tuple[tuple[SimulationResult, SimulationResult], ...]:
    """Pair every truth result with the planning result it re-evaluates.

    ``_optional_power_truth`` derives each truth request id from its planning
    request, so the request id -- not tuple position -- is the declared
    correspondence between the two fidelities.  The truth result must carry
    that same id: a result is bound to its request, so trusting slot order
    would silently compare the wrong pair if the two tuples ever disagree.
    A truth result that did not pass carries no comparable numbers and is
    skipped rather than compared against a valid planning result.
    """

    planning_by_id = {result.request_id: result for result in planning_results}
    truth_by_request_id = {result.request_id: result for result in truth_results}
    pairs: list[tuple[SimulationResult, SimulationResult]] = []
    for truth_request in truth_requests:
        truth_result = truth_by_request_id.get(truth_request.request_id)
        if truth_result is None or truth_result.status is not SimulationStatus.PASSED:
            continue
        base_id = truth_request.request_id.removesuffix("__spice_truth")
        planning_result = planning_by_id.get(base_id)
        if planning_result is None or planning_result.status is not SimulationStatus.PASSED:
            continue
        pairs.append((planning_result, truth_result))
    return tuple(pairs)


def _optional_power_truth(
    ir: UnifiedIR,
    graph,
    requests: tuple[SimulationRequest, ...],
    *,
    simulation_executor: SimulationExecutor,
) -> tuple[tuple[SimulationRequest, ...], tuple[SimulationResult, ...]]:
    """Run optional external truth evidence for a power candidate.

    Ideal averaged simulation remains the planning model.  When explicitly
    requested, every candidate receives an independent SPICE gate and an
    unavailable or unsupported result is preserved as evidence rather than
    being converted into a verified result.
    """

    options = dict(ir.optimization.get("spice_verification", {}) or {})
    if not bool(options.get("enabled", False)):
        return (), ()
    backend = NgspiceSimulatorBackend(
        executable=str(options.get("executable", "ngspice")),
        timeout_s=float(options.get("timeout_s", 30.0)),
        backend_version=str(options.get("backend_version", "unknown")),
    )
    truth_requests = tuple(
        replace(
            request,
            request_id=f"{request.request_id}__spice_truth",
            fidelity="spice",
            metadata={**dict(request.metadata), "truth_backend": "ngspice"},
        )
        for request in requests
    )
    truth_results = simulation_executor.execute(backend, graph, truth_requests)
    return truth_requests, truth_results


def _power_tie_breaker(
    optimization,
    validation: Any,
    ir: UnifiedIR,
) -> float:
    design_count = sum(
        component.attributes.get("role") != "external_load"
        for component in optimization.candidate.components
    )
    weights = dict(ir.optimization.get("weights", {}))
    complexity_weight = float(weights.get("component_count", 0.03))
    cost_weight = float(weights.get("cost", 0.0))
    area_weight = float(weights.get("area_mm2", 0.0))
    return float(
        optimization.objective
        + complexity_weight * design_count
        + cost_weight * validation.cost_cny
        + area_weight * validation.area_mm2
        + validation.soft_penalty
    )


def _rank_power_evaluations(
    evaluations: list[PowerCandidateEvaluation],
    ir: UnifiedIR,
) -> list[PowerCandidateEvaluation]:
    objective_rows: list[dict[str, float]] = []
    for item in evaluations:
        design_count = sum(
            component.attributes.get("role") != "external_load"
            for component in item.optimization.candidate.components
        )
        row = {
            "electrical_fit": float(item.optimization.objective),
            "component_count": float(design_count),
            "bom_cost": float(item.validation.cost_cny),
            "area": float(item.validation.area_mm2),
            "soft_penalty": float(item.validation.soft_penalty),
            **{str(key): float(value) for key, value in item.validation.objective_values.items()},
        }
        objective_rows.append(row)
    decisions = rank_candidates(
        [item.validation for item in evaluations],
        objective_rows,
        [item.selection_score for item in evaluations],
    )
    return [
        replace(item, selection_rank=decision)
        for item, decision in zip(evaluations, decisions)
    ]


def _write_replay(
    path: Path,
    ir: UnifiedIR,
    evaluations: list[PowerCandidateEvaluation],
) -> None:
    rows = []
    for item in evaluations:
        task_issues = [issue for task in item.task_evaluations for issue in task.issues]
        rows.append(
            {
                "kind": "verified_power_synthesis",
                "ir": ir.as_dict(),
                "candidate": item.optimization.candidate.as_dict(),
                "parameters": item.optimization.parameters.as_dict(),
                "operating_point": _simulation_measurements(item.simulation_results),
                "simulation_requests": [
                    request.as_dict() for request in item.simulation_requests
                ],
                "simulation_results": [
                    result.as_dict() for result in item.simulation_results
                ],
                "simulation_task_evaluations": [
                    task.as_dict() for task in item.task_evaluations
                ],
                "optimization_run": (
                    item.optimization.optimization_run.as_dict()
                    if item.optimization.optimization_run is not None
                    else None
                ),
                "capabilities": [
                    resolution.selected.target.as_dict()
                    for resolution in item.capability_resolutions
                ],
                "label": {
                    "accepted": candidate_accepted(
                        item.validation,
                        task_evaluations=item.task_evaluations,
                        simulation_results=item.simulation_results,
                    ),
                    "objective": item.optimization.objective,
                    "selection_score": item.selection_score,
                    "issues": [*item.validation.issues, *task_issues],
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode_candidates(
    evaluations: list[PowerCandidateEvaluation],
) -> tuple[EpisodeCandidate, ...]:
    """Turn evaluated power candidates into episode inputs with their lineage."""

    return tuple(
        episode_candidate(
            f"dc:{item.optimization.candidate.name}:{item.optimization.candidate.graph.graph_hash}",
            item.optimization.candidate.graph,
            validation=item.validation,
            task_evaluations=item.task_evaluations,
            simulation_results=item.simulation_results,
            cost={
                "dc.electrical_fit": float(item.optimization.objective),
                **dict(item.validation.objective_values),
                "selection_score": float(item.selection_score),
            },
            provenance=_episode_provenance(item),
            records=tuple(dict(item.optimization.candidate.metadata).get("derivation") or ()),
        )
        for item in evaluations
    )


def _episode_provenance(item: PowerCandidateEvaluation) -> dict[str, Any]:
    candidate = item.optimization.candidate
    metadata = dict(candidate.metadata)
    derivation = [
        str(step) for step in (metadata.get("derivation") or ()) if isinstance(step, str)
    ]
    provenance = {
        "origin": str(metadata.get("origin", candidate.family or "")),
        "family": str(candidate.family or ""),
        "solver_id": str(candidate.solver_id or metadata.get("solver", "")),
        "capabilities": [
            resolution.selected.target.as_dict()
            for resolution in item.capability_resolutions
        ],
    }
    if derivation:
        provenance["derivation"] = derivation
    return provenance


def _native_search_certificate(
    certificate: TopologySearchCertificate,
) -> dict[str, Any] | None:
    """The embedded native graph-search certificate, when native search ran."""

    native = certificate.native_search
    if not isinstance(native, dict):
        return None
    search = native.get("search")
    return search if isinstance(search, dict) else None


def _search_budget(ir: UnifiedIR) -> dict[str, float]:
    """Search bounds the run was allowed to spend, as episode budget evidence."""

    options = dict(ir.optimization.get("graph_search", {}) or {})
    budget: dict[str, float] = {}
    for key in ("max_expansions", "beam_width", "max_candidates", "max_components", "max_nodes", "max_depth"):
        value = options.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            budget[key] = float(value)
    return budget


def _configured_seed(ir: UnifiedIR) -> int | None:
    seed = ir.optimization.get("seed")
    return int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _simulation_measurements(results: tuple[SimulationResult, ...]) -> dict[str, Any]:
    measurements: dict[str, Any] = {}
    for result in results:
        measurements.update(
            {
                str(key): getattr(quantity, "value", quantity)
                for key, quantity in result.scalars.items()
            }
        )
    return measurements


def _evaluate_task_request_groups(
    tasks: tuple[SimulationTask, ...],
    request_groups: tuple[tuple[SimulationRequest, ...], ...],
    results: tuple[SimulationResult, ...],
) -> tuple[SimulationTaskEvaluation, ...]:
    evaluations: list[SimulationTaskEvaluation] = []
    offset = 0
    for task, requests in zip(tasks, request_groups):
        group_results = results[offset : offset + len(requests)]
        offset += len(requests)
        evaluations.append(evaluate_simulation_task(task, group_results))
    if offset != len(results):
        raise ValueError("simulation result count does not match task request groups")
    return tuple(evaluations)
