"""Unified AC/DC candidate evaluation orchestrator.

The first version intentionally delegates numerical work to the mature AC
and power engines.  The orchestration boundary is shared now, so later graph
search and multi-fidelity work can replace either adapter without changing
PBDL, metric, constraint, or replay consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping

from .capabilities import build_default_registry
from .constraints import ConstraintReport, DeclarativeConstraintValidator
from .design_problem import DesignProblem
from .evidence import (
    EpisodeCandidate,
    episode_candidate,
    manifest_hash,
    spec_hash_for,
    write_episode_artifact,
)
from .pbdl_boundary import translate_pbdl_dict
from .pipeline import PowerDesignResult, design_from_pbdl
from .selection import CandidateDecision, rank_candidates
from .simulation import SimulationRequest, SimulationResult, SimulationStatus
from .simulation_tasks import (
    SimulationTask,
    SimulationTaskEvaluation,
    evaluate_simulation_task,
)
from .spec import SynthesisSpec
from .synthesis import CircuitSynthesizer, SynthesisResult, write_results


@dataclass(frozen=True)
class CandidateEvaluation:
    """Backend-neutral evaluation record for one design candidate."""

    candidate_id: str
    domain: str
    name: str
    candidate: Any
    graph: Any
    parameters: Mapping[str, Any]
    validation: ConstraintReport
    task_evaluations: tuple[SimulationTaskEvaluation, ...] = ()
    simulation_requests: tuple[SimulationRequest, ...] = ()
    simulation_results: tuple[SimulationResult, ...] = ()
    objectives: Mapping[str, float] = field(default_factory=dict)
    selection_score: float = 0.0
    selection_rank: CandidateDecision | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", dict(self.objectives or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def passed(self) -> bool:
        return (
            self.validation.passed
            and all(item.passed for item in self.task_evaluations)
            and all(item.status is SimulationStatus.PASSED for item in self.simulation_results)
        )

    @property
    def graph_hash(self) -> str | None:
        return getattr(self.graph, "graph_hash", None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "domain": self.domain,
            "name": self.name,
            "graph_hash": self.graph_hash,
            "parameters": dict(self.parameters),
            "validation": self.validation.as_dict(),
            "task_evaluations": [item.as_dict() for item in self.task_evaluations],
            "simulation_requests": [item.as_dict() for item in self.simulation_requests],
            "simulation_results": [item.as_dict() for item in self.simulation_results],
            "objectives": dict(self.objectives),
            "selection_score": self.selection_score,
            "selection_rank": (
                self.selection_rank.as_dict() if self.selection_rank is not None else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DesignOrchestrationResult:
    problem: DesignProblem
    native_result: Any
    evaluations: tuple[CandidateEvaluation, ...]
    selected: CandidateEvaluation

    @property
    def succeeded(self) -> bool:
        return self.selected.passed

    @property
    def selected_native(self) -> Any:
        return self.selected.candidate

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem.problem_id,
            "stage_index": self.problem.stage_index,
            "succeeded": self.succeeded,
            "selected_candidate_id": self.selected.candidate_id,
            "candidates": [item.as_dict() for item in self.evaluations],
        }


class DesignOrchestrator:
    """Run a compiled design problem through the appropriate backend adapter."""

    def __init__(
        self,
        *,
        synthesizer: CircuitSynthesizer | None = None,
        capability_registry=None,
    ) -> None:
        self.synthesizer = synthesizer or CircuitSynthesizer(
            capability_registry=capability_registry
        )
        self.capability_registry = capability_registry

    def run(
        self,
        problem: DesignProblem,
        *,
        stage_index: int | None = None,
        translated_spec: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        top_k: int | None = None,
        replay_ranker: Mapping[str, Any] | None = None,
    ) -> DesignOrchestrationResult:
        stage = problem.stage(stage_index) if stage_index is not None else problem
        if stage.intent_kind == "dc":
            return self._run_dc(
                stage,
                output_dir=output_dir,
                replay_ranker=replay_ranker,
            )
        return self._run_ac(
            stage,
            translated_spec=translated_spec,
            output_dir=output_dir,
            top_k=top_k,
            replay_ranker=replay_ranker,
        )

    def _run_ac(
        self,
        problem: DesignProblem,
        *,
        translated_spec: Mapping[str, Any] | None,
        output_dir: str | Path | None,
        top_k: int | None,
        replay_ranker: Mapping[str, Any] | None,
    ) -> DesignOrchestrationResult:
        spec_data = dict(translated_spec or translate_pbdl_dict(dict(problem.source_data)))
        spec = SynthesisSpec.from_dict(spec_data)
        if top_k is not None:
            spec = replace(spec, optimization=replace(spec.optimization, top_k=top_k))
        native_results = self.synthesizer.synthesize(spec)
        if not native_results:
            raise ValueError("AC backend returned no candidates")

        task = problem.tasks[0] if problem.tasks else None
        validator = DeclarativeConstraintValidator()
        evaluations: list[CandidateEvaluation] = []
        for index, result in enumerate(native_results, start=1):
            graph = result.circuit_graph()
            validation = validator.validate(
                problem.ir,
                _GraphCandidate(graph),
                result.simulation_results,
                parameter_values=result.parameters,
            )
            task_evaluations = (
                (evaluate_simulation_task(task, result.simulation_results),)
                if task is not None
                else ()
            )
            objectives = {
                **dict(result.metrics.objectives),
                **dict(validation.objective_values),
                "ac.score": float(result.metrics.score),
            }
            candidate_id = f"ac:{index:04d}:{result.template.name}"
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate_id,
                    domain="ac",
                    name=result.template.name,
                    candidate=result,
                    graph=graph,
                    parameters=result.parameters,
                    validation=validation,
                    task_evaluations=task_evaluations,
                    simulation_requests=result.simulation_requests,
                    simulation_results=result.simulation_results,
                    objectives=objectives,
                    selection_score=float(
                        result.metrics.selection_score
                        if result.metrics.selection_score is not None
                        else result.metrics.score
                    ),
                    metadata={
                        "backend": "CircuitSynthesizer",
                        "spec_name": spec.name,
                        "behavior_kind": str(spec.behavior.get("kind", "")),
                    },
                )
            )

        evaluations = _rank_evaluations(evaluations)
        evaluations = _apply_replay_ranker(evaluations, replay_ranker, problem)
        ordered_native = tuple(
            replace(item.candidate, constraint_report=item.validation)
            for item in evaluations
        )
        by_id = {item.candidate_id: native for item, native in zip(evaluations, ordered_native)}
        evaluations = [
            replace(item, candidate=by_id[item.candidate_id]) for item in evaluations
        ]
        selected = evaluations[0]
        native_result: Any = ordered_native
        if output_dir is not None:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            write_results(list(ordered_native), output)
            _write_common_artifacts(output, problem, evaluations)
            _write_unified_replay(output, problem, evaluations)
            _write_unified_episode(
                output,
                problem,
                evaluations,
                environment_manifest_hash=manifest_hash(
                    (self.capability_registry or build_default_registry()).manifest()
                ),
                random_seed=int(spec.optimization.seed),
            )
        return DesignOrchestrationResult(
            problem=problem,
            native_result=native_result,
            evaluations=tuple(evaluations),
            selected=selected,
        )

    def _run_dc(
        self,
        problem: DesignProblem,
        *,
        output_dir: str | Path | None,
        replay_ranker: Mapping[str, Any] | None,
    ) -> DesignOrchestrationResult:
        native = design_from_pbdl(
            dict(problem.source_data),
            output_dir,
            capability_registry=self.capability_registry,
            episode_spec_hash=spec_hash_for(problem.source_data),
        )
        evaluations = tuple(_power_evaluation(item) for item in native.evaluations)
        if not evaluations:
            raise ValueError("DC backend returned no candidates")
        evaluations = tuple(_apply_replay_ranker(list(evaluations), replay_ranker, problem))
        if replay_ranker and replay_ranker.get("enabled"):
            selected = evaluations[0]
        else:
            selected_id = _native_selected_power_id(native, evaluations)
            selected = next(item for item in evaluations if item.candidate_id == selected_id)
        if output_dir is not None:
            _write_common_artifacts(Path(output_dir), problem, evaluations)
            _write_unified_replay(Path(output_dir), problem, evaluations)
            # The power pipeline owns episode.jsonl for this stage: it holds
            # the native graph-search lineage, which the evaluated candidates
            # alone cannot reconstruct.
        return DesignOrchestrationResult(
            problem=problem,
            native_result=native,
            evaluations=evaluations,
            selected=selected,
        )


@dataclass(frozen=True)
class _GraphCandidate:
    graph: Any


def _power_evaluation(item: Any) -> CandidateEvaluation:
    optimization = item.optimization
    candidate = optimization.candidate
    graph = candidate.graph
    objectives = {
        "dc.electrical_fit": float(optimization.objective),
        **dict(item.validation.objective_values),
    }
    return CandidateEvaluation(
        candidate_id=f"dc:{candidate.name}:{graph.graph_hash}",
        domain="dc",
        name=candidate.name,
        candidate=candidate,
        graph=graph,
        parameters=optimization.parameters.as_dict(),
        validation=item.validation,
        task_evaluations=tuple(item.task_evaluations),
        simulation_requests=tuple(item.simulation_requests),
        simulation_results=tuple(item.simulation_results),
        objectives=objectives,
        selection_score=float(item.selection_score),
        selection_rank=item.selection_rank,
        metadata={"backend": "PowerPipeline"},
    )


def _native_selected_power_id(
    native: PowerDesignResult,
    evaluations: tuple[CandidateEvaluation, ...],
) -> str:
    graph_hash = native.optimization.candidate.graph.graph_hash
    for item in evaluations:
        if item.graph_hash == graph_hash:
            return item.candidate_id
    return evaluations[0].candidate_id


def _rank_evaluations(
    evaluations: list[CandidateEvaluation],
) -> list[CandidateEvaluation]:
    decisions = rank_candidates(
        [item.validation for item in evaluations],
        [item.objectives for item in evaluations],
        [item.selection_score for item in evaluations],
    )
    ranked = [replace(item, selection_rank=decision) for item, decision in zip(evaluations, decisions)]
    ranked.sort(
        key=lambda item: (
            item.selection_rank.sort_key if item.selection_rank is not None else (float("inf"),),
            item.name,
            item.candidate_id,
        )
    )
    return ranked


def _apply_replay_ranker(
    evaluations: list[CandidateEvaluation],
    replay_ranker: Mapping[str, Any] | None,
    problem: DesignProblem,
) -> list[CandidateEvaluation]:
    """Apply an optional ML score after hard feasibility and Pareto ranking.

    This is an explicit experiment hook.  The learned score cannot change a
    candidate's feasibility tier, hard violation, or Pareto front; it only
    breaks ties within the deterministic selection prefix.
    """
    if not replay_ranker or not replay_ranker.get("enabled"):
        return evaluations
    from .ranking import score_replay_candidate_ranker
    from .replay import candidate_evaluation_to_replay_row

    scored: list[tuple[CandidateEvaluation, float]] = []
    for evaluation in evaluations:
        row = candidate_evaluation_to_replay_row(evaluation, problem=problem)
        score = score_replay_candidate_ranker(dict(replay_ranker), row)
        scored.append((evaluation, score))

    enriched = [
        replace(
            evaluation,
            metadata={
                **dict(evaluation.metadata),
                "replay_ranker_enabled": True,
                "replay_rank_score": float(score),
            },
        )
        for evaluation, score in scored
    ]
    score_by_id = {
        evaluation.candidate_id: float(score)
        for evaluation, score in scored
    }

    def sort_key(evaluation: CandidateEvaluation) -> tuple[Any, ...]:
        decision = evaluation.selection_rank
        if decision is None:
            prefix = (
                evaluation.validation.feasibility_rank,
                evaluation.validation.unavailable_hard_count,
                evaluation.validation.hard_violation,
                0,
            )
            suffix: tuple[Any, ...] = (
                evaluation.validation.soft_penalty,
                evaluation.selection_score,
            )
        else:
            selection_key = decision.sort_key
            prefix = selection_key[:4]
            suffix = selection_key[4:]
        return (*prefix, -score_by_id[evaluation.candidate_id], *suffix, evaluation.name, evaluation.candidate_id)

    return sorted(enriched, key=sort_key)


def _write_common_artifacts(
    output: Path,
    problem: DesignProblem,
    evaluations: tuple[CandidateEvaluation, ...] | list[CandidateEvaluation],
) -> None:
    (output / "design_problem.json").write_text(
        json.dumps(problem.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "candidate_evaluations.json").write_text(
        json.dumps(
            {"candidates": [item.as_dict() for item in evaluations]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_unified_replay(
    output: Path,
    problem: DesignProblem,
    evaluations: tuple[CandidateEvaluation, ...] | list[CandidateEvaluation],
) -> None:
    from .replay import export_candidate_evaluation_replay

    export_candidate_evaluation_replay(
        evaluations,
        output / "design_replay.jsonl",
        problem=problem,
        append=False,
    )


def _write_unified_episode(
    output: Path,
    problem: DesignProblem,
    evaluations: tuple[CandidateEvaluation, ...] | list[CandidateEvaluation],
    *,
    environment_manifest_hash: str,
    random_seed: int | None = None,
) -> None:
    """Write the DesignEpisode stream beside the unified replay stream."""

    write_episode_artifact(
        output / "episode.jsonl",
        _episode_candidates(evaluations),
        spec_hash=spec_hash_for(problem.source_data),
        environment_manifest_hash=environment_manifest_hash,
        random_seed=random_seed,
    )


def _episode_candidates(
    evaluations: tuple[CandidateEvaluation, ...] | list[CandidateEvaluation],
) -> tuple[EpisodeCandidate, ...]:
    """Turn evaluated candidates into episode inputs, keeping known lineage."""

    candidates: list[EpisodeCandidate] = []
    for item in evaluations:
        metadata = dict(item.metadata)
        provenance = {
            key: value
            for key, value in {
                "backend": metadata.get("backend"),
                "spec_name": metadata.get("spec_name"),
                "origin": metadata.get("origin"),
            }.items()
            if value
        }
        candidates.append(
            episode_candidate(
                item.candidate_id,
                item.graph,
                validation=item.validation,
                task_evaluations=item.task_evaluations,
                simulation_results=item.simulation_results,
                cost={**dict(item.objectives), "selection_score": float(item.selection_score)},
                provenance=provenance,
                records=tuple(metadata.get("derivation") or ()),
            )
        )
    return tuple(candidates)


__all__ = [
    "CandidateEvaluation",
    "DesignOrchestrationResult",
    "DesignOrchestrator",
]
