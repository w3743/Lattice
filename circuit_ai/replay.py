from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .formatting import db20
from .graph_learning import graph_to_record, load_graph_rows, train_graph_classifier, write_jsonl
from .graph_templates import GraphCircuitTemplate
from .learned import HoldoutSplit, save_model
from .quality import train_graph_quality_model
from .ranking import train_replay_pairwise_ranker
from .spec import SynthesisSpec
from .synthesis import SynthesisResult


@dataclass(frozen=True)
class ReplayExportSummary:
    rows_written: int
    accepted_rows: int
    rejected_rows: int
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_written": self.rows_written,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "path": self.path,
        }


def candidate_evaluation_to_replay_row(
    evaluation: Any,
    *,
    problem: Any | None = None,
) -> dict[str, Any]:
    """Serialize one backend-neutral CandidateEvaluation for replay."""
    validation = getattr(evaluation, "validation", None)
    graph = getattr(evaluation, "graph", None)
    graph_payload = graph.as_dict() if graph is not None and hasattr(graph, "as_dict") else None
    feasibility = getattr(getattr(validation, "feasibility", None), "value", "indeterminate")
    simulation_results = tuple(getattr(evaluation, "simulation_results", ()) or ())
    task_evaluations = tuple(getattr(evaluation, "task_evaluations", ()) or ())
    passed = bool(getattr(evaluation, "passed", False))
    execution_failed = any(
        getattr(getattr(item, "status", None), "value", None)
        in {"failed", "unavailable", "timeout"}
        for item in simulation_results
    )
    evidence_status = "verified" if passed else (
        "execution_failed" if execution_failed or feasibility == "execution_failed" else feasibility
    )
    problem_id = str(getattr(problem, "problem_id", "") or "")
    metadata = dict(getattr(evaluation, "metadata", {}) or {})
    behavior_kind = str(metadata.get("behavior_kind", "")) or _problem_behavior_kind(problem)
    component_count = len(getattr(graph, "components", ()) or ())
    score = _finite_replay_number(getattr(evaluation, "selection_score", 0.0))
    label = {
        "accepted": passed,
        "score": score,
        "spec_name": problem_id or str(metadata.get("spec_name", "")),
        "behavior_kind": behavior_kind,
        "component_count": component_count,
        "feasibility": feasibility,
        "feasibility_rank": int(getattr(validation, "feasibility_rank", 2)),
        "evidence_status": evidence_status,
        "execution_failed": execution_failed,
    }
    if validation is not None:
        label.update(
            {
                "hard_violation": _finite_replay_number(getattr(validation, "hard_violation", 0.0)),
                "soft_penalty": _finite_replay_number(getattr(validation, "soft_penalty", 0.0)),
            }
        )
    row = {
        "schema": "circuit_ai.design_replay",
        "schema_version": 1,
        "kind": "candidate_evaluation",
        "problem_id": problem_id,
        "domain": str(getattr(evaluation, "domain", "unknown")),
        "candidate_id": str(getattr(evaluation, "candidate_id", "")),
        "name": str(getattr(evaluation, "name", "")),
        "component_count": component_count,
        "graph": graph_payload,
        "graph_hash": getattr(graph, "graph_hash", None),
        "topology_hash": getattr(graph, "topology_hash", None),
        "parameters": _json_safe(getattr(evaluation, "parameters", {})),
        "objectives": _json_safe(getattr(evaluation, "objectives", {})),
        "selection_score": score,
        "selection_rank": (
            evaluation.selection_rank.as_dict()
            if getattr(evaluation, "selection_rank", None) is not None
            else None
        ),
        "validation": validation.as_dict() if validation is not None else None,
        "simulation_requests": [
            item.as_dict() for item in getattr(evaluation, "simulation_requests", ())
        ],
        "simulation_results": [item.as_dict() for item in simulation_results],
        "simulation_task_evaluations": [item.as_dict() for item in task_evaluations],
        "metadata": _json_safe(metadata),
        "label": label,
    }
    candidate = getattr(evaluation, "candidate", None)
    target = getattr(candidate, "target", None)
    response = getattr(candidate, "response", None)
    template = getattr(candidate, "template", None)
    if target is not None and response is not None:
        frequencies = np.asarray(getattr(target, "frequencies_hz", ()), dtype=float)
        response_array = np.asarray(response)
        target_values = np.asarray(getattr(target, "values", ()))
        if frequencies.size and response_array.size == frequencies.size:
            row.update(
                {
                    "template": str(getattr(template, "name", row["name"])),
                    "component_count": int(getattr(template, "component_count", 0)),
                    "frequency_hz": frequencies.tolist(),
                    "magnitude_db": db20(response_array).tolist(),
                    "phase_deg": np.rad2deg(np.unwrap(np.angle(response_array))).tolist(),
                    "target": {
                        "frequency_hz": frequencies.tolist(),
                        "magnitude_db": db20(target_values).tolist(),
                        "phase_deg": np.rad2deg(np.unwrap(np.angle(target_values))).tolist(),
                        "has_phase": bool(getattr(target, "has_phase", True)),
                        "analysis": getattr(
                            getattr(target, "analysis", None), "as_dict", lambda: {}
                        )(),
                    },
                }
            )
            row["error_features"] = replay_error_features_from_arrays(
                np.asarray(row["magnitude_db"]),
                np.asarray(row["phase_deg"]),
                np.asarray(row["target"]["magnitude_db"]),
                np.asarray(row["target"]["phase_deg"]),
                has_phase=bool(row["target"]["has_phase"]),
            )
            if isinstance(template, GraphCircuitTemplate):
                row["graph_record"] = graph_to_record(template)
    return row


def export_candidate_evaluation_replay(
    evaluations: Sequence[Any],
    path: str | Path,
    *,
    problem: Any | None = None,
    append: bool = False,
) -> ReplayExportSummary:
    """Write AC/DC evaluations to the unified design replay stream."""
    rows = [candidate_evaluation_to_replay_row(item, problem=problem) for item in evaluations]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    accepted = sum(bool(row["label"]["accepted"]) for row in rows)
    return ReplayExportSummary(
        rows_written=len(rows),
        accepted_rows=accepted,
        rejected_rows=len(rows) - accepted,
        path=str(path),
    )


def result_to_graph_replay_row(
    result: SynthesisResult,
    spec: SynthesisSpec,
    accept_rmse_db: float = 1.0,
    accept_max_abs_db: float = 6.0,
) -> dict[str, Any] | None:
    if not isinstance(result.template, GraphCircuitTemplate):
        return None

    simulation_ok = all(
        item.status.value == "passed"
        for item in result.simulation_results
    ) if result.simulation_results else True
    constraint_ok = result.constraint_report is None or result.constraint_report.passed
    accepted = (
        result.metrics.rmse_db <= accept_rmse_db
        and result.metrics.max_abs_db <= accept_max_abs_db
        and np.all(np.isfinite(result.response))
        and simulation_ok
        and constraint_ok
    )
    response_db = db20(result.response)
    response_phase_deg = np.rad2deg(np.unwrap(np.angle(result.response)))
    target_db = db20(result.target.values)
    target_phase_deg = np.rad2deg(np.unwrap(np.angle(result.target.values)))
    error_features = replay_error_features_from_arrays(
        response_db,
        response_phase_deg,
        target_db,
        target_phase_deg,
        has_phase=result.target.has_phase,
    )
    row = {
        "template": result.template.name,
        "graph": graph_to_record(result.template),
        "component_count": result.template.component_count,
        "parameters": result.parameters,
        "frequency_hz": result.target.frequencies_hz.tolist(),
        "magnitude_db": response_db.tolist(),
        "phase_deg": response_phase_deg.tolist(),
        "target": {
            "frequency_hz": result.target.frequencies_hz.tolist(),
            "magnitude_db": target_db.tolist(),
            "phase_deg": target_phase_deg.tolist(),
            "has_phase": result.target.has_phase,
            "analysis": result.target.analysis.as_dict(),
        },
        "error_features": error_features,
        "label": {
            "accepted": bool(accepted),
            "score": result.metrics.score,
            "rmse_db": result.metrics.rmse_db,
            "max_abs_db": result.metrics.max_abs_db,
            "phase_rmse_deg": result.metrics.phase_rmse_deg,
            "spec_name": spec.name,
            "behavior_kind": str(spec.behavior.get("kind", "")),
            "evidence_status": (
                "verified" if accepted else
                "execution_failed" if not simulation_ok else
                "known_infeasible" if constraint_ok is False else
                "indeterminate"
            ),
        },
    }
    if result.metrics.robustness is not None:
        row["label"]["robustness"] = result.metrics.robustness.as_dict()
    if result.metrics.differentiable is not None:
        row["label"]["differentiable"] = result.metrics.differentiable.as_dict()
    return row


def replay_error_features_from_arrays(
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    target_magnitude_db: np.ndarray,
    target_phase_deg: np.ndarray,
    has_phase: bool,
) -> dict[str, float | bool]:
    db_error = np.asarray(magnitude_db, dtype=float) - np.asarray(target_magnitude_db, dtype=float)
    features: dict[str, float | bool] = {
        "has_phase": bool(has_phase),
        "rmse_db": float(np.sqrt(np.mean(db_error**2))),
        "max_abs_db": float(np.max(np.abs(db_error))),
        "mean_db_error": float(np.mean(db_error)),
        "std_db_error": float(np.std(db_error)),
        "p95_abs_db": float(np.percentile(np.abs(db_error), 95)),
    }
    if has_phase:
        phase_error = np.asarray(phase_deg, dtype=float) - np.asarray(target_phase_deg, dtype=float)
        phase_error = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(phase_error))))
        features.update(
            {
                "phase_rmse_deg": float(np.sqrt(np.mean(phase_error**2))),
                "phase_max_abs_deg": float(np.max(np.abs(phase_error))),
            }
        )
    else:
        features.update({"phase_rmse_deg": 0.0, "phase_max_abs_deg": 0.0})
    return features


def attach_replay_error_features(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("target")
    if not isinstance(target, dict):
        return row
    new_row = dict(row)
    new_row["error_features"] = replay_error_features_from_arrays(
        np.asarray(new_row["magnitude_db"], dtype=float),
        np.asarray(new_row.get("phase_deg", [0.0] * len(new_row["magnitude_db"])), dtype=float),
        np.asarray(target["magnitude_db"], dtype=float),
        np.asarray(target.get("phase_deg", [0.0] * len(target["magnitude_db"])), dtype=float),
        has_phase=bool(target.get("has_phase", True)),
    )
    return new_row


def export_graph_replay_rows(
    results: list[SynthesisResult],
    spec: SynthesisSpec,
    path: str | Path,
    append: bool = True,
    include_rejected: bool = True,
    accept_rmse_db: float = 1.0,
    accept_max_abs_db: float = 6.0,
) -> ReplayExportSummary:
    rows = []
    accepted = 0
    rejected = 0
    for result in results:
        row = result_to_graph_replay_row(result, spec, accept_rmse_db, accept_max_abs_db)
        if row is None:
            continue
        if row["label"]["accepted"]:
            accepted += 1
        else:
            rejected += 1
            if not include_rejected:
                continue
        rows.append(row)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    return ReplayExportSummary(
        rows_written=len(rows),
        accepted_rows=accepted,
        rejected_rows=rejected if include_rejected else 0,
        path=str(path),
    )


def load_training_rows_from_replay(path: str | Path, accepted_only: bool = True) -> list[dict[str, Any]]:
    rows = load_graph_rows(path)
    if not accepted_only:
        return rows
    return [row for row in rows if row.get("label", {}).get("accepted", True)]


def summarize_replay_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    accepted = 0
    rejected = 0
    unlabeled = 0
    for row in rows:
        label = row.get("label")
        if not isinstance(label, dict) or "accepted" not in label:
            unlabeled += 1
        elif bool(label["accepted"]):
            accepted += 1
        else:
            rejected += 1
    return {
        "total": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "unlabeled": unlabeled,
    }


def resample_rows_to_common_grid(
    rows: list[dict[str, Any]],
    points: int = 96,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    mins = [min(row["frequency_hz"]) for row in rows]
    maxs = [max(row["frequency_hz"]) for row in rows]
    lo = float(f_min_hz if f_min_hz is not None else max(mins))
    hi = float(f_max_hz if f_max_hz is not None else min(maxs))
    if lo <= 0 or hi <= lo:
        raise ValueError("replay rows do not share a valid overlapping positive frequency range")

    common = np.logspace(np.log10(lo), np.log10(hi), int(points))
    resampled: list[dict[str, Any]] = []
    for row in rows:
        src_f = np.asarray(row["frequency_hz"], dtype=float)
        log_src = np.log10(src_f)
        log_dst = np.log10(common)
        phase = row.get("phase_deg", [0.0] * len(src_f))
        new_row = dict(row)
        new_row["frequency_hz"] = common.tolist()
        new_row["magnitude_db"] = np.interp(
            log_dst,
            log_src,
            np.asarray(row["magnitude_db"], dtype=float),
        ).tolist()
        new_row["phase_deg"] = np.interp(log_dst, log_src, np.asarray(phase, dtype=float)).tolist()
        target = row.get("target")
        if isinstance(target, dict):
            target_freq = np.asarray(target.get("frequency_hz", row["frequency_hz"]), dtype=float)
            target_log_src = np.log10(target_freq)
            target_phase = target.get("phase_deg", [0.0] * len(target_freq))
            new_target = dict(target)
            new_target["frequency_hz"] = common.tolist()
            new_target["magnitude_db"] = np.interp(
                log_dst,
                target_log_src,
                np.asarray(target["magnitude_db"], dtype=float),
            ).tolist()
            new_target["phase_deg"] = np.interp(
                log_dst,
                target_log_src,
                np.asarray(target_phase, dtype=float),
            ).tolist()
            new_row["target"] = new_target
            new_row = attach_replay_error_features(new_row)
        resampled.append(new_row)
    return resampled


def merge_jsonl(inputs: list[str | Path], output: str | Path, dedupe: bool = True) -> int:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in inputs:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = json.dumps(
                    {
                        "template": row.get("template"),
                        "parameters": row.get("parameters"),
                        "frequency_hz": row.get("frequency_hz"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if dedupe and key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    write_jsonl(rows, output)
    return len(rows)


def train_from_replay(
    replay_path: str | Path,
    model_path: str | Path,
    accepted_only: bool = True,
    resample_points: int | None = 96,
    hidden_layer_sizes: tuple[int, ...] = (96, 48),
    max_iter: int = 1000,
    seed: int = 59,
    use_phase: bool = True,
    validation_fraction: float = 0.2,
    top_k: int = 3,
    use_rejected_as_classifier_classes: bool = False,
    quality_rerank_weight: float = 0.35,
) -> dict[str, Any]:
    raw_rows = load_graph_rows(replay_path)
    label_summary = summarize_replay_labels(raw_rows)
    classifier_accepted_only = accepted_only or not use_rejected_as_classifier_classes
    rows = load_training_rows_from_replay(replay_path, accepted_only=classifier_accepted_only)
    if not rows:
        raise ValueError("no replay rows available for training")
    classifier_rows = [row for row in rows if _is_graph_classifier_row(row)]
    if resample_points is not None and classifier_rows:
        classifier_rows = resample_rows_to_common_grid(classifier_rows, points=resample_points)
    classifier_rows = [_classifier_row(row) for row in classifier_rows]
    bundle = (
        train_graph_classifier(
            classifier_rows,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            seed=seed,
            use_phase=use_phase,
            validation_fraction=validation_fraction,
            top_k=top_k,
        )
        if classifier_rows
        else _disabled_graph_classifier("no spectral graph rows available")
    )
    quality_model = train_graph_quality_model(
        raw_rows,
        validation_fraction=validation_fraction,
        seed=seed + 101,
    )
    ranking_model = train_replay_pairwise_ranker(
        raw_rows,
        validation_fraction=validation_fraction,
        seed=seed + 211,
    )
    bundle["quality_model"] = quality_model
    bundle["ranking_model"] = ranking_model
    bundle["quality_rerank_weight"] = float(quality_rerank_weight)
    bundle["training_source"] = {
        "kind": "replay",
        "path": str(replay_path),
        "accepted_only": classifier_accepted_only,
        "classifier_accepted_only": classifier_accepted_only,
        "resample_points": resample_points,
        "rows": len(classifier_rows),
        "raw_rows": label_summary["total"],
        "label_summary": label_summary,
        "quality_model_rows": quality_model.get("rows", 0),
        "quality_model_enabled": bool(quality_model.get("enabled", False)),
        "ranking_model_pairs": ranking_model.get("pairs", 0),
        "ranking_model_enabled": bool(ranking_model.get("enabled", False)),
        "rejected_rows_used_as_positive_classes": (
            not classifier_accepted_only and label_summary["rejected"] > 0
        ),
        "holdout": {
            "classifier": _holdout_fields(bundle),
            "quality_model": _holdout_fields(quality_model),
            "ranking_model": _holdout_fields(ranking_model),
        },
    }
    save_model(bundle, model_path)
    return bundle


def _holdout_fields(model: dict[str, Any]) -> dict[str, Any]:
    """Copy the recorded holdout contract of a trained model bundle."""
    return {
        "split_strategy": model.get("split_strategy"),
        "group_key": model.get("group_key"),
        "train_groups": list(model.get("train_groups", ())),
        "validation_groups": list(model.get("validation_groups", ())),
    }


def _is_graph_classifier_row(row: dict[str, Any]) -> bool:
    graph = row.get("graph_record", row.get("graph"))
    return bool(
        row.get("template")
        and isinstance(graph, dict)
        and "slots" in graph
        and row.get("frequency_hz")
        and isinstance(row.get("target"), dict)
    )


def _classifier_row(row: dict[str, Any]) -> dict[str, Any]:
    graph = row.get("graph_record")
    if not isinstance(graph, dict):
        return row
    copied = dict(row)
    copied["graph"] = graph
    return copied


def _disabled_graph_classifier(message: str) -> dict[str, Any]:
    return {
        "model": None,
        "frequency_hz": [],
        "use_phase": True,
        "classes": [],
        "model_kind": "graph_topology_proposer",
        "graph_catalog": {},
        "training_rows": 0,
        "validation": {
            "validation_fraction": 0.0,
            "validation_rows": 0,
            "validation_accuracy": None,
            "validation_top_k_accuracy": None,
        },
        "validation_accuracy": None,
        "message": message,
        **HoldoutSplit.all_training().as_dict(),
    }


def _finite_replay_number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def _problem_behavior_kind(problem: Any | None) -> str:
    source = getattr(problem, "source_data", {})
    if not isinstance(source, dict):
        return ""
    behavior = source.get("behavior")
    if isinstance(behavior, dict) and behavior.get("kind"):
        return str(behavior["kind"])
    targets = source.get("targets", [])
    if isinstance(targets, list) and targets and isinstance(targets[0], dict):
        for key in ("filter_kind", "target_kind"):
            if targets[0].get(key):
                return str(targets[0][key])
    analyses = source.get("analyses", [])
    if isinstance(analyses, list) and analyses and isinstance(analyses[0], dict):
        return str(analyses[0].get("kind", ""))
    return ""


def _json_safe(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value
