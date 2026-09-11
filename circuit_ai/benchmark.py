from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .graph_learning import SklearnGraphProposer
from .graph_templates import GraphCircuitTemplate
from .proposers import HeuristicProposer, TopologyProposer
from .spec import SynthesisSpec
from .synthesis import CircuitSynthesizer
from .topology import GraphSearchProposer


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    spec: SynthesisSpec


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    behavior_kind: str
    status: str
    success: bool
    template: str | None
    is_graph: bool
    component_count: int | None
    score: float | None
    pareto_rank: int | None
    rmse_db: float | None
    max_abs_db: float | None
    phase_rmse_deg: float | None
    elapsed_s: float
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "behavior_kind": self.behavior_kind,
            "status": self.status,
            "success": self.success,
            "template": self.template,
            "is_graph": self.is_graph,
            "component_count": self.component_count,
            "score": self.score,
            "pareto_rank": self.pareto_rank,
            "rmse_db": self.rmse_db,
            "max_abs_db": self.max_abs_db,
            "phase_rmse_deg": self.phase_rmse_deg,
            "elapsed_s": self.elapsed_s,
            "message": self.message,
        }


def generate_benchmark_cases(
    count_per_kind: int = 2,
    seed: int = 101,
    points: int = 64,
    max_iterations: int = 30,
    max_components: int = 5,
) -> list[BenchmarkCase]:
    rng = np.random.default_rng(seed)
    cases: list[BenchmarkCase] = []
    library = {
        "allowed": ["R", "C", "L", "opamp"],
        "parameter_ranges": {
            "R": [10, 2_000_000],
            "C": [1e-11, 1e-4],
            "L": [1e-6, 2.0],
        },
    }

    for index in range(count_per_kind):
        cutoff = float(10 ** rng.uniform(2.3, 4.7))
        cases.append(
            _case_from_dict(
                f"bench_lowpass_{index + 1}",
                {
                    "kind": "lowpass",
                    "cutoff_hz": cutoff,
                    "gain": 1.0,
                    "order": 1,
                    "frequency_range_hz": [cutoff / 100.0, cutoff * 100.0],
                },
                library,
                points,
                max_iterations,
                max_components,
                seed + index,
            )
        )

    for index in range(count_per_kind):
        cutoff = float(10 ** rng.uniform(2.3, 4.7))
        cases.append(
            _case_from_dict(
                f"bench_highpass_{index + 1}",
                {
                    "kind": "highpass",
                    "cutoff_hz": cutoff,
                    "gain": 1.0,
                    "order": 1,
                    "frequency_range_hz": [cutoff / 100.0, cutoff * 100.0],
                },
                library,
                points,
                max_iterations,
                max_components,
                seed + 100 + index,
            )
        )

    for index in range(count_per_kind):
        center = float(10 ** rng.uniform(3.0, 4.7))
        q = float(rng.uniform(2.0, 8.0))
        cases.append(
            _case_from_dict(
                f"bench_bandpass_{index + 1}",
                {
                    "kind": "bandpass",
                    "center_hz": center,
                    "q": q,
                    "gain": 1.0,
                    "frequency_range_hz": [center / 100.0, center * 100.0],
                },
                library,
                points,
                max_iterations,
                max_components,
                seed + 200 + index,
            )
        )

    return cases


def _case_from_dict(
    name: str,
    behavior: dict[str, Any],
    library: dict[str, Any],
    points: int,
    max_iterations: int,
    max_components: int,
    seed: int,
) -> BenchmarkCase:
    spec = SynthesisSpec.from_dict(
        {
            "name": name,
            "ports": 2,
            "behavior": behavior,
            "library": library,
            "optimization": {
                "points": points,
                "top_k": 1,
                "max_iterations": max_iterations,
                "max_components": max_components,
                "seed": seed,
            },
        }
    )
    return BenchmarkCase(name=name, spec=spec)


def load_cases(paths: list[str | Path]) -> list[BenchmarkCase]:
    cases = []
    for path in paths:
        spec = SynthesisSpec.from_json_file(path)
        cases.append(BenchmarkCase(name=spec.name, spec=spec))
    return cases


def build_benchmark_proposer(
    kind: str,
    graph_model: str | Path | None = None,
    graph_model_only: bool = False,
    graph_candidates: int = 32,
    internal_nodes: int = 1,
) -> TopologyProposer:
    kind = kind.strip().lower()
    proposer: TopologyProposer = HeuristicProposer()
    if kind == "heuristic":
        return proposer
    if kind == "graph-model":
        if graph_model is None:
            raise ValueError("graph-model proposer requires --graph-model")
        proposer = SklearnGraphProposer(
            graph_model,
            fallback=None if graph_model_only else proposer,
        )
        return proposer
    if kind == "graph-search":
        return GraphSearchProposer(
            fallback=proposer,
            max_candidates=graph_candidates,
            internal_nodes=internal_nodes,
            include_fallback=True,
        )
    if kind == "graph-search-only":
        return GraphSearchProposer(
            fallback=None,
            max_candidates=graph_candidates,
            internal_nodes=internal_nodes,
            include_fallback=False,
        )
    raise ValueError(f"unknown benchmark proposer: {kind}")


def evaluate_case(
    case: BenchmarkCase,
    proposer: TopologyProposer,
    accept_rmse_db: float = 1.0,
    accept_max_abs_db: float = 6.0,
    top_k: int = 1,
    max_iterations: int | None = None,
) -> BenchmarkResult:
    spec = case.spec
    if top_k != spec.optimization.top_k or max_iterations is not None:
        spec = SynthesisSpec(
            name=spec.name,
            ports=spec.ports,
            analysis=spec.analysis,
            behavior=spec.behavior,
            library=spec.library,
            optimization=replace(
                spec.optimization,
                top_k=top_k,
                max_iterations=max_iterations or spec.optimization.max_iterations,
            ),
        )

    started = time.perf_counter()
    try:
        results = CircuitSynthesizer(proposer=proposer).synthesize(spec)
        elapsed = time.perf_counter() - started
        best = results[0]
        success = best.metrics.rmse_db <= accept_rmse_db and best.metrics.max_abs_db <= accept_max_abs_db
        return BenchmarkResult(
            name=case.name,
            behavior_kind=str(spec.behavior["kind"]),
            status="ok",
            success=success,
            template=best.template.name,
            is_graph=isinstance(best.template, GraphCircuitTemplate),
            component_count=best.metrics.component_count,
            score=best.metrics.score,
            pareto_rank=best.metrics.pareto_rank,
            rmse_db=best.metrics.rmse_db,
            max_abs_db=best.metrics.max_abs_db,
            phase_rmse_deg=best.metrics.phase_rmse_deg,
            elapsed_s=elapsed,
            message="",
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should record task failures.
        elapsed = time.perf_counter() - started
        return BenchmarkResult(
            name=case.name,
            behavior_kind=str(spec.behavior.get("kind", "unknown")),
            status="error",
            success=False,
            template=None,
            is_graph=False,
            component_count=None,
            score=None,
            pareto_rank=None,
            rmse_db=None,
            max_abs_db=None,
            phase_rmse_deg=None,
            elapsed_s=elapsed,
            message=str(exc),
        )


def run_benchmark(
    cases: list[BenchmarkCase],
    proposer: TopologyProposer,
    accept_rmse_db: float = 1.0,
    accept_max_abs_db: float = 6.0,
    top_k: int = 1,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    results = [
        evaluate_case(
            case,
            proposer,
            accept_rmse_db=accept_rmse_db,
            accept_max_abs_db=accept_max_abs_db,
            top_k=top_k,
            max_iterations=max_iterations,
        )
        for case in cases
    ]
    return {"summary": summarize_results(results), "results": [result.as_dict() for result in results]}


def summarize_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "graph_rate": 0.0,
        }

    successes = [result for result in results if result.success]
    errors = [result for result in results if result.status != "ok"]
    ok_results = [result for result in results if result.status == "ok"]
    rmse_values = [result.rmse_db for result in ok_results if result.rmse_db is not None]
    max_values = [result.max_abs_db for result in ok_results if result.max_abs_db is not None]
    components = [result.component_count for result in ok_results if result.component_count is not None]
    elapsed = [result.elapsed_s for result in results]

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({result.behavior_kind for result in results}):
        subset = [result for result in results if result.behavior_kind == kind]
        by_kind[kind] = {
            "total": len(subset),
            "success_rate": _safe_ratio([result.success for result in subset]),
            "graph_rate": _safe_ratio([result.is_graph for result in subset]),
            "mean_rmse_db": _safe_mean([result.rmse_db for result in subset if result.rmse_db is not None]),
        }

    return {
        "total": total,
        "successes": len(successes),
        "failures": total - len(successes),
        "success_rate": len(successes) / total,
        "error_rate": len(errors) / total,
        "graph_rate": _safe_ratio([result.is_graph for result in ok_results]),
        "mean_rmse_db": _safe_mean(rmse_values),
        "median_rmse_db": _safe_median(rmse_values),
        "mean_max_abs_db": _safe_mean(max_values),
        "median_components": _safe_median(components),
        "mean_elapsed_s": _safe_mean(elapsed),
        "by_kind": by_kind,
    }


def write_benchmark_report(report: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_mean(values: list[float | int]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_median(values: list[float | int]) -> float | None:
    return float(np.median(values)) if values else None


def _safe_ratio(values: list[bool]) -> float:
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0
