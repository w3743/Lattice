from __future__ import annotations

import argparse
from pathlib import Path

from circuit_ai.benchmark import (
    build_benchmark_proposer,
    generate_benchmark_cases,
    load_cases,
    run_benchmark,
    write_benchmark_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatable circuit synthesis benchmarks.")
    parser.add_argument("--spec", action="append", type=Path, default=[], help="explicit spec JSON file")
    parser.add_argument("--generated", action="store_true", help="generate low/high/band-pass benchmark cases")
    parser.add_argument("--count-per-kind", type=int, default=2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--proposer",
        choices=["heuristic", "graph-search", "graph-search-only", "graph-model"],
        default="heuristic",
    )
    parser.add_argument("--graph-model", type=Path, default=None)
    parser.add_argument("--graph-model-only", action="store_true")
    parser.add_argument("--graph-candidates", type=int, default=32)
    parser.add_argument("--internal-nodes", type=int, default=1)
    parser.add_argument("--accept-rmse-db", type=float, default=1.0)
    parser.add_argument("--accept-max-db", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark/report.json"))
    args = parser.parse_args()

    cases = load_cases(args.spec)
    if args.generated or not cases:
        cases.extend(
            generate_benchmark_cases(
                count_per_kind=args.count_per_kind,
                seed=args.seed,
                points=args.points,
                max_iterations=args.max_iterations,
                max_components=args.max_components,
            )
        )

    proposer = build_benchmark_proposer(
        args.proposer,
        graph_model=args.graph_model,
        graph_model_only=args.graph_model_only,
        graph_candidates=args.graph_candidates,
        internal_nodes=args.internal_nodes,
    )
    report = run_benchmark(
        cases,
        proposer,
        accept_rmse_db=args.accept_rmse_db,
        accept_max_abs_db=args.accept_max_db,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
    )
    write_benchmark_report(report, args.out)
    summary = report["summary"]
    mean_rmse = summary["mean_rmse_db"]
    mean_rmse_text = "n/a" if mean_rmse is None else f"{mean_rmse:.4g}dB"
    print(
        f"Benchmark: total={summary['total']} "
        f"success_rate={100.0 * summary['success_rate']:.1f}% "
        f"graph_rate={100.0 * summary['graph_rate']:.1f}% "
        f"mean_rmse={mean_rmse_text}"
    )
    print(f"Wrote {args.out}")
    return 0 if summary["success_rate"] >= 0.5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
