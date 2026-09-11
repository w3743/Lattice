from __future__ import annotations

import argparse
import json
from pathlib import Path

from .feasibility import FeasibilityError, analyze_feasibility
from .graph_learning import SklearnGraphProposer
from .learned import SklearnTemplateProposer
from .pbdl_boundary import translate_pbdl_dict
from .proposers import DataDrivenProposer, HeuristicProposer
from .spec import SynthesisSpec
from .spice import NgspiceVerifier
from .synthesis import CircuitSynthesizer, write_results
from .surrogate import export_verified_surrogate_rows
from .topology import GraphSearchProposer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize a circuit from port behavior.")
    parser.add_argument("spec", type=Path, help="PBDL JSON design specification")
    parser.add_argument("--top", type=int, default=None, help="number of candidates to print")
    parser.add_argument("--out", type=Path, default=None, help="directory for report.json and best.spice")
    parser.add_argument(
        "--dataset-proposer",
        type=Path,
        default=None,
        help="JSONL synthetic behavior dataset for nearest-neighbor topology proposal",
    )
    parser.add_argument("--model-proposer", type=Path, default=None, help="trained proposer .joblib model")
    parser.add_argument("--graph-model-proposer", type=Path, default=None, help="trained typed-graph proposer .joblib")
    parser.add_argument(
        "--graph-model-only",
        action="store_true",
        help="use only the trained graph proposer, without heuristic/template fallback",
    )
    parser.add_argument("--graph-search", action="store_true", help="enable generated R/C/L graph search")
    parser.add_argument("--graph-candidates", type=int, default=None, help="generated graph candidate limit")
    parser.add_argument("--internal-nodes", type=int, default=None, help="internal nodes for graph search")
    parser.add_argument("--spice-verify", action="store_true", help="run external ngspice verification")
    parser.add_argument("--spice-executable", default="ngspice", help="ngspice executable path/name")
    parser.add_argument("--spice-timeout", type=float, default=30.0, help="ngspice timeout in seconds")
    parser.add_argument(
        "--surrogate-data",
        type=Path,
        default=None,
        help="append successful external-SPICE response traces to a surrogate JSONL dataset",
    )
    parser.add_argument("--feasibility-only", action="store_true", help="only run feasibility checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_data = json.loads(args.spec.read_text(encoding="utf-8"))
    spec = SynthesisSpec.from_dict(translate_pbdl_dict(source_data))
    if args.top is not None:
        spec = SynthesisSpec(
            name=spec.name,
            ports=spec.ports,
            analysis=spec.analysis,
            behavior=spec.behavior,
            library=spec.library,
            optimization=type(spec.optimization)(
                **{**spec.optimization.__dict__, "top_k": args.top}
            ),
        )

    if args.feasibility_only:
        report = analyze_feasibility(spec)
        print_feasibility(report)
        return 0 if report.passed else 2

    proposer = HeuristicProposer()
    if args.dataset_proposer:
        proposer = DataDrivenProposer(args.dataset_proposer, fallback=proposer)
    if args.model_proposer:
        proposer = SklearnTemplateProposer(args.model_proposer, fallback=proposer)
    if args.graph_model_proposer:
        proposer = SklearnGraphProposer(
            args.graph_model_proposer,
            fallback=None if args.graph_model_only else proposer,
        )
    graph_options = spec.optimization.graph_search
    use_graph_search = args.graph_search or bool(graph_options.get("enabled", False))
    if use_graph_search:
        proposer = GraphSearchProposer(
            fallback=proposer,
            max_candidates=int(args.graph_candidates or graph_options.get("max_candidates", 48)),
            internal_nodes=int(args.internal_nodes or graph_options.get("internal_nodes", 1)),
            include_fallback=bool(graph_options.get("include_fallback", True)),
        )
    try:
        results = CircuitSynthesizer(proposer=proposer).synthesize(spec)
    except FeasibilityError as exc:
        print_feasibility(exc.report)
        return 2
    print(f"Synthesis: {spec.name}")
    for index, result in enumerate(results, start=1):
        metrics = result.metrics
        print(
            f"{index}. {result.template.name} "
            f"pareto={metrics.pareto_rank} score={metrics.score:.4g} rmse={metrics.rmse_db:.4g}dB "
            f"max={metrics.max_abs_db:.4g}dB components={metrics.component_count}"
        )
        if metrics.robustness is not None:
            robust = metrics.robustness
            print(
                f"   robustness: yield={100.0 * robust.yield_fraction:.1f}% "
                f"finite={100.0 * robust.finite_fraction:.1f}% "
                f"p95_rmse={robust.rmse_db_p95:.4g}dB "
                f"worst_max={robust.max_abs_db_worst:.4g}dB"
            )
        if metrics.differentiable is not None:
            diff = metrics.differentiable
            if diff.initial_loss is None or diff.final_loss is None:
                print(f"   differentiable: {diff.status} ({diff.message})")
            else:
                print(
                    f"   differentiable: {diff.status} improved={diff.improved} "
                    f"loss={diff.initial_loss:.4g}->{diff.final_loss:.4g} "
                    f"backend={diff.backend}"
                )
        print(f"   {result.template.summary(result.parameters)}")

    spice_verifications = None
    spice_options = spec.optimization.spice_verification
    use_spice = args.spice_verify or bool(spice_options.get("enabled", False))
    if use_spice:
        verify_dir = args.out / "spice" if args.out else None
        verifier = NgspiceVerifier(
            spice_options.get("executable", args.spice_executable),
            timeout_s=float(spice_options.get("timeout_s", args.spice_timeout)),
            points_per_decade=int(spice_options.get("points_per_decade", 80)),
        )
        spice_verifications = [verifier.verify(result, spec, output_dir=verify_dir) for result in results]
        for index, verification in enumerate(spice_verifications, start=1):
            if verification.rmse_db is None:
                print(f"   spice {index}: {verification.status} ({verification.message})")
            else:
                print(
                    f"   spice {index}: {verification.status} "
                    f"rmse={verification.rmse_db:.4g}dB "
                    f"max={verification.max_abs_db:.4g}dB "
                    f"points={verification.points}"
                )
    if args.surrogate_data:
        if spice_verifications is None:
            print("--surrogate-data requires --spice-verify or optimization.spice_verification.enabled")
            return 2
        try:
            summary = export_verified_surrogate_rows(results, spice_verifications, args.surrogate_data)
        except ValueError as exc:
            print(f"Surrogate data export failed: {exc}")
            return 2
        print(
            f"Surrogate data: wrote {summary['rows_written']} validated rows "
            f"(skipped {summary['skipped_rows']}) to {args.surrogate_data}"
        )

    if args.out:
        write_results(results, args.out, spice_verifications=spice_verifications)
        print(f"Wrote {args.out / 'report.json'}")
        print(f"Wrote {args.out / 'best.spice'}")

    return 0


def print_feasibility(report) -> None:
    status = "passed" if report.passed else "failed"
    print(f"Feasibility: {status} enforce={report.enforce}")
    if not report.issues:
        print("   no issues")
        return
    for issue in report.issues:
        print(f"   {issue.severity}: {issue.code}: {issue.message}")


if __name__ == "__main__":
    raise SystemExit(main())
