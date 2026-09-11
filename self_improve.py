from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from circuit_ai.graph_learning import SklearnGraphProposer
from circuit_ai.proposers import HeuristicProposer
from circuit_ai.replay import export_graph_replay_rows, train_from_replay
from circuit_ai.spec import SynthesisSpec
from circuit_ai.synthesis import CircuitSynthesizer
from circuit_ai.topology import GraphSearchProposer
from train_template_proposer import parse_hidden


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthesis tasks, export graph replay data, and retrain.")
    parser.add_argument("specs", nargs="+", type=Path, help="synthesis spec JSON files")
    parser.add_argument("--replay", type=Path, default=Path("data/replay_graphs.jsonl"))
    parser.add_argument("--model-out", type=Path, default=Path("models/graph_proposer_replay.joblib"))
    parser.add_argument("--seed-model", type=Path, default=None, help="optional existing graph proposer model")
    parser.add_argument("--graph-candidates", type=int, default=32)
    parser.add_argument("--internal-nodes", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=4, help="number of synthesis candidates to keep per spec")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--accept-rmse-db", type=float, default=1.0)
    parser.add_argument("--accept-max-db", type=float, default=6.0)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--overwrite-replay", action="store_true")
    parser.add_argument("--hidden", type=parse_hidden, default=(96, 48))
    parser.add_argument("--max-train-iter", type=int, default=1000)
    parser.add_argument("--replay-train-points", type=int, default=96)
    parser.add_argument(
        "--train-with-rejected",
        action="store_true",
        help="legacy mode: also train the topology classifier with rejected rows as class examples",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--eval-top-k", type=int, default=3, help="top-k metric for model evaluation")
    args = parser.parse_args()

    all_summaries = []
    first_write = True

    for spec_path in args.specs:
        spec = SynthesisSpec.from_json_file(spec_path)
        spec = SynthesisSpec(
            name=spec.name,
            ports=spec.ports,
            analysis=spec.analysis,
            behavior=spec.behavior,
            library=spec.library,
            optimization=replace(
                spec.optimization,
                top_k=args.top_k,
                max_iterations=args.max_iterations or spec.optimization.max_iterations,
            ),
        )
        fallback = HeuristicProposer()
        if args.seed_model:
            fallback = SklearnGraphProposer(args.seed_model, fallback=fallback)
        proposer = GraphSearchProposer(
            fallback=fallback,
            max_candidates=args.graph_candidates,
            internal_nodes=args.internal_nodes,
            include_fallback=True,
        )
        results = CircuitSynthesizer(proposer=proposer).synthesize(spec)
        summary = export_graph_replay_rows(
            results,
            spec,
            args.replay,
            append=True if not first_write else not args.overwrite_replay,
            include_rejected=args.include_rejected,
            accept_rmse_db=args.accept_rmse_db,
            accept_max_abs_db=args.accept_max_db,
        )
        first_write = False
        all_summaries.append({"spec": str(spec_path), **summary.as_dict()})
        best = results[0]
        print(
            f"{spec.name}: best={best.template.name} "
            f"rmse={best.metrics.rmse_db:.4g}dB max={best.metrics.max_abs_db:.4g}dB "
            f"replay_rows={summary.rows_written}"
        )

    bundle = train_from_replay(
        args.replay,
        args.model_out,
        accepted_only=not args.train_with_rejected,
        resample_points=args.replay_train_points,
        hidden_layer_sizes=args.hidden,
        max_iter=args.max_train_iter,
        validation_fraction=args.validation_fraction,
        top_k=args.eval_top_k,
        use_rejected_as_classifier_classes=args.train_with_rejected,
    )
    print(f"Wrote replay dataset to {args.replay}")
    print(f"Wrote replay-trained graph proposer to {args.model_out}")
    print(f"Training rows: {bundle['training_source']['rows']}")
    print(f"Training accuracy: {bundle['training_accuracy']:.3f}")
    print(
        "Replay labels: "
        f"{bundle['training_source']['label_summary']['accepted']} accepted, "
        f"{bundle['training_source']['label_summary']['rejected']} rejected"
    )
    quality = bundle.get("quality_model", {})
    print(f"Quality model: {'enabled' if quality.get('enabled') else 'disabled'} ({quality.get('message', '')})")
    ranking = bundle.get("ranking_model", {})
    print(
        "Ranking model: "
        f"{'enabled' if ranking.get('enabled') else 'disabled'} "
        f"pairs={ranking.get('pairs', 0)} "
        f"group_top1={ranking.get('group_top1_accuracy')}"
    )
    if bundle.get("validation_accuracy") is None:
        print("Validation accuracy: unavailable")
    else:
        print(f"Validation accuracy: {bundle['validation_accuracy']:.3f}")
        print(f"Validation top-{bundle['top_k']} accuracy: {bundle['validation_top_k_accuracy']:.3f}")
    print(json.dumps({"summaries": all_summaries}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
