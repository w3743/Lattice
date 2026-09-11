from __future__ import annotations

import argparse
from pathlib import Path

from circuit_ai.graph_learning import load_graph_rows, save_graph_model, train_graph_classifier
from train_template_proposer import parse_hidden


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a learned typed-graph topology proposer.")
    parser.add_argument("dataset", type=Path, help="JSONL file from generate_graph_dataset.py")
    parser.add_argument("--out", type=Path, default=Path("models/graph_proposer.joblib"))
    parser.add_argument("--hidden", type=parse_hidden, default=(96, 48), help="comma-separated hidden layer sizes")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--no-phase", action="store_true", help="train with magnitude only")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows = load_graph_rows(args.dataset)
    bundle = train_graph_classifier(
        rows,
        hidden_layer_sizes=args.hidden,
        max_iter=args.max_iter,
        seed=args.seed,
        use_phase=not args.no_phase,
        validation_fraction=args.validation_fraction,
        top_k=args.top_k,
    )
    save_graph_model(bundle, args.out)
    print(f"Wrote learned graph proposer to {args.out}")
    print(f"Rows: {len(rows)}")
    print(f"Graph classes: {len(bundle['classes'])}")
    print(f"Training accuracy: {bundle['training_accuracy']:.3f}")
    print(f"Training top-{bundle['top_k']} accuracy: {bundle['training_top_k_accuracy']:.3f}")
    if bundle.get("validation_accuracy") is None:
        print("Validation accuracy: unavailable (not enough rows per class)")
    else:
        print(f"Validation accuracy: {bundle['validation_accuracy']:.3f}")
        print(f"Validation top-{bundle['top_k']} accuracy: {bundle['validation_top_k_accuracy']:.3f}")
    if bundle.get("convergence_warnings", 0):
        print("Training note: optimizer reached its iteration budget; physical verification still gates candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
