from __future__ import annotations

import argparse
from pathlib import Path

from circuit_ai.learned import load_jsonl_rows, save_model, train_template_classifier


def parse_hidden(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a learned topology template proposer.")
    parser.add_argument("dataset", type=Path, help="JSONL file from generate_dataset.py")
    parser.add_argument("--out", type=Path, default=Path("models/template_proposer.joblib"))
    parser.add_argument("--hidden", type=parse_hidden, default=(64, 32), help="comma-separated hidden layer sizes")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--no-phase", action="store_true", help="train with magnitude only")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl_rows(args.dataset)
    bundle = train_template_classifier(
        rows,
        hidden_layer_sizes=args.hidden,
        max_iter=args.max_iter,
        seed=args.seed,
        use_phase=not args.no_phase,
        validation_fraction=args.validation_fraction,
        top_k=args.top_k,
    )
    save_model(bundle, args.out)
    classes = ", ".join(bundle["classes"])
    print(f"Wrote learned proposer to {args.out}")
    print(f"Rows: {len(rows)}")
    print(f"Classes: {classes}")
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
