from __future__ import annotations

import argparse
from pathlib import Path

from circuit_ai.surrogate import (
    load_surrogate_rows,
    save_response_surrogate,
    train_response_surrogate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a validated circuit-response surrogate from external-SPICE JSONL rows."
    )
    parser.add_argument("rows", type=Path, help="JSONL response dataset")
    parser.add_argument("--out", type=Path, default=Path("models/validated_response_surrogate.joblib"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--estimators", type=int, default=64)
    parser.add_argument("--min-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--trust-uncertainty-db", type=float, default=1.0)
    args = parser.parse_args(argv)

    rows = load_surrogate_rows(args.rows)
    bundle = train_response_surrogate(
        rows,
        validation_fraction=args.validation_fraction,
        n_estimators=args.estimators,
        min_samples_leaf=args.min_leaf,
        seed=args.seed,
        trust_uncertainty_db=args.trust_uncertainty_db,
    )
    save_response_surrogate(bundle, args.out)
    print(f"trained {len(bundle['models'])} topology surrogate(s) from {bundle['training_rows']} validated rows")
    for name, record in sorted(bundle["models"].items()):
        metrics = record["validation"]
        print(
            f"{name}: rows={record['training_rows']} "
            f"validation_mag_rmse={metrics['magnitude_rmse_db']} "
            f"validation_phase_rmse={metrics['phase_rmse_deg']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
