from __future__ import annotations

import argparse
from pathlib import Path

from circuit_ai.dataset import generate_rows, write_jsonl
from circuit_ai.spec import LibrarySpec


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic circuit behavior data.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_linear.jsonl"))
    parser.add_argument("--samples-per-template", type=int, default=64)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--f-min", type=float, default=10.0)
    parser.add_argument("--f-max", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    library = LibrarySpec.from_dict(
        {
            "allowed": ["R", "C", "L", "opamp"],
            "parameter_ranges": {
                "R": [1.0, 10_000_000.0],
                "C": [1e-12, 1e-2],
                "L": [1e-9, 10.0],
            },
        }
    )
    rows = generate_rows(
        library,
        samples_per_template=args.samples_per_template,
        f_min_hz=args.f_min,
        f_max_hz=args.f_max,
        points=args.points,
        seed=args.seed,
    )
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
