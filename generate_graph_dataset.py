from __future__ import annotations

import argparse
from pathlib import Path

from circuit_ai.graph_learning import generate_graph_rows, write_jsonl
from circuit_ai.spec import LibrarySpec


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic typed-graph circuit data.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_graphs.jsonl"))
    parser.add_argument("--samples-per-graph", type=int, default=16)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--f-min", type=float, default=10.0)
    parser.add_argument("--f-max", type=float, default=1_000_000.0)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--internal-nodes", type=int, default=1)
    parser.add_argument("--graph-limit", type=int, default=64)
    parser.add_argument(
        "--behavior-kind",
        default="all",
        help="ranking hint for graph catalog: all, lowpass, highpass, bandpass, or *",
    )
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    library = LibrarySpec.from_dict(
        {
            "allowed": ["R", "C", "L"],
            "parameter_ranges": {
                "R": [1.0, 10_000_000.0],
                "C": [1e-12, 1e-2],
                "L": [1e-9, 10.0],
            },
        }
    )
    rows = generate_graph_rows(
        library,
        samples_per_graph=args.samples_per_graph,
        f_min_hz=args.f_min,
        f_max_hz=args.f_max,
        points=args.points,
        seed=args.seed,
        max_components=args.max_components,
        internal_nodes=args.internal_nodes,
        graph_limit=args.graph_limit,
        behavior_kind=args.behavior_kind,
    )
    write_jsonl(rows, args.out)
    graph_count = len({row["template"] for row in rows})
    print(f"Wrote {len(rows)} rows across {graph_count} graph topologies to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
