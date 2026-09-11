from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from circuit_ai.proposers import HeuristicProposer
from circuit_ai.spec import SynthesisSpec
from circuit_ai.synthesis import CircuitSynthesizer
from circuit_ai.topology import generate_linear_graph_templates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded minimum-component search over generated R/C/L graphs."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("--max-components", type=int, default=None)
    parser.add_argument("--internal-nodes", type=int, default=1)
    parser.add_argument("--graph-candidates", type=int, default=128)
    parser.add_argument("--accept-rmse-db", type=float, default=0.5)
    parser.add_argument("--accept-max-db", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base_spec = SynthesisSpec.from_json_file(args.spec)
    max_components = args.max_components or base_spec.optimization.max_components
    behavior_kind = str(base_spec.behavior["kind"]).strip().lower()
    history = []

    print(f"Bounded minimal search: {base_spec.name}")
    print(
        f"Acceptance: rmse <= {args.accept_rmse_db:g} dB, "
        f"max <= {args.accept_max_db:g} dB"
    )

    proof = {
        "spec": base_spec.name,
        "search_space": {
            "element_kinds": sorted(base_spec.library.allowed),
            "internal_nodes": args.internal_nodes,
            "max_components": max_components,
            "graph_candidates_per_count": args.graph_candidates,
        },
        "acceptance": {
            "rmse_db": args.accept_rmse_db,
            "max_abs_db": args.accept_max_db,
        },
        "history": history,
        "result": None,
    }

    for count in range(2, max_components + 1):
        templates = generate_linear_graph_templates(
            base_spec.library,
            max_components=count,
            min_components=count,
            internal_nodes=args.internal_nodes,
            limit=args.graph_candidates,
            behavior_kind=behavior_kind,
        )
        if not templates:
            print(f"{count} components: no legal generated graphs")
            history.append({"components": count, "candidates": 0, "best": None})
            continue

        spec = SynthesisSpec(
            name=base_spec.name,
            ports=base_spec.ports,
            analysis=base_spec.analysis,
            behavior=base_spec.behavior,
            library=base_spec.library,
            optimization=replace(base_spec.optimization, max_components=count, top_k=1),
        )
        result = CircuitSynthesizer(proposer=HeuristicProposer(templates=templates)).synthesize(spec)[0]
        best = {
            "template": result.template.name,
            "rmse_db": result.metrics.rmse_db,
            "max_abs_db": result.metrics.max_abs_db,
            "score": result.metrics.score,
            "parameters": result.parameters,
            "netlist": result.netlist("minimal_candidate"),
        }
        history.append({"components": count, "candidates": len(templates), "best": best})
        print(
            f"{count} components: best={result.template.name} "
            f"rmse={result.metrics.rmse_db:.4g}dB max={result.metrics.max_abs_db:.4g}dB "
            f"candidates={len(templates)}"
        )

        if result.metrics.rmse_db <= args.accept_rmse_db and result.metrics.max_abs_db <= args.accept_max_db:
            proof["result"] = {
                "status": "proved_within_bounded_search",
                "minimum_components": count,
                "best": best,
            }
            print(f"Bounded proof: minimum is {count} components within this search space.")
            break

    if proof["result"] is None:
        proof["result"] = {"status": "not_found_within_bounded_search"}
        print("No accepted circuit found within bounded search space.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(proof, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}")

    return 0 if proof["result"]["status"] == "proved_within_bounded_search" else 2


if __name__ == "__main__":
    raise SystemExit(main())
