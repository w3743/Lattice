from __future__ import annotations

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.graph_templates import GraphCircuitTemplate
from circuit_ai.topology import GraphSearchProposer, generate_linear_graph_templates


def test_graph_generator_contains_rc_lowpass_topology() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000},
            "library": {"allowed": ["R", "C"]},
            "optimization": {"max_components": 2},
        }
    )

    templates = generate_linear_graph_templates(
        spec.library,
        max_components=2,
        behavior_kind="lowpass",
        limit=12,
    )

    assert any(
        isinstance(template, GraphCircuitTemplate)
        and {f"{slot.kind}:{'-'.join(slot.normalized_edge())}" for slot in template.slots}
        == {"R:in-out", "C:0-out"}
        for template in templates
    )


def test_graph_search_synthesizes_lowpass_without_hand_template() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "graph_lowpass",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {"points": 64, "max_components": 2, "max_iterations": 25, "top_k": 1},
        }
    )

    proposer = GraphSearchProposer(fallback=None, max_candidates=12, include_fallback=False)
    result = CircuitSynthesizer(proposer=proposer).synthesize(spec)[0]

    assert isinstance(result.template, GraphCircuitTemplate)
    assert result.metrics.rmse_db < 0.1
    assert "generated R/C/L graph candidate" in result.netlist()
