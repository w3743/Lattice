from __future__ import annotations

from circuit_ai.proposers import HeuristicProposer
from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.topology import generate_linear_graph_templates


def test_exact_component_graph_generation_supports_minimal_search() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "minimal_lowpass",
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
            "optimization": {"points": 48, "max_iterations": 18, "top_k": 1, "seed": 10},
        }
    )
    templates = generate_linear_graph_templates(
        spec.library,
        min_components=2,
        max_components=2,
        behavior_kind="lowpass",
        limit=12,
    )
    assert templates
    assert all(template.component_count == 2 for template in templates)

    result = CircuitSynthesizer(proposer=HeuristicProposer(templates=templates)).synthesize(spec)[0]
    assert result.metrics.rmse_db < 0.2
