from __future__ import annotations

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.graph_learning import (
    SklearnGraphProposer,
    generate_graph_rows,
    load_graph_rows,
    save_graph_model,
    train_graph_classifier,
    write_jsonl,
)
from circuit_ai.graph_templates import GraphCircuitTemplate
from circuit_ai.spec import LibrarySpec


def test_graph_rows_round_trip_and_trainable_proposer(tmp_path) -> None:
    library = LibrarySpec.from_dict(
        {
            "allowed": ["R", "C"],
            "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
        }
    )
    rows = generate_graph_rows(
        library,
        samples_per_graph=5,
        points=32,
        seed=5,
        max_components=2,
        graph_limit=8,
        behavior_kind="lowpass",
    )
    dataset = tmp_path / "graphs.jsonl"
    write_jsonl(rows, dataset)
    loaded = load_graph_rows(dataset)
    assert loaded
    assert "graph" in loaded[0]

    bundle = train_graph_classifier(loaded, hidden_layer_sizes=(16,), max_iter=200, seed=5)
    assert bundle["model_kind"] == "graph_topology_proposer"
    assert bundle["validation"]["validation_rows"] > 0
    assert bundle["validation_top_k_accuracy"] is not None
    model_path = tmp_path / "graph_proposer.joblib"
    save_graph_model(bundle, model_path)

    spec = SynthesisSpec.from_dict(
        {
            "name": "learned_graph_lowpass",
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
            "optimization": {"points": 48, "max_iterations": 16, "top_k": 1, "seed": 12},
        }
    )

    proposer = SklearnGraphProposer(model_path, fallback=None, top_n=4)
    proposed = proposer.propose(spec)
    assert any(isinstance(template, GraphCircuitTemplate) for template in proposed)

    result = CircuitSynthesizer(proposer=proposer).synthesize(spec)[0]
    assert isinstance(result.template, GraphCircuitTemplate)
    assert result.metrics.rmse_db < 0.2
