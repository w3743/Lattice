from __future__ import annotations

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.dataset import generate_rows
from circuit_ai.learned import SklearnTemplateProposer, save_model, train_template_classifier
from circuit_ai.spec import LibrarySpec
from circuit_ai.templates import RCHighPass, RCLowPass, RLCBandPass


def test_trained_template_proposer_feeds_synthesis(tmp_path) -> None:
    library = LibrarySpec.from_dict(
        {
            "allowed": ["R", "C", "L"],
            "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4], "L": [1e-6, 1]},
        }
    )
    rows = generate_rows(
        library,
        samples_per_template=18,
        points=48,
        seed=9,
        templates=[RCLowPass(), RCHighPass(), RLCBandPass()],
    )
    bundle = train_template_classifier(rows, hidden_layer_sizes=(24,), max_iter=250, seed=9)
    assert bundle["validation"]["validation_rows"] > 0
    assert bundle["validation_accuracy"] is not None
    assert bundle["validation_top_k_accuracy"] is not None
    assert bundle["training_top_k_accuracy"] >= bundle["training_accuracy"]
    model_path = tmp_path / "template_proposer.joblib"
    save_model(bundle, model_path)

    spec = SynthesisSpec.from_dict(
        {
            "name": "learned_lowpass",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C", "L"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4], "L": [1e-6, 1]},
            },
            "optimization": {"points": 56, "max_iterations": 20, "top_k": 1, "seed": 6},
        }
    )

    proposer = SklearnTemplateProposer(model_path, fallback=None)
    result = CircuitSynthesizer(proposer=proposer).synthesize(spec)[0]

    assert result.metrics.rmse_db < 0.2
