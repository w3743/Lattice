from __future__ import annotations

import numpy as np

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.dataset import generate_rows, write_jsonl
from circuit_ai.mna import LinearCircuit
from circuit_ai.proposers import DataDrivenProposer, HeuristicProposer
from circuit_ai.spec import LibrarySpec
from circuit_ai.templates import CircuitTemplate


def test_data_driven_proposer_can_feed_synthesizer(tmp_path) -> None:
    library = LibrarySpec.from_dict(
        {
            "allowed": ["R", "C", "L", "opamp"],
            "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4], "L": [1e-6, 1]},
        }
    )
    rows = generate_rows(library, samples_per_template=2, points=24, seed=1)
    dataset = tmp_path / "synthetic.jsonl"
    write_jsonl(rows, dataset)
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_data_driven",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C", "opamp"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {"points": 48, "max_iterations": 15, "top_k": 1, "seed": 5},
        }
    )

    proposer = DataDrivenProposer(dataset, fallback=HeuristicProposer())
    result = CircuitSynthesizer(proposer=proposer).synthesize(spec)[0]

    assert result.metrics.rmse_db < 0.1


def test_generate_rows_skips_non_finite_template_responses() -> None:
    class NonFiniteTemplate(CircuitTemplate):
        name = "nonfinite"
        description = "non-finite test template"
        required_elements = frozenset()
        supported_kinds = frozenset({"*"})
        component_count = 0
        params = ()

        def response(self, values, frequencies_hz):
            return np.full(frequencies_hz.shape, np.nan + 1j * np.nan)

        def netlist(self, values, title="generated"):
            return ".end"

        def to_circuit(self, values):
            return LinearCircuit(elements=(), voltage_sources=())

    rows = generate_rows(
        LibrarySpec.from_dict({"allowed": ["R"]}),
        samples_per_template=3,
        templates=[NonFiniteTemplate()],
    )

    assert rows == []
