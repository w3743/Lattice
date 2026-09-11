from __future__ import annotations

from circuit_ai import (
    CircuitSynthesizer,
    PartCatalog,
    SynthesisSpec,
    discretize_template_candidates,
    e_series_values,
)
from circuit_ai.graph_templates import ElementSlot, GraphCircuitTemplate


def test_discretization_uses_catalog_neighbors_and_preserves_part_identity() -> None:
    template = GraphCircuitTemplate(
        (
            ElementSlot("R", "R", "in", "out"),
            ElementSlot("C", "C", "out", "0"),
        )
    )
    catalog = PartCatalog.from_dict(
        {
            "source": "test_parts",
            "records": [
                {
                    "name": "R1k",
                    "part_number": "R-1K",
                    "category": "resistor",
                    "nominal_value": 1000.0,
                    "status": "active",
                },
                {
                    "name": "R2k",
                    "part_number": "R-2K",
                    "category": "resistor",
                    "nominal_value": 2000.0,
                    "status": "active",
                },
                {
                    "name": "C10n",
                    "part_number": "C-10N",
                    "category": "capacitor",
                    "nominal_value": 1e-8,
                    "status": "active",
                },
                {
                    "name": "C22n",
                    "part_number": "C-22N",
                    "category": "capacitor",
                    "nominal_value": 2.2e-8,
                    "status": "active",
                },
            ],
        }
    )
    results = discretize_template_candidates(
        template,
        {"R": 1400.0, "C": 1.2e-8},
        [(100.0, 1e6), (1e-10, 1e-4)],
        {
            "enabled": True,
            "catalog": catalog.as_dict(),
            "policy": {"require_concrete": True},
            "neighborhood": 1,
            "max_combinations": 4,
        },
    )
    assert results
    assert all(item.feasible for item in results)
    assert results[0].catalog_snapshot == catalog.snapshot
    assert {item.part_id for item in results[0].selections} == {"R-1K", "C-10N"}
    assert len(results) == 4


def test_synthesis_can_resimulate_an_e_series_discrete_candidate() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "discrete_lowpass",
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
            "optimization": {
                "points": 32,
                "max_iterations": 3,
                "top_k": 1,
                "seed": 5,
                "discretization": {
                    "enabled": True,
                    "series": "E24",
                    "neighborhood": 1,
                    "max_combinations": 4,
                },
            },
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]
    assert result.discretization is not None
    assert result.discretization.feasible
    assert result.discretization.source == "e_series:E24"
    assert result.parameters["R"] in e_series_values(100.0, 1e6, "E24")
    assert result.parameters["C"] in e_series_values(1e-10, 1e-4, "E24")
    assert result.as_dict()["discretization"]["feasible"] is True


def test_catalog_selection_is_carried_into_export_graph_and_ratings() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "catalog_lowpass",
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
            "optimization": {
                "points": 16,
                "max_iterations": 1,
                "top_k": 1,
                "discretization": {
                    "enabled": True,
                    "catalog": {
                        "records": [
                            {
                                "name": "R1k",
                                "part_number": "R-1K",
                                "category": "resistor",
                                "nominal_value": 1000.0,
                                "status": "active",
                                "metadata": {
                                    "ratings": [
                                        {"quantity": "power", "unit": "W", "maximum": 0.25}
                                    ]
                                },
                            },
                            {
                                "name": "C10n",
                                "part_number": "C-10N",
                                "category": "capacitor",
                                "nominal_value": 1e-8,
                                "status": "active",
                            },
                        ]
                    },
                    "policy": {"require_concrete": True},
                    "neighborhood": 0,
                },
            },
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]
    graph = result.circuit_graph()
    assert result.discretization is not None and result.discretization.feasible
    assert {item.attributes.get("selected_part_id") for item in graph.components} >= {"R-1K", "C-10N"}
    resistor = next(item for item in graph.components if item.model.kind == "R")
    assert resistor.ratings[0].maximum == 0.25
