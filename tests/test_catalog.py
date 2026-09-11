from __future__ import annotations

import numpy as np
import pytest

from circuit_ai import CatalogProposer, CircuitSynthesizer, SynthesisSpec, standard_topology_catalog
from circuit_ai.optimizers import ParameterOptimizationResult


class FixedFirstOptimizer:
    def optimize(self, template, spec, target, score_response):
        parameters = {
            param.name: float(np.sqrt(lo * hi))
            for param, (lo, hi) in zip(template.params, template.parameter_bounds(spec.library))
        }
        response = template.analyze(parameters, target.frequencies_hz, target.analysis)
        score_response(response)
        return ParameterOptimizationResult(
            parameters=parameters,
            response=response,
            success=True,
            message="fixed catalog optimizer",
            evaluations=1,
        )


def test_standard_topology_catalog_contains_metadata() -> None:
    records = standard_topology_catalog()
    by_name = {record.name: record for record in records}

    assert "rc_lowpass" in by_name
    assert "sallen_key_lowpass" in by_name
    assert "sallen_key_gain_lowpass" in by_name
    assert "transimpedance_amplifier" in by_name
    assert "output_resistor_impedance" in by_name
    assert by_name["rc_lowpass"].family == "passive_rc"
    assert by_name["sallen_key_lowpass"].family == "active_rc"
    assert by_name["transimpedance_amplifier"].family == "active_transimpedance"
    assert by_name["output_resistor_impedance"].family == "termination"
    assert "sallen_key" in by_name["sallen_key_lowpass"].tags
    assert "current_input" in by_name["transimpedance_amplifier"].tags
    assert "output_matching" in by_name["output_resistor_impedance"].tags
    assert "lowpass" in by_name["rc_lowpass"].behaviors
    assert by_name["rc_lowpass"].topology_risk < by_name["rlc_bandpass"].topology_risk
    assert by_name["shunt_resistor_impedance"].analysis_kinds == frozenset({"input_impedance"})


def test_catalog_proposer_filters_by_analysis_behavior_and_library() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "catalog_impedance",
            "ports": 2,
            "analysis": {"kind": "input_impedance"},
            "behavior": {
                "kind": "constant_impedance",
                "ohms": 1000.0,
            },
            "library": {"allowed": ["R"]},
            "optimization": {"max_components": 2},
        }
    )

    proposed = CatalogProposer().propose(spec)

    assert [template.name for template in proposed] == ["shunt_resistor_impedance"]


def test_catalog_proposer_includes_sallen_key_for_active_second_order_lowpass() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "catalog_active_lowpass",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000.0,
                "order": 2,
            },
            "library": {"allowed": ["R", "C", "opamp"]},
            "optimization": {"max_components": 7},
        }
    )

    names = [template.name for template in CatalogProposer().propose(spec)]

    assert "sallen_key_lowpass" in names
    assert "sallen_key_gain_lowpass" in names


def test_catalog_proposer_filters_transimpedance_topology() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "catalog_tia",
            "ports": 2,
            "analysis": {"kind": "transimpedance"},
            "behavior": {"kind": "transimpedance", "transimpedance_ohm": 1000.0},
            "library": {"allowed": ["R", "C", "opamp"]},
            "optimization": {"max_components": 3},
        }
    )

    names = [template.name for template in CatalogProposer().propose(spec)]

    assert names == ["transimpedance_amplifier"]


def test_catalog_proposer_filters_output_impedance_topology() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "catalog_output_impedance",
            "ports": 2,
            "analysis": {"kind": "output_impedance"},
            "behavior": {"kind": "constant_output_impedance", "ohms": 50.0},
            "library": {"allowed": ["R"]},
            "optimization": {"max_components": 1},
        }
    )

    names = [template.name for template in CatalogProposer().propose(spec)]

    assert names == ["output_resistor_impedance"]


def test_catalog_required_elements_force_active_topology() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "required_opamp",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1_000.0},
            "library": {"allowed": ["R", "C", "opamp"], "required": ["opamp"]},
            "optimization": {"max_components": 8},
        }
    )

    names = [template.name for template in CatalogProposer().propose(spec)]

    assert "rc_lowpass" not in names
    assert "gain_rc_lowpass" in names


def test_synthesis_report_includes_catalog_and_resource_objectives() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "catalog_cost_lowpass",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [1000, 1001], "C": [1e-9, 1.1e-9]},
                "unit_costs": {"R": 0.02, "C": 0.04},
                "unit_areas_mm2": {"R": 1.2, "C": 1.8},
            },
            "optimization": {
                "points": 8,
                "top_k": 1,
                "weights": {"cost": 1.0, "area_mm2": 0.1, "topology_risk": 0.01},
            },
        }
    )

    result = CircuitSynthesizer(optimizer=FixedFirstOptimizer()).synthesize(spec)[0]
    report = result.as_dict()

    assert result.template.name == "rc_lowpass"
    assert result.template.component_inventory() == {"R": 1, "C": 1}
    assert result.metrics.estimated_cost == pytest.approx(0.06)
    assert result.metrics.estimated_area_mm2 == pytest.approx(3.0)
    assert result.metrics.topology_risk == pytest.approx(0.05)
    assert result.metrics.objectives["estimated_cost"] == pytest.approx(0.06)
    assert result.metrics.objectives["estimated_area_mm2"] == pytest.approx(3.0)
    assert result.metrics.objectives["topology_risk"] == pytest.approx(0.05)
    assert result.metrics.objectives["optimizer_failure"] == 0.0
    assert report["topology"]["family"] == "passive_rc"
    assert report["component_inventory"] == {"R": 1, "C": 1}
