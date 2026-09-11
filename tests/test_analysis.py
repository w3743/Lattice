from __future__ import annotations

import numpy as np

from circuit_ai import AnalysisRequest, CircuitSynthesizer, SynthesisSpec
from circuit_ai.analysis import LinearACAnalyzer
from circuit_ai.mna import CurrentSource, LinearCircuit, LinearElement, VoltageControlledVoltageSource, VoltageSource


def test_linear_ac_analyzer_measures_input_impedance() -> None:
    freqs = np.logspace(1, 5, 24)
    circuit = LinearCircuit(
        elements=(LinearElement("R1", "R", "in", "0", 1234.0),),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )

    result = LinearACAnalyzer().analyze(
        circuit,
        AnalysisRequest.input_impedance(),
        freqs,
    )

    assert result.request.kind == "input_impedance"
    assert np.max(np.abs(result.values - 1234.0)) < 1e-9


def test_synthesizer_can_target_constant_input_impedance() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_constant_input_impedance",
            "ports": 2,
            "analysis": {"kind": "input_impedance"},
            "behavior": {
                "kind": "constant_impedance",
                "ohms": 1000.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R"],
                "parameter_ranges": {"R": [100, 10000]},
            },
            "optimization": {"points": 24, "max_iterations": 18, "top_k": 1, "seed": 22},
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.template.name == "shunt_resistor_impedance"
    assert result.target.analysis.kind == "input_impedance"
    assert abs(result.parameters["R"] - 1000.0) / 1000.0 < 0.02
    assert result.metrics.rmse_db < 0.05
    assert result.as_dict()["analysis"]["kind"] == "input_impedance"


def test_linear_ac_analyzer_measures_output_impedance() -> None:
    freqs = np.logspace(1, 5, 24)
    circuit = LinearCircuit(
        elements=(LinearElement("Rout", "R", "out", "0", 50.0),),
        voltage_sources=(),
    )

    result = LinearACAnalyzer().analyze(
        circuit,
        AnalysisRequest.output_impedance(),
        freqs,
    )

    assert result.request.kind == "output_impedance"
    assert np.max(np.abs(result.values - 50.0)) < 1e-9


def test_synthesizer_can_target_constant_output_impedance() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_constant_output_impedance",
            "ports": 2,
            "analysis": {"kind": "output_impedance"},
            "behavior": {
                "kind": "constant_output_impedance",
                "ohms": 50.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R"],
                "parameter_ranges": {"R": [10, 1000]},
            },
            "optimization": {"points": 24, "max_iterations": 18, "top_k": 1, "seed": 23},
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.template.name == "output_resistor_impedance"
    assert result.target.analysis.kind == "output_impedance"
    assert abs(result.parameters["R"] - 50.0) / 50.0 < 0.02
    assert result.metrics.rmse_db < 0.05
    assert result.as_dict()["analysis"]["kind"] == "output_impedance"


def test_linear_ac_analyzer_measures_transimpedance() -> None:
    freqs = np.logspace(1, 5, 24)
    circuit = LinearCircuit(
        elements=(
            LinearElement("Rf", "R", "out", "in", 1000.0),
            LinearElement("Cf", "C", "out", "in", 1e-9),
        ),
        voltage_sources=(),
        current_sources=(CurrentSource("Iin", "0", "in", 1.0),),
        controlled_voltage_sources=(
            VoltageControlledVoltageSource("Eop", "out", "0", "0", "in", 1_000_000.0),
        ),
    )

    result = LinearACAnalyzer().analyze(
        circuit,
        AnalysisRequest.transimpedance(),
        freqs,
    )

    expected = -1000.0 / (1.0 + 2j * np.pi * freqs * 1000.0 * 1e-9) * (1_000_000.0 / 1_000_001.0)
    assert result.request.kind == "transimpedance"
    assert np.max(np.abs(result.values - expected)) < 1e-8


def test_synthesizer_can_target_transimpedance_amplifier() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "test_tia",
            "ports": 2,
            "analysis": {"kind": "transimpedance"},
            "behavior": {
                "kind": "transimpedance",
                "transimpedance_ohm": 1000.0,
                "cutoff_hz": 100000.0,
                "frequency_range_hz": [10, 1000000],
            },
            "library": {
                "allowed": ["R", "C", "opamp"],
                "parameter_ranges": {"R": [100, 10000], "C": [1e-12, 1e-5]},
            },
            "optimization": {"points": 48, "max_iterations": 30, "top_k": 1, "seed": 31},
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.template.name == "transimpedance_amplifier"
    assert result.target.analysis.kind == "transimpedance"
    assert result.metrics.rmse_db < 0.2
    assert result.as_dict()["topology"]["family"] == "active_transimpedance"
