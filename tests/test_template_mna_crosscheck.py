from __future__ import annotations

import numpy as np
import pytest

from circuit_ai.templates import CircuitTemplate, default_templates


TEMPLATE_VALUES = {
    "rc_lowpass": {"R": 10_000.0, "C": 15.915494309e-9},
    "rc_highpass": {"R": 10_000.0, "C": 15.915494309e-9},
    "rlc_bandpass": {"R": 1000.0, "L": 10e-3, "C": 25.33029591e-9},
    "shunt_resistor_impedance": {"R": 1000.0},
    "output_resistor_impedance": {"R": 50.0},
    "buffered_cascade_rc_lowpass": {
        "R1": 10_000.0,
        "C1": 15.915494309e-9,
        "R2": 22_000.0,
        "C2": 7.234315595e-9,
    },
    "sallen_key_lowpass": {
        "R1": 10_000.0,
        "R2": 10_000.0,
        "C1": 15.915494309e-9,
        "C2": 15.915494309e-9,
    },
    "sallen_key_gain_lowpass": {
        "R1": 10_000.0,
        "R2": 10_000.0,
        "C1": 15.915494309e-9,
        "C2": 15.915494309e-9,
        "Rg": 10_000.0,
        "Rf": 5_860.0,
    },
    "gain_rc_lowpass": {"R": 10_000.0, "C": 15.915494309e-9, "Rg": 10_000.0, "Rf": 20_000.0},
    "gain_rc_highpass": {"R": 10_000.0, "C": 15.915494309e-9, "Rg": 10_000.0, "Rf": 20_000.0},
    "transimpedance_amplifier": {"Rf": 1000.0, "Cf": 1e-9},
}


def test_circuit_template_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        CircuitTemplate()


@pytest.mark.parametrize("template", default_templates(), ids=lambda template: template.name)
def test_hand_template_formula_matches_mna_export(template: CircuitTemplate) -> None:
    frequencies = np.logspace(1, 6, 96)
    values = TEMPLATE_VALUES[template.name]

    analytic = template.response(values, frequencies)
    simulated = template.mna_response(values, frequencies)

    scale = max(1.0, float(np.max(np.abs(analytic))))
    assert np.max(np.abs(simulated - analytic)) / scale < 1e-9
