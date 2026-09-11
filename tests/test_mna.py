from __future__ import annotations

import numpy as np

from circuit_ai.mna import (
    CurrentSource,
    LinearCircuit,
    LinearElement,
    MNASimulator,
    VoltageControlledVoltageSource,
    VoltageSource,
)


def test_mna_rc_lowpass_matches_closed_form() -> None:
    freqs = np.logspace(1, 5, 60)
    r = 10_000.0
    c = 15.915494309e-9
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", r),
            LinearElement("C1", "C", "out", "0", c),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )

    simulated = MNASimulator().transfer(circuit, "out", "Vin", freqs)
    expected = 1.0 / (1.0 + 2j * np.pi * freqs * r * c)

    assert np.max(np.abs(simulated - expected)) < 1e-10


def test_mna_rlc_bandpass_matches_closed_form() -> None:
    freqs = np.logspace(2, 6, 80)
    r = 1000.0
    l = 10e-3
    c = 25.33029591e-9
    circuit = LinearCircuit(
        elements=(
            LinearElement("C1", "C", "in", "n1", c),
            LinearElement("L1", "L", "n1", "out", l),
            LinearElement("R1", "R", "out", "0", r),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )

    simulated = MNASimulator().transfer(circuit, "out", "Vin", freqs)
    s = 2j * np.pi * freqs
    expected = (s * r * c) / (s**2 * l * c + s * r * c + 1.0)

    assert np.max(np.abs(simulated - expected)) < 1e-10


def test_mna_voltage_controlled_voltage_source_gain() -> None:
    freqs = np.logspace(1, 5, 20)
    circuit = LinearCircuit(
        elements=(),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        controlled_voltage_sources=(
            VoltageControlledVoltageSource("E1", "out", "0", "in", "0", 2.5),
        ),
    )

    simulated = MNASimulator().transfer(circuit, "out", "Vin", freqs)

    assert np.max(np.abs(simulated - 2.5)) < 1e-12


def test_mna_current_source_drives_resistor() -> None:
    freqs = np.logspace(1, 5, 12)
    circuit = LinearCircuit(
        elements=(LinearElement("R1", "R", "in", "0", 1000.0),),
        voltage_sources=(),
        current_sources=(CurrentSource("Iin", "0", "in", 1e-3),),
    )

    solutions = MNASimulator().solve_ac(circuit, freqs)
    voltages = np.asarray([solution["in"] for solution in solutions])

    assert np.max(np.abs(voltages - 1.0)) < 1e-12
