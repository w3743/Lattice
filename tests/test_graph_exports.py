from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from circuit_ai.analysis import AnalysisRequest
from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.graph_exports import export_circuit_graph
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.mna import (
    LinearCircuit,
    LinearElement,
    VoltageControlledVoltageSource,
    VoltageSource,
)
from circuit_ai.simulation import legacy_ac_simulation_request
from circuit_ai.topology_grammar import PowerTopologyGrammar


def _graph():
    return linear_circuit_to_graph(
        LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "out", 1000.0),
                LinearElement("C1", "C", "out", "0", 1e-7),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        ),
        name="export_graph",
    )


def test_all_exports_share_graph_and_topology_hashes() -> None:
    graph = _graph()
    request = legacy_ac_simulation_request(
        graph,
        AnalysisRequest.voltage_transfer(),
        np.logspace(1, 5, 8),
    )
    bundle = export_circuit_graph(
        graph,
        formats=("svg", "kicad", "spice"),
        spice_request=request,
    )
    assert bundle.graph_hash == graph.graph_hash
    assert bundle.topology_hash == graph.topology_hash
    assert bundle.svg and "R1" in bundle.svg and "C1" in bundle.svg
    assert bundle.kicad_schematic and "CircuitAI:R" in bundle.kicad_schematic
    assert bundle.spice_netlist and "R1 in out" in bundle.spice_netlist


def test_export_reports_missing_spice_request_without_affecting_other_formats() -> None:
    bundle = export_circuit_graph(_graph(), formats=("svg", "spice"))
    assert bundle.svg
    assert bundle.spice_netlist is None
    assert "spice_request is required" in bundle.diagnostics[0]


def test_spice_export_handles_controlled_voltage_sources() -> None:
    graph = linear_circuit_to_graph(
        LinearCircuit(
            elements=(LinearElement("R1", "R", "out", "0", 1000.0),),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Eop", "out", "0", "in", "0", 2.0),
            ),
        ),
        name="export_vcvs",
    )
    request = legacy_ac_simulation_request(
        graph,
        AnalysisRequest.voltage_transfer(),
        np.logspace(1, 4, 5),
    )

    bundle = export_circuit_graph(graph, formats=("spice",), spice_request=request)

    assert bundle.spice_netlist
    assert ".param p_1_Eop=" in bundle.spice_netlist
    assert "Eop out 0 in 0 {p_1_Eop}" in bundle.spice_netlist


def test_power_graph_gets_structural_exports_when_linear_backend_is_inapplicable() -> None:
    spec_path = next(Path(".").glob("*/examples/5v_to_10v_dc_boost.json"))
    ir = pbdl_to_ir(json.loads(spec_path.read_text(encoding="utf-8")))
    candidate = PowerTopologyGrammar().search(ir).candidates[0]

    bundle = export_circuit_graph(candidate.graph, formats=("svg", "kicad"))

    assert bundle.svg and "ideal_switch" in bundle.svg
    assert bundle.kicad_schematic and "CircuitAI:SWITCH" in bundle.kicad_schematic
    assert any("structural graph export" in item for item in bundle.diagnostics)
