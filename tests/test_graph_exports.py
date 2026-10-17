from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from circuit_ai.analysis import AnalysisRequest
from circuit_ai.graph import (ComponentInstance, ParameterBinding, ParameterVariable,
                              TerminalConnection, linear_circuit_to_graph, primitive_model_ref)
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


def _nonlinear_graph():
    graph = _graph()
    components = (
        ComponentInstance("diode", "D1", primitive_model_ref("diode"),
            (TerminalConnection("cathode", "0"), TerminalConnection("anode", "out"))),
        ComponentInstance("mos", "M1", primitive_model_ref("nmos_channel"),
            (TerminalConnection("source", "0"), TerminalConnection("gate", "in"), TerminalConnection("drain", "out")),
            (ParameterBinding("width_m", "um", variable_id="width"),)),
    )
    return replace(graph, components=graph.components + components,
                   variables=(ParameterVariable("width", "continuous", "um", lower=1, upper=100),))


def _kicad_cli():
    requested = os.environ.get("LATTICE_KICAD_CLI") or shutil.which("kicad-cli")
    if requested:
        return requested
    if os.name == "nt":
        import winreg
        for hive, base in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                           (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                           (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")):
            try:
                with winreg.OpenKey(hive, base) as key:
                    for index in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            with winreg.OpenKey(key, winreg.EnumKey(key, index)) as entry:
                                if "kicad" not in winreg.QueryValueEx(entry, "DisplayName")[0].lower():
                                    continue
                                path = Path(winreg.QueryValueEx(entry, "InstallLocation")[0]) / "bin" / "kicad-cli.exe"
                                if path.is_file():
                                    return str(path)
                        except OSError:
                            continue
            except OSError:
                continue
    pytest.skip("KiCad CLI unavailable; native schematic round-trip not verified")


def _native_connections(schematic, tmp_path):
    path = tmp_path / "schematic.kicad_sch"
    output = tmp_path / "schematic.xml"
    path.write_text(schematic, encoding="utf-8")
    result = subprocess.run([_kicad_cli(), "sch", "export", "netlist", "--format", "kicadxml",
                             "-o", str(output), str(path)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file(), result.stdout + result.stderr
    parsed = ET.parse(output)
    return {(pin.attrib["ref"], pin.attrib["pin"]): net.attrib["name"].removeprefix("/")
            for net in parsed.findall(".//nets/net") for pin in net.findall("node")}


# Independent public pin-number expectations, not imported from the renderer.
_TERMINAL_NUMBERS = {
    **{kind: {"p": "1", "n": "2"} for kind in ("R", "C", "L", "voltage_source", "current_source")},
    "diode": {"anode": "1", "cathode": "2"}, "ideal_diode": {"anode": "1", "cathode": "2"},
    "nmos_channel": {"drain": "1", "gate": "2", "source": "3"}, "ideal_switch": {"a": "1", "b": "2"},
    "vcvs": {"p": "1", "n": "2", "control_p": "3", "control_n": "4"},
    "ideal_transformer": {"primary_p": "1", "primary_n": "2", "secondary_p": "3", "secondary_n": "4"},
}


def _expected_connections(graph):
    return {(c.reference, _TERMINAL_NUMBERS[c.model.kind][pin.terminal]): pin.net_id
            for c in graph.components for pin in c.connections}


@pytest.mark.parametrize("kind", ["linear", "nonlinear", "vcvs", "isolated_power"])
def test_kicad_native_round_trip_preserves_every_pin_and_net(kind, tmp_path):
    if kind == "linear":
        graph = _graph()
    elif kind == "nonlinear":
        graph = _nonlinear_graph()
    elif kind == "vcvs":
        graph = linear_circuit_to_graph(LinearCircuit(
            elements=(LinearElement("RL", "R", "out", "0", 1000),),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1),),
            controlled_voltage_sources=(VoltageControlledVoltageSource("Eop", "out", "0", "in", "0", 2),)))
    else:
        path = Path("测试集/pbdl/classic/dc_flyback_12v_to_5v.json")
        ir = pbdl_to_ir(json.loads(path.read_text(encoding="utf-8")))
        graph = PowerTopologyGrammar().search(ir).candidates[0].graph
    schematic = export_circuit_graph(graph, {"width": 10.0} if kind == "nonlinear" else {}, formats=("kicad",)).kicad_schematic
    assert schematic is not None
    assert _native_connections(schematic, tmp_path) == _expected_connections(graph)


def test_nonlinear_svg_uses_complete_named_pin_map_and_bound_values():
    graph = _nonlinear_graph()
    bundle = export_circuit_graph(graph, {"width": 12.0}, formats=("svg", "kicad"))
    assert "width_m=12.0 um" in bundle.svg and "width_m=12.0 um" in bundle.kicad_schematic
    root = ET.fromstring(bundle.svg)
    actual = {(group.attrib["data-component"], pin.attrib["data-number"]): pin.attrib["data-net"]
              for group in root.findall("{*}g") for pin in group.findall("{*}g")}
    assert actual == _expected_connections(graph)
    assert '(lib_id "CircuitAI:NMOS_CHANNEL")' in bundle.kicad_schematic


def test_unknown_model_is_not_silently_exported_as_a_switch():
    graph = _nonlinear_graph()
    part = graph.components[-1]
    part = replace(part, model=replace(part.model, kind="unknown_device"))
    graph = replace(graph, components=graph.components[:-1] + (part,))
    bundle = export_circuit_graph(graph, formats=("svg", "kicad"))
    assert bundle.kicad_schematic is None
    assert any("no registered symbol" in item for item in bundle.diagnostics)
    assert bundle.svg is not None and 'data-terminal="source"' in bundle.svg


def test_legacy_vcvs_polarity_survives_native_kicad_round_trip(tmp_path):
    from circuit_ai.kicad import render_circuit_kicad_schematic
    circuit = LinearCircuit(elements=(LinearElement("R1", "R", "out", "0", 1000),),
        voltage_sources=(VoltageSource("V1", "in", "0", 1),),
        controlled_voltage_sources=(VoltageControlledVoltageSource("E1", "out", "0", "in", "0", 2),))
    expected = _expected_connections(linear_circuit_to_graph(circuit))
    assert _native_connections(render_circuit_kicad_schematic(circuit), tmp_path) == expected
