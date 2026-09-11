"""Unified CircuitGraph-driven export facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from typing import Any, Iterable, Mapping

from .graph import CircuitGraph, graph_to_linear_circuit
from .kicad import (
    _Pin,
    _PlacedSymbol,
    _Point,
    _library_symbols,
    _pin_connection_blocks,
    _q,
    _stable_uuid,
    _symbol_block,
    render_circuit_kicad_schematic,
)
from .schematic import render_circuit_svg
from .simulation import (
    NgspiceSimulatorBackend,
    SimulationRequest,
    build_ngspice_netlist,
)


@dataclass(frozen=True)
class CircuitGraphExportBundle:
    graph_hash: str
    topology_hash: str
    spice_netlist: str | None = None
    svg: str | None = None
    kicad_schematic: str | None = None
    diagnostics: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_hash": self.graph_hash,
            "topology_hash": self.topology_hash,
            "spice_netlist": self.spice_netlist,
            "svg": self.svg,
            "kicad_schematic": self.kicad_schematic,
            "diagnostics": list(self.diagnostics),
        }


def export_circuit_graph(
    graph: CircuitGraph,
    parameter_values: Mapping[str, Any] | None = None,
    *,
    formats: Iterable[str] = ("spice", "svg", "kicad"),
    spice_request: SimulationRequest | None = None,
    spice_executable: str = "ngspice",
    model_bindings: tuple[dict[str, Any], ...] = (),
    title: str | None = None,
) -> CircuitGraphExportBundle:
    """Export one validated graph without reconstructing a second netlist."""

    graph.require_valid()
    requested = {str(item).casefold() for item in formats}
    values = dict(parameter_values or {})
    bound_ids = {
        parameter.variable_id
        for component in graph.components
        for parameter in component.parameters
        if parameter.variable_id is not None
    }
    bound_values = {key: value for key, value in values.items() if key in bound_ids}
    circuit = None
    diagnostics: list[str] = []
    spice = svg = kicad = None

    if requested & {"svg", "kicad"}:
        try:
            circuit = graph_to_linear_circuit(graph, bound_values)
        except (ValueError, TypeError, KeyError) as exc:
            diagnostics.append(
                "linear graph materialization unavailable; using structural graph export: "
                + str(exc)
            )
            if "svg" in requested:
                svg = _render_structural_svg(graph, title or graph.name)
            if "kicad" in requested:
                kicad = _render_structural_kicad(graph, title or graph.name)
    if "svg" in requested and svg is None and circuit is not None:
        svg = render_circuit_svg(
            circuit,
            title=title or graph.name,
            subtitle=f"graph_hash={graph.graph_hash[:12]}",
        )
    if "kicad" in requested and kicad is None and circuit is not None:
        kicad = render_circuit_kicad_schematic(
            circuit,
            title=title or graph.name,
            project_name=graph.graph_id,
            model_bindings=model_bindings,
        )
    if "spice" in requested:
        if spice_request is None:
            diagnostics.append("spice export skipped: spice_request is required")
        else:
            try:
                backend = NgspiceSimulatorBackend(executable=spice_executable)
                compiled = backend.compile(graph)
                request = replace(
                    spice_request,
                    graph_id=graph.graph_id,
                    parameter_values=bound_values or {
                        key: value
                        for key, value in spice_request.parameter_values.items()
                        if key in bound_ids
                    },
                    fidelity="spice",
                )
                spice = build_ngspice_netlist(compiled, request, "result.raw")
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                diagnostics.append(f"spice export unavailable for graph models: {exc}")

    unknown = requested - {"spice", "svg", "kicad"}
    if unknown:
        diagnostics.append(f"unsupported export formats: {sorted(unknown)}")
    return CircuitGraphExportBundle(
        graph_hash=graph.graph_hash,
        topology_hash=graph.topology_hash,
        spice_netlist=spice,
        svg=svg,
        kicad_schematic=kicad,
        diagnostics=tuple(diagnostics),
    )


__all__ = ["CircuitGraphExportBundle", "export_circuit_graph"]


def _render_structural_svg(graph: CircuitGraph, title: str) -> str:
    nets = sorted(graph.nets, key=lambda item: (item.is_reference is False, item.net_id))
    net_positions = {net.net_id: (100 + index * 150, 430) for index, net in enumerate(nets)}
    width = max(760, 180 + max(len(nets) - 1, 1) * 150)
    height = max(520, 150 + len(graph.components) * 58)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.sub{font-size:12px;fill:#667085}.wire{stroke:#263238;stroke-width:2}.part{fill:#fff;stroke:#176b61;stroke-width:2}.label{font-size:13px;font-weight:700}.net{font-size:12px;fill:#52606d}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="32" y="36">{escape(title)}</text>',
        f'<text class="sub" x="32" y="58">CircuitGraph {escape(graph.graph_hash[:16])}</text>',
    ]
    for index, component in enumerate(graph.components):
        connections = [item.net_id for item in component.connections]
        connected = [net_positions[net] for net in connections if net in net_positions]
        if not connected:
            continue
        center_x = sum(point[0] for point in connected) / len(connected)
        center_y = 110 + index * 58
        for point in connected:
            lines.append(f'<line class="wire" x1="{center_x:.1f}" y1="{center_y:.1f}" x2="{point[0]}" y2="{point[1]}"/>')
        kind = escape(component.model.kind)
        reference = escape(component.reference)
        value = escape(_graph_component_value(component))
        lines.extend([
            f'<rect class="part" x="{center_x - 52:.1f}" y="{center_y - 18:.1f}" width="104" height="36" rx="6"/>',
            f'<text class="label" x="{center_x - 46:.1f}" y="{center_y - 2:.1f}">{reference} [{kind}]</text>',
            f'<text class="net" x="{center_x - 46:.1f}" y="{center_y + 13:.1f}">{value}</text>',
        ])
    for net, point in net_positions.items():
        lines.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="4" fill="#263238"/>')
        lines.append(f'<text class="net" x="{point[0] - 20}" y="{point[1] + 24}">{escape(net)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def _render_structural_kicad(graph: CircuitGraph, title: str) -> str:
    symbols: list[_PlacedSymbol] = []
    model_to_lib = {
        "R": "CircuitAI:R",
        "C": "CircuitAI:C",
        "L": "CircuitAI:L",
        "voltage_source": "CircuitAI:VDC",
        "current_source": "CircuitAI:IDC",
        "vcvs": "CircuitAI:ESOURCE",
        "ideal_switch": "CircuitAI:SWITCH",
        "ideal_diode": "CircuitAI:DIODE",
        "ideal_transformer": "CircuitAI:TRANSFORMER",
    }
    for index, component in enumerate(graph.components, start=1):
        nodes = [item.net_id for item in component.connections]
        pins = _structural_pins(nodes, component.model.kind)
        symbols.append(
            _PlacedSymbol(
                lib_id=model_to_lib.get(component.model.kind, "CircuitAI:SWITCH"),
                reference=component.reference,
                value=f"{component.model.kind} {_graph_component_value(component)}",
                source_name=component.instance_id,
                position=_Point(76.2, 30.48 + (index - 1) * 20.32),
                rotation=0,
                pins=pins,
            )
        )
    project_name = graph.graph_id
    lines = [
        "(kicad_sch",
        "\t(version 20250114)",
        '\t(generator "circuit_ai")',
        '\t(generator_version "0.1")',
        f"\t(uuid {_q(_stable_uuid(project_name, 'root'))})",
        "\t(paper \"A4\")",
        "\t(title_block",
        f"\t\t(title {_q(title)})",
        f"\t\t(comment 1 {_q('CircuitGraph ' + graph.graph_hash)})",
        "\t)",
        *_library_symbols(),
    ]
    for index, symbol in enumerate(symbols, start=1):
        lines.extend(_symbol_block(symbol, project_name, index))
    for index, symbol in enumerate(symbols, start=1):
        lines.extend(_pin_connection_blocks(symbol, project_name, index))
    lines.extend([
        "\t(sheet_instances",
        "\t\t(path \"/\"",
        "\t\t\t(page \"1\")",
        "\t\t)",
        "\t)",
        "\t(embedded_fonts no)",
        ")",
    ])
    return "\n".join(lines) + "\n"


def _structural_pins(nodes: list[str], kind: str) -> tuple[_Pin, ...]:
    if len(nodes) == 4 and kind == "ideal_transformer":
        offsets = ((-5.08, -2.54), (-5.08, 2.54), (5.08, -2.54), (5.08, 2.54))
    elif len(nodes) >= 4:
        offsets = ((0.0, 5.08), (0.0, -5.08), (-5.08, 5.08), (-5.08, -5.08))
    else:
        offsets = ((-3.81, 0.0), (3.81, 0.0))
    return tuple(
        _Pin(node, _Point(*offset), _Point(-6.35 if offset[0] < 0 else 6.35, 0.0))
        for node, offset in zip(nodes, offsets)
    )


def _graph_component_value(component: Any) -> str:
    for parameter in component.parameters:
        value = parameter.value if parameter.value is not None else parameter.variable_id
        if value is not None:
            return f"{parameter.name}={value}"
    return ""
