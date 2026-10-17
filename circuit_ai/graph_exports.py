"""Unified CircuitGraph-driven export facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
import math
from typing import Any, Iterable, Mapping

from .graph import CircuitGraph, graph_to_linear_circuit
from .kicad import (
    EMBEDDED_SYMBOL_PINS,
    _Pin,
    _PlacedSymbol,
    _Point,
    _library_symbols,
    _pin_connection_blocks,
    _q,
    _stable_uuid,
    _symbol_block,
    _sim_metadata,
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


@dataclass(frozen=True)
class ResolvedSymbol:
    lib_id: str
    terminals: tuple[str, ...]
    pins: tuple[_Pin, ...]


class SymbolResolver:
    """Model terminal names -> embedded pin numbers and sheet coordinates.

    Connection list order is not pin order. Geometry comes from the same table
    that writes the embedded library; an unknown model is never drawn as a switch.
    """

    _models = {
        "R": ("R", ("p", "n")), "C": ("C", ("p", "n")), "L": ("L", ("p", "n")),
        "voltage_source": ("VDC", ("p", "n")), "current_source": ("IDC", ("p", "n")),
        "vcvs": ("ESOURCE", ("p", "n", "control_p", "control_n")),
        "ideal_switch": ("SWITCH", ("a", "b")), "ideal_diode": ("DIODE", ("anode", "cathode")),
        "diode": ("JUNCTION_DIODE", ("anode", "cathode")),
        "nmos_channel": ("NMOS_CHANNEL", ("drain", "gate", "source")),
        "ideal_transformer": ("TRANSFORMER", ("primary_p", "primary_n", "secondary_p", "secondary_n")),
    }

    def resolve(self, component) -> ResolvedSymbol:
        kind = component.model.kind
        if kind not in self._models:
            raise ValueError(f"no registered symbol for model {kind!r}")
        name, terminals = self._models[kind]
        connections = {item.terminal: item.net_id for item in component.connections}
        if set(connections) != set(terminals):
            raise ValueError(f"symbol terminal contract mismatch for {component.reference!r}")
        pins = []
        for terminal, definition in zip(terminals, EMBEDDED_SYMBOL_PINS[name], strict=True):
            _, _, _, x, y, angle = definition
            theta = math.radians(angle)
            offset = _Point(round(-6.35 * math.cos(theta), 6), round(6.35 * math.sin(theta), 6))
            pins.append(_Pin(connections[terminal], _Point(x, -y), offset))
        return ResolvedSymbol(f"CircuitAI:{name}", terminals, tuple(pins))


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
                svg = _render_structural_svg(graph, title or graph.name, bound_values)
    if "svg" in requested and svg is None and circuit is not None:
        svg = render_circuit_svg(
            circuit,
            title=title or graph.name,
            subtitle=f"graph_hash={graph.graph_hash[:12]}",
        )
    if "kicad" in requested:
        try:
            kicad = _render_structural_kicad(graph, title or graph.name, bound_values, model_bindings)
        except (ValueError, TypeError, KeyError) as exc:
            diagnostics.append(f"KiCad export unavailable: {exc}")
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


__all__ = ["CircuitGraphExportBundle", "SymbolResolver", "export_circuit_graph"]


def _render_structural_svg(graph: CircuitGraph, title: str, values=None) -> str:
    """An explicit terminal/net view when a device has no pictorial renderer."""
    resolver = SymbolResolver()
    height = 100 + sum(84 + len(c.model.terminals) * 30 for c in graph.components)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img">',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.label{font-size:16px;font-weight:700}.detail{font-size:12px;fill:#52606d}.wire{stroke:#176b61;stroke-width:2}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="32" y="36">{escape(title)}</text>',
        f'<text class="detail" x="32" y="60">Terminal connectivity / CircuitGraph {escape(graph.graph_hash[:16])}</text>',
    ]
    y = 80
    for component in graph.components:
        try:
            resolved = resolver.resolve(component)
            terminals = resolved.terminals
        except ValueError:
            # Preserve every declared terminal in the inspection view even when
            # no KiCad symbol is registered. Do not invent a substitute device.
            terminals = tuple(t.name for t in component.model.terminals)
        connections = {c.terminal: c.net_id for c in component.connections}
        card_height = 68 + len(terminals) * 30
        lines.extend([
            f'<g data-component="{escape(component.reference, quote=True)}">',
            f'<rect x="32" y="{y}" width="936" height="{card_height}" rx="8" fill="#f7faf9" stroke="#ccd8d3"/>',
            f'<text class="label" x="52" y="{y+26}">{escape(component.reference)} [{escape(component.model.kind)}]</text>',
            f'<text class="detail" x="52" y="{y+47}">{escape(_graph_component_value(component, values))}</text>',
        ])
        for number, terminal in enumerate(terminals, 1):
            net = connections[terminal]
            row = y + 58 + number * 30
            lines.extend([
                f'<g class="pin" data-number="{number}" data-terminal="{escape(terminal, quote=True)}" data-net="{escape(net, quote=True)}">',
                f'<text class="detail" x="64" y="{row-4}">{number}: {escape(terminal)}</text>',
                f'<line class="wire" x1="260" y1="{row-8}" x2="530" y2="{row-8}"/>',
                f'<circle cx="260" cy="{row-8}" r="3" fill="#176b61"/>',
                f'<text class="detail" x="550" y="{row-4}">{escape(net)}</text>',
                '</g>',
            ])
        lines.append('</g>')
        y += card_height + 16
    lines.append('</svg>')
    return "\n".join(lines)


def _render_structural_kicad(graph: CircuitGraph, title: str, values=None, model_bindings=()) -> str:
    symbols: list[_PlacedSymbol] = []
    resolver = SymbolResolver()
    bindings = {str(item.get("component", item.get("source_name", ""))): item for item in model_bindings}
    for index, component in enumerate(graph.components, start=1):
        resolved = resolver.resolve(component)
        binding = bindings.get(component.instance_id) or bindings.get(component.reference)
        metadata = _sim_metadata(binding, tuple(str(i + 1) for i in range(len(resolved.pins))))
        symbols.append(
            _PlacedSymbol(
                lib_id=resolved.lib_id,
                reference=component.reference,
                value=f"{component.model.kind} {_graph_component_value(component, values)}",
                source_name=component.instance_id,
                position=_Point(76.2, 30.48 + (index - 1) * 35.56),
                rotation=0,
                pins=resolved.pins,
                sim_library=metadata[0], sim_name=metadata[1], sim_pins=metadata[2], sim_device=metadata[3],
                label_x_offset=25.4 if component.model.kind in {"R", "C", "L", "voltage_source", "current_source", "vcvs"} else 0.0,
            )
        )
    project_name = graph.graph_id
    lines = [
        "(kicad_sch",
        "\t(version 20250114)",
        '\t(generator "circuit_ai")',
        '\t(generator_version "0.1")',
        f"\t(uuid {_q(_stable_uuid(project_name, 'root'))})",
        f'\t(paper "User" 210 {max(297, 65 + len(symbols) * 35.56):.2f})',
        "\t(title_block",
        f"\t\t(title {_q(title)})",
        f"\t\t(comment 1 {_q('CircuitGraph ' + graph.graph_hash)})",
        "\t)",
        *_library_symbols(),
    ]
    for index, symbol in enumerate(symbols, start=1):
        lines.extend(_symbol_block(symbol, project_name, index))
    for index, symbol in enumerate(symbols, start=1):
        lines.extend(_pin_connection_blocks(symbol, project_name, index, normalize_ground=False))
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


def _graph_component_value(component: Any, values=None) -> str:
    labels = []
    for parameter in component.parameters:
        value = parameter.value if parameter.value is not None else (values or {}).get(parameter.variable_id, parameter.variable_id)
        if value is not None:
            labels.append(f"{parameter.name}={value} {parameter.unit}".strip())
    return "; ".join(labels)
