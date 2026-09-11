from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from typing import Iterable

from .formatting import eng
from .mna import (
    CurrentSource,
    GROUND_NAMES,
    LinearCircuit,
    LinearElement,
    VoltageControlledVoltageSource,
    VoltageSource,
)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def add(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def sub(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def mul(self, value: float) -> "Point":
        return Point(self.x * value, self.y * value)


@dataclass(frozen=True)
class _Branch:
    name: str
    kind: str
    n1: str
    n2: str
    value: str
    role: str
    control_plus: str | None = None
    control_minus: str | None = None


def render_circuit_svg(
    circuit: LinearCircuit,
    *,
    title: str = "Circuit schematic",
    subtitle: str = "",
    footer: Iterable[str] = (),
) -> str:
    """Render a compact SVG schematic from the unified LinearCircuit IR."""

    layout = _layout_nodes(circuit)
    width = layout["width"]
    height = layout["height"]
    node_points: dict[str, Point] = layout["nodes"]

    branches = _branches(circuit)
    branch_counts: dict[tuple[str, str], int] = {}
    branch_indices: dict[tuple[str, str], int] = {}
    for branch in branches:
        key = _branch_key(branch.n1, branch.n2)
        branch_counts[key] = branch_counts.get(key, 0) + 1

    body: list[str] = [_svg_header(width, height)]
    body.append(f"<text class=\"title\" x=\"32\" y=\"34\">{escape(title)}</text>")
    if subtitle:
        body.append(f"<text class=\"subtitle\" x=\"32\" y=\"58\">{escape(subtitle)}</text>")

    body.append("<g class=\"wires\">")
    for branch in branches:
        key = _branch_key(branch.n1, branch.n2)
        index = branch_indices.get(key, 0)
        branch_indices[key] = index + 1
        count = branch_counts[key]
        offset = (index - (count - 1) / 2.0) * 24.0
        body.extend(_draw_branch(branch, node_points, offset))
    body.append("</g>")

    body.append("<g class=\"nodes\">")
    for node, point in node_points.items():
        body.append(_circle(point.x, point.y, 4, "node-dot"))
        label = _node_label(node)
        if node == "in":
            body.append(_text(point.x - 20, point.y - 18, "IN", "port-label"))
        elif node == "out":
            body.append(_text(point.x + 10, point.y - 18, "OUT", "port-label"))
        body.append(_text(point.x - 12, point.y + 24, label, "node-label"))
    body.append("</g>")

    footer_lines = [line for line in footer if line]
    if footer_lines:
        body.append("<g class=\"footer\">")
        for index, line in enumerate(footer_lines):
            body.append(_text(32, height - 36 + index * 18, line, "footer-text"))
        body.append("</g>")

    body.append("</svg>")
    return "\n".join(body)


def render_result_svg(result) -> str:
    """Render a synthesis result, including key optimization metrics."""

    circuit = result.template.to_circuit(result.parameters)
    metrics = result.metrics
    subtitle = (
        f"{result.template.name} | score={metrics.score:.4g}, "
        f"rmse={metrics.rmse_db:.4g} dB, components={metrics.component_count}"
    )
    footer = [
        f"Pareto rank: {metrics.pareto_rank}" if metrics.pareto_rank is not None else "",
        f"Optimizer: {metrics.optimizer_message}",
    ]
    return render_circuit_svg(
        circuit,
        title=result.template.description or result.template.name,
        subtitle=subtitle,
        footer=footer,
    )


def render_results_index_html(results, schematic_files: list[str]) -> str:
    rows = []
    for index, (result, filename) in enumerate(zip(results, schematic_files), start=1):
        metrics = result.metrics
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><a href=\"{escape(filename)}\">{escape(result.template.name)}</a></td>"
            f"<td>{metrics.rmse_db:.4g}</td>"
            f"<td>{metrics.max_abs_db:.4g}</td>"
            f"<td>{metrics.component_count}</td>"
            f"<td>{'' if metrics.pareto_rank is None else metrics.pareto_rank}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<title>Circuit AI schematics</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;color:#172033;background:#f7f8fb}",
            "h1{font-size:24px;margin:0 0 16px}",
            "table{border-collapse:collapse;background:white;border:1px solid #d8deea}",
            "th,td{padding:8px 12px;border-bottom:1px solid #e5e9f2;text-align:left}",
            "th{background:#edf1f7}",
            "a{color:#1358a8;text-decoration:none}",
            "iframe{display:block;width:100%;height:560px;border:1px solid #d8deea;background:white;margin-top:18px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Circuit AI schematics</h1>",
            "<table>",
            "<thead><tr><th>#</th><th>Template</th><th>RMSE dB</th><th>Max dB</th><th>Parts</th><th>Pareto</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
            f"<iframe src=\"{escape(schematic_files[0])}\"></iframe>" if schematic_files else "",
            "</body>",
            "</html>",
        ]
    )


def _layout_nodes(circuit: LinearCircuit) -> dict:
    nodes = list(circuit.nodes())
    nodes.sort(key=_node_sort_key)
    if not nodes:
        nodes = ["in", "out"]

    width = max(760, 180 + max(len(nodes) - 1, 1) * 130)
    height = 430
    left = 95.0
    right = width - 95.0
    span = right - left
    points: dict[str, Point] = {}
    for index, node in enumerate(nodes):
        x = left + (span * index / max(len(nodes) - 1, 1))
        y = 150.0 + _node_lane(node, index) * 58.0
        points[node] = Point(x, y)
    return {"width": int(width), "height": height, "nodes": points}


def _branches(circuit: LinearCircuit) -> list[_Branch]:
    branches: list[_Branch] = []
    for element in circuit.elements:
        branches.append(
            _Branch(
                name=element.name,
                kind=element.kind.upper(),
                n1=element.n1,
                n2=element.n2,
                value=_element_value(element),
                role="element",
            )
        )
    for source in circuit.voltage_sources:
        branches.append(
            _Branch(
                name=source.name,
                kind="V",
                n1=source.n_plus,
                n2=source.n_minus,
                value=_complex_value(source.value),
                role="voltage_source",
            )
        )
    for source in circuit.current_sources:
        branches.append(
            _Branch(
                name=source.name,
                kind="I",
                n1=source.n_plus,
                n2=source.n_minus,
                value=_complex_value(source.value),
                role="current_source",
            )
        )
    for source in circuit.controlled_voltage_sources:
        branches.append(
            _Branch(
                name=source.name,
                kind="E",
                n1=source.n_plus,
                n2=source.n_minus,
                value=f"gain={_complex_value(source.gain)}",
                role="vcvs",
                control_plus=source.control_plus,
                control_minus=source.control_minus,
            )
        )
    return branches


def _draw_branch(branch: _Branch, nodes: dict[str, Point], offset: float) -> list[str]:
    p1 = _node_point(branch.n1, nodes)
    p2 = _node_point(branch.n2, nodes, reference=p1)
    p1, p2 = _offset_points(p1, p2, offset)
    unit, normal, length = _basis(p1, p2)
    if length < 1.0:
        return []

    center = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
    symbol_len = min(78.0, max(42.0, length * 0.58))
    start = center.sub(unit.mul(symbol_len / 2.0))
    end = center.add(unit.mul(symbol_len / 2.0))

    svg = [
        _line(p1, start, "wire"),
        _line(end, p2, "wire"),
        *_draw_symbol(branch, start, end, center, unit, normal),
    ]
    label_pos = center.add(normal.mul(-24.0 if branch.role == "voltage_source" else 22.0))
    svg.append(_text(label_pos.x, label_pos.y, f"{branch.name} {branch.value}", "part-label"))
    if _is_ground(branch.n1):
        svg.extend(_draw_ground(p1))
    if _is_ground(branch.n2):
        svg.extend(_draw_ground(p2))
    if branch.role == "vcvs" and branch.control_plus and branch.control_minus:
        svg.extend(_draw_control_hint(branch, nodes, center))
    return svg


def _draw_symbol(branch: _Branch, start: Point, end: Point, center: Point, unit: Point, normal: Point) -> list[str]:
    if branch.kind == "R":
        return [_polyline(_zigzag(start, end, unit, normal), "resistor")]
    if branch.kind == "C":
        gap = 9.0
        plate = 25.0
        a = center.sub(unit.mul(gap))
        b = center.add(unit.mul(gap))
        return [
            _line(start, a, "wire"),
            _line(b, end, "wire"),
            _line(a.sub(normal.mul(plate / 2.0)), a.add(normal.mul(plate / 2.0)), "capacitor"),
            _line(b.sub(normal.mul(plate / 2.0)), b.add(normal.mul(plate / 2.0)), "capacitor"),
        ]
    if branch.kind == "L":
        return [_path(_inductor_path(start, end, unit, normal), "inductor")]
    if branch.kind == "V":
        return [
            _circle(center.x, center.y, 22, "source"),
            _line(start, center.sub(unit.mul(22.0)), "wire"),
            _line(center.add(unit.mul(22.0)), end, "wire"),
            _text(center.x - 5, center.y - 8, "+", "polarity"),
            _text(center.x - 4, center.y + 14, "-", "polarity"),
        ]
    if branch.kind == "I":
        arrow_start = center.sub(unit.mul(10.0))
        arrow_end = center.add(unit.mul(10.0))
        arrow_left = arrow_end.sub(unit.mul(7.0)).add(normal.mul(5.0))
        arrow_right = arrow_end.sub(unit.mul(7.0)).sub(normal.mul(5.0))
        return [
            _circle(center.x, center.y, 22, "source"),
            _line(start, center.sub(unit.mul(22.0)), "wire"),
            _line(center.add(unit.mul(22.0)), end, "wire"),
            _line(arrow_start, arrow_end, "part"),
            _polyline([arrow_left, arrow_end, arrow_right], "part"),
        ]
    if branch.kind == "E":
        r = 27.0
        points = [
            center.sub(unit.mul(r)),
            center.add(normal.mul(r)),
            center.add(unit.mul(r)),
            center.sub(normal.mul(r)),
        ]
        return [
            _polygon(points, "vcvs"),
            _line(start, center.sub(unit.mul(r)), "wire"),
            _line(center.add(unit.mul(r)), end, "wire"),
            _text(center.x - 5, center.y - 7, "+", "polarity"),
            _text(center.x - 4, center.y + 15, "-", "polarity"),
        ]
    return [_line(start, end, "part")]


def _draw_control_hint(branch: _Branch, nodes: dict[str, Point], center: Point) -> list[str]:
    cp = _node_point(branch.control_plus or "0", nodes)
    cm = _node_point(branch.control_minus or "0", nodes, reference=cp)
    sense_mid = Point((cp.x + cm.x) / 2.0, (cp.y + cm.y) / 2.0)
    return [
        _line(sense_mid, center, "control-wire"),
        _text((sense_mid.x + center.x) / 2.0 + 8, (sense_mid.y + center.y) / 2.0 - 8, "control", "hint-label"),
    ]


def _node_point(node: str, nodes: dict[str, Point], reference: Point | None = None) -> Point:
    if _is_ground(node):
        if reference is None:
            return Point(95.0, 300.0)
        return Point(reference.x, 305.0)
    return nodes[node]


def _offset_points(p1: Point, p2: Point, offset: float) -> tuple[Point, Point]:
    if abs(offset) < 0.01:
        return p1, p2
    _, normal, _ = _basis(p1, p2)
    delta = normal.mul(offset)
    return p1.add(delta), p2.add(delta)


def _basis(p1: Point, p2: Point) -> tuple[Point, Point, float]:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = max(math.hypot(dx, dy), 1e-9)
    unit = Point(dx / length, dy / length)
    normal = Point(-unit.y, unit.x)
    return unit, normal, length


def _zigzag(start: Point, end: Point, unit: Point, normal: Point) -> list[Point]:
    length = math.hypot(end.x - start.x, end.y - start.y)
    points = [start]
    steps = 8
    amplitude = 10.0
    for idx in range(1, steps):
        along = start.add(unit.mul(length * idx / steps))
        sign = -1.0 if idx % 2 else 1.0
        points.append(along.add(normal.mul(amplitude * sign)))
    points.append(end)
    return points


def _inductor_path(start: Point, end: Point, unit: Point, normal: Point) -> str:
    length = math.hypot(end.x - start.x, end.y - start.y)
    loops = 4
    step = length / loops
    parts = [f"M {start.x:.2f} {start.y:.2f}"]
    for idx in range(loops):
        a = start.add(unit.mul(step * idx))
        b = start.add(unit.mul(step * (idx + 1)))
        c = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0).add(normal.mul(18.0))
        parts.append(f"Q {c.x:.2f} {c.y:.2f} {b.x:.2f} {b.y:.2f}")
    return " ".join(parts)


def _draw_ground(point: Point) -> list[str]:
    y = point.y + 10.0
    return [
        _line(Point(point.x, point.y), Point(point.x, y), "wire"),
        _line(Point(point.x - 16, y), Point(point.x + 16, y), "ground"),
        _line(Point(point.x - 10, y + 6), Point(point.x + 10, y + 6), "ground"),
        _line(Point(point.x - 4, y + 12), Point(point.x + 4, y + 12), "ground"),
    ]


def _element_value(element: LinearElement) -> str:
    unit = {"R": "Ohm", "C": "F", "L": "H"}.get(element.kind.upper(), "")
    return eng(element.value, unit)


def _complex_value(value: complex) -> str:
    real = float(value.real)
    imag = float(value.imag)
    if abs(imag) < 1e-12:
        return f"{real:.6g}"
    if abs(real) < 1e-12:
        return f"{imag:.6g}j"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.6g}{sign}{abs(imag):.6g}j"


def _branch_key(n1: str, n2: str) -> tuple[str, str]:
    a = "0" if _is_ground(n1) else n1
    b = "0" if _is_ground(n2) else n2
    return tuple(sorted((a, b)))


def _node_sort_key(node: str) -> tuple[int, str]:
    priority = {
        "in": 0,
        "input": 0,
        "n1": 20,
        "n2": 30,
        "n3": 40,
        "nbuf": 55,
        "neg": 70,
        "out": 100,
        "output": 100,
    }
    return (priority.get(node, 50), node)


def _node_lane(node: str, index: int) -> int:
    if node in {"neg", "fb"}:
        return 1
    if node.startswith("n") and index % 2 == 0:
        return -1
    return 0


def _node_label(node: str) -> str:
    return node


def _is_ground(node: str) -> bool:
    return node in GROUND_NAMES


def _svg_header(width: int, height: int) -> str:
    return "\n".join(
        [
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\">",
            "<style>",
            ".canvas{fill:#fbfcfe}",
            ".title{font:700 20px Arial,sans-serif;fill:#172033}",
            ".subtitle,.footer-text{font:12px Arial,sans-serif;fill:#566173}",
            ".wire,.part{stroke:#28364f;stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}",
            ".control-wire{stroke:#778397;stroke-width:1.6;fill:none;stroke-dasharray:5 4}",
            ".resistor,.inductor{stroke:#20304a;stroke-width:2.2;fill:none;stroke-linecap:round;stroke-linejoin:round}",
            ".capacitor,.ground{stroke:#20304a;stroke-width:2.2;fill:none;stroke-linecap:round}",
            ".source{stroke:#20304a;stroke-width:2;fill:#ffffff}",
            ".vcvs{stroke:#20304a;stroke-width:2;fill:#ffffff}",
            ".node-dot{fill:#20304a}",
            ".part-label{font:12px Consolas,monospace;fill:#172033}",
            ".node-label,.hint-label{font:11px Arial,sans-serif;fill:#566173}",
            ".port-label{font:700 12px Arial,sans-serif;fill:#1358a8}",
            ".polarity{font:700 15px Arial,sans-serif;fill:#172033}",
            "</style>",
            f"<rect class=\"canvas\" x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\"/>",
        ]
    )


def _line(a: Point, b: Point, cls: str) -> str:
    return f"<line class=\"{cls}\" x1=\"{a.x:.2f}\" y1=\"{a.y:.2f}\" x2=\"{b.x:.2f}\" y2=\"{b.y:.2f}\"/>"


def _polyline(points: list[Point], cls: str) -> str:
    encoded = " ".join(f"{point.x:.2f},{point.y:.2f}" for point in points)
    return f"<polyline class=\"{cls}\" points=\"{encoded}\"/>"


def _polygon(points: list[Point], cls: str) -> str:
    encoded = " ".join(f"{point.x:.2f},{point.y:.2f}" for point in points)
    return f"<polygon class=\"{cls}\" points=\"{encoded}\"/>"


def _path(data: str, cls: str) -> str:
    return f"<path class=\"{cls}\" d=\"{data}\"/>"


def _circle(x: float, y: float, radius: float, cls: str) -> str:
    return f"<circle class=\"{cls}\" cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"{radius:.2f}\"/>"


def _text(x: float, y: float, text: str, cls: str) -> str:
    return f"<text class=\"{cls}\" x=\"{x:.2f}\" y=\"{y:.2f}\">{escape(str(text))}</text>"
