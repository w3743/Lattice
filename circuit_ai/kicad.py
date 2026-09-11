from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import uuid

from .formatting import eng
from .mna import (
    CurrentSource,
    LinearCircuit,
    LinearElement,
    VoltageControlledVoltageSource,
    VoltageSource,
)


SCHEMATIC_VERSION = "20250114"
_UUID_NAMESPACE = uuid.UUID("7d39345a-cd89-4c6c-b3b6-c8d1a80c90e9")


@dataclass(frozen=True)
class _Point:
    x: float
    y: float

    def add(self, other: "_Point") -> "_Point":
        return _Point(self.x + other.x, self.y + other.y)


@dataclass(frozen=True)
class _Pin:
    node: str
    offset: _Point
    label_offset: _Point


@dataclass(frozen=True)
class _PlacedSymbol:
    lib_id: str
    reference: str
    value: str
    source_name: str
    position: _Point
    rotation: int
    pins: tuple[_Pin, ...]
    sim_library: str = ""
    sim_name: str = ""
    sim_pins: str = ""
    sim_device: str = ""


def render_result_kicad_schematic(result, *, project_name: str = "best") -> str:
    circuit = result.template.to_circuit(result.parameters)
    title = result.template.description or result.template.name
    comment = f"{result.template.name}, score={result.metrics.score:.4g}, rmse={result.metrics.rmse_db:.4g} dB"
    return render_circuit_kicad_schematic(
        circuit,
        title=title,
        project_name=project_name,
        comment=comment,
        model_bindings=result.model_bindings,
    )


def render_circuit_kicad_schematic(
    circuit: LinearCircuit,
    *,
    title: str = "Circuit AI schematic",
    project_name: str = "best",
    comment: str = "",
    model_bindings: tuple[dict[str, str], ...] = (),
) -> str:
    symbols = _place_symbols(circuit, model_bindings)
    lines = [
        "(kicad_sch",
        f"\t(version {SCHEMATIC_VERSION})",
        "\t(generator \"circuit_ai\")",
        "\t(generator_version \"0.1\")",
        f"\t(uuid {_q(_stable_uuid(project_name, 'root'))})",
        "\t(paper \"A4\")",
        "\t(title_block",
        f"\t\t(title {_q(title)})",
    ]
    if comment:
        lines.append(f"\t\t(comment 1 {_q(comment)})")
    lines.extend(
        ["\t)"]
    )
    lines.extend(_library_symbols())

    for index, symbol in enumerate(symbols, start=1):
        lines.extend(_symbol_block(symbol, project_name, index))
    for index, symbol in enumerate(symbols, start=1):
        lines.extend(_pin_connection_blocks(symbol, project_name, index))
    lines.extend(
        [
            "\t(sheet_instances",
            "\t\t(path \"/\"",
            "\t\t\t(page \"1\")",
            "\t\t)",
            "\t)",
            "\t(embedded_fonts no)",
            ")",
        ]
    )
    return "\n".join(lines) + "\n"


def render_kicad_project(project_name: str = "best") -> str:
    data = {
        "meta": {
            "filename": f"{project_name}.kicad_pro",
            "version": 1,
        },
        "libraries": {
            "pinned_footprint_libs": [],
            "pinned_symbol_libs": [],
        },
        "schematic": {
            "legacy_lib_dir": "",
            "legacy_lib_list": [],
        },
    }
    return json.dumps(data, indent=2) + "\n"


def materialize_model_bindings(bindings: tuple[dict[str, str], ...], output_dir: str | Path) -> tuple[dict[str, str], ...]:
    """Copy referenced model files into a portable KiCad project directory."""
    output_path = Path(output_dir)
    model_dir = output_path / "models"
    materialized: list[dict[str, str]] = []
    for binding in bindings:
        item = dict(binding)
        source = Path(item.get("library", ""))
        if source.is_file():
            model_dir.mkdir(parents=True, exist_ok=True)
            destination = model_dir / source.name
            shutil.copy2(source, destination)
            item["library"] = f"models/{destination.name}"
        materialized.append(item)
    return tuple(materialized)


def write_result_kicad_project(result, output_dir: str | Path, *, project_name: str = "best") -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schematic_path = output_path / f"{project_name}.kicad_sch"
    project_path = output_path / f"{project_name}.kicad_pro"
    schematic_path.write_text(
        render_result_kicad_schematic(result, project_name=project_name),
        encoding="utf-8",
    )
    project_path.write_text(render_kicad_project(project_name), encoding="utf-8")
    return schematic_path, project_path


def _place_symbols(circuit: LinearCircuit, model_bindings: tuple[dict[str, str], ...] = ()) -> list[_PlacedSymbol]:
    symbols: list[_PlacedSymbol] = []
    counts: dict[str, int] = {}
    bindings = {str(item.get("component", item.get("source_name", ""))): item for item in model_bindings}

    def binding_for(source_name: str, reference: str) -> dict[str, str] | None:
        return bindings.get(source_name) or bindings.get(reference)

    def next_ref(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        return f"{prefix}{counts[prefix]}"

    row = 0
    for element in circuit.elements:
        row += 1
        source_name = element.name
        reference = next_ref(element.kind.upper())
        symbols.append(_two_pin_symbol(element, reference, row, binding_for(source_name, reference)))
    for source in circuit.voltage_sources:
        row += 1
        symbols.append(_voltage_source_symbol(source, next_ref("V"), row))
    for source in circuit.current_sources:
        row += 1
        symbols.append(_current_source_symbol(source, next_ref("I"), row))
    for source in circuit.controlled_voltage_sources:
        row += 1
        reference = next_ref("E")
        symbols.append(_vcvs_symbol(source, reference, row, binding_for(source.name, reference)))
    return symbols


def _two_pin_symbol(element: LinearElement, reference: str, row: int, binding: dict[str, str] | None = None) -> _PlacedSymbol:
    kind = element.kind.upper()
    lib_id = {"R": "CircuitAI:R", "C": "CircuitAI:C", "L": "CircuitAI:L"}[kind]
    unit = {"R": "Ohm", "C": "F", "L": "H"}[kind]
    sim_library, sim_name, sim_pins, sim_device = _sim_metadata(binding, ("1", "2"))
    return _PlacedSymbol(
        lib_id=lib_id,
        reference=reference,
        value=f"{element.name}={eng(element.value, unit)}",
        source_name=element.name,
        position=_row_position(row),
        rotation=90,
        pins=(
            _Pin(element.n1, _Point(-3.81, 0.0), _Point(-6.35, 0.0)),
            _Pin(element.n2, _Point(3.81, 0.0), _Point(6.35, 0.0)),
        ),
        sim_library=sim_library,
        sim_name=sim_name,
        sim_pins=sim_pins,
        sim_device=sim_device,
    )


def _voltage_source_symbol(source: VoltageSource, reference: str, row: int) -> _PlacedSymbol:
    return _PlacedSymbol(
        lib_id="CircuitAI:VDC",
        reference=reference,
        value=f"{source.name}={_complex_value(source.value)}",
        source_name=source.name,
        position=_row_position(row),
        rotation=90,
        pins=(
            _Pin(source.n_plus, _Point(-5.08, 0.0), _Point(-7.62, 0.0)),
            _Pin(source.n_minus, _Point(5.08, 0.0), _Point(7.62, 0.0)),
        ),
    )


def _current_source_symbol(source: CurrentSource, reference: str, row: int) -> _PlacedSymbol:
    return _PlacedSymbol(
        lib_id="CircuitAI:IDC",
        reference=reference,
        value=f"{source.name}={_complex_value(source.value)}",
        source_name=source.name,
        position=_row_position(row),
        rotation=90,
        pins=(
            _Pin(source.n_plus, _Point(-5.08, 0.0), _Point(-7.62, 0.0)),
            _Pin(source.n_minus, _Point(5.08, 0.0), _Point(7.62, 0.0)),
        ),
    )


def _vcvs_symbol(
    source: VoltageControlledVoltageSource,
    reference: str,
    row: int,
    binding: dict[str, str] | None = None,
) -> _PlacedSymbol:
    sim_library, sim_name, sim_pins, sim_device = _sim_metadata(binding, ("1", "2", "3", "4"))
    return _PlacedSymbol(
        lib_id="CircuitAI:ESOURCE",
        reference=reference,
        value=f"{source.name} gain={_complex_value(source.gain)}",
        source_name=source.name,
        position=_row_position(row),
        rotation=0,
        pins=(
            _Pin(source.n_plus, _Point(0.0, 5.08), _Point(0.0, 7.62)),
            _Pin(source.n_minus, _Point(0.0, -5.08), _Point(0.0, -7.62)),
            _Pin(source.control_plus, _Point(-5.08, 5.08), _Point(-7.62, 7.62)),
            _Pin(source.control_minus, _Point(-5.08, -5.08), _Point(-7.62, -7.62)),
        ),
        sim_library=sim_library,
        sim_name=sim_name,
        sim_pins=sim_pins,
        sim_device=sim_device,
    )


def _sim_metadata(
    binding: dict[str, str] | None,
    expected_pin_numbers: tuple[str, ...],
) -> tuple[str, str, str, str]:
    if not binding:
        return "", "", "", ""
    library = str(binding.get("library", "")).strip()
    name = str(binding.get("name", "")).strip()
    pins = str(binding.get("pins", "")).strip()
    device = str(binding.get("device", "X")).strip() or "X"
    if not library or not name:
        raise ValueError("a model binding requires both library and name")
    if pins:
        pairs = [token.split("=", 1) for token in pins.split()]
        if any(len(pair) != 2 or not pair[0].strip() or not pair[1].strip() for pair in pairs):
            raise ValueError(f"invalid Sim.Pins mapping for model {name!r}: {pins!r}")
        symbols = tuple(pair[0].strip() for pair in pairs)
        if len(set(symbols)) != len(symbols) or set(symbols) != set(expected_pin_numbers):
            expected = " ".join(expected_pin_numbers)
            raise ValueError(f"Sim.Pins for model {name!r} must map each symbol pin exactly once; expected {expected}")
    return library, name, pins, device


def _row_position(row: int) -> _Point:
    return _Point(76.2, 30.48 + (row - 1) * 20.32)


def _library_symbols() -> list[str]:
    symbols = ["\t(lib_symbols"]
    symbols.extend(_passive_library_symbol("R", "Resistor", _resistor_graphics()))
    symbols.extend(_passive_library_symbol("C", "Capacitor", _capacitor_graphics()))
    symbols.extend(_passive_library_symbol("L", "Inductor", _inductor_graphics()))
    symbols.extend(_source_library_symbol("VDC", "Voltage source", _voltage_source_graphics(), "V"))
    symbols.extend(_source_library_symbol("IDC", "Current source", _current_source_graphics(), "I"))
    symbols.extend(_esource_library_symbol())
    symbols.extend(_power_library_symbol("SWITCH", "Ideal PWM switch", _switch_graphics()))
    symbols.extend(_power_library_symbol("DIODE", "Ideal diode", _diode_graphics()))
    symbols.extend(_transformer_library_symbol())
    symbols.append("\t)")
    return symbols


def _power_library_symbol(name: str, description: str, graphics: list[str]) -> list[str]:
    """Embedded two-pin symbol used by the explainable power experts."""
    return [
        f"\t\t(symbol {_q('CircuitAI:' + name)}",
        "\t\t\t(pin_numbers (hide yes))",
        "\t\t\t(pin_names (offset 0) (hide yes))",
        "\t\t\t(exclude_from_sim no)",
        "\t\t\t(in_bom yes)",
        "\t\t\t(on_board yes)",
        "\t\t\t(in_pos_files yes)",
        "\t\t\t(duplicate_pin_numbers_are_jumpers no)",
        *_library_property("Reference", name[0], 2.032, 0, 90, hidden=False),
        *_library_property("Value", name, 0, 0, 90, hidden=False),
        *_library_property("Footprint", "", 0, 0, 0, hidden=True),
        *_library_property("Datasheet", "", 0, 0, 0, hidden=True),
        *_library_property("Description", description, 0, 0, 0, hidden=True),
        f"\t\t\t(symbol {_q(name + '_0_1')}",
        *graphics,
        "\t\t\t)",
        f"\t\t\t(symbol {_q(name + '_1_1')}",
        *_passive_pin("1", 0, 3.81, 270),
        *_passive_pin("2", 0, -3.81, 90),
        "\t\t\t)",
        "\t\t\t(embedded_fonts no)",
        "\t\t)",
    ]


def _transformer_library_symbol() -> list[str]:
    return [
        "\t\t(symbol \"CircuitAI:TRANSFORMER\"",
        "\t\t\t(pin_numbers (hide yes))",
        "\t\t\t(pin_names (offset 0) (hide yes))",
        "\t\t\t(exclude_from_sim no)",
        "\t\t\t(in_bom yes)",
        "\t\t\t(on_board yes)",
        "\t\t\t(in_pos_files yes)",
        "\t\t\t(duplicate_pin_numbers_are_jumpers no)",
        *_library_property("Reference", "T", 0, -5.08, 0, hidden=False),
        *_library_property("Value", "TRANSFORMER", 0, 5.08, 0, hidden=False),
        *_library_property("Footprint", "", 0, 0, 0, hidden=True),
        *_library_property("Datasheet", "", 0, 0, 0, hidden=True),
        *_library_property("Description", "Ideal transformer", 0, 0, 0, hidden=True),
        "\t\t\t(symbol \"TRANSFORMER_0_1\"",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -1.27 -3.81) (xy -1.27 3.81))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy 1.27 -3.81) (xy 1.27 3.81))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t)",
        "\t\t\t(symbol \"TRANSFORMER_1_1\"",
        *_passive_pin("1", -5.08, -2.54, 0),
        *_passive_pin("2", -5.08, 2.54, 0),
        *_passive_pin("3", 5.08, -2.54, 180),
        *_passive_pin("4", 5.08, 2.54, 180),
        "\t\t\t)",
        "\t\t\t(embedded_fonts no)",
        "\t\t)",
    ]


def _passive_library_symbol(name: str, description: str, graphics: list[str]) -> list[str]:
    return [
        f"\t\t(symbol {_q('CircuitAI:' + name)}",
        "\t\t\t(pin_numbers (hide yes))",
        "\t\t\t(pin_names (offset 0) (hide yes))",
        "\t\t\t(exclude_from_sim no)",
        "\t\t\t(in_bom yes)",
        "\t\t\t(on_board yes)",
        "\t\t\t(in_pos_files yes)",
        "\t\t\t(duplicate_pin_numbers_are_jumpers no)",
        *_library_property("Reference", name, 2.032, 0, 90, hidden=False),
        *_library_property("Value", name, 0, 0, 90, hidden=False),
        *_library_property("Footprint", "", 0, 0, 0, hidden=True),
        *_library_property("Datasheet", "", 0, 0, 0, hidden=True),
        *_library_property("Description", description, 0, 0, 0, hidden=True),
        f"\t\t\t(symbol {_q(name + '_0_1')}",
        *graphics,
        "\t\t\t)",
        f"\t\t\t(symbol {_q(name + '_1_1')}",
        *_passive_pin("1", 0, 3.81, 270),
        *_passive_pin("2", 0, -3.81, 90),
        "\t\t\t)",
        "\t\t\t(embedded_fonts no)",
        "\t\t)",
    ]


def _source_library_symbol(name: str, description: str, graphics: list[str], value: str) -> list[str]:
    return [
        f"\t\t(symbol {_q('CircuitAI:' + name)}",
        "\t\t\t(pin_numbers (hide yes))",
        "\t\t\t(pin_names (offset 0) (hide yes))",
        "\t\t\t(exclude_from_sim no)",
        "\t\t\t(in_bom yes)",
        "\t\t\t(on_board yes)",
        "\t\t\t(in_pos_files yes)",
        "\t\t\t(duplicate_pin_numbers_are_jumpers no)",
        *_library_property("Reference", value, 2.54, 2.54, 0, hidden=False),
        *_library_property("Value", value, 2.54, -2.54, 0, hidden=False),
        *_library_property("Footprint", "", 0, 0, 0, hidden=True),
        *_library_property("Datasheet", "", 0, 0, 0, hidden=True),
        *_library_property("Description", description, 0, 0, 0, hidden=True),
        f"\t\t\t(symbol {_q(name + '_0_1')}",
        *graphics,
        "\t\t\t)",
        f"\t\t\t(symbol {_q(name + '_1_1')}",
        *_passive_pin("1", 0, 5.08, 270),
        *_passive_pin("2", 0, -5.08, 90),
        "\t\t\t)",
        "\t\t\t(embedded_fonts no)",
        "\t\t)",
    ]


def _esource_library_symbol() -> list[str]:
    return [
        "\t\t(symbol \"CircuitAI:ESOURCE\"",
        "\t\t\t(pin_numbers (hide yes))",
        "\t\t\t(pin_names (offset 1.016))",
        "\t\t\t(exclude_from_sim no)",
        "\t\t\t(in_bom yes)",
        "\t\t\t(on_board yes)",
        "\t\t\t(in_pos_files yes)",
        "\t\t\t(duplicate_pin_numbers_are_jumpers no)",
        *_library_property("Reference", "E", 2.54, 4.064, 0, hidden=False),
        *_library_property("Value", "ESOURCE", 2.54, -4.064, 0, hidden=False),
        *_library_property("Footprint", "", 0, 0, 0, hidden=True),
        *_library_property("Datasheet", "", 0, 0, 0, hidden=True),
        *_library_property("Description", "Voltage controlled voltage source", 0, 0, 0, hidden=True),
        "\t\t\t(symbol \"ESOURCE_0_1\"",
        *[
            "\t\t\t\t(polyline",
            "\t\t\t\t\t(pts (xy 0 2.54) (xy -2.54 0) (xy 0 -2.54) (xy 2.54 0) (xy 0 2.54))",
            "\t\t\t\t\t(stroke (width 0.254) (type default))",
            "\t\t\t\t\t(fill (type background))",
            "\t\t\t\t)",
            "\t\t\t\t(text \"V\"",
            "\t\t\t\t\t(at -5.08 0 0)",
            "\t\t\t\t\t(effects (font (size 1.27 1.27)))",
            "\t\t\t\t)",
        ],
        "\t\t\t)",
        "\t\t\t(symbol \"ESOURCE_1_1\"",
        *_named_pin("1", "N+", "passive", 0, 5.08, 270),
        *_named_pin("2", "N-", "passive", 0, -5.08, 90),
        *_named_pin("3", "C+", "input", -5.08, 5.08, 270),
        *_named_pin("4", "C-", "input", -5.08, -5.08, 90),
        "\t\t\t)",
        "\t\t\t(embedded_fonts no)",
        "\t\t)",
    ]


def _library_property(name: str, value: str, x: float, y: float, rotation: int, *, hidden: bool) -> list[str]:
    lines = [
        f"\t\t\t(property {_q(name)} {_q(value)}",
        f"\t\t\t\t(at {_num(x)} {_num(y)} {rotation})",
        "\t\t\t\t(effects",
        "\t\t\t\t\t(font (size 1.27 1.27))",
    ]
    if hidden:
        lines.append("\t\t\t\t\t(hide yes)")
    lines.extend(
        [
            "\t\t\t\t)",
            "\t\t\t)",
        ]
    )
    return lines


def _passive_pin(number: str, x: float, y: float, rotation: int) -> list[str]:
    return _named_pin(number, "", "passive", x, y, rotation)


def _named_pin(number: str, name: str, electrical_type: str, x: float, y: float, rotation: int) -> list[str]:
    return [
        f"\t\t\t\t(pin {electrical_type} line",
        f"\t\t\t\t\t(at {_num(x)} {_num(y)} {rotation})",
        "\t\t\t\t\t(length 1.27)",
        f"\t\t\t\t\t(name {_q(name)}",
        "\t\t\t\t\t\t(effects (font (size 1.27 1.27)))",
        "\t\t\t\t\t)",
        f"\t\t\t\t\t(number {_q(number)}",
        "\t\t\t\t\t\t(effects (font (size 1.27 1.27)))",
        "\t\t\t\t\t)",
        "\t\t\t\t)",
    ]


def _resistor_graphics() -> list[str]:
    return [
        "\t\t\t\t(rectangle",
        "\t\t\t\t\t(start -1.016 -2.54)",
        "\t\t\t\t\t(end 1.016 2.54)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
    ]


def _capacitor_graphics() -> list[str]:
    return [
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -2.032 0.762) (xy 2.032 0.762))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -2.032 -0.762) (xy 2.032 -0.762))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
    ]


def _inductor_graphics() -> list[str]:
    return [
        "\t\t\t\t(arc",
        "\t\t\t\t\t(start 0 2.54)",
        "\t\t\t\t\t(mid 1.27 1.905)",
        "\t\t\t\t\t(end 0 1.27)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(arc",
        "\t\t\t\t\t(start 0 1.27)",
        "\t\t\t\t\t(mid 1.27 0.635)",
        "\t\t\t\t\t(end 0 0)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(arc",
        "\t\t\t\t\t(start 0 0)",
        "\t\t\t\t\t(mid 1.27 -0.635)",
        "\t\t\t\t\t(end 0 -1.27)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(arc",
        "\t\t\t\t\t(start 0 -1.27)",
        "\t\t\t\t\t(mid 1.27 -1.905)",
        "\t\t\t\t\t(end 0 -2.54)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
    ]


def _voltage_source_graphics() -> list[str]:
    return [
        "\t\t\t\t(circle",
        "\t\t\t\t\t(center 0 0)",
        "\t\t\t\t\t(radius 2.54)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type background))",
        "\t\t\t\t)",
        "\t\t\t\t(text \"+\"",
        "\t\t\t\t\t(at 0 1.905 0)",
        "\t\t\t\t\t(effects (font (size 1.27 1.27)))",
        "\t\t\t\t)",
        "\t\t\t\t(text \"-\"",
        "\t\t\t\t\t(at 0 -1.905 0)",
        "\t\t\t\t\t(effects (font (size 1.27 1.27)))",
        "\t\t\t\t)",
    ]


def _current_source_graphics() -> list[str]:
    return [
        "\t\t\t\t(circle",
        "\t\t\t\t\t(center 0 0)",
        "\t\t\t\t\t(radius 2.54)",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type background))",
        "\t\t\t\t)",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy 0 -1.524) (xy 0 1.524))",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -0.762 0.762) (xy 0 1.524) (xy 0.762 0.762))",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
    ]


def _switch_graphics() -> list[str]:
    return [
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -2.54 1.27) (xy 0.762 -1.27))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -2.54 -1.27) (xy 2.54 -1.27))",
        "\t\t\t\t\t(stroke (width 0.254) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
    ]


def _diode_graphics() -> list[str]:
    return [
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy -1.524 -2.032) (xy -1.524 2.032) (xy 1.524 0))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
        "\t\t\t\t(polyline",
        "\t\t\t\t\t(pts (xy 1.524 -2.032) (xy 1.524 2.032))",
        "\t\t\t\t\t(stroke (width 0.508) (type default))",
        "\t\t\t\t\t(fill (type none))",
        "\t\t\t\t)",
    ]


def _symbol_block(symbol: _PlacedSymbol, project_name: str, index: int) -> list[str]:
    x, y = symbol.position.x, symbol.position.y
    key = f"symbol-{index}-{symbol.reference}-{symbol.source_name}"
    return [
        "\t(symbol",
        f"\t\t(lib_id {_q(symbol.lib_id)})",
        f"\t\t(at {_num(x)} {_num(y)} {symbol.rotation})",
        "\t\t(unit 1)",
        "\t\t(exclude_from_sim no)",
        "\t\t(in_bom yes)",
        "\t\t(on_board yes)",
        "\t\t(dnp no)",
        f"\t\t(uuid {_q(_stable_uuid(project_name, key))})",
        *_property_block("Reference", symbol.reference, x, y - 4.318, 0, hidden=False),
        *_property_block("Value", symbol.value, x, y + 4.318, 0, hidden=False),
        *_property_block("Footprint", "", x, y, 0, hidden=True),
        *_property_block("Datasheet", "", x, y, 0, hidden=True),
        *_property_block("CircuitAI.Name", symbol.source_name, x, y + 7.62, 0, hidden=True),
        *(_property_block("Sim.Library", symbol.sim_library, x, y + 9.0, 0, hidden=True) if symbol.sim_library else []),
        *(_property_block("Sim.Name", symbol.sim_name, x, y + 10.0, 0, hidden=True) if symbol.sim_name else []),
        *(_property_block("Sim.Pins", symbol.sim_pins, x, y + 11.0, 0, hidden=True) if symbol.sim_pins else []),
        *(_property_block("Sim.Device", symbol.sim_device, x, y + 12.0, 0, hidden=True) if symbol.sim_device else []),
        "\t\t(instances",
        f"\t\t\t(project {_q(project_name)}",
        "\t\t\t\t(path \"/\"",
        f"\t\t\t\t\t(reference {_q(symbol.reference)})",
        "\t\t\t\t\t(unit 1)",
        "\t\t\t\t)",
        "\t\t\t)",
        "\t\t)",
        "\t)",
    ]


def _property_block(name: str, value: str, x: float, y: float, rotation: int, *, hidden: bool) -> list[str]:
    lines = [
        f"\t\t(property {_q(name)} {_q(value)}",
        f"\t\t\t(at {_num(x)} {_num(y)} {rotation})",
        "\t\t\t(effects",
        "\t\t\t\t(font",
        "\t\t\t\t\t(size 1.27 1.27)",
        "\t\t\t\t)",
    ]
    if hidden:
        lines.append("\t\t\t\t(hide yes)")
    lines.extend(
        [
            "\t\t\t)",
            "\t\t)",
        ]
    )
    return lines


def _pin_connection_blocks(symbol: _PlacedSymbol, project_name: str, symbol_index: int) -> list[str]:
    lines: list[str] = []
    for pin_index, pin in enumerate(symbol.pins, start=1):
        pin_point = symbol.position.add(pin.offset)
        label_point = symbol.position.add(pin.offset).add(pin.label_offset)
        net = _node_name(pin.node)
        key = f"{symbol_index}-{pin_index}-{symbol.reference}-{net}"
        lines.extend(
            [
                "\t(wire",
                f"\t\t(pts (xy {_num(pin_point.x)} {_num(pin_point.y)}) (xy {_num(label_point.x)} {_num(label_point.y)}))",
                "\t\t(stroke (width 0) (type solid))",
                f"\t\t(uuid {_q(_stable_uuid(project_name, 'wire-' + key))})",
                "\t)",
                f"\t(label {_q(net)}",
                f"\t\t(at {_num(label_point.x)} {_num(label_point.y)} 0)",
                "\t\t(effects",
                "\t\t\t(font",
                "\t\t\t\t(size 1.27 1.27)",
                "\t\t\t)",
                "\t\t\t(justify left bottom)",
                "\t\t)",
                f"\t\t(uuid {_q(_stable_uuid(project_name, 'label-' + key))})",
                "\t)",
            ]
        )
    return lines


def _node_name(node: str) -> str:
    return "0" if node in {"0", "gnd", "GND"} else node


def _complex_value(value: complex) -> str:
    real = float(value.real)
    imag = float(value.imag)
    if abs(imag) < 1e-12:
        return f"{real:.6g}"
    if abs(real) < 1e-12:
        return f"{imag:.6g}j"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.6g}{sign}{abs(imag):.6g}j"


def _stable_uuid(project_name: str, key: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, f"{project_name}:{key}"))


def _q(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
