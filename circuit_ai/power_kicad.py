"""KiCad renderer for the explainable ideal boost expert."""

from __future__ import annotations

from dataclasses import replace

from .kicad import (
    _Pin,
    _PlacedSymbol,
    _Point,
    _library_symbols,
    _num,
    _pin_connection_blocks,
    _q,
    _stable_uuid,
    _symbol_block,
    _two_pin_symbol,
    _voltage_source_symbol,
    render_kicad_project,
)
from .mna import VoltageSource
from .power import BoostOptimizationResult, FlybackOptimizationResult


def render_boost_kicad_schematic(result: BoostOptimizationResult, *, project_name: str = "boost") -> str:
    if result.candidate.family == "isolated_flyback":
        return _render_flyback_kicad_schematic(result, project_name=project_name)
    if result.candidate.family in {"dc_buck", "dc_sepic"}:
        return _render_generated_power_kicad_schematic(result, project_name=project_name)
    components = {component.name: component for component in result.candidate.components}
    in_node = result.candidate.metadata["input_port"]
    # The metadata stores port names, while the components already contain the
    # resolved electrical nodes.  Resolve nodes from the component graph.
    in_net = components["L1"].nodes[0]
    ground = result.candidate.metadata["reference_node"]
    out_net = components["C1"].nodes[0]

    symbols: list[_PlacedSymbol] = []
    source = _voltage_source_symbol(
        VoltageSource("Vin", in_net, ground, result.operating_point.input_voltage_v),
        "V1",
        1,
    )
    symbols.append(replace(source, position=_Point(50.8, 76.2)))
    symbols.append(_custom_component(components["L1"], "L1", "L", _Point(81.28, 50.8), result))
    symbols.append(_custom_component(components["Q1"], "Q1", "SWITCH", _Point(111.76, 101.6), result))
    symbols.append(_custom_component(components["D1"], "D1", "DIODE", _Point(142.24, 50.8), result))
    symbols.append(_custom_component(components["C1"], "C1", "C", _Point(172.72, 101.6), result))
    symbols.append(_custom_component(components["Rload"], "R1", "R", _Point(203.2, 101.6), result))

    lines = [
        "(kicad_sch",
        "\t(version 20250114)",
        "\t(generator \"circuit_ai\")",
        "\t(generator_version \"0.1\")",
        f"\t(uuid {_q(_stable_uuid(project_name, 'root'))})",
        "\t(paper \"A4\")",
        "\t(title_block",
        "\t\t(title \"Ideal asynchronous boost converter\")",
        f"\t\t(comment 1 {_q('Expert: ' + result.candidate.name)})",
        f"\t\t(comment 2 {_q('Vin=%.4g V, Vout=%.4g V, D=%.5f' % (result.operating_point.input_voltage_v, result.operating_point.output_voltage_v, result.parameters.duty_cycle))})",
        "\t)",
    ]
    lines.extend(_library_symbols())
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


def write_boost_kicad_project(result: BoostOptimizationResult, output_dir, *, project_name: str = "boost"):
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    schematic = output / f"{project_name}.kicad_sch"
    project = output / f"{project_name}.kicad_pro"
    schematic.write_text(render_boost_kicad_schematic(result, project_name=project_name), encoding="utf-8")
    project.write_text(render_kicad_project(project_name), encoding="utf-8")
    return schematic, project


def _render_flyback_kicad_schematic(result: FlybackOptimizationResult, *, project_name: str) -> str:
    components = {component.name: component for component in result.candidate.components}
    transformer = components["T1"]
    source = _voltage_source_symbol(
        VoltageSource("Vin", transformer.nodes[0], transformer.nodes[3], result.operating_point.input_voltage_v),
        "V1",
        1,
    )
    symbols = [
        replace(source, position=_Point(50.8, 76.2)),
        _custom_transformer(transformer, "T1", _Point(96.52, 76.2), result),
        _custom_component(components["Q1"], "Q1", "SWITCH", _Point(127.0, 106.68), result),
        _custom_component(components["D1"], "D1", "DIODE", _Point(157.48, 76.2), result),
        _custom_component(components["C1"], "C1", "C", _Point(187.96, 106.68), result),
        _custom_component(components["Rload"], "R1", "R", _Point(218.44, 106.68), result),
    ]
    lines = [
        "(kicad_sch",
        "\t(version 20250114)",
        "\t(generator \"circuit_ai\")",
        "\t(generator_version \"0.1\")",
        f"\t(uuid {_q(_stable_uuid(project_name, 'root'))})",
        "\t(paper \"A4\")",
        "\t(title_block",
        "\t\t(title \"Ideal isolated flyback converter\")",
        f"\t\t(comment 1 {_q('Expert: ' + result.candidate.name)})",
        f"\t\t(comment 2 {_q('Vin=%.4g V, Vout=%.4g V, D=%.5f, N=%.5f' % (result.operating_point.input_voltage_v, result.operating_point.output_voltage_v, result.parameters.duty_cycle, result.parameters.turns_ratio))})",
        "\t)",
    ]
    lines.extend(_library_symbols())
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


def _render_generated_power_kicad_schematic(result, *, project_name: str) -> str:
    metadata = result.candidate.metadata
    source = _voltage_source_symbol(
        VoltageSource(
            "Vin",
            str(metadata["input_node"]),
            str(metadata["input_reference_node"]),
            result.operating_point.input_voltage_v,
        ),
        "V1",
        1,
    )
    symbols = [replace(source, position=_Point(45.72, 76.2))]
    prefix_by_kind = {
        "R": "R",
        "C": "C",
        "L": "L",
        "ideal_switch": "Q",
        "ideal_diode": "D",
    }
    lib_by_kind = {
        "R": "R",
        "C": "C",
        "L": "L",
        "ideal_switch": "SWITCH",
        "ideal_diode": "DIODE",
    }
    counts: dict[str, int] = {}
    for index, component in enumerate(result.candidate.components, start=1):
        prefix = prefix_by_kind[component.kind]
        counts[prefix] = counts.get(prefix, 0) + 1
        reference = f"{prefix}{counts[prefix]}"
        column = (index - 1) % 4
        row = (index - 1) // 4
        symbols.append(
            _custom_component(
                component,
                reference,
                lib_by_kind[component.kind],
                _Point(86.36 + 38.1 * column, 55.88 + 50.8 * row),
                result,
            )
        )

    title = {
        "dc_buck": "Grammar-generated ideal buck converter",
        "dc_sepic": "Grammar-generated ideal SEPIC converter",
    }[result.candidate.family]
    lines = [
        "(kicad_sch",
        "\t(version 20250114)",
        "\t(generator \"circuit_ai\")",
        "\t(generator_version \"0.1\")",
        f"\t(uuid {_q(_stable_uuid(project_name, 'root'))})",
        "\t(paper \"A4\")",
        "\t(title_block",
        f"\t\t(title {_q(title)})",
        f"\t\t(comment 1 {_q('Grammar: ' + result.candidate.name)})",
        f"\t\t(comment 2 {_q('Vin=%.4g V, Vout=%.4g V, D=%.5f' % (result.operating_point.input_voltage_v, result.operating_point.output_voltage_v, result.parameters.duty_cycle))})",
        "\t)",
    ]
    lines.extend(_library_symbols())
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


def _custom_component(component, reference: str, lib_name: str, position: _Point, result: BoostOptimizationResult) -> _PlacedSymbol:
    value = component.kind
    if component.kind == "L":
        value = f"{component.name}={result.parameters.inductance_h:.6g} H"
    elif component.kind == "C":
        value = f"{component.name}={result.parameters.capacitance_f:.6g} F"
    elif component.kind == "R":
        value = f"{component.name}={result.parameters.load_ohm:.6g} Ohm"
    elif component.kind == "ideal_switch":
        value = f"ideal PWM, D={result.parameters.duty_cycle:.6g}, f={result.parameters.switching_frequency_hz:.6g} Hz"
    elif component.kind == "ideal_diode":
        value = "ideal diode"
    pins = (
        _Pin(component.nodes[0], _Point(-3.81, 0.0), _Point(-6.35, 0.0)),
        _Pin(component.nodes[1], _Point(3.81, 0.0), _Point(6.35, 0.0)),
    )
    return _PlacedSymbol(
        lib_id=f"CircuitAI:{lib_name}",
        reference=reference,
        value=value,
        source_name=component.name,
        position=position,
        rotation=90,
        pins=pins,
    )


def _custom_transformer(component, reference: str, position: _Point, result: FlybackOptimizationResult) -> _PlacedSymbol:
    return _PlacedSymbol(
        lib_id="CircuitAI:TRANSFORMER",
        reference=reference,
        value=f"ideal transformer N={result.parameters.turns_ratio:.6g}",
        source_name=component.name,
        position=position,
        rotation=0,
        pins=(
            _Pin(component.nodes[0], _Point(-5.08, -2.54), _Point(-7.62, -2.54)),
            _Pin(component.nodes[1], _Point(-5.08, 2.54), _Point(-7.62, 2.54)),
            _Pin(component.nodes[2], _Point(5.08, -2.54), _Point(7.62, -2.54)),
            _Pin(component.nodes[3], _Point(5.08, 2.54), _Point(7.62, 2.54)),
        ),
    )
