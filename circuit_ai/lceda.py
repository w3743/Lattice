"""Read-back checks for native JLCEDA Pro exports.

This module verifies an editor-produced Protel2 netlist. It does not generate
projects or infer physical pin numbers from the order of graph connections.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from typing import Mapping, Sequence
import zipfile

from .graph import CircuitGraph


@dataclass(frozen=True)
class LcedaNetlist:
    components: Mapping[str, Mapping[str, str]]
    pin_nets: Mapping[tuple[str, str], str]


def verify_epro2_library_references(archive: bytes) -> dict[str, int]:
    """Check embedded device/symbol/footprint references in native EPRO2.

    Does not claim 3D-model closure, pad geometry correctness, or PCB validity.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        sources = [name for name in package.namelist() if name.endswith(".epru")]
        if len(sources) != 1:
            raise ValueError("expected one native EPRU document stream")
        source = package.read(sources[0]).decode("utf-8")
    documents: dict[str, str] = {}
    references: list[tuple[str, str]] = []
    kind = None
    types = {"Device": "DEVICE", "Symbol": "SYMBOL", "Footprint": "FOOTPRINT"}
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not line.endswith("|") and index != len(lines) - 1:
            raise ValueError("native record missing terminator")
        record = line[:-1] if line.endswith("|") else line
        header, payload = (json.loads(part) for part in record.split("||", 1))
        if header["type"] == "DOCHEAD":
            uuid, kind = payload["uuid"], payload["docType"]
            if uuid in documents:
                raise ValueError(f"duplicate native document UUID: {uuid}")
            documents[uuid] = kind
        elif kind == "DEVICE" and header["type"] == "META":
            for key, value in payload.get("attributes", {}).items():
                if key in types and value:
                    references.append((types[key], value))
        elif kind in {"SCH_PAGE", "PCB"} and header["type"] == "ATTR":
            key, value = payload.get("key"), payload.get("value")
            if key in types and value:
                references.append((types[key], value))
    if "SCH_PAGE" not in documents.values() or not references:
        raise ValueError("archive has no schematic with library references")
    for kind, uuid in references:
        if documents.get(uuid) != kind:
            raise ValueError(f"unresolved embedded {kind} reference: {uuid}")
    return {kind: sum(value == kind for value in documents.values()) for kind in types.values()}


def parse_protel2_netlist(text: str) -> LcedaNetlist:
    """Parse JLCEDA's Protel2 export; reject incomplete/ambiguous records."""
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines.pop(0).strip() != "PROTEL NETLIST 2.0":
        raise ValueError("expected PROTEL NETLIST 2.0 header")
    components: dict[str, dict[str, str]] = {}
    nets: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        marker = lines[index].strip()
        index += 1
        if not marker:
            continue
        if marker not in {"[", "("}:
            raise ValueError(f"unexpected netlist record: {marker!r}")
        end = "]" if marker == "[" else ")"
        block: list[str] = []
        while index < len(lines) and lines[index].strip() != end:
            block.append(lines[index].strip())
            index += 1
        if index == len(lines):
            raise ValueError(f"unterminated {marker!r} record")
        index += 1
        if marker == "[":
            if not block or block[-1] != "*":
                raise ValueError("component record missing terminator")
            block.pop()
            properties: dict[str, str] = {}
            cursor = 0
            while cursor < len(block) and any(block[cursor:]):
                if cursor + 1 >= len(block):
                    raise ValueError("component property missing value")
                key, value = block[cursor:cursor + 2]
                if not key or key in properties:
                    raise ValueError(f"empty or duplicate component property: {key!r}")
                properties[key] = value
                cursor += 2
            reference = properties.get("DESIGNATOR")
            if not reference or reference in components:
                raise ValueError(f"missing or duplicate component reference: {reference!r}")
            components[reference] = properties
        else:
            if not block or not block[0] or block[0] in nets:
                raise ValueError("missing or duplicate net name")
            if len(block) < 2 or any(not entry for entry in block[1:]):
                raise ValueError("empty net or pin record")
            nets[block[0]] = block[1:]

    pins: dict[tuple[str, str], str] = {}
    for net, entries in nets.items():
        for entry in entries:
            identifier = entry.split()[0]
            candidates = [ref for ref in components if identifier.startswith(ref + "-")]
            if len(candidates) != 1:
                raise ValueError(f"unknown or ambiguous component pin: {identifier!r}")
            ref = candidates[0]
            pin = identifier[len(ref) + 1:]
            if not pin or (ref, pin) in pins:
                raise ValueError(f"empty or duplicate pin: {identifier!r}")
            pins[ref, pin] = net
    return LcedaNetlist(components, pins)


def verify_lceda_connectivity(
    graph: CircuitGraph,
    netlist: LcedaNetlist,
    terminal_pins: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    net_names: Mapping[str, str] | None = None,
) -> None:
    """Require exact component and physical-pin connectivity after export.

    ``terminal_pins`` uses graph references, named terminals and explicit
    physical pin numbers. One terminal may have multiple physical pins.
    ``net_names`` explicitly translates graph net IDs to editor net names.
    """
    graph.require_valid()
    references = {component.reference for component in graph.components}
    if set(terminal_pins) != references or set(netlist.components) != references:
        raise ValueError("component set differs between graph, pin map and editor")
    names = {net.net_id: net.name for net in graph.nets}
    if net_names is not None:
        if set(net_names) - names.keys():
            raise ValueError("net name mapping contains unknown graph net IDs")
        names.update(net_names)
    if any(not name for name in names.values()) or len(set(names.values())) != len(names):
        raise ValueError("exported net names must be nonempty and unique")
    expected: dict[tuple[str, str], str] = {}
    for component in graph.components:
        mapping = terminal_pins[component.reference]
        if set(mapping) != {connection.terminal for connection in component.connections}:
            raise ValueError(f"terminal map incomplete for {component.reference}")
        for connection in component.connections:
            numbers = mapping[connection.terminal]
            if isinstance(numbers, str) or not numbers or len(set(numbers)) != len(numbers):
                raise ValueError(f"invalid physical pin list for {component.reference}")
            for number in numbers:
                if not isinstance(number, str) or not number.strip():
                    raise ValueError("physical pin numbers must be nonempty strings")
                key = component.reference, number
                if key in expected and expected[key] != names[connection.net_id]:
                    raise ValueError(f"physical pin assigned to different nets: {key}")
                expected[key] = names[connection.net_id]
    if expected != netlist.pin_nets:
        missing = sorted(expected.keys() - netlist.pin_nets.keys())
        extra = sorted(netlist.pin_nets.keys() - expected.keys())
        wrong = sorted(key for key in expected.keys() & netlist.pin_nets.keys()
                       if expected[key] != netlist.pin_nets[key])
        raise ValueError(f"pin/net mismatch: missing={missing}, extra={extra}, wrong={wrong}")
