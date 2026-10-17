from dataclasses import replace
from pathlib import Path
import io
import json
import zipfile

import pytest

from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.lceda import (parse_protel2_netlist, verify_lceda_connectivity,
                              verify_epro2_library_references)
from circuit_ai.mna import LinearCircuit, LinearElement


NETLIST = """PROTEL NETLIST 2.0
[
DESIGNATOR
R1
Value
1kΩ
Channel ID


*
]
[
DESIGNATOR
C1
Value
100nF
*
]
(
IN
R1-1 resistor-1 Input
)
(
OUT
R1-2 resistor-2 Input
C1-1 capacitor-1 Passive
)
(
GND
C1-2 capacitor-2 Passive
)
"""


def graph():
    return linear_circuit_to_graph(LinearCircuit(elements=(
        LinearElement("R1", "R", "in", "out", 1000),
        LinearElement("C1", "C", "out", "0", 1e-7),
    ), voltage_sources=()))


PINS = {"R1": {"p": ("1",), "n": ("2",)}, "C1": {"p": ("1",), "n": ("2",)}}
NAMES = {"in": "IN", "out": "OUT", "0": "GND"}


def test_actual_4160_archive_import_readback():
    text = (Path(__file__).parent / "fixtures/lceda_rc_protel2.net").read_text(encoding="utf-8")
    netlist = parse_protel2_netlist(text)
    verify_lceda_connectivity(graph(), netlist, PINS, net_names=NAMES)
    assert netlist.components["R1"]["Supplier Part"] == "C21190"
    assert netlist.components["R1"]["Value"] == "1kΩ"
    assert netlist.components["C1"]["FOOTPRINT"] == "C0603"
    assert netlist.components["C1"]["PARTTYPE"] == "CC0603KRX7R9BB104"


def test_editor_netlist_preserves_blank_properties_and_named_terminal_order():
    netlist = parse_protel2_netlist("\ufeff" + NETLIST.replace("\n", "\r\n"))
    assert netlist.components["R1"]["Channel ID"] == ""
    g = graph()
    reordered = replace(g, components=tuple(
        replace(c, connections=tuple(reversed(c.connections))) for c in g.components))
    verify_lceda_connectivity(reordered, netlist, PINS, net_names=NAMES)


@pytest.mark.parametrize("bad", [
    NETLIST.replace("R1-1 resistor-1 Input", "R1-2 resistor-2 Input"),
    NETLIST.replace("C1-2 capacitor-2 Passive", "U7-2 capacitor-2 Passive"),
    NETLIST.rstrip()[:-1],
    NETLIST.replace("PROTEL NETLIST 2.0", "PROTEL NETLIST 1.0"),
])
def test_rejects_corrupt_or_unknown_editor_records(bad):
    with pytest.raises(ValueError):
        parse_protel2_netlist(bad)


@pytest.mark.parametrize("bad", [
    NETLIST.replace("IN\n", "TEMP\n").replace("OUT\n", "IN\n").replace("TEMP\n", "OUT\n"),
    NETLIST.replace("C1-1 capacitor-1 Passive\n", ""),
    NETLIST.replace("C1-1 capacitor-1 Passive", "C1-1 capacitor-1 Passive\nC1-3 capacitor-3 Passive"),
])
def test_rejects_swapped_nets_missing_and_extra_pins(bad):
    with pytest.raises(ValueError, match="pin/net mismatch"):
        verify_lceda_connectivity(graph(), parse_protel2_netlist(bad), PINS, net_names=NAMES)


def test_rejects_implicit_net_merge_and_conflicting_physical_pad():
    netlist = parse_protel2_netlist(NETLIST)
    with pytest.raises(ValueError, match="unique"):
        verify_lceda_connectivity(graph(), netlist, PINS, net_names={**NAMES, "in": "OUT"})
    with pytest.raises(ValueError, match="different nets"):
        verify_lceda_connectivity(graph(), netlist,
                                  {**PINS, "R1": {"p": ("1",), "n": ("1",)}}, net_names=NAMES)


def test_requires_every_physical_pad_of_multi_pad_terminal():
    pins = {**PINS, "C1": {"p": ("1",), "n": ("2", "EP")}}
    with pytest.raises(ValueError, match="missing"):
        verify_lceda_connectivity(graph(), parse_protel2_netlist(NETLIST), pins, net_names=NAMES)
    full = NETLIST.replace("C1-2 capacitor-2 Passive", "C1-2 capacitor-2 Passive\nC1-EP capacitor-EP Passive")
    verify_lceda_connectivity(graph(), parse_protel2_netlist(full), pins, net_names=NAMES)


@pytest.mark.parametrize("footprint,accepted", [("f", True), ("missing", False)])
def test_native_archive_must_embed_referenced_libraries(footprint, accepted):
    records = [
        ("DOCHEAD", {"uuid": "d", "docType": "DEVICE"}),
        ("META", {"attributes": {"Symbol": "s", "Footprint": footprint}}),
        ("DOCHEAD", {"uuid": "s", "docType": "SYMBOL"}),
        ("DOCHEAD", {"uuid": "f", "docType": "FOOTPRINT"}),
        ("DOCHEAD", {"uuid": "page", "docType": "SCH_PAGE"}),
        ("ATTR", {"key": "Device", "value": "d"}),
    ]
    # The verified client omits the final record terminator in EPRO2 archives.
    source = "|\n".join(json.dumps({"type": kind}) + "||" + json.dumps(data)
                         for kind, data in records)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("design.epru", source)
    if accepted:
        assert verify_epro2_library_references(stream.getvalue()) == {
            "DEVICE": 1, "SYMBOL": 1, "FOOTPRINT": 1}
    else:
        with pytest.raises(ValueError, match="unresolved embedded FOOTPRINT"):
            verify_epro2_library_references(stream.getvalue())
