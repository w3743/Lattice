from __future__ import annotations

from dataclasses import replace
import json
import string

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from circuit_ai.graph import (
    CIRCUIT_GRAPH_SCHEMA,
    CircuitGraph,
    CircuitGraphMigrationError,
    GraphValidationError,
    ParameterBinding,
    TerminalConnection,
    graph_to_linear_circuit,
    graph_to_topology_candidate,
    linear_circuit_to_graph,
    migrate_circuit_graph_payload,
    topology_candidate_to_graph,
)
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.mna import (
    CurrentSource,
    LinearCircuit,
    LinearElement,
    MNASimulator,
    VoltageControlledVoltageSource,
    VoltageSource,
)
from circuit_ai.templates import default_templates
from circuit_ai.topology_grammar import PowerTopologyGrammar


FINITE_POSITIVE = st.floats(
    min_value=1e-15,
    max_value=1e15,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)
FINITE_SIGNED = st.floats(
    min_value=-1e12,
    max_value=1e12,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)
GRAPH_VALUE = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), FINITE_SIGNED, st.text(max_size=24)),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=12), children, max_size=5),
    ),
    max_leaves=20,
)


def _linear_graph():
    circuit = LinearCircuit(
        elements=(
            LinearElement("C1", "C", "in", "n1", 10e-9),
            LinearElement("L1", "L", "n1", "out", 1e-3),
            LinearElement("R1", "R", "out", "0", 1000.0),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
    )
    return linear_circuit_to_graph(circuit, name="rlc_graph")


def _power_spec(vin: float, vout: float, *, isolated: bool = False) -> dict:
    input_reference = "0"
    output_reference = "out_0" if isolated else "0"
    spec = {
        "name": "graph_power",
        "ports": [
            {
                "name": "input",
                "role": "input",
                "terminals": [
                    {"name": "in", "quantity": "voltage"},
                    {"name": input_reference, "quantity": "ground"},
                ],
            },
            {
                "name": "output",
                "role": "output",
                "terminals": [
                    {"name": "out", "quantity": "voltage"},
                    {"name": output_reference, "quantity": "ground"},
                ],
            },
        ],
        "relations": [],
        "analyses": [
            {"kind": "dc_transfer", "source_port": "input", "output_port": "output"}
        ],
        "targets": [
            {
                "target_kind": "dc",
                "input_voltage_v": vin,
                "output_voltage_v": vout,
                "output_current_a": 1.0,
            }
        ],
        "constraints": {
            "element_types": [
                "R",
                "C",
                "L",
                "ideal_switch",
                "ideal_diode",
                "ideal_transformer",
            ],
            "max_component_count": 8,
            "parameter_ranges": {"R": [1.0, 1e6], "C": [1e-12, 1e-2], "L": [1e-9, 1.0]},
        },
    }
    if isolated:
        spec["relations"] = [
            {
                "kind": "galvanic_isolation",
                "source_port": "input",
                "response_port": "output",
            }
        ]
    return spec


def test_graph_schema_round_trip_preserves_all_hashes() -> None:
    graph = _linear_graph()

    serialized = json.loads(json.dumps(graph.as_dict()))
    restored = CircuitGraph.from_dict(serialized)

    assert restored.validate().passed
    assert restored.as_dict() == graph.as_dict()
    assert restored.document_hash == graph.document_hash
    assert restored.graph_hash == graph.graph_hash
    assert restored.topology_hash == graph.topology_hash


@settings(max_examples=60, deadline=None)
@given(
    resistance=FINITE_POSITIVE,
    capacitance=FINITE_POSITIVE,
    inductance=FINITE_POSITIVE,
    source_real=FINITE_SIGNED,
    source_imag=FINITE_SIGNED,
)
def test_generated_linear_circuits_round_trip_exactly(
    resistance: float,
    capacitance: float,
    inductance: float,
    source_real: float,
    source_imag: float,
) -> None:
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", resistance),
            LinearElement("C1", "C", "out", "0", capacitance),
            LinearElement("L1", "L", "in", "aux", inductance),
        ),
        voltage_sources=(
            VoltageSource("Vin", "in", "0", complex(source_real, source_imag)),
        ),
    )

    graph = linear_circuit_to_graph(circuit, output_node="out")
    wire_payload = json.loads(json.dumps(graph.as_dict(), allow_nan=False))
    restored_graph = CircuitGraph.from_dict(wire_payload)

    assert restored_graph.validate().passed
    assert graph_to_linear_circuit(restored_graph) == circuit
    assert restored_graph.as_dict() == graph.as_dict()
    assert restored_graph.document_hash == graph.document_hash
    assert restored_graph.graph_hash == graph.graph_hash
    assert restored_graph.topology_hash == graph.topology_hash


@settings(max_examples=50, deadline=None)
@given(value=GRAPH_VALUE)
def test_generated_extension_values_are_wire_round_trip_safe(value) -> None:
    graph = replace(_linear_graph(), extensions={"generated": value})

    wire_payload = json.loads(json.dumps(graph.as_dict(), allow_nan=False))
    restored = CircuitGraph.from_dict(wire_payload)

    assert restored.as_dict() == graph.as_dict()
    assert restored.document_hash == graph.document_hash


def test_graph_rejects_unknown_schema_version() -> None:
    payload = _linear_graph().as_dict()
    payload["schema_version"] = 2

    with pytest.raises(CircuitGraphMigrationError, match="schema_version=2"):
        CircuitGraph.from_dict(payload)


def test_v0_payload_migrates_to_v1_without_mutating_or_losing_unknown_data() -> None:
    legacy = _linear_graph().as_dict()
    legacy.pop("schema")
    legacy["schema_version"] = 0
    legacy["preferred_solver_id"] = legacy.pop("preferred_solver_ids")[0]
    legacy["vendor_annotation"] = {"tool": "legacy_synth", "revision": 7}
    original = json.loads(json.dumps(legacy))

    migrated = migrate_circuit_graph_payload(legacy)
    restored = CircuitGraph.from_dict(legacy)

    assert legacy == original
    assert migrated["schema"] == CIRCUIT_GRAPH_SCHEMA
    assert migrated["schema_version"] == 1
    assert migrated["preferred_solver_ids"] == ["linear_mna"]
    assert restored.extensions["unknown_top_level_fields"]["vendor_annotation"] == {
        "tool": "legacy_synth",
        "revision": 7,
    }
    assert restored.validate().passed


def test_current_payload_preserves_unknown_top_level_fields_in_extensions() -> None:
    payload = _linear_graph().as_dict()
    payload["producer_specific_evidence"] = {"score": 0.125}

    restored = CircuitGraph.from_dict(payload)

    assert restored.extensions["unknown_top_level_fields"]["producer_specific_evidence"] == {
        "score": 0.125
    }


def test_validation_reports_terminal_and_net_contract_errors() -> None:
    graph = _linear_graph()
    first = graph.components[0]
    invalid_component = replace(
        first,
        connections=(
            TerminalConnection("wrong_terminal", "missing_net"),
            first.connections[1],
        ),
    )
    invalid = replace(graph, components=(invalid_component, *graph.components[1:]))

    report = invalid.validate()

    assert not report.passed
    assert {item.code for item in report.errors} >= {"terminal_contract", "missing_net"}
    with pytest.raises(GraphValidationError):
        invalid.require_valid()


def test_structural_hash_is_invariant_to_order_and_internal_identifiers() -> None:
    graph = _linear_graph()
    reordered = replace(
        graph,
        domains=tuple(reversed(graph.domains)),
        ports=tuple(reversed(graph.ports)),
        nets=tuple(reversed(graph.nets)),
        components=tuple(reversed(graph.components)),
    )
    renamed = _rename_internal_net_and_instances(graph, "n1", "middle")

    for equivalent in (reordered, renamed):
        assert equivalent.validate().passed
        assert equivalent.graph_hash == graph.graph_hash
        assert equivalent.topology_hash == graph.topology_hash
    assert renamed.document_hash != graph.document_hash


@settings(max_examples=60, deadline=None)
@given(
    component_order=st.permutations((0, 1, 2, 3)),
    net_order=st.permutations((0, 1, 2, 3)),
    suffix=st.text(alphabet=string.ascii_lowercase + string.digits, max_size=12),
)
def test_generated_reordering_and_internal_renaming_preserve_structural_hashes(
    component_order: list[int],
    net_order: list[int],
    suffix: str,
) -> None:
    graph = _linear_graph()
    reordered = replace(
        graph,
        components=tuple(graph.components[index] for index in component_order),
        nets=tuple(graph.nets[index] for index in net_order),
    )
    equivalent = _rename_internal_net_and_instances(
        reordered,
        "n1",
        f"internal_{suffix}",
        instance_prefix=f"generated_{suffix}",
    )

    assert equivalent.validate().passed
    assert equivalent.graph_hash == graph.graph_hash
    assert equivalent.topology_hash == graph.topology_hash


@settings(max_examples=60, deadline=None)
@given(value=FINITE_POSITIVE)
def test_generated_parameter_changes_preserve_topology_hash(value: float) -> None:
    graph = _linear_graph()
    resistor = next(item for item in graph.components if item.model.kind == "R")
    changed_resistor = replace(
        resistor,
        parameters=(ParameterBinding("value", "ohm", value),),
    )
    changed = replace(
        graph,
        components=tuple(
            changed_resistor if item.instance_id == resistor.instance_id else item
            for item in graph.components
        ),
    )

    assert changed.validate().passed
    assert changed.topology_hash == graph.topology_hash


def test_parameter_change_affects_graph_hash_but_not_topology_hash() -> None:
    graph = _linear_graph()
    resistor = next(item for item in graph.components if item.model.kind == "R")
    changed_resistor = replace(
        resistor,
        parameters=(ParameterBinding("value", "ohm", 2200.0),),
    )
    changed = replace(
        graph,
        components=tuple(
            changed_resistor if item.instance_id == resistor.instance_id else item
            for item in graph.components
        ),
    )

    assert changed.graph_hash != graph.graph_hash
    assert changed.topology_hash == graph.topology_hash


def test_external_port_identity_is_part_of_topology() -> None:
    graph = _linear_graph()
    input_port = graph.ports[0]
    renamed_port = replace(input_port, port_id="source", name="source")
    changed = replace(graph, ports=(renamed_port, *graph.ports[1:]))

    assert changed.topology_hash != graph.topology_hash


def test_linear_circuit_round_trip_preserves_every_supported_primitive() -> None:
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", 1000.0),
            LinearElement("C1", "C", "out", "0", 1e-9),
            LinearElement("L1", "L", "in", "aux", 1e-3),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "0", 1 + 0.25j),),
        current_sources=(CurrentSource("Iin", "0", "aux", 1e-3),),
        controlled_voltage_sources=(
            VoltageControlledVoltageSource("E1", "sense", "0", "out", "0", 2.5),
        ),
    )

    graph = linear_circuit_to_graph(circuit, output_node="sense")
    restored = graph_to_linear_circuit(CircuitGraph.from_dict(graph.as_dict()))

    assert restored == circuit


@pytest.mark.parametrize("template", default_templates(), ids=lambda item: item.name)
def test_every_ac_template_converts_to_graph_without_information_loss(template) -> None:
    defaults = {"R": 1000.0, "C": 1e-9, "L": 1e-3}
    values = {parameter.name: defaults[parameter.element_type] for parameter in template.params}
    circuit = template.to_circuit(values)
    graph = template.to_graph(values)

    restored = graph_to_linear_circuit(graph)

    assert graph.validate().passed
    assert restored == circuit


def test_linear_graph_round_trip_preserves_mna_response() -> None:
    graph = _linear_graph()
    original = graph_to_linear_circuit(graph)
    restored = graph_to_linear_circuit(CircuitGraph.from_dict(graph.as_dict()))
    frequencies = np.logspace(1, 6, 32)

    first = MNASimulator().transfer(original, "out", "Vin", frequencies)
    second = MNASimulator().transfer(restored, "out", "Vin", frequencies)

    assert np.max(np.abs(first - second)) < 1e-12


def test_linear_adapter_normalizes_all_mna_ground_aliases() -> None:
    circuit = LinearCircuit(
        elements=(
            LinearElement("R1", "R", "in", "out", 1000.0),
            LinearElement("C1", "C", "out", "GND", 1e-9),
        ),
        voltage_sources=(VoltageSource("Vin", "in", "gnd", 1.0),),
    )

    graph = linear_circuit_to_graph(circuit)
    restored = graph_to_linear_circuit(graph)
    frequencies = np.logspace(1, 6, 24)

    assert graph.validate().passed
    assert {item.net_id for item in graph.nets} == {"0", "in", "out"}
    assert all(
        connection.net_id not in {"gnd", "GND"}
        for component in graph.components
        for connection in component.connections
    )
    original_response = MNASimulator().transfer(circuit, "out", "Vin", frequencies)
    restored_response = MNASimulator().transfer(restored, "out", "Vin", frequencies)
    assert np.max(np.abs(original_response - restored_response)) < 1e-12


@pytest.mark.parametrize(
    ("vin", "vout", "isolated"),
    ((12.0, 5.0, False), (5.0, 10.0, False), (5.0, 10.0, True)),
)
def test_power_candidates_convert_to_valid_graphs(vin, vout, isolated) -> None:
    ir = pbdl_to_ir(_power_spec(vin, vout, isolated=isolated))
    candidates = PowerTopologyGrammar().search(ir).candidates

    assert candidates
    for candidate in candidates:
        graph = topology_candidate_to_graph(candidate, ir)
        restored = graph_to_topology_candidate(CircuitGraph.from_dict(graph.as_dict()))

        assert graph.validate().passed
        assert restored.family == candidate.family
        assert [item.kind for item in restored.components] == [
            item.kind for item in candidate.components
        ]
        assert [item.nodes for item in restored.components] == [
            item.nodes for item in candidate.components
        ]


def test_isolated_graph_has_symmetric_domains_and_distinct_references() -> None:
    ir = pbdl_to_ir(_power_spec(5.0, 10.0, isolated=True))
    candidate = PowerTopologyGrammar().search(ir).candidates[0]
    graph = topology_candidate_to_graph(candidate, ir)

    domains = {item.domain_id: item for item in graph.domains}
    assert domains["input_domain"].isolated_from == ("output_domain",)
    assert domains["output_domain"].isolated_from == ("input_domain",)
    assert domains["input_domain"].reference_net_id != domains["output_domain"].reference_net_id


def test_graph_constants_are_emitted_in_public_payload() -> None:
    payload = _linear_graph().as_dict()

    assert payload["schema"] == CIRCUIT_GRAPH_SCHEMA
    assert payload["schema_version"] == 1


def _rename_internal_net_and_instances(
    graph,
    old_net: str,
    new_net: str,
    *,
    instance_prefix: str = "instance",
):
    nets = tuple(
        replace(item, net_id=new_net, name=new_net) if item.net_id == old_net else item
        for item in graph.nets
    )
    components = []
    for index, component in enumerate(graph.components):
        connections = tuple(
            replace(item, net_id=new_net) if item.net_id == old_net else item
            for item in component.connections
        )
        components.append(
            replace(
                component,
                instance_id=f"{instance_prefix}_{index}",
                reference=f"X{index}",
                connections=connections,
            )
        )
    return replace(graph, nets=nets, components=tuple(components))
