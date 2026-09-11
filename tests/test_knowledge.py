from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from circuit_ai.ir import pbdl_to_ir
from circuit_ai.knowledge import DEFAULT_POWER_KNOWLEDGE, load_topology_knowledge
from circuit_ai.topology_grammar import PowerTopologyGrammar


def _step_up_spec() -> dict:
    return {
        "name": "knowledge_test",
        "ports": [
            {
                "name": "input",
                "role": "input",
                "terminals": [
                    {"name": "in", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
            {
                "name": "output",
                "role": "output",
                "terminals": [
                    {"name": "out", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
        ],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [{
            "target_kind": "dc",
            "input_voltage_v": 5.0,
            "output_voltage_v": 10.0,
            "output_current_a": 1.0,
        }],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_diode"],
            "max_component_count": 8,
        },
    }


def _default_yaml() -> dict:
    return yaml.safe_load(DEFAULT_POWER_KNOWLEDGE.read_text(encoding="utf-8"))


def test_default_knowledge_loads_functional_modules_and_productions() -> None:
    knowledge = load_topology_knowledge()

    assert knowledge.knowledge_id == "circuit-ai-power-blocks-v1"
    assert "pwm_switch" in knowledge.modules
    assert all(module.graph.validate().passed for module in knowledge.modules.values())
    assert knowledge.modules["pwm_switch"].graph.graph_id == "module:pwm_switch"
    assert {item.family for item in knowledge.productions} == {
        "dc_buck",
        "dc_boost",
        "dc_sepic",
        "isolated_flyback",
    }


def test_candidate_records_knowledge_provenance_and_slot_derivation() -> None:
    search = PowerTopologyGrammar().search(pbdl_to_ir(_step_up_spec()))

    candidate = search.candidates[0]
    assert candidate.metadata["origin"] == "functional_block_knowledge_base"
    assert candidate.metadata["knowledge_id"] == "circuit-ai-power-blocks-v1"
    assert candidate.metadata["knowledge_sources"]
    assert all(item.startswith("slot:") for item in candidate.metadata["derivation"])
    assert candidate.solver_id == "ideal_boost_averaged"
    assert candidate.graph is not None and candidate.graph.validate().passed
    assert candidate.graph.provenance[0].source == "functional_block_knowledge_base"


def test_loader_rejects_incomplete_slot_bindings(tmp_path) -> None:
    data = _default_yaml()
    data["productions"][0]["slots"][0]["bindings"].pop("b")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="bindings do not match"):
        load_topology_knowledge(path)


def test_new_production_can_be_registered_without_python_connection_code(tmp_path) -> None:
    data = _default_yaml()
    alternate = deepcopy(data["productions"][1])
    alternate["name"] = "alternate_boost_from_yaml"
    alternate["metadata"] = {"variant": "knowledge_extension_test"}
    data["knowledge_id"] = "extended-test-knowledge"
    data["productions"].append(alternate)
    path = tmp_path / "extended.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    search = PowerTopologyGrammar(path).search(pbdl_to_ir(_step_up_spec()))

    assert "alternate_boost_from_yaml" in search.certificate.registered_productions
    assert any(
        item.name == "alternate_boost_from_yaml" and item.reason == "duplicate electrical graph"
        for item in search.certificate.rejected
    )
