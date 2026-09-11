from __future__ import annotations

import json

import pytest

from circuit_ai.experts import UnsupportedTopology
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.pipeline import design_from_pbdl
from circuit_ai.topology_grammar import PowerTopologyGrammar


def _power_spec(
    vin: float,
    vout: float,
    *,
    allowed: list[str] | None = None,
    max_components: int = 8,
) -> dict:
    return {
        "name": "grammar_power_test",
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
            "element_types": allowed or ["R", "C", "L", "ideal_switch", "ideal_diode"],
            "max_component_count": max_components,
        },
        "optimization": {"max_iterations": 2, "popsize": 4, "seed": 17},
    }


def test_step_up_grammar_generates_multiple_executable_families() -> None:
    ir = pbdl_to_ir(_power_spec(5.0, 10.0, max_components=8))

    search = PowerTopologyGrammar().search(ir)

    assert [candidate.family for candidate in search.candidates] == ["dc_boost", "dc_sepic"]
    assert search.certificate.accepted_candidates == 2
    assert search.certificate.exhaustive_within_registered_productions
    assert all(candidate.metadata["derivation"] for candidate in search.candidates)


def test_component_bound_prunes_sepic_without_disabling_boost() -> None:
    ir = pbdl_to_ir(_power_spec(5.0, 10.0, max_components=5))

    search = PowerTopologyGrammar().search(ir)

    assert [candidate.family for candidate in search.candidates] == ["dc_boost"]
    assert any(item.name == "grammar_ideal_sepic" for item in search.certificate.rejected)


def test_pipeline_selects_grammar_generated_buck_and_writes_audit_files(tmp_path) -> None:
    result = design_from_pbdl(_power_spec(12.0, 5.0), tmp_path)

    assert result.succeeded
    assert result.optimization.candidate.family == "dc_buck"
    assert result.optimization.candidate.metadata["origin"] == "functional_block_knowledge_base"
    assert abs(result.optimization.operating_point.output_voltage_v - 5.0) < 1e-5
    assert (tmp_path / "topology_search.json").exists()
    assert (tmp_path / "simulation_tasks.json").exists()
    assert (tmp_path / "replay.jsonl").exists()
    assert (tmp_path / "best.kicad_sch").exists()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["topology_search"]["accepted_candidates"] == 1
    assert report["candidates"][0]["candidate"]["family"] == "dc_buck"


def test_library_constraint_can_exclude_every_registered_topology() -> None:
    spec = _power_spec(
        5.0,
        10.0,
        allowed=["R", "C", "ideal_switch", "ideal_diode"],
    )

    with pytest.raises(UnsupportedTopology, match="declared search bounds"):
        design_from_pbdl(spec)


def test_step_up_pipeline_records_boost_and_sepic_replay_rows(tmp_path) -> None:
    result = design_from_pbdl(_power_spec(5.0, 10.0, max_components=8), tmp_path)

    assert result.succeeded
    assert result.optimization.candidate.family == "dc_boost"
    assert {item.candidate.family for item in result.evaluations} == {"dc_boost", "dc_sepic"}
    replay_rows = [
        json.loads(line)
        for line in (tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(replay_rows) == 2
    assert all(row["kind"] == "verified_power_synthesis" for row in replay_rows)


def test_native_power_graph_search_constructs_boost_and_records_certificate() -> None:
    spec = _power_spec(5.0, 10.0, max_components=5)
    spec["optimization"]["graph_search"] = {
        "native_enabled": True,
        "max_candidates": 2,
        "beam_width": 8,
        "max_expansions": 220,
    }

    search = PowerTopologyGrammar().search(pbdl_to_ir(spec))

    native = [
        candidate
        for candidate in search.candidates
        if candidate.metadata.get("origin") == "native_power_graph_search"
    ]
    assert native
    assert all(candidate.graph is not None for candidate in native)
    assert all(candidate.metadata.get("solver") == "ideal_boost_averaged" for candidate in native)
    assert all(candidate.metadata.get("derivation") for candidate in native)
    assert search.certificate.exhaustive_within_registered_productions is False
    assert search.certificate.native_search is not None
    assert search.certificate.native_search["search"]["grammar_version"] == "power_graph_actions.v1"


def test_pipeline_can_opt_into_native_power_graph_search(tmp_path) -> None:
    spec = _power_spec(5.0, 10.0, max_components=5)
    spec["optimization"]["graph_search"] = {
        "native_enabled": True,
        "max_candidates": 1,
        "beam_width": 8,
        "max_expansions": 220,
    }

    result = design_from_pbdl(spec, tmp_path)

    assert result.succeeded
    assert any(
        item.candidate.metadata.get("origin") == "native_power_graph_search"
        for item in result.evaluations
    )
    payload = json.loads((tmp_path / "topology_search.json").read_text(encoding="utf-8"))
    assert payload["native_search"]["accepted_candidates"] >= 1
