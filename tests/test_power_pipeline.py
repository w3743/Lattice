from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from circuit_ai.ir import pbdl_to_ir
from circuit_ai.constraints import FeasibilityStatus
from circuit_ai.pipeline import design_from_pbdl
from circuit_ai.power import BoostParameters, simulate_ideal_boost, solve_ideal_boost_dc
from circuit_ai.simulation import SimulationStatus


BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


def test_pbdl_is_losslessly_projected_into_unified_ir() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    ir = pbdl_to_ir(data)

    assert ir.intent_kind == "dc"
    assert [port.name for port in ir.ports] == ["input", "output"]
    assert ir.primary_target["output_voltage_v"] == 10.0
    assert ir.constraints["required_elements"] == ["L", "C", "ideal_switch", "ideal_diode"]


def test_ideal_boost_dc_and_transient_are_finite() -> None:
    params = BoostParameters(0.5, 100e-6, 100e-6, 100e3, 100.0)
    dc = solve_ideal_boost_dc(5.0, params)
    trace = simulate_ideal_boost(5.0, params, cycles=20)

    assert abs(dc.output_voltage_v - 10.0) < 1e-12
    assert abs(dc.output_current_a - 0.1) < 1e-12
    assert np.isfinite(trace.output_voltage_v).all()
    assert np.isfinite(trace.inductor_current_a).all()
    assert trace.output_voltage_v[-1] > 0.0


def test_boost_pipeline_writes_validated_kicad_project(tmp_path) -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    result = design_from_pbdl(data, tmp_path)

    assert result.succeeded
    assert result.optimization.candidate.name == "ideal_asynchronous_boost"
    assert result.validation.feasibility is FeasibilityStatus.VERIFIED_FEASIBLE
    assert result.validation.as_dict()["schema"] == "circuit_ai.constraint_report"
    voltage_check = next(
        item
        for item in result.validation.evaluations
        if item.metric_id == "port.output.voltage.dc"
    )
    assert voltage_check.actual is not None
    assert voltage_check.actual_unit == "V"
    assert voltage_check.source_path
    assert voltage_check.evidence_level != "unverified"
    assert result.validation.cost_cny < 2.0
    assert result.kicad_schematic is not None and result.kicad_schematic.exists()
    schematic = result.kicad_schematic.read_text(encoding="utf-8")
    assert 'lib_id "CircuitAI:SWITCH"' in schematic
    assert 'lib_id "CircuitAI:DIODE"' in schematic
    assert (tmp_path / "ir.json").exists()
    assert (tmp_path / "transient.json").exists()
    assert (tmp_path / "best.svg").exists()
    assert result.graph_svg is not None and result.graph_svg.exists()
    assert result.graph_kicad_schematic is not None and result.graph_kicad_schematic.exists()
    graph_export = json.loads((tmp_path / "circuit_graph_export.json").read_text(encoding="utf-8"))
    assert graph_export["graph_hash"] == result.optimization.candidate.graph.graph_hash
    assert "ideal_switch" in (tmp_path / "circuit_graph.svg").read_text(encoding="utf-8")
    assert (tmp_path / "capability_manifest.json").exists()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    selected_ids = {
        item["selected"]["capability_id"]
        for item in report["capabilities"]["selected"]
    }
    assert "power.optimizer.ideal_boost_averaged" in selected_ids
    assert "simulation.power.ideal_boost_averaged" in selected_ids
    assert "power.exporter.kicad" in selected_ids
    assert result.simulation_requests[0].analysis.kind == "dc_transfer"
    assert result.simulation_results[0].status is SimulationStatus.PASSED
    assert result.simulation_results[0].scalars["output_voltage_v"].value == report[
        "operating_point"
    ]["output_voltage_v"]
    assert report["simulation_requests"][0]["request_hash"]
    assert report["simulation_results"][0]["evidence_hash"]
    optimization_run = result.optimization.optimization_run
    assert optimization_run is not None
    assert optimization_run.problem_id == "power.dc_boost"
    assert optimization_run.truth_evaluated
    assert optimization_run.evaluations > 0
    assert report["optimization_run"]["problem_id"] == "power.dc_boost"
    simulation_evidence = json.loads(
        (tmp_path / "simulation_tasks.json").read_text(encoding="utf-8")
    )
    assert simulation_evidence["requests"][0]["request_hash"]
    assert simulation_evidence["results"][0]["evidence_hash"]


def test_requested_power_spice_gate_does_not_accept_missing_executable() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data.setdefault("optimization", {})["spice_verification"] = {
        "enabled": True,
        "executable": "definitely_missing_ngspice_for_power_test",
    }
    result = design_from_pbdl(data)
    truth = [item for item in result.simulation_results if item.backend_id == "ngspice"]
    assert truth
    assert truth[0].status is SimulationStatus.UNAVAILABLE
    assert not result.succeeded
    assert not result.validation.passed


def test_power_replay_marks_missing_high_fidelity_evidence_rejected(tmp_path) -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data.setdefault("optimization", {})["spice_verification"] = {
        "enabled": True,
        "executable": "definitely_missing_ngspice_for_replay_test",
    }

    design_from_pbdl(data, tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    assert all(row["label"]["accepted"] is False for row in rows)


def test_isolated_high_current_spec_selects_flyback(tmp_path) -> None:
    data = {
        "name": "isolated_5v_to_10v_10a",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "out_0", "quantity": "ground"}]},
        ],
        "relations": [{"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [{"target_kind": "dc", "input_voltage_v": 5, "output_voltage_v": 10, "output_current_a": 10}],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_transformer", "ideal_diode"],
            "max_component_count": 5,
            "parameter_ranges": {"R": [100, 1_000_000], "C": [1e-12, 1e-4], "L": [1e-9, 1]},
        },
        "optimization": {"max_iterations": 5, "seed": 11},
    }
    result = design_from_pbdl(data, tmp_path)

    assert result.succeeded
    assert result.optimization.candidate.name == "ideal_isolated_flyback"
    assert result.validation.feasibility is FeasibilityStatus.VERIFIED_FEASIBLE
    assert result.validation.as_dict()["schema"] == "circuit_ai.constraint_report"
    assert abs(result.optimization.operating_point.output_voltage_v - 10.0) < 0.2
    assert abs(result.optimization.operating_point.output_current_a - 10.0) < 0.2
    optimization_run = result.optimization.optimization_run
    assert optimization_run is not None
    assert optimization_run.problem_id == "power.isolated_flyback"
    assert optimization_run.truth_evaluated
    assert 'lib_id "CircuitAI:TRANSFORMER"' in result.kicad_schematic.read_text(encoding="utf-8")
    assert "isolation barrier" in (tmp_path / "best.svg").read_text(encoding="utf-8")
