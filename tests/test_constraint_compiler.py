from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from circuit_ai.constraints import (
    ConstraintEvaluator,
    ConstraintSeverity,
    FeasibilityStatus,
    compile_constraint_program,
)
from circuit_ai.experts import ExpertTopologySelector
from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.metrics import MetricContext, MetricEngine
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.pbdl_boundary import canonicalize_pbdl_dict
from circuit_ai.simulation import Quantity, ResultAxis, SimulationResult, SimulationStatus, Waveform


BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")
LOWPASS_SPEC = Path("端口描述语言/examples/lowpass_1khz.json")


def _result(graph, *, scalars=None, waveforms=None):
    models = {(item.model.model_id, item.model.version): item.model for item in graph.components}
    return SimulationResult(
        request_id="constraint_request",
        request_hash="request_hash",
        graph_hash=graph.graph_hash,
        backend_id="linear_mna",
        backend_version="1.0.0",
        model_manifest=tuple(models[key] for key in sorted(models)),
        status=SimulationStatus.PASSED,
        scalars=scalars or {},
        waveforms=waveforms or {},
        fidelity="linear_frequency_domain",
    )


def _rc_graph():
    return linear_circuit_to_graph(
        LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "out", 1000.0),
                LinearElement("C1", "C", "out", "0", 1.0 / (2.0 * np.pi * 1000.0 * 1000.0)),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        ),
        name="constraint_rc",
    )


def test_pbdl_round_trip_preserves_declarative_constraint_fields_and_dc_tolerance() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["ports"][1]["variable_constraints"] = [
        {
            "id": "output_voltage_soft",
            "variable": "v",
            "analysis": "dc_operating_point",
            "operator": "equal",
            "value": 10.0,
            "unit": "V",
            "severity": "soft",
            "tolerance": {"relative": 0.02},
            "weight": 3.0,
            "reduction": "mean",
            "required_evidence_rank": 40,
        }
    ]
    data["constraints"]["objectives"] = [
        {"id": "cost", "metric": "resource.bom_cost", "direction": "minimize"}
    ]
    canonical = canonicalize_pbdl_dict(data)
    constraint = canonical["ports"][1]["variable_constraints"][0]
    assert constraint["id"] == "output_voltage_soft"
    assert constraint["severity"] == "soft"
    assert constraint["tolerance"]["relative"] == 0.02
    assert constraint["required_evidence_rank"] == 40
    assert canonical["targets"][0]["tolerance"]["relative"] == 0.01
    assert canonical["constraints"]["objectives"][0]["metric"] == "resource.bom_cost"


def test_boost_compiles_and_evaluates_all_legacy_targets_and_resources() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    ir = pbdl_to_ir(data)
    graph = ExpertTopologySelector().select(ir).graph
    program = compile_constraint_program(ir, graph)
    output_voltage = next(
        item for item in program.constraints if item.constraint_id == "target.0.output_voltage_v"
    )
    assert output_voltage.tolerance.relative == 0.01
    assert {item.constraint_id for item in program.constraints} >= {
        "structure.allowed_elements",
        "structure.required_elements",
        "resource.max_component_count",
        "resource.max_cost",
        "resource.max_area",
        "resource.max_power",
    }
    result = _result(
        graph,
        scalars={
            "input_voltage_v": Quantity(5.0, "V"),
            "input_current_a": Quantity(0.2, "A"),
            "output_voltage_v": Quantity(10.0, "V"),
            "output_current_a": Quantity(0.1, "A"),
            "output_power_w": Quantity(1.0, "W"),
            "efficiency": Quantity(1.0, "1"),
            "predicted_ripple_mv": Quantity(20.0, "mV"),
        },
    )
    metrics = MetricEngine().extract(
        program.metrics,
        MetricContext(graph=graph, ir=ir, simulation_results=(result,)),
    )
    report = ConstraintEvaluator().evaluate(program.constraints, metrics)
    assert report.feasibility is FeasibilityStatus.VERIFIED_FEASIBLE
    assert report.cost_cny == pytest.approx(0.47)
    assert report.area_mm2 == pytest.approx(59.0)


def test_four_port_constraints_remain_independent() -> None:
    ports = []
    expected = {"p1": 5.0, "p2": 10.0, "p3": 3.3, "p4": -2.0}
    for index, (name, value) in enumerate(expected.items()):
        ports.append(
            {
                "name": name,
                "terminals": [
                    {"name": f"n{index}", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
                "variable_constraints": [
                    {
                        "id": f"constraint_{name}",
                        "variable": "v",
                        "analysis": "dc_operating_point",
                        "operator": "equal",
                        "value": value,
                        "unit": "V",
                    }
                ],
            }
        )
    data = {
        "name": "four_port",
        "ports": ports,
        "analyses": [{"kind": "dc_transfer", "source_port": "p1", "output_port": "p2"}],
        "targets": [],
        "constraints": {"element_types": ["R"], "max_component_count": 2},
    }
    ir = pbdl_to_ir(data)
    graph = _rc_graph()
    program = compile_constraint_program(ir, graph)
    port_constraints = [item for item in program.constraints if item.constraint_id.startswith("constraint_")]
    assert len(port_constraints) == 4
    assert len({item.metric.metric_id for item in port_constraints}) == 4
    result = _result(
        graph,
        scalars={
            "input_voltage_v": Quantity(5.0, "V"),
            "output_voltage_v": Quantity(10.0, "V"),
            "port.p3.voltage_v": Quantity(3.3, "V"),
            "port.p4.voltage_v": Quantity(-2.0, "V"),
        },
    )
    metrics = MetricEngine().extract(
        program.metrics, MetricContext(graph=graph, ir=ir, simulation_results=(result,))
    )
    report = ConstraintEvaluator().evaluate(program.constraints, metrics)
    assert all(
        item.status.value == "passed"
        for item in report.evaluations
        if item.constraint_id.startswith("constraint_")
    )


def test_explicit_objective_compiles_without_becoming_a_hard_constraint() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data["constraints"]["objectives"] = [
        {
            "id": "minimize_cost",
            "metric": "resource.bom_cost",
            "direction": "minimize",
            "weight": None,
        }
    ]
    ir = pbdl_to_ir(data)
    program = compile_constraint_program(ir, ExpertTopologySelector().select(ir).graph)
    objective = next(item for item in program.constraints if item.constraint_id == "minimize_cost")
    assert objective.severity is ConstraintSeverity.OBJECTIVE


def test_same_evaluator_handles_rc_frequency_response() -> None:
    data = json.loads(LOWPASS_SPEC.read_text(encoding="utf-8"))
    ir = pbdl_to_ir(data)
    graph = _rc_graph()
    program = compile_constraint_program(ir, graph)
    analysis = ir.primary_analysis
    frequencies = np.geomspace(
        float(analysis["frequency_hz"][0]),
        float(analysis["frequency_hz"][1]),
        int(analysis["frequency_hz"][2]),
    )
    target_module = importlib.import_module("端口描述语言.targets")
    target = target_module.target_from_dict(ir.primary_target)
    waveform = Waveform(
        observable_id="voltage_transfer",
        unit="1",
        axes=(ResultAxis("frequency", "Hz", tuple(float(item) for item in frequencies)),),
        values=tuple(complex(item) for item in target.evaluate(frequencies)),
    )
    result = _result(graph, waveforms={"voltage_transfer": waveform})
    metrics = MetricEngine().extract(
        program.metrics, MetricContext(graph=graph, ir=ir, simulation_results=(result,))
    )
    report = ConstraintEvaluator().evaluate(program.constraints, metrics)
    assert report.feasibility is FeasibilityStatus.VERIFIED_FEASIBLE
    assert report.objective_values["target.0.minimize_rmse_db"] == pytest.approx(0.0, abs=1e-12)
    assert report.cost_cny == 0.0
