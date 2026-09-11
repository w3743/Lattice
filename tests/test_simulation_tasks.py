from __future__ import annotations

from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.power import BoostOperatingPoint
from circuit_ai.simulation_tasks import evaluate_simulation_task, simulation_tasks_from_ir


def _spec() -> dict:
    return {
        "name": "simulation_contract_test",
        "ports": [
            {
                "name": "input",
                "role": "input",
                "terminals": [
                    {"name": "in", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
                "variable_constraints": [{
                    "variable": "v",
                    "analysis": "dc_operating_point",
                    "operator": "equal",
                    "value": 5.0,
                    "unit": "V",
                }],
            },
            {
                "name": "output",
                "role": "output",
                "terminals": [
                    {"name": "out", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
                "variable_constraints": [{
                    "variable": "i",
                    "analysis": "dc_operating_point",
                    "operator": "minimum",
                    "minimum": 9.5,
                    "unit": "A",
                }],
            },
        ],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [{
            "target_kind": "dc",
            "input_voltage_v": 5.0,
            "output_voltage_v": 10.0,
            "output_current_a": 10.0,
            "efficiency": 0.9,
            "ripple_mv": 100.0,
        }],
    }


def _operating_point(output_current: float = 10.0) -> BoostOperatingPoint:
    return BoostOperatingPoint(
        input_voltage_v=5.0,
        output_voltage_v=10.0,
        input_current_a=20.0,
        output_current_a=output_current,
        output_power_w=100.0,
        efficiency=1.0,
        predicted_ripple_mv=20.0,
        inductor_ripple_a=1.0,
    )


def _graph():
    return linear_circuit_to_graph(
        LinearCircuit(
            elements=(LinearElement("R1", "R", "in", "out", 1000.0),),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        ),
        name="task_request_graph",
    )


def test_dc_transfer_task_includes_dc_operating_point_port_constraints() -> None:
    task = simulation_tasks_from_ir(pbdl_to_ir(_spec()))[0]

    assert task.analysis_kind == "dc_transfer"
    assert {metric.name for metric in task.metrics} >= {
        "output_voltage_v",
        "output_current_a",
        "efficiency",
        "ripple_mv",
        "input.v",
        "output.i",
    }


def test_task_evaluation_accepts_matching_solver_measurements() -> None:
    task = simulation_tasks_from_ir(pbdl_to_ir(_spec()))[0]

    evaluation = evaluate_simulation_task(task, _operating_point())

    assert evaluation.passed
    assert all(item.passed for item in evaluation.observations)


def test_task_evaluation_rejects_a_failed_port_constraint() -> None:
    task = simulation_tasks_from_ir(pbdl_to_ir(_spec()))[0]

    evaluation = evaluate_simulation_task(task, _operating_point(output_current=8.0))

    assert not evaluation.passed
    assert any("output_current_a" in issue or "8 A" in issue for issue in evaluation.issues)


def test_dc_task_compiles_to_dc_request_without_frequency_sweep() -> None:
    spec = _spec()
    spec["ports"][0]["excitation"] = {
        "waveform": {"kind": "sine", "frequency_hz": 1e6, "amplitude_v": 1.0}
    }
    task = simulation_tasks_from_ir(pbdl_to_ir(spec))[0]

    request = task.to_requests(
        _graph(),
        parameter_values={"duty_cycle": 0.5},
        fidelity="analytic",
    )[0]

    assert request.analysis.kind == "dc_transfer"
    assert request.analysis.sweeps == ()
    assert len(request.excitations) == 1
    assert request.excitations[0].kind == "dc_voltage"
    assert request.excitations[0].parameters["value"] == 5.0
    assert {item.observable_id for item in request.requested_observables} >= {
        "output_voltage_v",
        "output_current_a",
        "efficiency",
    }


def test_ac_task_compiles_to_frequency_request_and_complex_transfer_observable() -> None:
    spec = {
        "name": "ac_task",
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
            {
                "kind": "voltage_transfer",
                "source_port": "input",
                "output_port": "output",
                "frequency_hz": [10.0, 1e6, 41],
            }
        ],
        "targets": [{"target_kind": "filter", "filter_kind": "lowpass"}],
    }
    task = simulation_tasks_from_ir(pbdl_to_ir(spec))[0]

    request = task.to_requests(_graph(), fidelity="linear_frequency_domain")[0]

    assert request.analysis.kind == "small_signal_ac"
    assert request.analysis.sweeps[0].variable == "frequency"
    assert request.analysis.sweeps[0].scale == "log"
    assert len(request.analysis.sweeps[0].materialize()) == 41
    assert request.excitations[0].kind == "ac_voltage"
    assert request.requested_observables[0].quantity == "voltage_gain"
    assert request.requested_observables[0].representation == "complex"
