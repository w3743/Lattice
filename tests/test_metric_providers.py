from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_ai.experts import ExpertTopologySelector
from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.ir import pbdl_to_ir
from circuit_ai.metrics import MetricContext, MetricEngine, MetricSpec, MetricStatus
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.simulation import (
    Quantity,
    ResultAxis,
    SimulationResult,
    SimulationStatus,
    Waveform,
)


BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


def _linear_graph():
    return linear_circuit_to_graph(
        LinearCircuit(
            elements=(LinearElement("R1", "R", "in", "out", 1000.0),),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        ),
        name="metric_provider_graph",
    )


def _result(graph, *, scalars=None, waveforms=None, status=SimulationStatus.PASSED):
    manifest = tuple(
        {item.model.model_id: item.model for item in graph.components}.values()
    )
    return SimulationResult(
        request_id="request_1",
        request_hash="request_hash",
        graph_hash=graph.graph_hash,
        backend_id="linear_mna",
        backend_version="1.0.0",
        model_manifest=manifest,
        status=status,
        scalars=scalars or {},
        waveforms=waveforms or {},
        fidelity="linear_frequency_domain",
    )


def test_scalar_provider_preserves_evidence_and_converts_units() -> None:
    graph = _linear_graph()
    result = _result(graph, scalars={"output_voltage_v": Quantity(5000.0, "mV")})
    spec = MetricSpec(
        "port.output.voltage",
        "dc_port_metrics",
        "output_voltage_v",
        "voltage",
        "V",
    )
    metric = MetricEngine().extract((spec,), MetricContext(graph=graph, simulation_results=(result,))).get(spec.metric_id)
    assert metric is not None and metric.available
    assert metric.value == pytest.approx(5.0)
    assert metric.evidence_level == "numerical"
    assert metric.evidence[0].evidence_hash == result.evidence_hash


def test_missing_observable_from_successful_result_is_explicit() -> None:
    graph = _linear_graph()
    spec = MetricSpec("missing", "simulation.scalar", "absent", "voltage", "V")
    metric = MetricEngine().extract(
        (spec,), MetricContext(graph=graph, simulation_results=(_result(graph),))
    ).get("missing")
    assert metric is not None and metric.status is MetricStatus.MISSING
    assert "absent" in metric.diagnostics[0]


def test_frequency_response_provider_computes_rmse_against_declared_samples() -> None:
    graph = _linear_graph()
    waveform = Waveform(
        observable_id="voltage_transfer",
        unit="1",
        axes=(ResultAxis("frequency", "Hz", (10.0, 100.0, 1000.0)),),
        values=(1.0 + 0j, 0.5 + 0j, 0.1 + 0j),
    )
    target_db = (0.0, 20.0 * __import__("math").log10(0.5), -20.0)
    spec = MetricSpec(
        "ac.transfer.rmse_db",
        "frequency_response",
        "voltage_transfer",
        "magnitude_error",
        "dB",
        reduction="rmse_db",
        selectors={"target_magnitude_db": target_db},
    )
    result = _result(graph, waveforms={"voltage_transfer": waveform})
    metric = MetricEngine().extract(
        (spec,), MetricContext(graph=graph, simulation_results=(result,))
    ).get(spec.metric_id)
    assert metric is not None and metric.value == pytest.approx(0.0, abs=1e-12)


def test_transient_ripple_uses_declared_steady_state_window() -> None:
    graph = _linear_graph()
    waveform = Waveform(
        observable_id="output_voltage",
        unit="V",
        axes=(ResultAxis("time", "s", tuple(index * 1e-6 for index in range(10))),),
        values=(0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 9.9, 10.1, 9.9),
    )
    spec = MetricSpec(
        "port.output.voltage.peak_to_peak",
        "transient_ripple",
        "output_voltage",
        "voltage_ripple",
        "mV",
        reduction="steady_state_peak_to_peak",
        selectors={"window_fraction": 0.3},
    )
    result = _result(graph, waveforms={"output_voltage": waveform})
    metric = MetricEngine().extract(
        (spec,), MetricContext(graph=graph, simulation_results=(result,))
    ).get(spec.metric_id)
    assert metric is not None and metric.value == pytest.approx(200.0)


def test_resource_provider_excludes_external_load_and_uses_declared_catalog_values() -> None:
    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    ir = pbdl_to_ir(data)
    candidate = ExpertTopologySelector().select(ir)
    specs = (
        MetricSpec("resource.component_count", "graph.resource", "component_count", "count", "1"),
        MetricSpec("resource.bom_cost", "graph.resource", "bom_cost", "cost", "CNY"),
        MetricSpec("resource.area", "graph.resource", "area", "area", "mm2"),
    )
    metrics = MetricEngine().extract(specs, MetricContext(graph=candidate.graph, ir=ir))
    assert metrics.get("resource.component_count").value == 4.0
    assert metrics.get("resource.bom_cost").value == pytest.approx(0.47)
    assert metrics.get("resource.area").value == pytest.approx(59.0)


def test_graph_parameter_provider_reads_materialized_binding() -> None:
    graph = _linear_graph()
    spec = MetricSpec(
        "component.R1.value",
        "graph.parameter",
        "R1.value",
        "resistance",
        "kohm",
    )
    metric = MetricEngine().extract((spec,), MetricContext(graph=graph)).get(spec.metric_id)
    assert metric is not None and metric.value == pytest.approx(1.0)


def test_device_stress_provider_reduces_explicit_component_waveform() -> None:
    graph = _linear_graph()
    waveform = Waveform(
        observable_id="device.R1.current",
        unit="A",
        axes=(ResultAxis("time", "s", (0.0, 1.0, 2.0)),),
        values=(-2.0, 3.0, -4.0),
    )
    result = _result(graph, waveforms={waveform.observable_id: waveform})
    spec = MetricSpec(
        "device.R1.current.maximum_abs",
        "device_stress",
        waveform.observable_id,
        "current",
        "A",
        reduction="maximum_abs",
    )
    metric = MetricEngine().extract(
        (spec,), MetricContext(graph=graph, simulation_results=(result,))
    ).get(spec.metric_id)
    assert metric is not None and metric.value == pytest.approx(4.0)
