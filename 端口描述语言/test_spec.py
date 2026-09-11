"""端口描述语言测试。"""

from __future__ import annotations

import math

import numpy as np

from 端口描述语言 import (
    CircuitSpec,
    FilterTarget,
    AmplifierTarget,
    ImpedanceTarget,
    SampledTarget,
    MaskTarget,
    DcTarget,
    DcTransfer,
    Port,
    PortBundle,
    Terminal,
    PortVariable,
    VariableConstraint,
    AcTransfer,
    Impedance,
    Tolerance,
    Constraints,
    OperatingPoint,
    load_spec,
    save_spec,
)
from 端口描述语言.targets import _butterworth_poles


def test_terminal_creation() -> None:
    t = Terminal("in", "voltage")
    assert t.name == "in"
    assert t.quantity == "voltage"

    t2 = Terminal.from_dict("gnd")
    assert t2.quantity == "unspecified"

    t3 = Terminal.from_dict({"name": "vcc", "quantity": "voltage"})
    assert t3.name == "vcc"


def test_port_creation() -> None:
    p = Port.single_ended("input")
    assert p.voltage_terminal().name == "in"
    assert p.ref_terminal().name == "0"

    p_diff = Port.differential("signal", "p", "n")
    assert len(p_diff.terminals) == 2
    assert all(t.quantity == "voltage" for t in p_diff.terminals)


def test_electrical_port_has_potential_flow_and_derived_power_variables() -> None:
    port = Port.electrical("supply", "vin_plus", "vin_return")
    assert [(item.name, item.role) for item in port.variables] == [
        ("v", "potential"),
        ("i", "flow"),
        ("p", "derived"),
    ]
    assert port.variables[0].expression == "V(vin_plus) - V(vin_return)"
    assert port.variables[1].expression == "current entering vin_plus"


def test_port_variables_accept_independent_analysis_scoped_constraints() -> None:
    port = Port.electrical(
        "output",
        "vout_plus",
        "vout_return",
        constraints=(
            VariableConstraint("v", {"kind": "dc_operating_point"}, "equal", value=20, unit="V"),
            VariableConstraint("i", {"kind": "dc_operating_point"}, "interval", minimum=-10, maximum=0, unit="A"),
            VariableConstraint("v", {"kind": "small_signal_ac", "frequency_hz": [10, 1e6]}, "function", data={"magnitude_db": [0, -3]}),
        ),
    )
    restored = Port.from_dict(port.as_dict())
    assert restored.variable_constraints[0].value == 20
    assert restored.variable_constraints[1].minimum == -10
    assert restored.variable_constraints[2].analysis["kind"] == "small_signal_ac"


def test_port_bundle() -> None:
    bundle = PortBundle.standard_two_port()
    assert bundle["input"].voltage_terminal().name == "in"
    assert bundle["output"].voltage_terminal().name == "out"
    assert len(bundle) == 2


def test_filter_target_lowpass() -> None:
    target = FilterTarget(kind="lowpass", order=1, cutoff_hz=1000.0)
    freqs = np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0])

    resp = target.evaluate(freqs)
    mag_db = target.evaluate_magnitude_db(freqs)

    # 通带接近 0dB
    assert abs(mag_db[0]) < 0.01
    # 截止频率 -3dB
    assert abs(mag_db[2] + 3.0) < 0.1
    # 阻带衰减 > 30dB
    assert mag_db[4] < -35.0


def test_filter_target_highpass() -> None:
    target = FilterTarget(kind="highpass", order=1, cutoff_hz=1000.0)
    freqs = np.array([10.0, 1000.0, 100000.0])
    mag_db = target.evaluate_magnitude_db(freqs)

    # 低频衰减
    assert mag_db[0] < -35.0
    # 截止附近 -3dB
    assert abs(mag_db[1] + 3.0) < 0.1
    # 高频通带
    assert abs(mag_db[2]) < 0.01


def test_filter_target_bandpass() -> None:
    target = FilterTarget(
        kind="bandpass",
        order=2,
        cutoff_hz=10000.0,
        q=5.0,
    )
    freqs = np.array([100.0, 10000.0, 1000000.0])
    mag_db = target.evaluate_magnitude_db(freqs)

    # 中心频率 0dB
    assert abs(mag_db[1]) < 0.1
    # 远离中心衰减
    assert mag_db[0] < -30.0
    assert mag_db[2] < -30.0


def test_filter_target_bandstop() -> None:
    target = FilterTarget(
        kind="bandstop",
        order=2,
        cutoff_hz=1000.0,
        q=5.0,
    )
    freqs = np.array([10.0, 1000.0, 100000.0])
    mag_db = target.evaluate_magnitude_db(freqs)

    # 通带
    assert abs(mag_db[0]) < 0.5
    assert abs(mag_db[2]) < 0.5
    # 阻带
    assert mag_db[1] < -10.0


def test_filter_target_allpass() -> None:
    target = FilterTarget(kind="allpass", order=1, cutoff_hz=1000.0)
    freqs = np.logspace(1, 5, 100)
    mag_db = target.evaluate_magnitude_db(freqs)

    # 全通：幅度始终为 0dB
    assert np.max(np.abs(mag_db)) < 1e-9


def test_butterworth_poles() -> None:
    for order in [1, 2, 3, 4]:
        poles = _butterworth_poles(order)
        assert len(poles) == order
        # 所有极点应在左半平面
        assert all(p.real < 0 for p in poles)


def test_butterworth_order2_vs_order1() -> None:
    """二阶 Butterworth 应比一阶在阻带更陡。"""
    target1 = FilterTarget(kind="lowpass", order=1, cutoff_hz=1000.0)
    target2 = FilterTarget(kind="lowpass", order=2, cutoff_hz=1000.0)

    freqs = np.array([10000.0])  # 10× 截止频率

    # 一阶: -20dB/decade → 约 -20dB
    # 二阶: -40dB/decade → 约 -40dB
    assert target1.evaluate_magnitude_db(freqs)[0] < -15.0
    assert target2.evaluate_magnitude_db(freqs)[0] < -35.0
    assert target2.evaluate_magnitude_db(freqs)[0] < target1.evaluate_magnitude_db(freqs)[0]


def test_filter_target_gain_db() -> None:
    target = FilterTarget(kind="lowpass", order=1, cutoff_hz=1000.0, gain_db=6.0)
    freqs = np.array([10.0])
    mag_db = target.evaluate_magnitude_db(freqs)

    assert abs(mag_db[0] - 6.0) < 0.01


def test_filter_target_order2_q() -> None:
    """Q=0.5 时下垂更陡，2阶 Butterworth (Q=0.707) 平坦。

    Butterworth 的 Q 由逼近理论固定为 1/√2。
    q 参数仅在非 Butterworth 响应的简单实现(非级联)时使用，
    或当目标指定其他响应类型(Chebyshev等)时生效。
    """
    freqs = np.logspace(2, 4, 200)

    # 二阶 Butterworth，Q 固定 0.707
    bt = FilterTarget(kind="lowpass", order=2, cutoff_hz=1000.0)
    mag_bt = bt.evaluate_magnitude_db(freqs)

    # 通带基本平坦（Butterworth 无波纹）
    assert abs(mag_bt[0]) < 0.1
    # 在截止频率附近约 -3dB
    fc_index = np.argmin(np.abs(freqs - 1000.0))
    assert abs(mag_bt[fc_index] + 3.0) < 1.0
    # 高频衰减 > 30dB（二阶 -40dB/decade）
    assert mag_bt[-1] < -30.0


def test_amplifier_target() -> None:
    target = AmplifierTarget(
        gain_db=20.0,
        bandwidth_hz=(10.0, 100_000.0),
    )
    freqs = np.array([1000.0])  # 在带内
    mag_db = target.evaluate_magnitude_db(freqs)

    assert abs(mag_db[0] - 20.0) < 0.5


def test_impedance_target_constant() -> None:
    target = ImpedanceTarget(ohms=50.0)
    freqs = np.array([10.0, 100000.0])
    values = target.evaluate(freqs)

    assert abs(values[0] - 50.0) < 1e-9
    assert abs(values[1] - 50.0) < 1e-9


def test_sampled_target() -> None:
    target = SampledTarget(
        frequency_hz=(10.0, 100.0, 1000.0, 10000.0),
        magnitude_db=(0.0, -0.1, -3.0, -20.0),
    )
    freqs = np.array([10.0, 1000.0])
    mag_db = target.evaluate_magnitude_db(freqs)

    assert abs(mag_db[0]) < 0.01
    assert abs(mag_db[1] + 3.0) < 0.1


def test_mask_target() -> None:
    target = MaskTarget(
        frequency_hz=(10.0, 1000.0, 100000.0),
        lower_bound_db=(-1.0, -4.0, -50.0),
        upper_bound_db=(1.0, 0.0, -20.0),
    )
    freqs = np.array([10.0, 1000.0, 100000.0])

    # 在通带内
    assert np.all(target.satisfied(freqs, np.array([0.0, -2.0, -30.0])))
    # 超出上限
    assert not target.satisfied(freqs, np.array([2.0, -2.0, -30.0]))[0]
    # 低于下限
    assert not target.satisfied(freqs, np.array([0.0, -5.0, -30.0]))[1]


def test_tolerance() -> None:
    tol = Tolerance(relative=0.01, absolute=0.5)
    target = np.array([10.0, 1.0])
    actual = np.array([10.05, 0.6])
    result = tol.is_satisfied(actual, target)

    assert np.all(result)  # 0.05 < 0.1 (rel) or < 0.5 (abs)


def test_circuit_spec_creation() -> None:
    spec = CircuitSpec.lowpass_filter(cutoff_hz=1000.0)
    assert spec.name == "lowpass_filter"
    assert len(spec.analyses) == 1
    assert len(spec.targets) == 1
    assert spec.analyses[0].kind == "voltage_transfer"
    assert isinstance(spec.targets[0], FilterTarget)


def test_circuit_spec_constant_impedance() -> None:
    spec = CircuitSpec.constant_impedance(ohms=50.0)
    assert spec.analyses[0].kind == "impedance"
    assert spec.constraints.element_types == ("R",)


def test_circuit_spec_validate() -> None:
    spec = CircuitSpec.lowpass_filter(cutoff_hz=1000.0)
    assert spec.is_valid()
    assert spec.validate() == []


def test_roundtrip_json() -> None:
    spec = CircuitSpec.lowpass_filter(cutoff_hz=5000.0, order=2, gain_db=3.0)
    d = spec.as_dict()
    restored = CircuitSpec.from_dict(d)

    assert restored.name == spec.name
    assert restored.constraints.max_component_count == spec.constraints.max_component_count
    assert isinstance(restored.targets[0], FilterTarget)
    assert restored.targets[0].cutoff_hz == 5000.0  # type: ignore[attr-defined]


def test_ac_transfer_frequency_range() -> None:
    analysis = AcTransfer(
        source_port="input",
        output_port="output",
        frequency_hz=(10.0, 1_000_000.0, 100),
    )
    assert analysis.frequency_hz == (10.0, 1_000_000.0, 100)


def test_constraints_with_power_and_noise() -> None:
    c = Constraints(
        element_types=("R", "C", "opamp"),
        max_component_count=6,
        max_power_mw=50.0,
        noise_floor_dbm=-90.0,
        preferred_values="E24",
    )
    d = c.as_dict()
    assert d["max_power_mw"] == 50.0
    assert d["noise_floor_dbm"] == -90.0
    assert d["preferred_values"] == "E24"


def test_operating_point() -> None:
    op = OperatingPoint(
        supply_voltage_v=5.0,
        supply_bipolar=False,
        temperature_c=85.0,
    )
    d = op.as_dict()
    assert d["supply_voltage_v"] == 5.0
    assert d["temperature_c"] == 85.0


def test_sensor_interface_spec() -> None:
    """传感器接口 spec 应该包含两个分析和一个滤波器目标。"""
    spec = CircuitSpec(
        name="sensor_interface",
        ports=PortBundle.standard_two_port(),
        analyses=(
            AcTransfer(frequency_hz=(1.0, 10_000_000.0, 200)),
            Impedance(port="output"),
        ),
        targets=(
            FilterTarget(kind="lowpass", cutoff_hz=100_000.0, order=2, gain_db=6.0),
            ImpedanceTarget(ohms=50.0, weight=0.3),
        ),
        constraints=Constraints(element_types=("R", "C", "opamp"), max_component_count=10),
        operating_point=OperatingPoint(supply_voltage_v=5.0),
    )
    assert spec.is_valid()
    assert len(spec.analyses) == 2
    assert len(spec.targets) == 2


def test_dc_transfer_spec() -> None:
    """直流升压变换 spec: 5V → 10V。"""
    spec = CircuitSpec(
        name="dc_boost",
        ports=PortBundle.standard_two_port(input_name="input", output_name="output"),
        analyses=(DcTransfer(source_port="input", output_port="output"),),
        targets=(DcTarget(input_voltage_v=5.0, output_voltage_v=10.0, output_current_a=0.1),),
        constraints=Constraints(element_types=("R", "C", "L", "opamp"), max_component_count=15),
    )
    assert spec.is_valid()
    d = spec.as_dict()
    restored = CircuitSpec.from_dict(d)
    assert isinstance(restored.targets[0], DcTarget)
    assert restored.targets[0].input_voltage_v == 5.0  # type: ignore
    assert restored.targets[0].output_voltage_v == 10.0  # type: ignore
