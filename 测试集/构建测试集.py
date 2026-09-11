"""Build the versioned PBDL classic-requirement regression suite.

The suite intentionally stores *requirements*, not vendor reference designs or
SPICE netlists.  This keeps the corpus redistributable while giving the
planner a broad, reviewable set of classical circuit intents.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PBDL_ROOT = ROOT / "pbdl"
CLASSIC_ROOT = PBDL_ROOT / "classic"
BASELINE_ROOT = PBDL_ROOT / "baseline"


SOURCES = {
    "analog_filters": {
        "title": "Analog Devices active anti-aliasing filter design note",
        "url": "https://www.analog.com/en/resources/design-notes/2021/11/09/04/34/high-frequency-active-antialiasing-filters.html",
        "use": "Filter and anti-aliasing requirement families.",
    },
    "receiver_filter": {
        "title": "Analog Devices AN-2545 high-IF receiver front end",
        "url": "https://www.analog.com/en/resources/app-notes/an-2545.html",
        "use": "Band-pass and ADC-interface requirement families.",
    },
    "power_topologies": {
        "title": "TI power-topology selection brief SLVAFJ4",
        "url": "https://www.ti.com/document-viewer/lit/html/slvafj4",
        "use": "Buck, boost, SEPIC, flyback and isolation requirement families.",
    },
    "flyback_selector": {
        "title": "TI Fly-Buck and Flyback topology selector",
        "url": "https://www.ti.com/tool/FLYBUCK-FLYBACK-DESIGN-CALC",
        "use": "Isolated-output and multi-output power-supply requirement families.",
    },
    "classic_textbook": {
        "title": "All About Circuits electronics textbook",
        "url": "https://www.allaboutcircuits.com/textbook/",
        "use": "Classical amplifier, comparator, oscillator, rectifier and driver families.",
    },
    "ieee_benchmarks": {
        "title": "IEEE P2427 analogue benchmark circuits",
        "url": "https://sagroups.ieee.org/2427/analogue-benchmark-circuits/",
        "use": "External benchmark inventory; no third-party netlist is copied into this suite.",
    },
    "analoggym": {
        "title": "AnalogGym: An Open and Practical Testing Suite for Analog Circuit Synthesis",
        "url": "https://arxiv.org/abs/2409.08534",
        "use": "External analog-synthesis benchmark inventory; PDK-dependent data remains external.",
    },
    "hf_circuit_benchmarks": {
        "title": "Circuit Design Benchmarks dataset card",
        "url": "https://huggingface.co/datasets/englund/circuit-design-benchmarks/viewer",
        "use": "Requirement-level benchmark inventory; values here are independently expressed in PBDL.",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _voltage_ports(*, isolated: bool = False, include_supply: bool = False) -> list[dict[str, Any]]:
    output_reference = "secondary_0" if isolated else "0"
    ports = [
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
                {"name": output_reference, "quantity": "ground"},
            ],
        },
    ]
    if include_supply:
        ports.append(
            {
                "name": "supply",
                "role": "power",
                "terminals": [
                    {"name": "vcc", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            }
        )
    return ports


def _base_constraints(elements: list[str], maximum: int = 8) -> dict[str, Any]:
    return {
        "element_types": elements,
        "parameter_ranges": {"R": [10.0, 10_000_000.0], "C": [1e-12, 1e-3], "L": [1e-9, 1.0]},
        "max_component_count": maximum,
    }


def _filter_case(
    name: str,
    title: str,
    *,
    kind: str,
    cutoff_hz: float,
    order: int = 1,
    gain_db: float = 0.0,
    q: float | None = None,
    bandwidth_hz: float | None = None,
    response: str = "butterworth",
    elements: list[str] | None = None,
    source_groups: list[str] | None = None,
    smoke: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    elements = elements or ["R", "C", "opamp"]
    target: dict[str, Any] = {
        "target_kind": "filter",
        "filter_kind": kind,
        "cutoff_hz": cutoff_hz,
        "order": order,
        "gain_db": gain_db,
        "response": response,
        "tolerance": {"absolute": 1.0},
    }
    if q is not None:
        target["q"] = q
    if bandwidth_hz is not None:
        target["bandwidth_hz"] = bandwidth_hz
    data = {
        "name": name,
        "description": title,
        "ports": _voltage_ports(),
        "analyses": [{"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [max(0.1, cutoff_hz / 100), cutoff_hz * 100, 64]}],
        "targets": [target],
        "constraints": _base_constraints(elements, 8 if order > 1 else 4),
        "operating_point": {"temperature_c": 25.0},
        "optimization": {"points": 64, "max_iterations": 10, "top_k": 1, "seed": 17},
    }
    meta = _case_meta(name, title, "filter", "frequency_domain", source_groups or ["analog_filters"], "supported", smoke)
    return data, meta


def _amplifier_case(
    name: str,
    title: str,
    *,
    gain_db: float,
    bandwidth_hz: tuple[float, float],
    source_groups: list[str] | None = None,
    current_input: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if current_input:
        ports = [
            {"name": "current_input", "role": "input", "terminals": [{"name": "iin", "quantity": "current"}, {"name": "0", "quantity": "ground"}]},
            {"name": "voltage_output", "role": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        ]
        analysis = {"kind": "voltage_transfer", "source_port": "current_input", "output_port": "voltage_output", "frequency_hz": [max(0.1, bandwidth_hz[0] / 10), bandwidth_hz[1] * 10, 64]}
    else:
        ports = _voltage_ports(include_supply=True)
        analysis = {"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [max(0.1, bandwidth_hz[0] / 10), bandwidth_hz[1] * 10, 64]}
    data = {
        "name": name,
        "description": title,
        "ports": ports,
        "analyses": [analysis],
        "targets": [{"target_kind": "amplifier", "gain_db": gain_db, "bandwidth_hz": list(bandwidth_hz), "tolerance": {"absolute": 1.0}}],
        "constraints": _base_constraints(["R", "C", "opamp"], 10),
        "operating_point": {"supply_voltage_v": 5.0, "temperature_c": 25.0},
        "optimization": {"points": 64, "max_iterations": 10, "top_k": 1, "seed": 19},
    }
    category = "transimpedance" if current_input else "amplifier"
    planning = "supported" if current_input else "unsupported"
    meta = _case_meta(
        name,
        title,
        category,
        "frequency_domain",
        source_groups or ["classic_textbook"],
        planning,
        False,
        expected_error_code=None if current_input else "capability_gap",
    )
    return data, meta


def _impedance_case(name: str, title: str, *, ohms: float, port: str = "input") -> tuple[dict[str, Any], dict[str, Any]]:
    data = {
        "name": name,
        "description": title,
        "ports": _voltage_ports(),
        "analyses": [{"kind": "impedance", "port": port, "frequency_hz": [10.0, 10_000_000.0, 64]}],
        "targets": [{"target_kind": "impedance", "ohms": ohms, "tolerance": {"relative": 0.05}}],
        "constraints": _base_constraints(["R"], 2),
        "operating_point": {"temperature_c": 25.0},
        "optimization": {"points": 64, "max_iterations": 10, "top_k": 1, "seed": 23},
    }
    category = "output_impedance" if port == "output" else "input_impedance"
    return data, _case_meta(name, title, category, "frequency_domain", ["receiver_filter"], "supported", False)


def _multi_analysis_case(
    name: str,
    title: str,
    *,
    analyses: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    ports: list[dict[str, Any]] | None = None,
    elements: list[str] | None = None,
    source_groups: list[str] | None = None,
    smoke: bool = False,
    planning: str = "supported",
    expected_error_code: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = {
        "name": name,
        "description": title,
        "ports": ports or _voltage_ports(),
        "analyses": analyses,
        "targets": targets,
        "constraints": _base_constraints(elements or ["R", "C", "opamp"], 12),
        "operating_point": {"supply_voltage_v": 5.0, "temperature_c": 25.0},
        "optimization": {"points": 64, "max_iterations": 10, "top_k": 1, "seed": 31},
    }
    return data, _case_meta(
        name,
        title,
        "multi_stage",
        "frequency_domain",
        source_groups or ["classic_textbook"],
        planning,
        smoke,
        expected_error_code=expected_error_code,
    )


def _multi_port_case(name: str, title: str, *, port_count: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    ports = _voltage_ports(include_supply=True)
    for index in range(4, port_count + 1):
        ports.append(
            {
                "name": f"aux_{index}",
                "role": "bidirectional",
                "terminals": [
                    {"name": f"aux_{index}_p", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            }
        )
    data = {
        "name": name,
        "description": title,
        "ports": ports,
        "analyses": [{
            "kind": "voltage_transfer",
            "source_port": "input",
            "output_port": "output",
            "frequency_hz": [10.0, 1_000_000.0, 64],
        }],
        "targets": [{
            "target_kind": "filter",
            "filter_kind": "lowpass",
            "cutoff_hz": 10_000.0,
            "order": 1,
            "tolerance": {"absolute": 1.0},
        }],
        "constraints": _base_constraints(["R", "C", "opamp"], 10),
        "operating_point": {"supply_voltage_v": 5.0, "temperature_c": 25.0},
    }
    meta = _case_meta(
        name,
        title,
        "multi_port",
        "frequency_domain",
        ["classic_textbook"],
        "unsupported",
        False,
        expected_error_code="capability_gap",
    )
    meta["validation_scope"] = "planner_boundary"
    meta["known_boundary"] = "execution_adapter_is_two_port"
    return data, meta


def _power_case(
    name: str,
    title: str,
    *,
    vin: float,
    vout: float,
    current: float,
    isolated: bool = False,
    smoke: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    elements = ["R", "C", "L", "ideal_switch", "ideal_diode"]
    if isolated:
        elements.append("ideal_transformer")
    data: dict[str, Any] = {
        "name": name,
        "description": title,
        "ports": _voltage_ports(isolated=isolated),
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [{"target_kind": "dc", "input_voltage_v": vin, "output_voltage_v": vout, "output_current_a": current, "efficiency": 0.8, "ripple_mv": 50.0, "tolerance": {"relative": 0.02}}],
        "constraints": {
            **_base_constraints(elements, 12),
            "required_elements": (
                ["C", "ideal_switch", "ideal_diode", "ideal_transformer"]
                if isolated
                else ["C", "L", "ideal_switch", "ideal_diode"]
            ),
            "max_power_mw": max(1000.0, vout * current * 1500.0),
        },
        "operating_point": {"supply_voltage_v": vin, "temperature_c": 25.0},
        "optimization": {"max_iterations": 10, "top_k": 1, "seed": 29},
    }
    if isolated:
        data["relations"] = [{"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}]
    category = "isolated_power" if isolated else "dc_power"
    return data, _case_meta(name, title, category, "dc", ["power_topologies", "flyback_selector"] if isolated else ["power_topologies"], "supported", smoke)


def _future_case(name: str, title: str, *, kind: str, domain: str, category: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = {
        "name": name,
        "description": title,
        "ports": _voltage_ports(include_supply=True),
        "analyses": [{"kind": kind, "domain": domain, "source_port": "input", "output_port": "output"}],
        "targets": [{"target_kind": "dc", "input_voltage_v": 5.0, "output_voltage_v": 2.5, "tolerance": {"relative": 0.01}}],
        "constraints": _base_constraints(["R", "C", "L", "opamp", "ideal_switch"], 16),
        "operating_point": {"supply_voltage_v": 5.0, "temperature_c": 25.0},
    }
    return data, _case_meta(
        name,
        title,
        category,
        domain,
        ["classic_textbook", "ieee_benchmarks"],
        "unsupported",
        False,
        expected_error_code="capability_gap",
    )


def _case_meta(
    name: str,
    title: str,
    category: str,
    domain: str,
    sources: list[str],
    planning: str,
    smoke: bool,
    *,
    expected_error_code: str | None = None,
) -> dict[str, Any]:
    return {
        "id": name,
        "title": title,
        "category": category,
        "domain": domain,
        "source_groups": sources,
        "expected_planning": planning,
        "execution_tier": "smoke" if smoke else "schema",
        "validation_scope": "execution" if smoke else "planner",
        "expected_error_code": expected_error_code,
    }


def _classic_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    cases.extend([
        _filter_case("ac_rc_lowpass_100hz", "传感器慢变信号：100 Hz 一阶低通", kind="lowpass", cutoff_hz=100.0, elements=["R", "C"]),
        _filter_case("ac_rc_lowpass_1khz", "通用采样前端：1 kHz 一阶低通", kind="lowpass", cutoff_hz=1000.0, elements=["R", "C"], smoke=True),
        _filter_case("ac_active_lowpass_10khz", "主动抗混叠：10 kHz 二阶 Butterworth 低通", kind="lowpass", cutoff_hz=10_000.0, order=2, q=0.70710678),
        _filter_case("ac_active_lowpass_20khz", "音频 ADC 前端：20 kHz 二阶低通", kind="lowpass", cutoff_hz=20_000.0, order=2, q=0.70710678),
        _filter_case("ac_bessel_lowpass_5khz", "脉冲保真前端：5 kHz 二阶 Bessel 低通", kind="lowpass", cutoff_hz=5_000.0, order=2, response="bessel"),
        _filter_case("ac_chebyshev_lowpass_10khz", "选择性滤波：10 kHz 二阶 Chebyshev 低通", kind="lowpass", cutoff_hz=10_000.0, order=2, response="chebyshev"),
        _filter_case("ac_audio_highpass_20hz", "音频耦合：20 Hz 一阶高通", kind="highpass", cutoff_hz=20.0, elements=["R", "C"]),
        _filter_case("ac_highpass_1khz", "交流耦合传感器：1 kHz 一阶高通", kind="highpass", cutoff_hz=1_000.0),
        _filter_case("ac_active_highpass_5khz", "信号调理：5 kHz 二阶高通", kind="highpass", cutoff_hz=5_000.0, order=2, q=0.70710678),
        _filter_case("ac_bandpass_1khz_q5", "窄带音频检测：中心 1 kHz、Q=5 带通", kind="bandpass", cutoff_hz=1_000.0, order=2, q=5.0, bandwidth_hz=200.0, elements=["R", "C", "L"]),
        _filter_case("ac_bandpass_10khz_q4", "中频选择：中心 10 kHz、Q=4 带通", kind="bandpass", cutoff_hz=10_000.0, order=2, q=4.0, bandwidth_hz=2_500.0, elements=["R", "C", "L"]),
        _filter_case("ac_if_bandpass_455khz", "AM 中频：中心 455 kHz、10 kHz 带宽带通", kind="bandpass", cutoff_hz=455_000.0, order=2, q=45.5, bandwidth_hz=10_000.0, elements=["R", "C", "L"], source_groups=["receiver_filter"]),
        _filter_case("ac_notch_50hz", "工频抑制：50 Hz 二阶带阻", kind="bandstop", cutoff_hz=50.0, order=2, q=10.0, bandwidth_hz=5.0),
        _filter_case("ac_allpass_phase_1khz", "相位均衡：1 kHz 一阶全通", kind="allpass", cutoff_hz=1_000.0),
    ])
    cases.extend([
        _filter_case("ac_rc_lowpass_10hz", "慢速传感器抗噪：10 Hz RC 低通", kind="lowpass", cutoff_hz=10.0, elements=["R", "C"]),
        _filter_case("ac_rc_lowpass_100khz", "采样保持前端：100 kHz RC 低通", kind="lowpass", cutoff_hz=100_000.0, elements=["R", "C"]),
        _filter_case("ac_rc_lowpass_1mhz", "高速接口抗混叠：1 MHz RC 低通", kind="lowpass", cutoff_hz=1_000_000.0, elements=["R", "C"]),
        _filter_case("ac_rc_highpass_100hz", "低频漂移隔离：100 Hz RC 高通", kind="highpass", cutoff_hz=100.0, elements=["R", "C"]),
        _filter_case("ac_rc_highpass_100khz", "高速交流耦合：100 kHz RC 高通", kind="highpass", cutoff_hz=100_000.0, elements=["R", "C"]),
        _filter_case("ac_active_lowpass_1khz", "低频测量抗噪：1 kHz 二阶主动低通", kind="lowpass", cutoff_hz=1_000.0, order=2, q=0.70710678),
        _filter_case("ac_active_highpass_20khz", "音频分频前端：20 kHz 二阶主动高通", kind="highpass", cutoff_hz=20_000.0, order=2, q=0.70710678),
        _filter_case("ac_bandpass_100hz_q2", "机械振动检测：100 Hz、Q=2 带通", kind="bandpass", cutoff_hz=100.0, order=2, q=2.0, bandwidth_hz=50.0, elements=["R", "C", "L"]),
        _filter_case("ac_bandpass_100khz_q10", "超声前端选择：100 kHz、Q=10 带通", kind="bandpass", cutoff_hz=100_000.0, order=2, q=10.0, bandwidth_hz=10_000.0, elements=["R", "C", "L"]),
        _filter_case("ac_notch_60hz", "电网干扰抑制：60 Hz 二阶带阻", kind="bandstop", cutoff_hz=60.0, order=2, q=12.0, bandwidth_hz=5.0),
        _filter_case("ac_notch_1khz", "开关纹波抑制：1 kHz 二阶带阻", kind="bandstop", cutoff_hz=1_000.0, order=2, q=20.0, bandwidth_hz=50.0),
        _filter_case("ac_allpass_phase_10khz", "通信相位补偿：10 kHz 一阶全通", kind="allpass", cutoff_hz=10_000.0),
        _filter_case("ac_bessel_lowpass_100khz", "高速采样保波形：100 kHz 二阶 Bessel 低通", kind="lowpass", cutoff_hz=100_000.0, order=2, response="bessel"),
        _filter_case("ac_chebyshev_lowpass_100khz", "高选择性采样滤波：100 kHz 二阶 Chebyshev 低通", kind="lowpass", cutoff_hz=100_000.0, order=2, response="chebyshev"),
        _filter_case("ac_filter_gain_12db", "有源滤波增益：12 dB、10 kHz 低通", kind="lowpass", cutoff_hz=10_000.0, order=2, gain_db=12.0, q=0.70710678),
    ])
    cases.extend([
        _amplifier_case("ac_noninverting_20db", "通用非反相放大器：20 dB，10 Hz 至 100 kHz", gain_db=20.0, bandwidth_hz=(10.0, 100_000.0)),
        _amplifier_case("ac_sensor_gain_40db", "低电平传感器放大：40 dB，1 Hz 至 10 kHz", gain_db=40.0, bandwidth_hz=(1.0, 10_000.0)),
        _amplifier_case("ac_audio_preamp_14db", "音频前置放大：14 dB，20 Hz 至 20 kHz", gain_db=14.0, bandwidth_hz=(20.0, 20_000.0)),
        _amplifier_case("ac_inverting_6db", "反相缓冲级：6 dB，100 Hz 至 1 MHz", gain_db=6.0, bandwidth_hz=(100.0, 1_000_000.0)),
        _amplifier_case("ac_tia_100kohm", "光电二极管跨阻放大：100 kohm，10 Hz 至 100 kHz", gain_db=100.0, bandwidth_hz=(10.0, 100_000.0), current_input=True),
        _amplifier_case("ac_tia_1mohm", "弱光电流读出：1 Mohm，1 Hz 至 10 kHz", gain_db=120.0, bandwidth_hz=(1.0, 10_000.0), current_input=True),
        _impedance_case("ac_input_match_50ohm", "射频源接口：50 ohm 输入阻抗", ohms=50.0),
        _impedance_case("ac_input_match_75ohm", "视频同轴接口：75 ohm 输入阻抗", ohms=75.0),
        _impedance_case("ac_audio_input_600ohm", "传统音频接口：600 ohm 输入阻抗", ohms=600.0),
        _impedance_case("ac_output_driver_50ohm", "测试仪输出：50 ohm 输出阻抗", ohms=50.0, port="output"),
    ])
    cases.extend([
        _amplifier_case("ac_noninverting_6db", "缓冲放大：6 dB、10 Hz 至 100 kHz", gain_db=6.0, bandwidth_hz=(10.0, 100_000.0)),
        _amplifier_case("ac_noninverting_40db", "高增益传感器：40 dB、1 Hz 至 10 kHz", gain_db=40.0, bandwidth_hz=(1.0, 10_000.0)),
        _amplifier_case("ac_inverting_20db", "反相信号链：20 dB、100 Hz 至 1 MHz", gain_db=20.0, bandwidth_hz=(100.0, 1_000_000.0)),
        _amplifier_case("ac_inverting_40db", "反相高增益：40 dB、10 Hz 至 100 kHz", gain_db=40.0, bandwidth_hz=(10.0, 100_000.0)),
        _amplifier_case("ac_audio_preamp_20db", "音频前置放大：20 dB、20 Hz 至 20 kHz", gain_db=20.0, bandwidth_hz=(20.0, 20_000.0)),
        _amplifier_case("ac_wideband_1mhz", "宽带电压放大：10 dB、1 kHz 至 1 MHz", gain_db=10.0, bandwidth_hz=(1_000.0, 1_000_000.0)),
        _amplifier_case("ac_output_buffer", "电压跟随器：0 dB、1 Hz 至 100 kHz", gain_db=0.0, bandwidth_hz=(1.0, 100_000.0)),
        _amplifier_case("ac_tia_10kohm", "光电流跨阻：10 kohm、10 Hz 至 100 kHz", gain_db=80.0, bandwidth_hz=(10.0, 100_000.0), current_input=True),
        _amplifier_case("ac_tia_10mohm", "弱电流检测：10 Mohm、1 Hz 至 1 kHz", gain_db=140.0, bandwidth_hz=(1.0, 1_000.0), current_input=True),
        _impedance_case("ac_input_match_10ohm", "低阻传感器输入：10 ohm 输入阻抗", ohms=10.0),
        _impedance_case("ac_input_match_1kohm", "控制接口输入：1 kohm 输入阻抗", ohms=1_000.0),
        _impedance_case("ac_input_match_1mohm", "高阻测量输入：1 Mohm 输入阻抗", ohms=1_000_000.0),
        _impedance_case("ac_output_driver_10ohm", "功率缓冲输出：10 ohm 输出阻抗", ohms=10.0, port="output"),
        _impedance_case("ac_output_driver_75ohm", "视频驱动输出：75 ohm 输出阻抗", ohms=75.0, port="output"),
        _impedance_case("ac_output_driver_600ohm", "音频驱动输出：600 ohm 输出阻抗", ohms=600.0, port="output"),
    ])
    cases.extend([
        _power_case("dc_boost_3v3_to_5v", "电池升压：3.3 V 到 5 V，500 mA", vin=3.3, vout=5.0, current=0.5),
        _power_case("dc_boost_5v_to_12v", "接口电源升压：5 V 到 12 V，300 mA", vin=5.0, vout=12.0, current=0.3, smoke=True),
        _power_case("dc_boost_12v_to_24v", "工业控制升压：12 V 到 24 V，500 mA", vin=12.0, vout=24.0, current=0.5),
        _power_case("dc_buck_24v_to_5v", "工业降压：24 V 到 5 V，1 A", vin=24.0, vout=5.0, current=1.0),
        _power_case("dc_buck_12v_to_3v3", "数字核心降压：12 V 到 3.3 V，2 A", vin=12.0, vout=3.3, current=2.0),
        _power_case("dc_buck_5v_to_1v8", "处理器内核降压：5 V 到 1.8 V，1 A", vin=5.0, vout=1.8, current=1.0),
        _power_case("dc_sepic_5v_to_9v", "宽输入电源：5 V 到 9 V，500 mA SEPIC 候选", vin=5.0, vout=9.0, current=0.5),
        _power_case("dc_sepic_9v_to_12v", "车载保持输出：9 V 到 12 V，300 mA SEPIC 候选", vin=9.0, vout=12.0, current=0.3),
        _power_case("dc_flyback_5v_to_12v", "隔离偏置电源：5 V 到 12 V，1 A", vin=5.0, vout=12.0, current=1.0, isolated=True, smoke=True),
        _power_case("dc_flyback_24v_to_5v", "隔离控制电源：24 V 到 5 V，500 mA", vin=24.0, vout=5.0, current=0.5, isolated=True),
    ])
    cases.extend([
        _power_case("dc_boost_1v8_to_5v", "低压电池升压：1.8 V 到 5 V，200 mA", vin=1.8, vout=5.0, current=0.2),
        _power_case("dc_boost_9v_to_24v", "仪器辅助升压：9 V 到 24 V，200 mA", vin=9.0, vout=24.0, current=0.2),
        _power_case("dc_boost_24v_to_48v", "工业母线升压：24 V 到 48 V，500 mA", vin=24.0, vout=48.0, current=0.5),
        _power_case("dc_buck_48v_to_12v", "工业母线降压：48 V 到 12 V，2 A", vin=48.0, vout=12.0, current=2.0),
        _power_case("dc_buck_12v_to_5v", "嵌入式主电源：12 V 到 5 V，3 A", vin=12.0, vout=5.0, current=3.0),
        _power_case("dc_buck_3v3_to_1v2", "FPGA 核心供电：3.3 V 到 1.2 V，2 A", vin=3.3, vout=1.2, current=2.0),
        _power_case("dc_sepic_3v3_to_12v", "电池宽输入升压：3.3 V 到 12 V，100 mA", vin=3.3, vout=12.0, current=0.1),
        _power_case("dc_sepic_12v_to_5v", "输入跨越输出的 SEPIC：12 V 到 5 V，500 mA", vin=12.0, vout=5.0, current=0.5),
        _power_case("dc_flyback_12v_to_5v", "隔离传感器电源：12 V 到 5 V，1 A", vin=12.0, vout=5.0, current=1.0, isolated=True),
        _power_case("dc_flyback_5v_to_15v", "隔离模拟电源：5 V 到 15 V，200 mA", vin=5.0, vout=15.0, current=0.2, isolated=True),
        _power_case("dc_flyback_24v_to_15v", "隔离栅极驱动电源：24 V 到 15 V，300 mA", vin=24.0, vout=15.0, current=0.3, isolated=True),
    ])
    cases.extend([
        _multi_analysis_case(
            "ac_sensor_filter_match_10khz",
            "传感器接口：10 kHz 低通加 1 kohm 输出阻抗",
            analyses=[
                {"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [1.0, 1_000_000.0, 64]},
                {"kind": "output_impedance", "port": "output", "frequency_hz": [1.0, 1_000_000.0, 64]},
            ],
            targets=[
                {"target_kind": "filter", "filter_kind": "lowpass", "cutoff_hz": 10_000.0, "order": 1, "tolerance": {"absolute": 1.0}},
                {"target_kind": "impedance", "ohms": 1_000.0, "tolerance": {"relative": 0.1}},
            ],
            smoke=True,
        ),
        _multi_analysis_case(
            "ac_amplifier_input_output_impedance",
            "放大器接口：20 dB 电压增益加输入输出阻抗",
            analyses=[
                {"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [10.0, 1_000_000.0, 64]},
                {"kind": "impedance", "port": "input", "frequency_hz": [10.0, 1_000_000.0, 64]},
                {"kind": "output_impedance", "port": "output", "frequency_hz": [10.0, 1_000_000.0, 64]},
            ],
            targets=[
                {"target_kind": "amplifier", "gain_db": 20.0, "bandwidth_hz": [10.0, 100_000.0], "tolerance": {"absolute": 1.0}},
                {"target_kind": "impedance", "ohms": 10_000.0, "tolerance": {"relative": 0.1}},
                {"target_kind": "impedance", "ohms": 50.0, "tolerance": {"relative": 0.1}},
            ],
        ),
        _multi_analysis_case(
            "ac_tia_output_match",
            "光电接收前端：跨阻放大加 50 ohm 输出匹配",
            analyses=[
                {"kind": "transimpedance", "source_port": "current_input", "output_port": "voltage_output", "frequency_hz": [10.0, 1_000_000.0, 64]},
                {"kind": "output_impedance", "port": "voltage_output", "frequency_hz": [10.0, 1_000_000.0, 64]},
            ],
            targets=[
                {"target_kind": "amplifier", "gain_db": 80.0, "bandwidth_hz": [10.0, 100_000.0], "tolerance": {"absolute": 1.0}},
                {"target_kind": "impedance", "ohms": 50.0, "tolerance": {"relative": 0.1}},
            ],
            ports=[
                {"name": "current_input", "role": "input", "terminals": [{"name": "iin", "quantity": "current"}, {"name": "0", "quantity": "ground"}]},
                {"name": "voltage_output", "role": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            ],
            elements=["R", "C", "opamp"],
            planning="mixed",
            expected_error_code="capability_gap",
        ),
        _multi_port_case("ac_three_port_supply_adapter", "带独立供电端口的三端口传感器接口", port_count=3),
        _multi_port_case("ac_four_port_auxiliary_network", "带两个辅助端口的四端口模拟网络", port_count=4),
    ])
    cases.extend([
        _future_case("future_comparator_2v5", "比较器：2.5 V 阈值、传播延迟和过驱动要求", kind="transient", domain="time_domain", category="comparator"),
        _future_case("future_schmitt_trigger", "施密特触发器：上升/下降阈值和迟滞", kind="transient", domain="time_domain", category="schmitt_trigger"),
        _future_case("future_wien_bridge_oscillator", "文氏桥振荡器：1 kHz 稳态正弦、幅值稳定", kind="periodic_steady_state", domain="time_domain", category="oscillator"),
        _future_case("future_rc_phase_shift_oscillator", "RC 移相振荡器：10 kHz 启振和周期稳态", kind="transient", domain="time_domain", category="oscillator"),
        _future_case("future_555_astable", "555 无稳态振荡器：1 kHz、60% 占空比", kind="transient", domain="time_domain", category="timer"),
        _future_case("future_h_bridge_motor", "H 桥电机驱动：24 V、2 A、PWM 和死区", kind="transient", domain="time_domain", category="motor_driver"),
        _future_case("future_linear_ldo", "线性稳压器：12 V 到 5 V、1 A、压差与热约束", kind="dc_operating_point", domain="dc", category="linear_regulator"),
        _future_case("future_bandgap_reference", "带隙基准：2.5 V、温漂和电源抑制比", kind="dc_operating_point", domain="dc", category="voltage_reference"),
        _future_case("future_rf_sparameter_match", "RF 匹配网络：100 MHz 双端口 S 参数和 50 ohm 匹配", kind="s_parameter", domain="frequency_domain", category="rf_matching"),
        _future_case("future_mixer_10mhz", "有源混频器：10 MHz 输入、变频产物和杂散约束", kind="periodic_steady_state", domain="time_domain", category="mixer"),
        _future_case("future_adc_sampling_transient", "ADC 采样前端：阶跃建立时间、过冲和采样瞬态", kind="transient", domain="time_domain", category="adc_frontend"),
        _future_case("future_rc_step_response", "RC 阶跃响应：上升时间、稳态误差和初始条件", kind="transient", domain="time_domain", category="transient"),
        _future_case("future_pwm_gate_driver", "PWM 栅极驱动：死区、上升沿和开关瞬态", kind="transient", domain="time_domain", category="gate_driver"),
        _future_case("future_noise_lna", "低噪声放大器：输入参考噪声和噪声增益", kind="noise", domain="noise", category="noise"),
        _future_case("future_thermal_power_stage", "功率级热设计：结温、热阻和功耗约束", kind="thermal", domain="dc", category="thermal"),
        _future_case("future_dc_bias_point", "晶体管偏置：直流工作点、静态电流和余量", kind="dc_operating_point", domain="dc", category="bias"),
        _future_case("future_rf_sparameter_100mhz", "RF 匹配网络：100 MHz S 参数和回波损耗", kind="s_parameter", domain="frequency_domain", category="rf_matching"),
        _future_case("future_rf_sparameter_2ghz", "RF 匹配网络：2 GHz S 参数和带宽", kind="s_parameter", domain="frequency_domain", category="rf_matching"),
        _future_case("future_periodic_wien_bridge", "文氏桥振荡器：1 kHz 周期稳态和启动过程", kind="periodic_steady_state", domain="time_domain", category="oscillator"),
        _future_case("future_periodic_switching_converter", "开关变换器：周期稳态纹波和开关应力", kind="periodic_steady_state", domain="time_domain", category="switching_power"),
        _future_case("future_multiport_dc_distribution", "四端口直流配电：多路输出、负载分配和交叉调整率", kind="dc_transfer", domain="dc", category="multi_port_power"),
    ])
    return cases


def _baseline_cases() -> list[dict[str, Any]]:
    examples = PROJECT_ROOT / "端口描述语言" / "examples"
    statuses = {
        "5v_to_10v_dc_boost.json": "supported",
        "amplifier_20db.json": "unsupported",
        "bandgap_2v5.json": "supported",
        "bandpass_10khz.json": "supported",
        "differential_amplifier.json": "unsupported",
        "emc_filter_mask.json": "unsupported",
        "h_bridge_motor_driver.json": "unsupported",
        "lowpass_1khz.json": "supported",
        "photodiode_tia.json": "unsupported",
        "sensor_interface.json": "supported",
    }
    entries = []
    for filename, planning in statuses.items():
        source = examples / filename
        destination = BASELINE_ROOT / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries.append({
            "id": source.stem,
            "title": f"现有 PBDL 基线：{source.stem}",
            "path": f"pbdl/baseline/{filename}",
            "category": "existing_baseline",
            "domain": "mixed",
            "source_groups": ["repository_baseline"],
            "expected_planning": planning,
            "execution_tier": "schema",
            "validation_scope": "planner",
            "expected_error_code": None,
        })
    return entries


def main() -> None:
    CLASSIC_ROOT.mkdir(parents=True, exist_ok=True)
    cases = []
    for data, meta in _classic_cases():
        path = CLASSIC_ROOT / f"{data['name']}.json"
        _write_json(path, data)
        meta["path"] = f"pbdl/classic/{path.name}"
        cases.append(meta)

    baseline = _baseline_cases()
    category_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    planning_counts: dict[str, int] = {}
    for item in cases + baseline:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
        domain_counts[item["domain"]] = domain_counts.get(item["domain"], 0) + 1
        planning_counts[item["expected_planning"]] = planning_counts.get(item["expected_planning"], 0) + 1
    manifest = {
        "schema": "circuit_ai.classic_requirement_suite",
        "schema_version": 1,
        "purpose": "Classical circuit requirements expressed as canonical PBDL regression cases.",
        "counts": {
            "classic": len(cases),
            "baseline": len(baseline),
            "total": len(cases) + len(baseline),
            "by_category": dict(sorted(category_counts.items())),
            "by_domain": dict(sorted(domain_counts.items())),
            "by_expected_planning": dict(sorted(planning_counts.items())),
        },
        "cases": sorted(cases + baseline, key=lambda item: item["id"]),
    }
    _write_json(ROOT / "清单.json", manifest)
    _write_json(ROOT / "来源.json", {"schema": "circuit_ai.requirement_sources", "sources": SOURCES})


if __name__ == "__main__":
    main()
