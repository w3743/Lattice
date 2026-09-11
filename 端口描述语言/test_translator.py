"""翻译器测试 —— 验证 PBDL → circuit_ai 的桥。"""

from 端口描述语言 import CircuitSpec, load_spec
from 端口描述语言.translator import to_circuit_ai_plan, to_circuit_ai_spec, TranslationError


def _try_synthesize(spec: CircuitSpec):
    """尝试用 circuit_ai 合成。返回 (成功?, 结果或错误消息)。"""
    try:
        from circuit_ai import SynthesisSpec, CircuitSynthesizer

        spec_dict = to_circuit_ai_spec(spec)
        ai_spec = SynthesisSpec.from_dict(spec_dict)
        results = CircuitSynthesizer().synthesize(ai_spec)
        best = results[0]
        return (
            True,
            f"✓ {best.template.name}  rmse={best.metrics.rmse_db:.3g}dB  "
            f"components={best.metrics.component_count}  score={best.metrics.score:.4g}",
        )
    except TranslationError as e:
        return (False, str(e))
    except ImportError:
        return (False, "circuit_ai 未安装")
    except Exception as e:
        return (False, f"合成失败: {e}")


def test_translate_lowpass() -> None:
    """低通滤波器：应成功翻译并合成。"""
    spec = CircuitSpec.lowpass_filter(cutoff_hz=1000.0, elements=("R", "C"))
    ok, msg = _try_synthesize(spec)
    print(f"  lowpass: {msg}")
    assert ok


def test_translate_highpass() -> None:
    """高通滤波器：应成功。"""
    spec = CircuitSpec.highpass_filter(cutoff_hz=5000.0, elements=("R", "C"))
    ok, msg = _try_synthesize(spec)
    print(f"  highpass: {msg}")
    assert ok


def test_translate_bandpass() -> None:
    """带通滤波器：应成功。"""
    spec = CircuitSpec.bandpass_filter(center_hz=10000.0, elements=("R", "C", "L"))
    ok, msg = _try_synthesize(spec)
    print(f"  bandpass: {msg}")
    assert ok


def test_translate_amplifier() -> None:
    """放大器：应成功。"""
    spec = CircuitSpec.amplifier(gain_db=20.0)
    ok, msg = _try_synthesize(spec)
    print(f"  amplifier: {msg}")
    assert ok


def test_translate_constant_impedance() -> None:
    """恒阻抗：应成功。"""
    spec = CircuitSpec.constant_impedance(ohms=1000.0)
    ok, msg = _try_synthesize(spec)
    print(f"  impedance: {msg}")
    assert ok


def test_dc_spec_gives_clear_error() -> None:
    """DC 目标：应返回明确的错误消息。"""
    from 端口描述语言 import DcTarget, DcTransfer, PortBundle, Constraints

    spec = CircuitSpec(
        name="dc_boost",
        ports=PortBundle.standard_two_port(),
        analyses=(DcTransfer(),),
        targets=(DcTarget(input_voltage_v=5.0, output_voltage_v=10.0),),
        constraints=Constraints(),
    )
    ok, msg = _try_synthesize(spec)
    print(f"  dc_boost error: {msg[:100]}...")
    assert not ok
    assert "DC" in msg


def test_transistor_spec_gives_clear_error() -> None:
    """晶体管元件：应返回明确的错误消息。"""
    from 端口描述语言 import Constraints

    spec = CircuitSpec(
        name="npn_test",
        ports=CircuitSpec.lowpass_filter(cutoff_hz=1000.0).ports,
        analyses=CircuitSpec.lowpass_filter(cutoff_hz=1000.0).analyses,
        targets=CircuitSpec.lowpass_filter(cutoff_hz=1000.0).targets,
        constraints=Constraints(element_types=("NPN", "R")),
    )
    ok, msg = _try_synthesize(spec)
    print(f"  npn error: {msg[:100]}...")
    assert not ok
    assert "NPN" in msg or "晶体管" in msg


def test_current_port_spec_gives_clear_error() -> None:
    """电流端口：应返回明确的错误消息。"""
    from 端口描述语言 import Port, PortBundle, Terminal

    ports = PortBundle(ports=(
        Port(name="input", terminals=(
            Terminal("pd", "current"), Terminal("0", "ground")
        )),
        Port(name="output", terminals=(
            Terminal("vout", "voltage"), Terminal("0", "ground")
        )),
    ))
    spec = CircuitSpec(
        name="tia",
        ports=ports,
        analyses=CircuitSpec.lowpass_filter(cutoff_hz=1000.0).analyses,
        targets=CircuitSpec.lowpass_filter(cutoff_hz=1000.0).targets,
        constraints=CircuitSpec.lowpass_filter(cutoff_hz=1000.0).constraints,
    )
    ok, msg = _try_synthesize(spec)
    print(f"  current_port error: {msg[:100]}...")
    assert not ok
    assert "电流" in msg or "current" in msg.lower()


def test_translate_current_input_transimpedance() -> None:
    """跨阻放大器：电流输入端口应翻译为 transimpedance 并合成。"""
    from 端口描述语言 import (
        AmplifierTarget,
        Constraints,
        Port,
        PortBundle,
        Terminal,
        Transimpedance,
    )

    ports = PortBundle(ports=(
        Port(name="current_input", terminals=(
            Terminal("pd", "current"), Terminal("0", "ground")
        )),
        Port(name="voltage_output", terminals=(
            Terminal("vout", "voltage"), Terminal("0", "ground")
        )),
    ))
    spec = CircuitSpec(
        name="tia",
        ports=ports,
        analyses=(Transimpedance(
            source_port="current_input",
            output_port="voltage_output",
            frequency_hz=(10.0, 1_000_000.0, 80),
        ),),
        targets=(AmplifierTarget(gain_db=60.0, bandwidth_hz=(10.0, 100_000.0)),),
        constraints=Constraints(
            element_types=("R", "C", "opamp"),
            parameter_ranges={"R": (100.0, 10000.0), "C": (1e-12, 1e-5)},
            max_component_count=3,
        ),
    )

    spec_dict = to_circuit_ai_spec(spec)
    assert spec_dict["analysis"]["kind"] == "transimpedance"
    assert spec_dict["behavior"]["kind"] == "transimpedance"
    assert spec_dict["behavior"]["transimpedance_ohm"] == 1000.0

    ok, msg = _try_synthesize(spec)
    print(f"  tia: {msg}")
    assert ok
    assert "transimpedance_amplifier" in msg


def test_multi_analysis_gives_clear_error() -> None:
    """多分析 spec：应返回明确的错误消息。"""
    ok, msg = _try_synthesize(
        load_spec("examples/sensor_interface.json")
    )
    print(f"  multi_analysis error: {msg[:120]}...")
    assert not ok
    assert "多" in msg or "analysis" in msg.lower()


def test_plan_photodiode_tia_blocks_multiport_downgrade() -> None:
    """三端口 TIA 不得把电源端口静默丢弃后降级成双端口执行。"""

    spec = load_spec("examples/photodiode_tia.json")
    plan = to_circuit_ai_plan(spec)

    assert len(plan.stages) == 2
    assert not plan.fully_supported

    assert all(not stage.supported for stage in plan.stages)
    assert all(stage.status == "unsupported" for stage in plan.stages)
    assert all("3 个端口" in (stage.missing_capability or "") for stage in plan.stages)
    assert all("静默丢弃" in (stage.suggestion or "") for stage in plan.stages)


def test_plan_sensor_interface_splits_filter_and_output_impedance_stages() -> None:
    """传感器接口 spec：滤波和输出阻抗应拆成两个可执行阶段。"""
    from circuit_ai import CircuitSynthesizer, SynthesisSpec

    spec = load_spec("examples/sensor_interface.json")
    plan = to_circuit_ai_plan(spec)

    assert len(plan.stages) == 2
    assert plan.fully_supported
    assert plan.stages[0].supported
    assert plan.stages[0].synthesis_spec["behavior"]["kind"] == "lowpass"
    assert plan.stages[1].supported
    assert plan.stages[1].analysis_kind == "output_impedance"
    assert plan.stages[1].synthesis_spec["analysis"]["kind"] == "output_impedance"
    assert plan.stages[1].synthesis_spec["behavior"]["kind"] == "constant_output_impedance"

    result = CircuitSynthesizer().synthesize(SynthesisSpec.from_dict(plan.stages[1].synthesis_spec))[0]
    assert result.template.name == "output_resistor_impedance"
    assert result.metrics.rmse_db < 0.05
