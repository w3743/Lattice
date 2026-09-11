"""PBDL → circuit_ai 翻译器。

将统一的端口描述语言翻译成 circuit_ai 可消费的 SynthesisSpec。

支持的翻译：
  voltage_transfer + filter     → lowpass/highpass/bandpass spec
  voltage_transfer + amplifier  → AC gain spec (理想运放放大器)
  impedance + impedance         → 恒阻抗 spec
  transimpedance + amplifier    → TIA spec
  dc_transfer + dc              → ❌ circuit_ai 不支持，返回明确错误

不支持时返回 TranslationError，包含明确的缺失能力说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TranslationError(Exception):
    """翻译失败的说明。"""
    spec_name: str
    missing_capability: str
    suggestion: str

    def __str__(self) -> str:
        return (
            f"无法合成 '{self.spec_name}': {self.missing_capability}\n"
            f"  建议: {self.suggestion}"
        )


@dataclass(frozen=True)
class TranslationStage:
    """复杂 PBDL 规格中的一个可执行或待实现阶段。"""

    name: str
    status: str
    analysis_index: int
    target_index: int | None
    analysis_kind: str
    target_kind: str | None
    synthesis_spec: dict[str, Any] | None = None
    missing_capability: str | None = None
    suggestion: str | None = None

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "analysis_index": self.analysis_index,
            "target_index": self.target_index,
            "analysis_kind": self.analysis_kind,
            "target_kind": self.target_kind,
            "synthesis_spec": self.synthesis_spec,
            "missing_capability": self.missing_capability,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class TranslationPlan:
    """多分析/多目标 PBDL 规格的分阶段翻译结果。"""

    name: str
    stages: tuple[TranslationStage, ...]

    @property
    def executable_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(stage.synthesis_spec for stage in self.stages if stage.synthesis_spec is not None)

    @property
    def unsupported_stages(self) -> tuple[TranslationStage, ...]:
        return tuple(stage for stage in self.stages if not stage.supported)

    @property
    def fully_supported(self) -> bool:
        return bool(self.stages) and not self.unsupported_stages

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fully_supported": self.fully_supported,
            "stages": [stage.as_dict() for stage in self.stages],
        }


def to_circuit_ai_plan(pbdl_spec) -> TranslationPlan:
    """将复杂 PBDL 规格拆成多个 circuit_ai 合成阶段。

    `to_circuit_ai_spec()` 保持严格：只接受单个可执行合成问题。
    本函数用于真实系统的上层规划：能执行的阶段产出 SynthesisSpec dict，
    暂不支持的阶段保留明确原因，供后续 DC/噪声/布局等专家接管。
    """
    from .spec import CircuitSpec

    stages: list[TranslationStage] = []
    analyses = list(pbdl_spec.analyses)
    targets = list(pbdl_spec.targets)

    for index, analysis in enumerate(analyses):
        target = targets[index] if index < len(targets) else None
        analysis_kind = _circuit_analysis_kind(analysis, pbdl_spec.ports)
        target_kind = _target_kind(target) if target is not None else None
        stage_name = f"{pbdl_spec.name}__stage_{index + 1}_{analysis_kind}"
        if target is None:
            stages.append(
                TranslationStage(
                    name=stage_name,
                    status="unsupported",
                    analysis_index=index,
                    target_index=None,
                    analysis_kind=analysis_kind,
                    target_kind=None,
                    missing_capability="缺少目标",
                    suggestion="为该 analysis 添加一个 target，或将它标记为 report-only 分析。",
                )
            )
            continue

        if len(pbdl_spec.ports) != 2:
            stages.append(
                TranslationStage(
                    name=stage_name,
                    status="unsupported",
                    analysis_index=index,
                    target_index=index,
                    analysis_kind=analysis_kind,
                    target_kind=target_kind,
                    missing_capability=(
                        f"当前执行适配器仅支持 2 个外部端口，当前需求包含 {len(pbdl_spec.ports)} 个端口"
                    ),
                    suggestion=(
                        "等待多端口联合求解器，或将需求拆分为明确的两端口阶段；"
                        "系统不会静默丢弃额外端口。"
                    ),
                )
            )
            continue

        single = CircuitSpec(
            name=stage_name,
            description=pbdl_spec.description,
            ports=pbdl_spec.ports,
            analyses=(analysis,),
            targets=(target,),
            constraints=pbdl_spec.constraints,
            operating_point=pbdl_spec.operating_point,
        )
        # Power-stage synthesis has its own IR/solver path.  Keep the direct
        # ``to_circuit_ai_spec`` contract strict for legacy AC callers, while
        # allowing the multi-stage planner to route a supported two-port DC
        # conversion into the power expert.
        if analysis_kind == "dc_transfer" and target_kind == "dc" and len(pbdl_spec.ports) == 2:
            synthesis_spec = {
                "_pbdl_power_stage": True,
                "_pbdl_spec": pbdl_spec.as_dict(),
                "_pbdl_stage": {
                    "source_spec": pbdl_spec.name,
                    "analysis_index": index,
                    "target_index": index,
                    "analysis_kind": analysis_kind,
                    "target_kind": target_kind,
                },
            }
            stages.append(
                TranslationStage(
                    name=stage_name,
                    status="supported",
                    analysis_index=index,
                    target_index=index,
                    analysis_kind=analysis_kind,
                    target_kind=target_kind,
                    synthesis_spec=synthesis_spec,
                )
            )
            continue
        try:
            synthesis_spec = to_circuit_ai_spec(single)
            synthesis_spec["_pbdl_stage"] = {
                "source_spec": pbdl_spec.name,
                "analysis_index": index,
                "target_index": index,
                "analysis_kind": analysis_kind,
                "target_kind": target_kind,
            }
            stages.append(
                TranslationStage(
                    name=stage_name,
                    status="supported",
                    analysis_index=index,
                    target_index=index,
                    analysis_kind=analysis_kind,
                    target_kind=target_kind,
                    synthesis_spec=synthesis_spec,
                )
            )
        except TranslationError as exc:
            stages.append(
                TranslationStage(
                    name=stage_name,
                    status="unsupported",
                    analysis_index=index,
                    target_index=index,
                    analysis_kind=analysis_kind,
                    target_kind=target_kind,
                    missing_capability=exc.missing_capability,
                    suggestion=exc.suggestion,
                )
            )

    for target_index in range(len(analyses), len(targets)):
        target = targets[target_index]
        stages.append(
            TranslationStage(
                name=f"{pbdl_spec.name}__unassigned_target_{target_index + 1}",
                status="unsupported",
                analysis_index=-1,
                target_index=target_index,
                analysis_kind="unassigned",
                target_kind=_target_kind(target),
                missing_capability="目标没有对应 analysis",
                suggestion="将 targets 与 analyses 一一配对，或拆成单独的 CircuitSpec。",
            )
        )

    return TranslationPlan(name=pbdl_spec.name, stages=tuple(stages))


def to_circuit_ai_spec(pbdl_spec) -> dict[str, Any]:
    """将 PBDL CircuitSpec 翻译为 circuit_ai 的 JSON dict。

    返回的 dict 可以直接传给 SynthesisSpec.from_dict()。

    Raises:
        TranslationError: 当 PBDL spec 包含 circuit_ai 不支持的能力时。
    """
    name = pbdl_spec.name
    analyses = list(pbdl_spec.analyses)
    targets = list(pbdl_spec.targets)
    constraints = pbdl_spec.constraints
    operating = pbdl_spec.operating_point

    # ── 验证能力边界 ──────────────────────────────
    _check_capabilities(name, analyses, targets, pbdl_spec.ports)

    # ── 提取主分析 ──────────────────────────────
    primary_analysis = analyses[0]
    primary_target = targets[0]

    # ── 构建 behavior ──────────────────────────────
    behavior = _build_behavior(primary_analysis, primary_target, pbdl_spec.ports)

    # ── 构建 library ──────────────────────────────
    library = _build_library(constraints)

    # ── 构建 optimization ──────────────────────────────
    optimization = _build_optimization(
        primary_analysis,
        primary_target,
        constraints,
        operating,
        pbdl_spec.optimization,
    )

    spec_dict = {
        "name": name,
        # The current execution IR is still a two-terminal adapter.  Keep the
        # full PBDL port bundle in the canonical spec; do not silently expose
        # this internal limitation as a change to the public schema.
        "ports": 2,
        "behavior": behavior,
        "library": library,
        "optimization": optimization,
    }

    # 非默认分析需要额外的 analysis 信息
    kind = _circuit_analysis_kind(primary_analysis, pbdl_spec.ports)
    if kind == "impedance":
        spec_dict["analysis"] = {"kind": "input_impedance"}
    elif kind == "output_impedance":
        spec_dict["analysis"] = {"kind": "output_impedance"}
    elif kind == "transimpedance":
        spec_dict["analysis"] = {"kind": "transimpedance", "source_name": "Iin"}

    if pbdl_spec.description:
        spec_dict["_description"] = pbdl_spec.description

    return spec_dict


def _check_capabilities(name, analyses, targets, ports) -> None:
    """检查 circuit_ai 能否处理这些分析和目标。"""

    # 检查分析类型
    for analysis in analyses:
        kind = _circuit_analysis_kind(analysis, ports)
        domain = getattr(analysis, "domain", "")

        if kind == "dc_transfer":
            raise TranslationError(
                spec_name=name,
                missing_capability="DC 分析",
                suggestion=(
                    "circuit_ai 目前只有频域 AC 仿真器（MNA）。"
                    "DC-DC 变换器、偏置网络、基准源需要 DC 求解器（牛顿迭代）。"
                    "这是电路综合的下一个技术跳跃。"
                ),
            )

        if kind not in ("voltage_transfer", "impedance", "output_impedance", "transimpedance"):
            raise TranslationError(
                spec_name=name,
                missing_capability=f"分析类型 '{kind}'",
                suggestion=f"circuit_ai 支持 voltage_transfer, impedance, output_impedance 和 transimpedance。",
            )

    # 检查目标类型
    for target in targets:
        kind = _target_kind(target)

        if kind == "dc":
            raise TranslationError(
                spec_name=name,
                missing_capability="DC 目标",
                suggestion="见上方 DC 分析说明。",
            )

        if kind not in ("filter", "amplifier", "impedance", "sampled", "mask"):
            raise TranslationError(
                spec_name=name,
                missing_capability=f"目标类型 '{kind}'",
                suggestion="circuit_ai 支持 filter, amplifier, impedance, sampled, mask。",
            )

    # 检查端口物理量
    for port in ports:
        for terminal in port.terminals:
            if terminal.quantity not in ("voltage", "ground", "unspecified"):
                if terminal.quantity == "current" and _current_port_is_supported(port.name, analyses, targets, ports):
                    continue
                raise TranslationError(
                    spec_name=name,
                    missing_capability=f"端口物理量类型 '{terminal.quantity}'",
                    suggestion=(
                        "circuit_ai 假定所有信号端口是电压端口。"
                        "电流端口（如 TIA 的电流输入）需要电流源激励 + 电流测量分析。"
                    ),
                )

    # 检查多分析 + 多目标组合
    if len(analyses) > 1:
        raise TranslationError(
            spec_name=name,
            missing_capability="多分析类型组合",
            suggestion=(
                "circuit_ai 单次合成只支持一个 analysis 类型。"
                "多目标（如 AC 增益 + DC 偏置）需要分解为多个合成步骤，再组合结果。"
            ),
        )

    # 如果目标多于分析，给出警告（非致命）
    if len(targets) > 1:
        _warn_unused_targets(name, targets[1:])


def _build_behavior(analysis, target, ports=None) -> dict[str, Any]:
    """从分析+目标构建 behavior dict。"""

    kind = _circuit_analysis_kind(analysis, ports)
    target_kind = _target_kind(target)

    if kind == "voltage_transfer":
        return _ac_behavior(analysis, target, target_kind)
    if kind in {"impedance", "output_impedance"}:
        return _impedance_behavior(analysis, target, kind)
    if kind == "transimpedance":
        return _transimpedance_behavior(analysis, target, target_kind)

    raise TranslationError(
        spec_name="unknown",
        missing_capability=f"分析类型 '{kind}'",
        suggestion="不支持的翻译。",
    )


def _ac_behavior(analysis, target, target_kind) -> dict[str, Any]:
    """频域传输行为。"""
    freq_range = getattr(analysis, "frequency_hz", (10.0, 1e6, 160))

    if target_kind == "filter":
        return _filter_behavior(target, freq_range)
    if target_kind == "amplifier":
        return _amplifier_behavior(target, freq_range)
    if target_kind in ("sampled", "mask"):
        raise TranslationError(
            spec_name="",
            missing_capability=f"目标类型 '{target_kind}' → behavior 翻译",
            suggestion=(
                f"sampled/mask 目标可以翻译为 circuit_ai 的 'samples' behavior。"
                f"此翻译尚未实现，但技术上可行（将采样点直接映射到 behavior.frequency_hz + magnitude_db）。"
            ),
        )

    raise TranslationError(
        spec_name="",
        missing_capability=f"目标类型 '{target_kind}'",
        suggestion="不支持的翻译。",
    )


def _filter_behavior(target, freq_range) -> dict[str, Any]:
    """滤波器目标 → behavior。"""
    filter_kind = getattr(target, "kind", "lowpass")
    behavior: dict[str, Any] = {
        "kind": filter_kind,
        "frequency_range_hz": [freq_range[0], freq_range[1]],
    }

    cutoff = getattr(target, "cutoff_hz", 1000.0)
    if filter_kind in ("lowpass", "highpass"):
        behavior["cutoff_hz"] = cutoff
    elif filter_kind == "bandpass":
        behavior["center_hz"] = cutoff

    order = getattr(target, "order", 1)
    if order and order > 1:
        behavior["order"] = int(order)

    gain_db = getattr(target, "gain_db", 0.0)
    if gain_db != 0.0:
        behavior["gain"] = 10.0 ** (gain_db / 20.0)

    q_val = getattr(target, "q", None)
    if q_val is not None:
        behavior["q"] = float(q_val)

    return behavior


def _amplifier_behavior(target, freq_range) -> dict[str, Any]:
    """放大器目标 → 等效低通 behavior。

    放大器 = 平直增益 + 带宽限制。
    高频滚降 = 等效低通滤波器，增益 = 放大器增益。
    """

    gain_db = getattr(target, "gain_db", 0.0)
    bw = getattr(target, "bandwidth_hz", (10.0, 100_000.0))
    f_hi = bw[1] if isinstance(bw, (list, tuple)) and len(bw) == 2 else bw

    return {
        "kind": "lowpass",
        "cutoff_hz": float(f_hi),
        "gain": 10.0 ** (gain_db / 20.0),
        "order": 1,
        "frequency_range_hz": [freq_range[0], freq_range[1]],
    }


def _impedance_behavior(analysis, target, analysis_kind: str = "impedance") -> dict[str, Any]:
    """阻抗分析+目标 → behavior。"""
    freq_range = getattr(analysis, "frequency_hz", (10.0, 1e6, 160))

    ohms = getattr(target, "ohms", None)
    if ohms is not None:
        return {
            "kind": "constant_output_impedance" if analysis_kind == "output_impedance" else "constant_impedance",
            "ohms": float(ohms),
            "frequency_range_hz": [freq_range[0], freq_range[1]],
        }

    raise TranslationError(
        spec_name="",
        missing_capability="频率相关阻抗目标",
        suggestion="circuit_ai 目前只支持 constant_impedance。频率相关的复阻抗需要扩展。",
    )


def _transimpedance_behavior(analysis, target, target_kind) -> dict[str, Any]:
    """跨阻分析+目标 → behavior。"""
    freq_range = getattr(analysis, "frequency_hz", (10.0, 1e6, 160))
    if target_kind != "amplifier":
        raise TranslationError(
            spec_name="",
            missing_capability=f"跨阻目标类型 '{target_kind}'",
            suggestion="TIA 当前从 amplifier target 翻译跨阻增益和带宽。",
        )

    gain_db = getattr(target, "gain_db", 60.0)
    bw = getattr(target, "bandwidth_hz", (10.0, 100_000.0))
    f_hi = bw[1] if isinstance(bw, (list, tuple)) and len(bw) == 2 else bw
    return {
        "kind": "transimpedance",
        "transimpedance_ohm": 10.0 ** (float(gain_db) / 20.0),
        "cutoff_hz": float(f_hi),
        "frequency_range_hz": [freq_range[0], freq_range[1]],
    }


def _build_library(constraints) -> dict[str, Any]:
    """约束 → library dict。"""
    elements = list(constraints.element_types)

    # 映射：将 opamp → circuit_ai 兼容
    mapped = []
    for e in elements:
        e_upper = e.upper()
        if e_upper in ("OPAMP", "E", "VCVS"):
            mapped.append("opamp")
        elif e_upper in ("R", "C", "L"):
            mapped.append(e_upper)
        elif e_upper in ("NPN", "PNP", "NMOS", "PMOS", "MOSFET_N", "MOSFET_P"):
            raise TranslationError(
                spec_name="",
                missing_capability=f"元件类型 '{e}'",
                suggestion=(
                    "circuit_ai 不支持晶体管。"
                    "需要 DC 偏置求解器和非线性元件模型。"
                    "用理想运放 (opamp) 替代晶体管级设计。"
                ),
            )
        else:
            raise TranslationError(
                spec_name="",
                missing_capability=f"元件类型 '{e}'",
                suggestion=f"circuit_ai 支持 R, C, L, opamp。'{e}' 不在支持列表中。",
            )

    library: dict[str, Any] = {
        "allowed": mapped,
        "required": list(constraints.required_elements),
    }

    if constraints.parameter_ranges:
        library["parameter_ranges"] = {
            k: list(v) for k, v in constraints.parameter_ranges.items()
        }

    if constraints.real_components:
        library["real_components"] = list(constraints.real_components)
    if constraints.model_bindings:
        library["model_bindings"] = [dict(item) for item in constraints.model_bindings]
    if constraints.unit_costs:
        library["unit_costs"] = dict(constraints.unit_costs)
    if constraints.unit_areas_mm2:
        library["unit_areas_mm2"] = dict(constraints.unit_areas_mm2)

    return library


def _build_optimization(analysis, target, constraints, operating, requested: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建 optimization 配置。"""
    freq_range = getattr(analysis, "frequency_hz", (10.0, 1e6, 160))

    opt: dict[str, Any] = dict(requested or {})
    opt.setdefault("points", int(freq_range[2]) if len(freq_range) >= 3 else 160)
    opt.setdefault("max_components", constraints.max_component_count)
    opt.setdefault("max_iterations", 50)
    opt.setdefault("top_k", 3)
    defaults: dict[str, Any] = {
        "points": int(freq_range[2]) if len(freq_range) >= 3 else 160,
        "max_components": constraints.max_component_count,
        "max_iterations": 50,
        "top_k": 3,
    }
    for key, value in defaults.items():
        opt.setdefault(key, value)

    # PBDL is the canonical user entry.  Frequency-domain synthesis therefore
    # searches small generated R/C/L graphs alongside the standard catalog by
    # default; users can still tighten or disable this explicit search bound.
    if _target_kind(target) in {"filter", "sampled", "mask"}:
        graph_search = dict(opt.get("graph_search", {}))
        graph_search.setdefault("enabled", True)
        graph_search.setdefault("max_candidates", 4)
        graph_search.setdefault("internal_nodes", 1)
        graph_search.setdefault("include_fallback", True)
        opt["graph_search"] = graph_search

    if constraints.max_power_mw is not None:
        opt["_max_power_mw_not_enforced"] = (
            f"约束 max_power_mw={constraints.max_power_mw}mW，"
            f"但 circuit_ai 当前不强制执行功耗约束。"
        )

    if constraints.noise_floor_dbm is not None:
        opt["_noise_floor_not_enforced"] = (
            f"约束 noise_floor_dbm={constraints.noise_floor_dbm}dBm，"
            f"但 circuit_ai 当前不强制执行噪声约束。"
        )

    if operating.supply_voltage_v is not None:
        opt["_supply_voltage_not_enforced"] = (
            f"供电 {operating.supply_voltage_v}V，"
            f"但 circuit_ai 当前不考虑供电电压约束。"
        )

    # tolerance 翻译为 weights
    tol = getattr(target, "tolerance", None)
    if tol is not None:
        abs_tol = getattr(tol, "absolute", None)
        if abs_tol is not None and abs_tol > 0:
            # 更紧的容差 → 更大的 rmse 权重
            opt["weights"] = {
                "rmse_db": 1.0 / max(abs_tol, 0.01),
            }

    return opt


def _warn_unused_targets(name, unused_targets) -> None:
    """记录未使用的目标。"""
    kinds = [_target_kind(t) for t in unused_targets]
    import warnings
    warnings.warn(
        f"spec '{name}' 有 {len(unused_targets)} 个未使用的目标 ({kinds})。"
        f"circuit_ai 单次合成只使用第一个 (analysis, target) 对。"
        f"多目标合成需要分别运行后组合结果。"
    )


def _target_kind(target) -> str:
    """获取目标的 kind 字符串。"""
    if isinstance(target, dict):
        return target.get("target_kind", target.get("kind", "unknown"))

    # DcTarget: 有 output_voltage_v 但无 cutoff_hz/gain_db（频域特征）
    if hasattr(target, "output_voltage_v") and not hasattr(target, "cutoff_hz") and not hasattr(target, "gain_db"):
        return "dc"

    # ImpedanceTarget: 有 ohms
    if hasattr(target, "ohms") and getattr(target, "ohms", None) is not None:
        return "impedance"

    # SampledTarget: 有 magnitude_db
    if hasattr(target, "magnitude_db") and getattr(target, "magnitude_db", None):
        return "sampled"

    # MaskTarget: 有 upper_bound_db
    if hasattr(target, "upper_bound_db") and getattr(target, "upper_bound_db", None):
        return "mask"

    # FilterTarget: 有 kind + cutoff_hz — 统一归为 "filter"
    if hasattr(target, "kind") and hasattr(target, "cutoff_hz"):
        return "filter"

    # AmplifierTarget: 有 gain_db + bandwidth_hz 但无 cutoff_hz
    if hasattr(target, "gain_db") and hasattr(target, "bandwidth_hz") and not hasattr(target, "cutoff_hz"):
        return "amplifier"

    # 最后尝试从 as_dict 获取
    if hasattr(target, "as_dict"):
        d = target.as_dict()
        return d.get("target_kind", d.get("kind", "unknown"))

    return "unknown"


def _circuit_analysis_kind(analysis, ports=None) -> str:
    kind = getattr(analysis, "kind", "")
    if kind == "impedance":
        port = getattr(analysis, "port", "input")
        if port == "output":
            return "output_impedance"
    if kind == "voltage_transfer" and ports is not None:
        source_port = getattr(analysis, "source_port", "")
        if _port_has_quantity(ports, source_port, "current"):
            return "transimpedance"
    return kind


def _current_port_is_supported(port_name, analyses, targets, ports) -> bool:
    if not targets:
        return False
    primary_target_kind = _target_kind(targets[0])
    if primary_target_kind != "amplifier":
        return False
    for analysis in analyses:
        if getattr(analysis, "source_port", None) != port_name:
            continue
        if _circuit_analysis_kind(analysis, ports) == "transimpedance":
            return True
    return False


def _port_has_quantity(ports, port_name: str, quantity: str) -> bool:
    if not port_name:
        return False
    for port in ports:
        if port.name != port_name:
            continue
        return any(terminal.quantity == quantity for terminal in port.terminals)
    return False
