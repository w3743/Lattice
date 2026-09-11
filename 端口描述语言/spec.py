"""电路规格 —— 统一的端口行为描述。

CircuitSpec = Ports + Analysis + Targets + Constraints + OperatingPoint

这五个概念加在一起，完整定义了"我要什么电路"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ports import Port, PortBundle, Terminal
from .analysis import Analysis, AcTransfer
from .functions import FunctionSpec, compile_function_graph
from .targets import Target, FilterTarget, target_from_dict
from .constraints import Constraints, OperatingPoint


@dataclass(frozen=True)
class CircuitSpec:
    """完整的电路规格。

    这是一个电路的"问题陈述"，不涉及如何解决。
    它告诉电路合成器：
      1. 这个电路有哪些端口（ports）
      2. 我要在端口上做什么测量（analyses）
      3. 测量结果应该是什么（targets）
      4. 有哪些限制条件（constraints）
      5. 在什么环境下工作（operating_point）

    analyses 可以包含多个测量（如同时测电压传输和输入阻抗）。
    每个 analysis 可以对应零个或多个 target（不含 target 的分析只做报告不做优化）。
    """

    name: str = "unnamed"
    description: str = ""

    # 端口定义
    ports: PortBundle = field(default_factory=PortBundle.standard_two_port)
    relations: tuple[dict[str, Any], ...] = ()
    functions: tuple[FunctionSpec, ...] = ()

    # 分析 + 目标对
    analyses: tuple[Analysis, ...] = ()
    targets: tuple[Target, ...] = ()

    # 约束
    constraints: Constraints = field(default_factory=Constraints)
    operating_point: OperatingPoint = field(default_factory=OperatingPoint)
    optimization: dict[str, Any] = field(default_factory=dict)

    # ── 便捷构造器 ──

    @classmethod
    def lowpass_filter(
        cls,
        cutoff_hz: float,
        name: str = "lowpass_filter",
        order: int = 1,
        gain_db: float = 0.0,
        elements: tuple[str, ...] = ("R", "C", "opamp"),
        **kwargs,
    ) -> "CircuitSpec":
        return cls(
            name=name,
            ports=PortBundle.standard_two_port(),
            analyses=(AcTransfer(frequency_hz=_auto_freq_range(cutoff_hz)),),
            targets=(
                FilterTarget(
                    kind="lowpass",
                    cutoff_hz=cutoff_hz,
                    order=order,
                    gain_db=gain_db,
                    response=kwargs.get("response", "butterworth"),
                    q=kwargs.get("q"),
                ),
            ),
            constraints=Constraints(
                element_types=elements,
                max_component_count=kwargs.get("max_components", 8),
            ),
            operating_point=OperatingPoint(
                supply_voltage_v=kwargs.get("supply_v"),
            ),
        )

    @classmethod
    def highpass_filter(
        cls,
        cutoff_hz: float,
        name: str = "highpass_filter",
        order: int = 1,
        gain_db: float = 0.0,
        elements: tuple[str, ...] = ("R", "C", "opamp"),
        **kwargs,
    ) -> "CircuitSpec":
        return cls(
            name=name,
            ports=PortBundle.standard_two_port(),
            analyses=(AcTransfer(frequency_hz=_auto_freq_range(cutoff_hz)),),
            targets=(
                FilterTarget(
                    kind="highpass",
                    cutoff_hz=cutoff_hz,
                    order=order,
                    gain_db=gain_db,
                ),
            ),
            constraints=Constraints(
                element_types=elements,
                max_component_count=kwargs.get("max_components", 8),
            ),
            operating_point=OperatingPoint(
                supply_voltage_v=kwargs.get("supply_v"),
            ),
        )

    @classmethod
    def bandpass_filter(
        cls,
        center_hz: float,
        name: str = "bandpass_filter",
        bandwidth_hz: float | None = None,
        q: float = 3.0,
        gain_db: float = 0.0,
        elements: tuple[str, ...] = ("R", "C", "L", "opamp"),
    ) -> "CircuitSpec":
        return cls(
            name=name,
            ports=PortBundle.standard_two_port(),
            analyses=(AcTransfer(frequency_hz=_auto_freq_range(center_hz)),),
            targets=(
                FilterTarget(
                    kind="bandpass",
                    cutoff_hz=center_hz,
                    bandwidth_hz=bandwidth_hz,
                    q=q,
                    gain_db=gain_db,
                ),
            ),
            constraints=Constraints(
                element_types=elements,
                max_component_count=6,
            ),
        )

    @classmethod
    def amplifier(
        cls,
        gain_db: float = 20.0,
        bandwidth_hz: tuple[float, float] = (10.0, 100_000.0),
        name: str = "amplifier",
    ) -> "CircuitSpec":
        from .targets import AmplifierTarget

        return cls(
            name=name,
            ports=PortBundle.standard_two_port(),
            analyses=(AcTransfer(frequency_hz=_auto_freq_range(
                center_hz=float(bandwidth_hz[0] * bandwidth_hz[1]) ** 0.5,
                decades=3.0,
            )),),
            targets=(AmplifierTarget(gain_db=gain_db, bandwidth_hz=bandwidth_hz),),
            constraints=Constraints(
                element_types=("R", "C", "opamp"),
                max_component_count=10,
            ),
        )

    @classmethod
    def constant_impedance(
        cls,
        ohms: float,
        name: str = "constant_impedance",
    ) -> "CircuitSpec":
        from .targets import ImpedanceTarget
        from .analysis import Impedance

        return cls(
            name=name,
            ports=PortBundle.standard_two_port(),
            analyses=(Impedance(port="input"),),
            targets=(ImpedanceTarget(ohms=ohms),),
            constraints=Constraints(
                element_types=("R",),
                max_component_count=2,
            ),
        )

    # ── 序列化 ──

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pbdl.circuit_spec",
            "schema_version": 2,
            "name": self.name,
            "description": self.description,
            "ports": self.ports.as_dict(),
            "relations": [dict(item) for item in self.relations],
            "functions": [item.as_dict() for item in self.functions],
            "analyses": [a.as_dict() for a in self.analyses],
            "targets": [t.as_dict() for t in self.targets],
            "constraints": self.constraints.as_dict(),
            "operating_point": self.operating_point.as_dict(),
            "optimization": dict(self.optimization),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CircuitSpec":
        from .analysis import analysis_from_dict

        ports = PortBundle.from_list(data.get("ports", []))
        functions = tuple(FunctionSpec.from_dict(item) for item in data.get("functions", []))
        raw_analyses = tuple(analysis_from_dict(item) for item in data.get("analyses", []))
        raw_targets = tuple(target_from_dict(item) for item in data.get("targets", []))
        if functions:
            compilation = compile_function_graph(functions, ports)
            # In v2, functions are authoritative. Legacy analyses/targets are
            # accepted only as a migration aid and are filled when omitted.
            if not raw_analyses:
                raw_analyses = tuple(analysis_from_dict(item) for item in compilation.analyses)
            if not raw_targets:
                raw_targets = tuple(target_from_dict(item) for item in compilation.targets)

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            ports=ports,
            relations=tuple(dict(item) for item in data.get("relations", [])),
            functions=functions,
            analyses=raw_analyses,
            targets=raw_targets,
            constraints=Constraints.from_dict(data.get("constraints")),
            operating_point=OperatingPoint.from_dict(data.get("operating_point")),
            optimization=dict(data.get("optimization", {})),
        )

    # ── 验证 ──

    def validate(self) -> list[str]:
        errors = []
        if not self.ports:
            errors.append("no ports defined")
        if not self.analyses:
            errors.append("no analyses defined")
        if not self.targets:
            errors.append("no targets defined")
        if self.functions and not self.analyses:
            errors.append("functions did not lower to an executable analysis")
        if self.constraints.max_component_count < 1:
            errors.append("max_component_count must be >= 1")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


def _auto_freq_range(
    center_hz: float,
    decades: float = 2.0,
) -> tuple[float, float, int]:
    """自动生成频率范围：中心频率上下各 decades 个数量级。"""
    factor = 10.0**decades
    return (
        float(center_hz / factor),
        float(center_hz * factor),
        160,
    )


# ── I/O ──


def save_spec(spec: CircuitSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_spec(path: str | Path) -> CircuitSpec:
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        package_relative = Path(__file__).resolve().parent / path
        if package_relative.exists():
            path = package_relative
    with path.open("r", encoding="utf-8") as fh:
        return CircuitSpec.from_dict(json.load(fh))
