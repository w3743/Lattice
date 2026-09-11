"""分析类型 —— 你要对电路做什么测量。

分析类型与目标类型分离：同一个电路可以同时有 AC 传输目标、噪声目标、阻抗目标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AnalysisDomain = Literal["frequency_domain", "time_domain", "dc", "noise"]


def frequency_grid(f_min_hz: float, f_max_hz: float, points: int) -> np.ndarray:
    return np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), points)


@dataclass(frozen=True)
class Analysis:
    """基础分析类型。"""
    domain: AnalysisDomain = "frequency_domain"
    description: str = ""
    # Preserve future analysis kinds even before a dedicated model exists.  The
    # planner can then report a capability gap without changing the PBDL input.
    kind: str = ""

    def as_dict(self) -> dict:
        data = {"domain": self.domain, "description": self.description}
        if self.kind:
            data["kind"] = self.kind
        return data


@dataclass(frozen=True)
class AcTransfer(Analysis):
    """频域传输——测量 output_port / input_port。

    输入端口由理想电压源驱动，测量输出端口上的电压（相对于参考端子）。

    这是最常见的分析类型：滤波器、放大器、阻抗匹配都属于频域传输。
    """

    kind: Literal["voltage_transfer"] = "voltage_transfer"
    source_port: str = "input"
    output_port: str = "output"
    frequency_hz: tuple[float, float, int] = (10.0, 1_000_000.0, 160)

    def as_dict(self) -> dict:
        return {
            **super().as_dict(),
            "kind": self.kind,
            "source_port": self.source_port,
            "output_port": self.output_port,
            "frequency_hz": list(self.frequency_hz),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcTransfer":
        freq = data.get("frequency_hz", [10.0, 1e6, 160])
        return cls(
            domain=data.get("domain", "frequency_domain"),
            description=data.get("description", ""),
            kind=data.get("kind", "voltage_transfer"),
            source_port=data.get("source_port", "input"),
            output_port=data.get("output_port", "output"),
            frequency_hz=(
                float(freq[0]),
                float(freq[1]),
                int(freq[2]),
            ),
        )


@dataclass(frozen=True)
class Impedance(Analysis):
    """阻抗分析——测量端口上的 V/I。

    适用于：恒阻抗目标、阻抗匹配网络。
    """

    kind: Literal["impedance"] = "impedance"
    port: str = "input"
    frequency_hz: tuple[float, float, int] = (10.0, 1_000_000.0, 160)

    def as_dict(self) -> dict:
        return {
            **super().as_dict(),
            "kind": self.kind,
            "port": self.port,
            "frequency_hz": list(self.frequency_hz),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Impedance":
        freq = data.get("frequency_hz", [10.0, 1e6, 160])
        return cls(
            domain=data.get("domain", "frequency_domain"),
            description=data.get("description", ""),
            kind=data.get("kind", "impedance"),
            port=data.get("port", "input"),
            frequency_hz=(float(freq[0]), float(freq[1]), int(freq[2])),
        )


@dataclass(frozen=True)
class Transimpedance(Analysis):
    """跨阻分析——测量 voltage_output / current_input。

    适用于：光电二极管 TIA、电流输出传感器接口。
    """

    kind: Literal["transimpedance"] = "transimpedance"
    source_port: str = "current_input"
    output_port: str = "voltage_output"
    frequency_hz: tuple[float, float, int] = (10.0, 1_000_000.0, 160)

    def as_dict(self) -> dict:
        return {
            **super().as_dict(),
            "kind": self.kind,
            "source_port": self.source_port,
            "output_port": self.output_port,
            "frequency_hz": list(self.frequency_hz),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transimpedance":
        freq = data.get("frequency_hz", [10.0, 1e6, 160])
        return cls(
            domain=data.get("domain", "frequency_domain"),
            description=data.get("description", ""),
            kind=data.get("kind", "transimpedance"),
            source_port=data.get("source_port", "current_input"),
            output_port=data.get("output_port", "voltage_output"),
            frequency_hz=(float(freq[0]), float(freq[1]), int(freq[2])),
        )


@dataclass(frozen=True)
class DcTransfer(Analysis):
    """直流传输分析——测量输入端口到输出端口的 DC 电压/电流关系。

    适用场景：DC-DC 变换器、线性稳压器、电平转换电路。
    """

    kind: str = "dc_transfer"
    source_port: str = "input"
    output_port: str = "output"
    domain: str = "dc"

    def as_dict(self) -> dict:
        return {
            **super().as_dict(),
            "kind": self.kind,
            "source_port": self.source_port,
            "output_port": self.output_port,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DcTransfer":
        return cls(
            source_port=data.get("source_port", "input"),
            output_port=data.get("output_port", "output"),
            description=data.get("description", ""),
        )


# 预定义分析类型注册表
ANALYSIS_REGISTRY: dict[str, type[Analysis]] = {
    "voltage_transfer": AcTransfer,
    "impedance": Impedance,
    "transimpedance": Transimpedance,
    "dc_transfer": DcTransfer,
}


def analysis_from_dict(data: dict) -> Analysis:
    kind = data.get("kind", "")
    cls = ANALYSIS_REGISTRY.get(kind, Analysis)
    if hasattr(cls, "from_dict"):
        return cls.from_dict(data)
    return cls(
        domain=data.get("domain", "frequency_domain"),
        description=data.get("description", ""),
        kind=str(kind),
    )
