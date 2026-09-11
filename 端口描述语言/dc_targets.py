"""DC 目标类型 —— 稳态直流变换目标。"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── DC 目标 ────────────────────────────────────────────


@dataclass(frozen=True)
class DcTarget:
    """直流目标——稳态电压、电流、功率。

    适用于：电源变换器、偏置网络、恒压/恒流源。
    """

    input_voltage_v: float | None = None
    output_voltage_v: float | None = None
    input_current_a: float | None = None
    output_current_a: float | None = None
    output_power_w: float | None = None
    efficiency: float | None = None  # 0.0~1.0
    ripple_mv: float | None = None   # 输出纹波
    weight: float = 1.0
    description: str = ""
    tolerance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "target_kind": "dc",
            "weight": self.weight,
            "description": self.description,
        }
        if self.input_voltage_v is not None:
            d["input_voltage_v"] = self.input_voltage_v
        if self.output_voltage_v is not None:
            d["output_voltage_v"] = self.output_voltage_v
        if self.input_current_a is not None:
            d["input_current_a"] = self.input_current_a
        if self.output_current_a is not None:
            d["output_current_a"] = self.output_current_a
        if self.output_power_w is not None:
            d["output_power_w"] = self.output_power_w
        if self.efficiency is not None:
            d["efficiency"] = self.efficiency
        if self.ripple_mv is not None:
            d["ripple_mv"] = self.ripple_mv
        if self.tolerance:
            d["tolerance"] = dict(self.tolerance)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DcTarget":
        return cls(
            input_voltage_v=float(data["input_voltage_v"]) if data.get("input_voltage_v") is not None else None,
            output_voltage_v=float(data["output_voltage_v"]) if data.get("output_voltage_v") is not None else None,
            input_current_a=float(data["input_current_a"]) if data.get("input_current_a") is not None else None,
            output_current_a=float(data["output_current_a"]) if data.get("output_current_a") is not None else None,
            output_power_w=float(data["output_power_w"]) if data.get("output_power_w") is not None else None,
            efficiency=float(data["efficiency"]) if data.get("efficiency") is not None else None,
            ripple_mv=float(data["ripple_mv"]) if data.get("ripple_mv") is not None else None,
            weight=float(data.get("weight", 1.0)),
            description=data.get("description", ""),
            tolerance={str(key): float(value) for key, value in data.get("tolerance", {}).items()},
        )
