"""约束条件和操作条件——电路设计的限制因素。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatingPoint:
    """电路的工作条件。

    这些不是"优化目标"，而是"在此条件下电路必须工作"。
    """

    supply_voltage_v: float | None = None
    supply_bipolar: bool = False  # ±Vcc
    temperature_c: float = 25.0
    process_corner: str = "typical"  # typical / slow / fast / slow_fast / fast_slow

    def as_dict(self) -> dict:
        return {
            "supply_voltage_v": self.supply_voltage_v,
            "supply_bipolar": self.supply_bipolar,
            "temperature_c": self.temperature_c,
            "process_corner": self.process_corner,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "OperatingPoint":
        if data is None:
            return cls()
        return cls(
            supply_voltage_v=float(data["supply_voltage_v"])
            if data.get("supply_voltage_v") is not None
            else None,
            supply_bipolar=bool(data.get("supply_bipolar", False)),
            temperature_c=float(data.get("temperature_c", 25.0)),
            process_corner=data.get("process_corner", "typical"),
        )


@dataclass(frozen=True)
class Constraints:
    """电路设计的硬性约束。

    元件库 (library) 约束：
      element_types:  允许的元件类型 ["R", "C", "L", "opamp", ...]
      parameter_ranges: {type: [min, max]}  元件值范围

    资源约束：
      max_component_count: 最大元件数
      max_power_mw:        最大功耗
      max_area_mm2:        最大面积（需要版图估算）
      max_cost_cny:        最大成本

    性能约束：
      stability_margin_deg: 最小相位裕度
      noise_floor_dbm:      最大噪声底

    制造约束：
      preferred_values:     优先使用标准值（E12/E24）
    """

    # 元件库
    element_types: tuple[str, ...] = ("R", "C", "opamp")
    required_elements: tuple[str, ...] = ()
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    # 资源
    max_component_count: int = 8
    max_power_mw: float | None = None
    max_area_mm2: float | None = None
    max_cost_cny: float | None = None

    # 性能
    stability_margin_deg: float | None = None
    noise_floor_dbm: float | None = None

    # 制造
    preferred_values: str | None = None  # "E12", "E24", "E48", "E96", None
    real_components: tuple[str, ...] = ()
    model_bindings: tuple[dict, ...] = ()
    unit_costs: dict[str, float] = field(default_factory=dict)
    unit_areas_mm2: dict[str, float] = field(default_factory=dict)
    metric_constraints: tuple[dict[str, Any], ...] = ()
    objectives: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict:
        d = {
            "element_types": list(self.element_types),
            "required_elements": list(self.required_elements),
            "max_component_count": self.max_component_count,
        }
        if self.stability_margin_deg is not None:
            d["stability_margin_deg"] = self.stability_margin_deg
        if self.parameter_ranges:
            d["parameter_ranges"] = {
                k: list(v) for k, v in self.parameter_ranges.items()
            }
        if self.max_power_mw is not None:
            d["max_power_mw"] = self.max_power_mw
        if self.max_area_mm2 is not None:
            d["max_area_mm2"] = self.max_area_mm2
        if self.max_cost_cny is not None:
            d["max_cost_cny"] = self.max_cost_cny
        if self.noise_floor_dbm is not None:
            d["noise_floor_dbm"] = self.noise_floor_dbm
        if self.preferred_values is not None:
            d["preferred_values"] = self.preferred_values
        if self.real_components:
            d["real_components"] = list(self.real_components)
        if self.model_bindings:
            d["model_bindings"] = [dict(item) for item in self.model_bindings]
        if self.unit_costs:
            d["unit_costs"] = dict(self.unit_costs)
        if self.unit_areas_mm2:
            d["unit_areas_mm2"] = dict(self.unit_areas_mm2)
        if self.metric_constraints:
            d["metric_constraints"] = [dict(item) for item in self.metric_constraints]
        if self.objectives:
            d["objectives"] = [dict(item) for item in self.objectives]
        return d

    @classmethod
    def from_dict(cls, data: dict | list[dict[str, Any]] | None) -> "Constraints":
        if data is None:
            return cls()
        if isinstance(data, list):
            data = {"metric_constraints": data}
        ranges = {}
        for k, v in data.get("parameter_ranges", {}).items():
            ranges[k] = (float(v[0]), float(v[1]))
        return cls(
            element_types=tuple(data.get("element_types", ["R", "C", "opamp"])),
            required_elements=tuple(data.get("required_elements", data.get("required", []))),
            parameter_ranges=ranges,
            max_component_count=int(data.get("max_component_count", 8)),
            max_power_mw=float(data["max_power_mw"])
            if data.get("max_power_mw") is not None
            else None,
            max_area_mm2=float(data["max_area_mm2"])
            if data.get("max_area_mm2") is not None
            else None,
            max_cost_cny=float(data["max_cost_cny"])
            if data.get("max_cost_cny") is not None
            else None,
            stability_margin_deg=float(data["stability_margin_deg"])
            if data.get("stability_margin_deg") is not None
            else None,
            noise_floor_dbm=float(data["noise_floor_dbm"])
            if data.get("noise_floor_dbm") is not None
            else None,
            preferred_values=data.get("preferred_values"),
            real_components=tuple(str(item) for item in data.get("real_components", [])),
            model_bindings=tuple(dict(item) for item in data.get("model_bindings", [])),
            unit_costs={str(k): float(v) for k, v in data.get("unit_costs", {}).items()},
            unit_areas_mm2={str(k): float(v) for k, v in data.get("unit_areas_mm2", {}).items()},
            metric_constraints=tuple(
                dict(item)
                for item in data.get("metric_constraints", data.get("declarations", ()))
            ),
            objectives=tuple(dict(item) for item in data.get("objectives", ())),
        )
