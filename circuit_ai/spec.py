from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .analysis import AnalysisRequest


DEFAULT_RANGES = {
    "R": (1.0, 10_000_000.0),
    "C": (1e-12, 1e-2),
    "L": (1e-9, 10.0),
    "gain": (0.01, 1_000.0),
}


ALIASES = {
    "r": "R",
    "resistor": "R",
    "resistors": "R",
    "c": "C",
    "capacitor": "C",
    "capacitors": "C",
    "l": "L",
    "inductor": "L",
    "inductors": "L",
    "opamp": "opamp",
    "op-amp": "opamp",
    "ideal_opamp": "opamp",
    "vcvs": "E",
    "controlled_source": "E",
    "e": "E",
}


def normalize_element(name: str) -> str:
    return ALIASES.get(name.strip().lower(), name.strip())


def inferred_component_family(identifier: str) -> str:
    text = identifier.casefold()
    if any(token in text for token in ("resistor", ":r", "device:r")):
        return "R"
    if any(token in text for token in ("capacitor", ":c", "device:c")):
        return "C"
    if any(token in text for token in ("inductor", ":l", "device:l")):
        return "L"
    if any(token in text for token in ("opamp", "operational", "amplifier")):
        return "opamp"
    if "diode" in text:
        return "ideal_diode"
    if "switch" in text:
        return "ideal_switch"
    if "transformer" in text:
        return "ideal_transformer"
    return ""


@dataclass(frozen=True)
class LibrarySpec:
    allowed: frozenset[str]
    required: frozenset[str] = frozenset()
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    unit_costs: dict[str, float] = field(default_factory=dict)
    unit_areas_mm2: dict[str, float] = field(default_factory=dict)
    real_components: tuple[str, ...] = ()
    model_bindings: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[str]) -> "LibrarySpec":
        if isinstance(data, list):
            allowed = data
            required = []
            ranges = {}
            costs = {}
            areas = {}
            real_components = []
            model_bindings = []
        else:
            allowed = data.get("allowed", [])
            required = _first_present(data, "required", "must_include", "required_elements", default=[])
            ranges = data.get("parameter_ranges", {})
            costs = _first_present(data, "unit_costs", "element_costs", "costs", default={})
            areas = _first_present(
                data,
                "unit_areas_mm2",
                "element_areas_mm2",
                "areas_mm2",
                "areas",
                default={},
            )
            real_components = _first_present(data, "real_components", "selected_components", default=[])
            model_bindings = data.get("model_bindings", [])

        normalized = frozenset(normalize_element(item) for item in allowed)
        normalized_required = frozenset(normalize_element(item) for item in required)
        if not normalized_required.issubset(normalized):
            missing = sorted(normalized_required - normalized)
            raise ValueError(f"required elements must also be allowed: {missing}")
        parsed_ranges: dict[str, tuple[float, float]] = {}
        for key, value in ranges.items():
            norm = normalize_element(key)
            if len(value) != 2:
                raise ValueError(f"parameter range for {key!r} must have [min, max]")
            lo, hi = float(value[0]), float(value[1])
            if lo <= 0 or hi <= lo:
                raise ValueError(f"invalid parameter range for {key!r}: {value!r}")
            parsed_ranges[norm] = (lo, hi)

        return cls(
            allowed=normalized,
            required=normalized_required,
            parameter_ranges=parsed_ranges,
            unit_costs=_parse_nonnegative_mapping(costs, "unit cost"),
            unit_areas_mm2=_parse_nonnegative_mapping(areas, "unit area"),
            real_components=tuple(str(item).strip() for item in real_components if str(item).strip()),
            model_bindings=tuple(dict(item) for item in model_bindings),
        )

    def range_for(self, element_type: str) -> tuple[float, float]:
        element_type = normalize_element(element_type)
        return self.parameter_ranges.get(element_type, DEFAULT_RANGES[element_type])

    def cost_for(self, element_type: str) -> float:
        return self.unit_costs.get(normalize_element(element_type), 0.0)

    def area_for(self, element_type: str) -> float:
        return self.unit_areas_mm2.get(normalize_element(element_type), 0.0)

    def supports_real_component_family(self, element_type: str) -> bool:
        if not self.real_components:
            return True
        family = normalize_element(element_type)
        return any(family == inferred_component_family(identifier) for identifier in self.real_components)


def _first_present(data: dict[str, Any], *keys: str, default):
    for key in keys:
        if key in data:
            return data[key]
    return default


def _parse_nonnegative_mapping(data: dict[str, Any], label: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in dict(data).items():
        norm = normalize_element(key)
        amount = float(value)
        if amount < 0:
            raise ValueError(f"{label} for {key!r} must be non-negative")
        parsed[norm] = amount
    return parsed


@dataclass(frozen=True)
class OptimizationSpec:
    f_min_hz: float
    f_max_hz: float
    points: int = 160
    max_components: int = 8
    top_k: int = 5
    max_iterations: int = 70
    seed: int = 7
    weights: dict[str, float] = field(default_factory=dict)
    graph_search: dict[str, Any] = field(default_factory=dict)
    fidelity: dict[str, Any] = field(default_factory=dict)
    fidelity_disagreement: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    spice_verification: dict[str, Any] = field(default_factory=dict)
    surrogate: dict[str, Any] = field(default_factory=dict)
    discretization: dict[str, Any] = field(default_factory=dict)
    differentiable: dict[str, Any] = field(default_factory=dict)
    feasibility: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], behavior: dict[str, Any]) -> "OptimizationSpec":
        if "frequency_range_hz" in data:
            f_min, f_max = data["frequency_range_hz"]
        elif "frequency_range_hz" in behavior:
            f_min, f_max = behavior["frequency_range_hz"]
        elif "cutoff_hz" in behavior:
            fc = float(behavior["cutoff_hz"])
            f_min, f_max = fc / 100.0, fc * 100.0
        elif "center_hz" in behavior:
            fc = float(behavior["center_hz"])
            f_min, f_max = fc / 100.0, fc * 100.0
        else:
            f_min, f_max = 10.0, 1_000_000.0

        if f_min <= 0 or f_max <= f_min:
            raise ValueError("frequency range must be positive and increasing")

        return cls(
            f_min_hz=float(f_min),
            f_max_hz=float(f_max),
            points=int(data.get("points", 160)),
            max_components=int(data.get("max_components", 8)),
            top_k=int(data.get("top_k", 5)),
            max_iterations=int(data.get("max_iterations", 70)),
            seed=int(data.get("seed", 7)),
            weights=dict(data.get("weights", {})),
            graph_search=dict(data.get("graph_search", {})),
            fidelity=dict(data.get("fidelity", {})),
            fidelity_disagreement=dict(data.get("fidelity_disagreement", {})),
            optimizer=dict(data.get("optimizer", {})),
            robustness=dict(data.get("robustness", {})),
            spice_verification=dict(data.get("spice_verification", {})),
            surrogate=dict(data.get("surrogate", {})),
            discretization=dict(data.get("discretization", {})),
            differentiable=dict(data.get("differentiable", {})),
            feasibility=dict(data.get("feasibility", {})),
        )


@dataclass(frozen=True)
class SynthesisSpec:
    name: str
    ports: int
    analysis: AnalysisRequest
    behavior: dict[str, Any]
    library: LibrarySpec
    optimization: OptimizationSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthesisSpec":
        behavior = dict(data["behavior"])
        analysis = AnalysisRequest.from_dict(data.get("analysis", behavior.get("analysis")))
        library = LibrarySpec.from_dict(data["library"])
        optimization = OptimizationSpec.from_dict(data.get("optimization", {}), behavior)
        ports = int(data.get("ports", 2))
        if ports != 2:
            raise ValueError(
                "the frequency-domain synthesis engine currently supports exactly two external terminals; "
                "use a PBDL multi-port specification for 3+ ports"
            )
        return cls(
            name=str(data.get("name", "unnamed_synthesis")),
            ports=ports,
            analysis=analysis,
            behavior=behavior,
            library=library,
            optimization=optimization,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SynthesisSpec":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
