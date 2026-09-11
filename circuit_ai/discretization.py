"""Continuous-to-discrete component binding for synthesis candidates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping

from .parts import PartCatalog, PartRecord, PartSelectionPolicy


@dataclass(frozen=True)
class DiscreteSelection:
    variable_id: str
    element_type: str
    requested_value: float
    selected_value: float | None
    source: str
    part_id: str | None = None
    part: Mapping[str, Any] | None = None
    relative_error: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable_id": self.variable_id,
            "element_type": self.element_type,
            "requested_value": self.requested_value,
            "selected_value": self.selected_value,
            "source": self.source,
            "part_id": self.part_id,
            "part": dict(self.part) if self.part is not None else None,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True)
class DiscretizationResult:
    parameter_values: Mapping[str, float]
    selections: tuple[DiscreteSelection, ...]
    feasible: bool
    source: str
    catalog_snapshot: str = ""
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", dict(self.parameter_values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_values": dict(self.parameter_values),
            "selections": [item.as_dict() for item in self.selections],
            "feasible": self.feasible,
            "source": self.source,
            "catalog_snapshot": self.catalog_snapshot,
            "diagnostics": list(self.diagnostics),
        }


def discretize_template_candidates(
    template: Any,
    values: Mapping[str, float],
    bounds: list[tuple[float, float]],
    options: Mapping[str, Any] | None = None,
    *,
    real_components: tuple[str, ...] = (),
) -> tuple[DiscretizationResult, ...]:
    """Return deterministic nearby discrete combinations for one template.

    The first result is the closest combination in log-distance.  Additional
    combinations form the local neighborhood that the caller can re-simulate
    and rank, which is important when several parts trade response error for
    cost or availability.
    """

    options = dict(options or {})
    if not bool(options.get("enabled", False)):
        return ()
    if len(values) != len(template.params) or len(bounds) != len(template.params):
        raise ValueError("template parameter values and bounds must have the same length")

    catalog, catalog_source = _load_catalog(options.get("catalog"))
    policy = PartSelectionPolicy.from_dict(options.get("policy", options))
    series = str(options.get("series", "E24")).upper()
    neighborhood = int(options.get("neighborhood", options.get("neighbors", 1)))
    max_combinations = int(options.get("max_combinations", 32))
    if neighborhood < 0 or max_combinations < 1:
        raise ValueError("discretization neighborhood must be non-negative and max_combinations positive")

    value_choices: list[list[tuple[float, PartRecord | None, str]]] = []
    diagnostics: list[str] = []
    for parameter, (lower, upper) in zip(template.params, bounds):
        requested = float(values[parameter.name])
        family = str(parameter.element_type)
        records = (
            catalog.records_for(
                family,
                policy=policy,
                real_components=real_components,
            )
            if catalog is not None
            else ()
        )
        numeric = sorted(
            {
                float(record.nominal_value)
                for record in records
                if record.nominal_value is not None
                and math.isfinite(float(record.nominal_value))
                and float(lower) <= float(record.nominal_value) <= float(upper)
            }
        )
        if numeric:
            source = "part_catalog"
            lookup = {
                float(record.nominal_value): record
                for record in records
                if record.nominal_value is not None
            }
            candidates = [(value, lookup.get(value), source) for value in numeric]
        elif catalog is not None and policy.require_concrete:
            candidates = []
            diagnostics.append(
                f"no numeric concrete part for {parameter.name!r} in family {family!r}"
            )
        else:
            source = f"e_series:{series}"
            candidates = [(value, None, source) for value in e_series_values(lower, upper, series)]
        if not candidates:
            value_choices.append([])
            continue
        ordered = sorted(
            candidates,
            key=lambda item: _log_distance(requested, item[0]),
        )
        value_choices.append(ordered[: max(1, 2 * neighborhood + 1)])

    if any(not choices for choices in value_choices):
        return (
            DiscretizationResult(
                parameter_values=dict(values),
                selections=(),
                feasible=False,
                source=catalog_source or f"e_series:{series}",
                catalog_snapshot=catalog.snapshot if catalog is not None else "",
                diagnostics=tuple(diagnostics),
            ),
        )

    combinations = list(product(*value_choices))
    combinations.sort(
        key=lambda combination: sum(
            _log_distance(float(values[parameter.name]), combination[index][0])
            for index, parameter in enumerate(template.params)
        )
    )

    results: list[DiscretizationResult] = []
    for combination in combinations[:max_combinations]:
        selected_values = {
            parameter.name: float(combination[index][0])
            for index, parameter in enumerate(template.params)
        }
        selections = tuple(
            _selection(
                parameter.name,
                parameter.element_type,
                float(values[parameter.name]),
                combination[index][0],
                combination[index][1],
                combination[index][2],
            )
            for index, parameter in enumerate(template.params)
        )
        results.append(
            DiscretizationResult(
                parameter_values=selected_values,
                selections=selections,
                feasible=True,
                source=_combined_source(item[2] for item in combination),
                catalog_snapshot=catalog.snapshot if catalog is not None else "",
                diagnostics=tuple(diagnostics),
            )
        )
    return tuple(results)


def e_series_values(lower: float, upper: float, series: str = "E24") -> tuple[float, ...]:
    """Generate IEC-style logarithmic E-series nominal values in a range."""

    bases = {
        "E3": (3,),
        "E6": (6,),
        "E12": (12,),
        "E24": (24,),
        "E48": (48,),
        "E96": (96,),
        "E192": (192,),
    }.get(series.upper())
    if bases is None:
        raise ValueError(f"unsupported E-series {series!r}")
    count = bases[0]
    if lower <= 0.0 or upper <= lower:
        raise ValueError("discretization bounds must be positive and increasing")
    values: set[float] = set()
    start = int(math.floor(math.log10(lower))) - 1
    stop = int(math.ceil(math.log10(upper))) + 1
    for decade in range(start, stop + 1):
        for index in range(count):
            value = 10.0 ** (decade + index / count)
            if lower <= value <= upper:
                values.add(float(f"{value:.12g}"))
    return tuple(sorted(values))


def _load_catalog(raw: Any) -> tuple[PartCatalog | None, str]:
    if raw is None or raw == {}:
        return None, ""
    if isinstance(raw, PartCatalog):
        return raw, "part_catalog"
    if isinstance(raw, (str, Path)):
        catalog = PartCatalog.from_index(raw)
        return catalog, "part_catalog"
    if isinstance(raw, Mapping):
        catalog = PartCatalog.from_dict(raw)
        return catalog, "part_catalog"
    raise ValueError("discretization.catalog must be a PartCatalog, path, or mapping")


def _selection(
    variable_id: str,
    element_type: str,
    requested: float,
    selected: float,
    record: PartRecord | None,
    source: str,
) -> DiscreteSelection:
    relative_error = abs(selected - requested) / max(abs(requested), 1e-30)
    return DiscreteSelection(
        variable_id=variable_id,
        element_type=element_type,
        requested_value=requested,
        selected_value=selected,
        source=source,
        part_id=record.identity if record is not None else None,
        part=record.as_dict() if record is not None else None,
        relative_error=float(relative_error),
    )


def _log_distance(first: float, second: float) -> float:
    return abs(math.log10(max(first, 1e-300)) - math.log10(max(second, 1e-300)))


def _combined_source(sources) -> str:
    unique = tuple(sorted(set(str(item) for item in sources)))
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + "+".join(unique)
