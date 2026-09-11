"""Operating envelopes and worst-case evaluation for power stage designs.

An ideal averaged design is solved at one operating point, but a requirement is
almost never a point: "36 V to 5 V" usually means 24-48 V in, 10-100 % load.
A design that only meets its target at nominal is not a design.

This module adds the missing layer without touching the point-design path:

* :class:`OperatingEnvelope` describes the allowed input voltage and load range.
* :func:`enumerate_corners` turns an envelope into explicit design corners.
* :func:`worst_case_summary` reduces per-corner results to the extremes a
  designer must design against, and records which corner produced each one, so
  the worst case is attributable rather than a bare number.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ENVELOPE_SCHEMA",
    "ENVELOPE_VERSION",
    "OperatingCorner",
    "OperatingEnvelope",
    "WorstCaseSummary",
    "enumerate_corners",
    "envelope_from_mapping",
    "worst_case_summary",
]


ENVELOPE_SCHEMA = "circuit_ai.operating_envelope"
ENVELOPE_VERSION = 1


@dataclass(frozen=True)
class OperatingCorner:
    """One operating point extracted from an envelope."""

    corner_id: str
    input_voltage_v: float
    load_fraction: float

    def __post_init__(self) -> None:
        if not self.corner_id.strip():
            raise ValueError("corner id must not be empty")
        if self.input_voltage_v <= 0.0:
            raise ValueError("corner input voltage must be positive")
        if not 0.0 < self.load_fraction:
            raise ValueError("corner load fraction must be positive")

    @property
    def is_nominal(self) -> bool:
        return self.corner_id == "nominal"

    def as_dict(self) -> dict[str, Any]:
        return {
            "corner_id": self.corner_id,
            "input_voltage_v": self.input_voltage_v,
            "load_fraction": self.load_fraction,
        }


@dataclass(frozen=True)
class OperatingEnvelope:
    """The input voltage and load range a design must survive.

    A degenerate envelope (one voltage, one load) reproduces single-point
    design exactly, so callers that declare nothing keep their current
    behaviour.
    """

    nominal_input_voltage_v: float
    min_input_voltage_v: float | None = None
    max_input_voltage_v: float | None = None
    nominal_load_fraction: float = 1.0
    min_load_fraction: float = 1.0
    max_load_fraction: float = 1.0
    schema: str = ENVELOPE_SCHEMA
    schema_version: int = ENVELOPE_VERSION

    def __post_init__(self) -> None:
        if self.schema != ENVELOPE_SCHEMA or self.schema_version != ENVELOPE_VERSION:
            raise ValueError("unsupported operating envelope schema")
        if self.nominal_input_voltage_v <= 0.0:
            raise ValueError("nominal input voltage must be positive")
        low = self.min_input_voltage_v if self.min_input_voltage_v is not None else self.nominal_input_voltage_v
        high = self.max_input_voltage_v if self.max_input_voltage_v is not None else self.nominal_input_voltage_v
        if low <= 0.0 or high <= 0.0:
            raise ValueError("input voltage limits must be positive")
        if low > high:
            raise ValueError("min input voltage must not exceed max input voltage")
        if not low <= self.nominal_input_voltage_v <= high:
            raise ValueError("nominal input voltage must lie inside the declared range")
        if self.min_load_fraction <= 0.0 or self.max_load_fraction <= 0.0:
            raise ValueError("load fractions must be positive")
        if self.min_load_fraction > self.max_load_fraction:
            raise ValueError("min load fraction must not exceed max load fraction")
        if not self.min_load_fraction <= self.nominal_load_fraction <= self.max_load_fraction:
            raise ValueError("nominal load fraction must lie inside the declared range")

    @property
    def low_input_voltage_v(self) -> float:
        return self.min_input_voltage_v if self.min_input_voltage_v is not None else self.nominal_input_voltage_v

    @property
    def high_input_voltage_v(self) -> float:
        return self.max_input_voltage_v if self.max_input_voltage_v is not None else self.nominal_input_voltage_v

    @property
    def is_degenerate(self) -> bool:
        """Whether this envelope collapses to a single operating point."""

        return (
            self.low_input_voltage_v == self.high_input_voltage_v
            and self.min_load_fraction == self.max_load_fraction
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "nominal_input_voltage_v": self.nominal_input_voltage_v,
            "min_input_voltage_v": self.min_input_voltage_v,
            "max_input_voltage_v": self.max_input_voltage_v,
            "nominal_load_fraction": self.nominal_load_fraction,
            "min_load_fraction": self.min_load_fraction,
            "max_load_fraction": self.max_load_fraction,
            "is_degenerate": self.is_degenerate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> OperatingEnvelope | None:
        if not data:
            return None
        nominal = data.get("nominal_input_voltage_v", data.get("input_voltage_v"))
        if nominal is None:
            return None
        return cls(
            nominal_input_voltage_v=float(nominal),
            min_input_voltage_v=_optional_float(data.get("min_input_voltage_v")),
            max_input_voltage_v=_optional_float(data.get("max_input_voltage_v")),
            nominal_load_fraction=float(data.get("nominal_load_fraction", 1.0)),
            min_load_fraction=float(data.get("min_load_fraction", 1.0)),
            max_load_fraction=float(data.get("max_load_fraction", 1.0)),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def enumerate_corners(envelope: OperatingEnvelope) -> tuple[OperatingCorner, ...]:
    """Expand an envelope into the corners a design must be checked at.

    Nominal first, then the extremes.  A degenerate envelope yields exactly one
    corner so single-point behaviour is preserved.
    """

    voltages: list[tuple[str, float]] = [("nominal", envelope.nominal_input_voltage_v)]
    if envelope.low_input_voltage_v != envelope.nominal_input_voltage_v:
        voltages.append(("input_min", envelope.low_input_voltage_v))
    if envelope.high_input_voltage_v != envelope.nominal_input_voltage_v:
        voltages.append(("input_max", envelope.high_input_voltage_v))

    loads: list[tuple[str, float]] = [("nominal", envelope.nominal_load_fraction)]
    if envelope.min_load_fraction != envelope.nominal_load_fraction:
        loads.append(("load_min", envelope.min_load_fraction))
    if envelope.max_load_fraction != envelope.nominal_load_fraction:
        loads.append(("load_max", envelope.max_load_fraction))

    corners: list[OperatingCorner] = []
    seen: set[tuple[float, float]] = set()
    for voltage_label, voltage in voltages:
        for load_label, load in loads:
            key = (float(voltage), float(load))
            if key in seen:
                continue
            seen.add(key)
            corner_id = (
                "nominal"
                if voltage_label == "nominal" and load_label == "nominal"
                else f"{voltage_label}__{load_label}"
            )
            corners.append(
                OperatingCorner(
                    corner_id=corner_id,
                    input_voltage_v=float(voltage),
                    load_fraction=float(load),
                )
            )
    return tuple(corners)


@dataclass(frozen=True)
class DutySchedule:
    """The duty cycle each corner needs to hold the target rail.

    This is the topology's *control authority*, reported separately from whether
    the built design exercises it.  A design with one fixed duty cannot hold a
    rail across a range; if every required duty is inside the usable band, then a
    controller that schedules duty against the input could, and the design is
    realisable.

    It is deliberately **not** a stability claim.  Whether a loop can be
    compensated -- gain and phase margin, crossover, transient recovery -- needs
    an analysis this project does not have, and ``justifies_stability_claim``
    says so in the record rather than leaving it to the reader.
    """

    entries: tuple[Mapping[str, Any], ...]
    usable_band: tuple[float, float]
    control_effort: tuple[float, float]
    #: Always false today.  Present so a report cannot be read as claiming a
    #: compensated loop, and so the field has to be changed deliberately.
    justifies_stability_claim: bool = False
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def all_reachable(self) -> bool:
        """Whether every corner's required duty lies inside the usable band.

        This is the feasibility result: the conversion is possible everywhere in
        the envelope, so the requirement is not asking for something the topology
        cannot deliver.
        """

        return bool(self.entries) and all(
            bool(dict(entry)["inside_usable_band"]) for entry in self.entries
        )

    @property
    def spread(self) -> float:
        """How much duty range the controller must cover."""

        return float(self.control_effort[1] - self.control_effort[0])

    @property
    def limiting_corner(self) -> str | None:
        """The corner with the least headroom, i.e. the binding one."""

        scored = [
            (
                min(float(dict(e)["headroom_to_upper"]), float(dict(e)["headroom_to_lower"])),
                str(dict(e)["corner_id"]),
            )
            for e in self.entries
        ]
        if not scored:
            return None
        return min(scored)[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_reachable": self.all_reachable,
            "usable_band": {"lower": self.usable_band[0], "upper": self.usable_band[1]},
            "control_effort": {"min_duty": self.control_effort[0], "max_duty": self.control_effort[1]},
            "spread": self.spread,
            "limiting_corner": self.limiting_corner,
            "justifies_stability_claim": self.justifies_stability_claim,
            "entries": [dict(entry) for entry in self.entries],
            "extensions": dict(self.extensions),
        }


@dataclass(frozen=True)
class WorstCaseSummary:
    """Extremes across corners, each attributable to the corner that caused it."""

    corner_count: int
    efficiency_min: float
    efficiency_min_corner: str
    efficiency_nominal: float
    loss_max_w: float
    loss_max_corner: str
    input_current_max_a: float
    input_current_max_corner: str
    duty_max: float
    duty_max_corner: str
    duty_min: float
    duty_min_corner: str
    ripple_max_mv: float
    ripple_max_corner: str
    output_voltage_min_v: float
    output_voltage_min_corner: str
    output_voltage_max_v: float
    output_voltage_max_corner: str
    nominal_output_voltage_v: float
    target_output_voltage_v: float | None = None
    tolerance_fraction: float | None = None
    regulation_violations: tuple[Mapping[str, Any], ...] = ()
    #: What duty each corner would need to hold the rail.  ``None`` when no
    #: schedule was computed, so the record never implies one it did not check.
    duty_schedule: DutySchedule | None = None
    corner_results: tuple[Mapping[str, Any], ...] = ()
    schema: str = "circuit_ai.worst_case_summary"
    schema_version: int = 1
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def regulates_output(self) -> bool | None:
        """Whether every corner holds the target rail within tolerance.

        ``None`` when no target was supplied, so "not checked" is never reported
        as "passed".  A fixed-duty design with no feedback loop is expected to
        fail this across a real input range, and that is the point of reporting
        it: the requirement demands regulation the design method cannot yet
        provide.
        """

        if self.target_output_voltage_v is None or self.tolerance_fraction is None:
            return None
        return not self.regulation_violations

    @property
    def output_voltage_spread_v(self) -> float:
        """How far the delivered rail moves across the envelope.

        A fixed-duty design has no feedback loop, so this number is the line and
        load regulation it actually achieves.  It is reported rather than judged:
        whether the spread is acceptable is the requirement's business, but it
        must not be hidden behind a nominal-only figure.
        """

        return self.output_voltage_max_v - self.output_voltage_min_v

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "corner_count": self.corner_count,
            "efficiency": {
                "nominal": self.efficiency_nominal,
                "min": self.efficiency_min,
                "min_corner": self.efficiency_min_corner,
            },
            "loss_max_w": self.loss_max_w,
            "loss_max_corner": self.loss_max_corner,
            "input_current_max_a": self.input_current_max_a,
            "input_current_max_corner": self.input_current_max_corner,
            "duty": {
                "min": self.duty_min,
                "min_corner": self.duty_min_corner,
                "max": self.duty_max,
                "max_corner": self.duty_max_corner,
            },
            "ripple_max_mv": self.ripple_max_mv,
            "ripple_max_corner": self.ripple_max_corner,
            "output_voltage": {
                "nominal": self.nominal_output_voltage_v,
                "min": self.output_voltage_min_v,
                "min_corner": self.output_voltage_min_corner,
                "max": self.output_voltage_max_v,
                "max_corner": self.output_voltage_max_corner,
                "spread": self.output_voltage_spread_v,
                "target": self.target_output_voltage_v,
                "tolerance_fraction": self.tolerance_fraction,
                "regulates": self.regulates_output,
                "violations": [dict(item) for item in self.regulation_violations],
            },
            "duty_schedule": (
                self.duty_schedule.as_dict() if self.duty_schedule is not None else None
            ),
            "corners": [dict(item) for item in self.corner_results],
            "extensions": dict(self.extensions),
        }


def worst_case_summary(
    corners: Iterable[OperatingCorner],
    evaluate: Callable[[OperatingCorner], Mapping[str, Any]],
    *,
    target_output_voltage_v: float | None = None,
    tolerance_fraction: float | None = None,
    duty_schedule: DutySchedule | None = None,
) -> WorstCaseSummary:
    """Evaluate every corner and reduce the results to design-relevant extremes.

    ``evaluate`` must return at least ``efficiency``, ``loss_w``,
    ``input_current_a``, ``duty``, ``predicted_ripple_mv`` and
    ``output_voltage_v`` for its corner.  A corner that raises is not silently
    dropped: the exception propagates, because a design that cannot be evaluated
    somewhere in its envelope is a finding, not a gap to paper over.

    Supplying ``target_output_voltage_v`` and ``tolerance_fraction`` turns the
    delivered rail into an acceptance check rather than a bare number, which is
    what makes a fixed-duty design's lack of regulation visible.
    """

    corner_list = tuple(corners)
    if not corner_list:
        raise ValueError("worst-case analysis needs at least one corner")

    rows: list[dict[str, Any]] = []
    for corner in corner_list:
        result = dict(evaluate(corner))
        missing = {
            "efficiency",
            "loss_w",
            "input_current_a",
            "duty",
            "predicted_ripple_mv",
            "output_voltage_v",
        } - set(result)
        if missing:
            raise ValueError(
                f"corner {corner.corner_id!r} result is missing {sorted(missing)}"
            )
        rows.append({"corner": corner.as_dict(), **result})

    nominal = next((row for row in rows if row["corner"]["corner_id"] == "nominal"), rows[0])
    efficiency_min_row = min(rows, key=lambda row: float(row["efficiency"]))
    loss_max_row = max(rows, key=lambda row: float(row["loss_w"]))
    current_max_row = max(rows, key=lambda row: float(row["input_current_a"]))
    duty_max_row = max(rows, key=lambda row: float(row["duty"]))
    duty_min_row = min(rows, key=lambda row: float(row["duty"]))
    ripple_max_row = max(rows, key=lambda row: float(row["predicted_ripple_mv"]))
    voltage_min_row = min(rows, key=lambda row: float(row["output_voltage_v"]))
    voltage_max_row = max(rows, key=lambda row: float(row["output_voltage_v"]))

    def corner_id(row: Mapping[str, Any]) -> str:
        return str(dict(row["corner"])["corner_id"])

    violations: list[dict[str, Any]] = []
    if target_output_voltage_v is not None or tolerance_fraction is not None:
        if target_output_voltage_v is None or tolerance_fraction is None:
            raise ValueError(
                "regulation needs both target_output_voltage_v and tolerance_fraction"
            )
        if target_output_voltage_v <= 0.0:
            raise ValueError("target output voltage must be positive")
        if not 0.0 <= tolerance_fraction < 1.0:
            raise ValueError("tolerance_fraction must be within [0, 1)")
        low = target_output_voltage_v * (1.0 - tolerance_fraction)
        high = target_output_voltage_v * (1.0 + tolerance_fraction)
        for row in rows:
            delivered = float(row["output_voltage_v"])
            if not low <= delivered <= high:
                violations.append(
                    {
                        "corner_id": corner_id(row),
                        "delivered_v": delivered,
                        "allowed_min_v": low,
                        "allowed_max_v": high,
                        "error_fraction": (delivered - target_output_voltage_v)
                        / target_output_voltage_v,
                    }
                )

    return WorstCaseSummary(
        corner_count=len(rows),
        efficiency_min=float(efficiency_min_row["efficiency"]),
        efficiency_min_corner=corner_id(efficiency_min_row),
        efficiency_nominal=float(nominal["efficiency"]),
        loss_max_w=float(loss_max_row["loss_w"]),
        loss_max_corner=corner_id(loss_max_row),
        input_current_max_a=float(current_max_row["input_current_a"]),
        input_current_max_corner=corner_id(current_max_row),
        duty_max=float(duty_max_row["duty"]),
        duty_max_corner=corner_id(duty_max_row),
        duty_min=float(duty_min_row["duty"]),
        duty_min_corner=corner_id(duty_min_row),
        ripple_max_mv=float(ripple_max_row["predicted_ripple_mv"]),
        ripple_max_corner=corner_id(ripple_max_row),
        output_voltage_min_v=float(voltage_min_row["output_voltage_v"]),
        output_voltage_min_corner=corner_id(voltage_min_row),
        output_voltage_max_v=float(voltage_max_row["output_voltage_v"]),
        output_voltage_max_corner=corner_id(voltage_max_row),
        nominal_output_voltage_v=float(nominal["output_voltage_v"]),
        target_output_voltage_v=target_output_voltage_v,
        tolerance_fraction=tolerance_fraction,
        regulation_violations=tuple(violations),
        duty_schedule=duty_schedule,
        corner_results=tuple(rows),
    )


def envelope_from_mapping(
    spec: Mapping[str, Any],
    *,
    fallback_input_voltage_v: float | None = None,
) -> OperatingEnvelope | None:
    """Build an envelope from a PBDL spec's ``operating_envelope`` block.

    Returns ``None`` when the spec declares no envelope, so callers fall back to
    single-point design.  ``fallback_input_voltage_v`` supplies the nominal rail
    (usually the target's input voltage) so the envelope block only has to
    declare the *ranges*.
    """

    block = spec.get("operating_envelope")
    if not isinstance(block, Mapping):
        return None
    nominal = block.get("nominal_input_voltage_v", fallback_input_voltage_v)
    if nominal is None:
        return None
    merged = {**dict(block), "nominal_input_voltage_v": nominal}
    return OperatingEnvelope.from_dict(merged)
