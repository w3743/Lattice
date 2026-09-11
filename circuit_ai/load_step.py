"""Load-step transient response for averaged power stages.

A load-step requirement -- "the rail must not sag more than X mV when the load
steps from A to B, and must recover within T" -- is one of the most common real
power requirements, and the project had no way to express, evaluate or report it.
The only transient model was *startup*: the output ramping toward its DC target.

What this model is
------------------
A charge-balance model of one load step.  At the instant the load changes, the
output capacitor supplies the difference while the loop slews the converter's
delivery current to the new demand.  The dominant term is the missing charge

    Q = dI * tau / 2      (triangle: the delivery deficit ramps linearly to zero)

so the excursion is ``dI * tau / (2 * C)`` and the recovery time is the slew
time itself.  That is the standard first-order sizing relation, and it is what
makes the answer actionable: a requirement that misses its budget needs either
more capacitance or a faster loop, and the report says which term dominates.

What this model is not
----------------------
``loop_response_s`` is an *assumed* closed-loop slew time, not a designed one.
This is therefore a **feasibility** analysis: it answers "is this budget
achievable, and at what capacitance", never "is the loop stable".  Gain and phase
margin, crossover and conditional stability need a compensator design and a
small-signal loop model that this project does not have.  The record says so in
``justifies_stability_claim`` rather than leaving it to the reader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "LOAD_STEP_SCHEMA",
    "LOAD_STEP_VERSION",
    "LoadStepRequirement",
    "LoadStepResult",
    "evaluate_load_step",
    "load_step_from_mapping",
]


LOAD_STEP_SCHEMA = "circuit_ai.load_step"
LOAD_STEP_VERSION = 1


@dataclass(frozen=True)
class LoadStepRequirement:
    """A load step and the excursion budget it must stay inside.

    ``from_fraction`` and ``to_fraction`` are multiples of the target's nominal
    output current, so a requirement is expressed the way it is written: a step
    from 10 % to 100 % of rated load.
    """

    from_fraction: float
    to_fraction: float
    max_undershoot_mv: float | None = None
    max_overshoot_mv: float | None = None
    max_recovery_us: float | None = None
    loop_response_s: float = 50e-6
    schema: str = LOAD_STEP_SCHEMA
    schema_version: int = LOAD_STEP_VERSION

    def __post_init__(self) -> None:
        if self.schema != LOAD_STEP_SCHEMA or self.schema_version != LOAD_STEP_VERSION:
            raise ValueError("unsupported load step schema")
        if self.from_fraction < 0.0 or self.to_fraction < 0.0:
            raise ValueError("load step fractions must be non-negative")
        if self.from_fraction == self.to_fraction:
            raise ValueError("a load step must change the load")
        if self.loop_response_s <= 0.0:
            raise ValueError("loop response time must be positive")
        for name in ("max_undershoot_mv", "max_overshoot_mv", "max_recovery_us"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when given")
        if (
            self.max_undershoot_mv is None
            and self.max_overshoot_mv is None
            and self.max_recovery_us is None
        ):
            raise ValueError("a load step requirement must state at least one budget")

    @property
    def is_release(self) -> bool:
        """Whether the step removes load rather than adding it."""

        return self.to_fraction < self.from_fraction

    @property
    def is_step_up(self) -> bool:
        return self.to_fraction > self.from_fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "from_fraction": self.from_fraction,
            "to_fraction": self.to_fraction,
            "max_undershoot_mv": self.max_undershoot_mv,
            "max_overshoot_mv": self.max_overshoot_mv,
            "max_recovery_us": self.max_recovery_us,
            "loop_response_s": self.loop_response_s,
            "direction": "release" if self.is_release else "step_up",
        }


@dataclass(frozen=True)
class LoadStepResult:
    """The evaluated excursion, its budget verdict, and the sizing terms."""

    requirement: LoadStepRequirement
    nominal_output_current_a: float
    delta_current_a: float
    capacitance_f: float
    excursion_mv: float
    recovery_us: float
    settled: bool
    #: Always false: this is a feasibility model, not a stability analysis.
    justifies_stability_claim: bool = False
    schema: str = LOAD_STEP_SCHEMA
    schema_version: int = LOAD_STEP_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every stated budget is met."""

        requirement = self.requirement
        excursion_budget = (
            requirement.max_undershoot_mv
            if requirement.is_step_up
            else requirement.max_overshoot_mv
        )
        breached = []
        if excursion_budget is not None:
            breached.append(self.excursion_mv > excursion_budget)
        if requirement.max_recovery_us is not None:
            breached.append(self.recovery_us > requirement.max_recovery_us)
        return not any(breached)

    @property
    def capacitance_for_budget_f(self) -> float | None:
        """Capacitance that would exactly meet the stated excursion budget.

        This is the actionable number: it says how much larger the output
        capacitor must be, given the assumed loop response.
        """

        requirement = self.requirement
        budget_mv = (
            requirement.max_undershoot_mv
            if requirement.is_step_up
            else requirement.max_overshoot_mv
        )
        if budget_mv is None or budget_mv <= 0.0:
            return None
        return self.delta_current_a * requirement.loop_response_s / (2.0 * budget_mv * 1e-3)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "passed": self.passed,
            "delta_current_a": self.delta_current_a,
            "nominal_output_current_a": self.nominal_output_current_a,
            "capacitance_f": self.capacitance_f,
            "excursion_mv": self.excursion_mv,
            "recovery_us": self.recovery_us,
            "settled": self.settled,
            "capacitance_for_budget_f": self.capacitance_for_budget_f,
            "justifies_stability_claim": self.justifies_stability_claim,
            "requirement": self.requirement.as_dict(),
            "extensions": dict(self.extensions),
        }
        return payload


def evaluate_load_step(
    requirement: LoadStepRequirement,
    *,
    nominal_output_current_a: float,
    output_voltage_v: float,
    capacitance_f: float,
    max_steps: int = 20000,
) -> LoadStepResult:
    """Simulate one load step from the charge-balance model.

    The capacitor voltage is integrated directly rather than using the closed
    form, so the reported excursion and recovery come from the same trajectory
    and can be inspected.  The delivery current slews linearly from the old to
    the new demand over ``loop_response_s``, which is the model's central
    assumption and is stated in the result.
    """

    if nominal_output_current_a <= 0.0:
        raise ValueError("load step needs a positive nominal output current")
    if output_voltage_v <= 0.0:
        raise ValueError("load step needs a positive output voltage")
    if capacitance_f <= 0.0:
        raise ValueError("load step needs a positive output capacitance")

    old_current = nominal_output_current_a * requirement.from_fraction
    new_current = nominal_output_current_a * requirement.to_fraction
    delta_current = new_current - old_current

    # Integration grid: fine enough to resolve the slew, long enough to settle.
    tau = requirement.loop_response_s
    total_time = max(6.0 * tau, 1e-6)
    timestep = total_time / max(200, min(max_steps, 4000))
    steps = int(total_time / timestep)

    voltage = output_voltage_v
    times: list[float] = []
    voltages: list[float] = []
    for index in range(steps):
        time = (index + 1) * timestep
        # Delivery ramps to the new demand over the loop response, then holds.
        fraction = min(1.0, time / tau)
        delivered = old_current + delta_current * fraction
        # The capacitor supplies whatever the load draws beyond delivery.
        load = new_current
        d_voltage = (delivered - load) / capacitance_f
        voltage += timestep * d_voltage
        times.append(time)
        voltages.append(voltage)

    trace = np.asarray(voltages, dtype=float)
    times_array = np.asarray(times, dtype=float)
    deviation = trace - output_voltage_v
    if requirement.is_step_up:
        excursion = float(max(0.0, -np.min(deviation)))
    else:
        excursion = float(max(0.0, np.max(deviation)))

    # Recovery: the last moment the rail is outside a 1 % band of the target.
    band = 0.01 * output_voltage_v
    outside = np.flatnonzero(np.abs(deviation) > band)
    recovery_s = float(times_array[outside[-1]]) if outside.size else 0.0
    settled = bool(outside.size == 0 or outside[-1] < len(times_array) - 1)

    # Closed-form cross-check of the excursion, kept so a caller can see the
    # sizing relation the numeric trajectory is consistent with.
    expected_excursion = abs(delta_current) * tau / (2.0 * capacitance_f)

    return LoadStepResult(
        requirement=requirement,
        nominal_output_current_a=nominal_output_current_a,
        delta_current_a=delta_current,
        capacitance_f=capacitance_f,
        excursion_mv=excursion * 1000.0,
        recovery_us=recovery_s * 1e6,
        settled=settled,
        extensions={
            "charge_balance_excursion_mv": expected_excursion * 1000.0,
            "output_voltage_v": output_voltage_v,
            "timestep_s": timestep,
            "sample_count": len(times),
        },
    )


def _is_load_step_block(value: Any) -> bool:
    """Whether a mapping is itself a load-step block rather than a container.

    A block is recognised by its own step fields, so passing the block directly
    and passing a spec that contains it both work.  Getting this wrong is silent:
    a container lookup on a block simply finds nothing and the requirement is
    reported as "not stated", which looks like a user omission rather than a bug.
    """

    if not isinstance(value, Mapping):
        return False
    return any(
        key in value
        for key in (
            "from_fraction",
            "to_fraction",
            "max_undershoot_mv",
            "max_overshoot_mv",
            "max_recovery_us",
            "loop_response_s",
        )
    )


def load_step_from_mapping(spec: Mapping[str, Any]) -> LoadStepRequirement | None:
    """Read a load-step budget from a spec, or from the block itself.

    Returns ``None`` when nothing is stated, so a spec without a budget claims
    nothing about one.
    """

    if _is_load_step_block(spec):
        block: Any = spec
    else:
        block = spec.get("load_step")
    if not isinstance(block, Mapping):
        return None
    return LoadStepRequirement(
        from_fraction=float(block.get("from_fraction", 0.1)),
        to_fraction=float(block.get("to_fraction", 1.0)),
        max_undershoot_mv=_optional_float(block.get("max_undershoot_mv")),
        max_overshoot_mv=_optional_float(block.get("max_overshoot_mv")),
        max_recovery_us=_optional_float(block.get("max_recovery_us")),
        loop_response_s=float(block.get("loop_response_s", 50e-6)),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
