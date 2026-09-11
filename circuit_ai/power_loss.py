"""Composable conduction and switching loss models for averaged power stages.

Why this module exists
----------------------
The ideal averaged converters in :mod:`circuit_ai.power` report
``efficiency == 1.0`` because they model no loss mechanism at all.  That makes
"maximise efficiency" meaningless as a design objective: with no loss term the
optimiser has nothing to trade against, and two designs at 5 W and 50 W come out
identical.

This module adds a *parameterised* loss model that is deliberately
topology-agnostic.  A converter family contributes only three data points:

* how the switch node voltage relates to the input voltage
  (``switch_voltage_factor``),
* how much of the cycle the main switch conducts (``switch_duty``),
* how much of the cycle the rectifier conducts (``rectifier_duty``).

Everything else -- conduction, diode drop, switching transition, switch
capacitance and output-capacitor ESR loss -- is shared.  Adding a new converter
family is therefore a registration, not a new solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "DEFAULT_LOSS_PARAMETERS",
    "LOSS_MODEL_SCHEMA",
    "LOSS_MODEL_VERSION",
    "LossBreakdown",
    "LossParameters",
    "SwitchPathDuty",
    "default_switch_path_duty",
    "evaluate_loss_model",
    "loss_parameters_from_options",
]


LOSS_MODEL_SCHEMA = "circuit_ai.power_loss_model"
LOSS_MODEL_VERSION = 1


@dataclass(frozen=True)
class LossParameters:
    """Physical loss coefficients of one averaged converter design.

    Defaults model a small 30-60 V class MOSFET with a Schottky rectifier and a
    ferrite-cored inductor.  They are *typical*, not datasheet-exact: the point
    is to make efficiency a function of the design variables and the power
    level, which is what the ideal model could not do.
    """

    switch_resistance_ohm: float = 0.05
    inductor_resistance_ohm: float = 0.10
    rectifier_forward_v: float = 0.45
    rectifier_resistance_ohm: float = 0.02
    switching_transition_s: float = 50e-9
    switch_output_capacitance_f: float = 150e-12
    output_capacitor_esr_ohm: float = 0.02
    #: Magnetising/primary inductance losses are not modelled; magnetic core
    #: loss needs a core model that this project does not have yet.  Recorded
    #: explicitly so nobody reads the efficiency number as including it.
    core_loss_modelled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "switch_resistance_ohm",
            "inductor_resistance_ohm",
            "rectifier_forward_v",
            "rectifier_resistance_ohm",
            "switching_transition_s",
            "switch_output_capacitance_f",
            "output_capacitor_esr_ohm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0.0:
                raise ValueError(f"loss parameter {name} must be a non-negative number")

    def as_dict(self) -> dict[str, Any]:
        return {
            "switch_resistance_ohm": self.switch_resistance_ohm,
            "inductor_resistance_ohm": self.inductor_resistance_ohm,
            "rectifier_forward_v": self.rectifier_forward_v,
            "rectifier_resistance_ohm": self.rectifier_resistance_ohm,
            "switching_transition_s": self.switching_transition_s,
            "switch_output_capacitance_f": self.switch_output_capacitance_f,
            "output_capacitor_esr_ohm": self.output_capacitor_esr_ohm,
            "core_loss_modelled": self.core_loss_modelled,
        }


#: Model a 36 V / 2 A class design by default.  Chosen so that a 10 W isolated
#: converter lands in the high-80s to low-90s percent range, which is what a
#: Schottky-rectified flyback actually achieves.
DEFAULT_LOSS_PARAMETERS = LossParameters()


@dataclass(frozen=True)
class SwitchPathDuty:
    """Which portion of the cycle each conducting path carries.

    Family-specific, and the only topology knowledge the loss model needs.
    ``switch_voltage_v`` is the blocking voltage the main switch actually sees,
    which is *not* simply the output rail: a boost switch blocks the output, a
    flyback switch blocks the input plus the reflected output.
    """

    switch_duty: float
    rectifier_duty: float
    switch_voltage_v: float

    def __post_init__(self) -> None:
        for name in ("switch_duty", "rectifier_duty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not -1e-9 <= value <= 1.0 + 1e-9:
                raise ValueError(f"{name} must be within [0, 1]")
        if not isinstance(self.switch_voltage_v, (int, float)) or self.switch_voltage_v < 0.0:
            raise ValueError("switch_voltage_v must be non-negative")


def default_switch_path_duty(
    total_duty: float,
    *,
    switch_voltage_v: float,
) -> SwitchPathDuty:
    """Switch and rectifier split the cycle; the caller supplies the stress."""

    duty = min(max(float(total_duty), 0.0), 1.0)
    return SwitchPathDuty(
        switch_duty=duty,
        rectifier_duty=max(0.0, 1.0 - duty),
        switch_voltage_v=float(switch_voltage_v),
    )


@dataclass(frozen=True)
class LossBreakdown:
    """Per-mechanism loss in watts, plus the resulting efficiency."""

    output_power_w: float
    input_power_w: float
    total_loss_w: float
    efficiency: float
    switch_conduction_w: float = 0.0
    inductor_conduction_w: float = 0.0
    rectifier_conduction_w: float = 0.0
    switching_transition_w: float = 0.0
    switch_capacitance_w: float = 0.0
    capacitor_esr_w: float = 0.0
    schema: str = LOSS_MODEL_SCHEMA
    schema_version: int = LOSS_MODEL_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mechanisms(self) -> dict[str, float]:
        return {
            "switch_conduction": self.switch_conduction_w,
            "inductor_conduction": self.inductor_conduction_w,
            "rectifier_conduction": self.rectifier_conduction_w,
            "switching_transition": self.switching_transition_w,
            "switch_capacitance": self.switch_capacitance_w,
            "capacitor_esr": self.capacitor_esr_w,
        }

    @property
    def dominant_mechanism(self) -> str | None:
        items = {name: value for name, value in self.mechanisms.items() if value > 0.0}
        if not items:
            return None
        return max(items, key=lambda name: items[name])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "output_power_w": self.output_power_w,
            "input_power_w": self.input_power_w,
            "total_loss_w": self.total_loss_w,
            "efficiency": self.efficiency,
            "losses": self.mechanisms,
            "dominant_mechanism": self.dominant_mechanism,
            "extensions": dict(self.extensions),
        }


def evaluate_loss_model(
    *,
    input_voltage_v: float,
    output_power_w: float,
    output_voltage_v: float,
    output_current_a: float,
    duty: float,
    switching_frequency_hz: float,
    inductance_h: float,
    capacitance_f: float,
    inductor_ripple_a: float,
    path: SwitchPathDuty,
    parameters: LossParameters = DEFAULT_LOSS_PARAMETERS,
) -> LossBreakdown:
    """Estimate losses for one averaged operating point.

    The load is taken as fixed (a converter is specified by its output), so the
    input current follows from the loss estimate rather than the other way
    round.  One fixed-point refinement is applied because conduction loss
    depends on the current that the loss itself raises.  The input rail enters
    through the input current; the *switched* voltage enters through
    ``path.switch_voltage_v``, which each family supplies.
    """

    if input_voltage_v <= 0.0:
        raise ValueError("loss model requires a positive input voltage")
    if output_power_w <= 0.0:
        raise ValueError("loss model requires positive output power")
    if switching_frequency_hz <= 0.0:
        raise ValueError("loss model requires a positive switching frequency")
    if inductance_h <= 0.0 or capacitance_f <= 0.0:
        raise ValueError("loss model requires positive inductance and capacitance")

    def once(assumed_efficiency: float) -> tuple[float, dict[str, float]]:
        input_power = output_power_w / max(assumed_efficiency, 1e-6)
        input_current = input_power / input_voltage_v
        # Conduction path currents, as RMS over the whole cycle.  The switch
        # carries a chopped current, the rectifier carries the load current for
        # the complementary interval.
        switch_rms = input_current * (path.switch_duty**0.5)
        rectifier_avg = output_current_a * path.rectifier_duty
        inductor_rms = input_current

        switch_conduction = parameters.switch_resistance_ohm * switch_rms**2
        inductor_conduction = parameters.inductor_resistance_ohm * inductor_rms**2
        rectifier_conduction = (
            parameters.rectifier_forward_v * rectifier_avg
            + parameters.rectifier_resistance_ohm * rectifier_avg**2
        )

        switch_voltage = path.switch_voltage_v
        transition = (
            0.5
            * switch_voltage
            * input_current
            * parameters.switching_transition_s
            * switching_frequency_hz
        )
        capacitance = (
            0.5
            * parameters.switch_output_capacitance_f
            * switch_voltage**2
            * switching_frequency_hz
        )
        ripple_current = inductor_ripple_a / (12.0**0.5)
        capacitor_esr = parameters.output_capacitor_esr_ohm * ripple_current**2

        losses = {
            "switch_conduction_w": max(0.0, switch_conduction),
            "inductor_conduction_w": max(0.0, inductor_conduction),
            "rectifier_conduction_w": max(0.0, rectifier_conduction),
            "switching_transition_w": max(0.0, transition),
            "switch_capacitance_w": max(0.0, capacitance),
            "capacitor_esr_w": max(0.0, capacitor_esr),
        }
        return sum(losses.values()), losses

    first_loss, _ = once(1.0)
    total_loss, mechanisms = once(output_power_w / (output_power_w + first_loss))
    input_power = output_power_w + total_loss
    efficiency = output_power_w / input_power if input_power > 0.0 else 0.0
    return LossBreakdown(
        output_power_w=output_power_w,
        input_power_w=input_power,
        total_loss_w=total_loss,
        efficiency=efficiency,
        **mechanisms,
    )


def loss_parameters_from_options(options: Mapping[str, Any] | None) -> LossParameters:
    """Build loss parameters from the ``optimization.loss_model`` spec block.

    Only keys that are present override the defaults, so a spec can tune the
    switch resistance without restating the rest of the model.
    """

    if not options:
        return DEFAULT_LOSS_PARAMETERS
    overrides: dict[str, Any] = {}
    for name in (
        "switch_resistance_ohm",
        "inductor_resistance_ohm",
        "rectifier_forward_v",
        "rectifier_resistance_ohm",
        "switching_transition_s",
        "switch_output_capacitance_f",
        "output_capacitor_esr_ohm",
    ):
        if name in options and options[name] is not None:
            overrides[name] = float(options[name])
    if not overrides:
        return DEFAULT_LOSS_PARAMETERS
    return LossParameters(**{**DEFAULT_LOSS_PARAMETERS.as_dict(), **overrides, "core_loss_modelled": False})
