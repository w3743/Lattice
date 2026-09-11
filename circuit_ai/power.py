"""Deterministic ideal power-stage solver and parameter optimizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from .experts import TopologyCandidate
from .ir import IRComponent, UnifiedIR
from .operating_envelope import (
    DutySchedule,
    OperatingCorner,
    OperatingEnvelope,
    WorstCaseSummary,
    enumerate_corners,
    worst_case_summary,
)
from .optimization import (
    DifferentialEvolutionConfig,
    DifferentialEvolutionDriver,
    EvaluationBudget,
    OptimizationProblem,
    OptimizationRunResult,
    OptimizationVariable,
    VariableKind,
    VariableScale,
)
from .power_loss import (
    LossParameters,
    default_switch_path_duty,
    evaluate_loss_model,
    loss_parameters_from_options,
)


@dataclass(frozen=True)
class BoostParameters:
    duty_cycle: float
    inductance_h: float
    capacitance_f: float
    switching_frequency_hz: float
    load_ohm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "duty_cycle": self.duty_cycle,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "switching_frequency_hz": self.switching_frequency_hz,
            "load_ohm": self.load_ohm,
        }


@dataclass(frozen=True)
class BoostOperatingPoint:
    input_voltage_v: float
    output_voltage_v: float
    input_current_a: float
    output_current_a: float
    output_power_w: float
    efficiency: float
    predicted_ripple_mv: float
    inductor_ripple_a: float

    def as_dict(self) -> dict[str, float]:
        return {
            "input_voltage_v": self.input_voltage_v,
            "output_voltage_v": self.output_voltage_v,
            "input_current_a": self.input_current_a,
            "output_current_a": self.output_current_a,
            "output_power_w": self.output_power_w,
            "efficiency": self.efficiency,
            "predicted_ripple_mv": self.predicted_ripple_mv,
            "inductor_ripple_a": self.inductor_ripple_a,
        }


@dataclass(frozen=True)
class TransientTrace:
    time_s: tuple[float, ...]
    output_voltage_v: tuple[float, ...]
    inductor_current_a: tuple[float, ...]
    switch_on: tuple[bool, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": list(self.time_s),
            "output_voltage_v": list(self.output_voltage_v),
            "inductor_current_a": list(self.inductor_current_a),
            "switch_on": list(self.switch_on),
        }


@dataclass(frozen=True)
class BoostOptimizationResult:
    candidate: TopologyCandidate
    parameters: PowerStageParameters
    operating_point: BoostOperatingPoint
    transient: TransientTrace
    optimizer_message: str
    objective: float
    optimization_run: OptimizationRunResult | None = None
    #: Worst-case behaviour across the declared operating envelope.  ``None``
    #: when the spec declares no envelope, so single-point artifacts are
    #: unchanged.
    envelope_analysis: WorstCaseSummary | None = None


@dataclass(frozen=True)
class FlybackParameters:
    duty_cycle: float
    inductance_h: float
    capacitance_f: float
    switching_frequency_hz: float
    load_ohm: float
    turns_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "duty_cycle": self.duty_cycle,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "switching_frequency_hz": self.switching_frequency_hz,
            "load_ohm": self.load_ohm,
            "turns_ratio": self.turns_ratio,
        }


@dataclass(frozen=True)
class FlybackOptimizationResult:
    candidate: TopologyCandidate
    parameters: PowerStageParameters
    operating_point: BoostOperatingPoint
    transient: TransientTrace
    optimizer_message: str
    objective: float
    optimization_run: OptimizationRunResult | None = None
    envelope_analysis: WorstCaseSummary | None = None


@dataclass(frozen=True)
class BuckOptimizationResult:
    candidate: TopologyCandidate
    parameters: PowerStageParameters
    operating_point: BoostOperatingPoint
    transient: TransientTrace
    optimizer_message: str
    objective: float
    optimization_run: OptimizationRunResult | None = None
    envelope_analysis: WorstCaseSummary | None = None


@dataclass(frozen=True)
class SepicOptimizationResult:
    candidate: TopologyCandidate
    parameters: PowerStageParameters
    operating_point: BoostOperatingPoint
    transient: TransientTrace
    optimizer_message: str
    objective: float
    optimization_run: OptimizationRunResult | None = None
    envelope_analysis: WorstCaseSummary | None = None


# Unit and scale of every decision variable shared by the ideal power stages.
# The order is the optimizer coordinate order and must stay stable.
POWER_STAGE_VARIABLE_DEFINITIONS: tuple[tuple[str, str, VariableScale], ...] = (
    ("duty_cycle", "1", VariableScale.LINEAR),
    ("inductance_h", "H", VariableScale.LOG10),
    ("capacitance_f", "F", VariableScale.LOG10),
    ("switching_frequency_hz", "Hz", VariableScale.LOG10),
)

# Declared by every stage so the variable schema is identical across families.
# Non-isolated stages pin it with equal bounds; `OptimizationVariable.fixed`
# then keeps it out of the optimizer coordinate vector.
POWER_STAGE_TURNS_RATIO_DEFINITION: tuple[str, str, VariableScale] = (
    "turns_ratio",
    "1",
    VariableScale.LOG10,
)

POWER_STAGE_DERIVED_PARAMETERS: tuple[str, ...] = ("load_ohm",)


@dataclass(frozen=True)
class PowerStageParameters:
    """Unified parameter view of one ideal power stage.

    Field names are the shared optimization variable ids plus the derived load
    resistance, so the same record describes every registered stage. It is the
    neutral boundary shape: capability registration and simulator backends
    exchange this record instead of a topology family's own parameter class.

    `turns_ratio` is `None` when the stage has no transformer, which keeps the
    serialized stage parameters identical to the family-specific view they
    replace.
    """

    duty_cycle: float
    inductance_h: float
    capacitance_f: float
    switching_frequency_hz: float
    load_ohm: float
    turns_ratio: float | None = None

    # Consumed by `AnalyticPowerSimulatorBackend._parameters`: these fields may
    # be absent from a simulation request instead of being invented, so a stage
    # without a transformer keeps its original parameter payload.
    optional_parameter_fields: ClassVar[frozenset[str]] = frozenset({"turns_ratio"})

    def as_dict(self) -> dict[str, float]:
        values = {
            "duty_cycle": self.duty_cycle,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "switching_frequency_hz": self.switching_frequency_hz,
            "load_ohm": self.load_ohm,
        }
        if self.turns_ratio is not None:
            values["turns_ratio"] = self.turns_ratio
        return values


def power_stage_parameters(parameters: BoostParameters | FlybackParameters) -> PowerStageParameters:
    """Adapt one family parameter set into the unified stage parameter view.

    Non-isolated families carry no turns ratio, so it stays `None`; downstream
    consumers never branch on family to decide whether it applies.
    """
    if isinstance(parameters, FlybackParameters):
        return PowerStageParameters(
            duty_cycle=parameters.duty_cycle,
            inductance_h=parameters.inductance_h,
            capacitance_f=parameters.capacitance_f,
            switching_frequency_hz=parameters.switching_frequency_hz,
            load_ohm=parameters.load_ohm,
            turns_ratio=parameters.turns_ratio,
        )
    return PowerStageParameters(
        duty_cycle=parameters.duty_cycle,
        inductance_h=parameters.inductance_h,
        capacitance_f=parameters.capacitance_f,
        switching_frequency_hz=parameters.switching_frequency_hz,
        load_ohm=parameters.load_ohm,
    )


def boost_parameters_from_values(
    values: Mapping[str, Any], load_ohm: float
) -> BoostParameters:
    """Adapt unified variable values into the non-isolated family parameters.

    This is the inverse of `power_stage_parameters` for the buck, boost and
    SEPIC stages, which all share the same physical parameter set.
    """
    return BoostParameters(
        float(values["duty_cycle"]),
        float(values["inductance_h"]),
        float(values["capacitance_f"]),
        float(values["switching_frequency_hz"]),
        load_ohm,
    )


def flyback_parameters_from_values(
    values: Mapping[str, Any], load_ohm: float
) -> FlybackParameters:
    """Adapt unified variable values into the isolated flyback parameters.

    Unlike the non-isolated stages this consumes `turns_ratio`, which is a free
    decision variable only when the stage declares a real turns-ratio range.
    """
    return FlybackParameters(
        float(values["duty_cycle"]),
        float(values["inductance_h"]),
        float(values["capacitance_f"]),
        float(values["switching_frequency_hz"]),
        load_ohm,
        float(values["turns_ratio"]),
    )


def _ideal_stage_dc(
    vin: float,
    duty_cycle: float,
    inductance_h: float,
    capacitance_f: float,
    switching_frequency_hz: float,
    load_ohm: float,
    voltage_from_input: Callable[[float, float], float],
    *,
    ripple_factor: Callable[[float, float], float],
    inductor_ripple_factor: Callable[[float, float], float],
    loss_parameters: LossParameters | None = None,
    switch_voltage_v: float = 0.0,
    duty_transform: Callable[[float], float] | None = None,
) -> BoostOperatingPoint:
    """One averaged-stage operating point for any converter family.

    The families in this module differ only in their ideal conversion ratio and
    in how the switch current paths divide the cycle -- every output quantity
    below has the same form for all of them.  Keeping that in one place is what
    makes a new family a registration rather than a fifth near-duplicate solver.

    Each family supplies ``voltage_from_input(vin, duty)`` and its ripple factors
    written to reproduce the original arithmetic exactly, rather than an
    algebraically equal rearrangement.  ``vin / (1 - duty)`` and
    ``vin * (1 / (1 - duty))`` differ in the last bit, and these numbers feed a
    stochastic optimiser whose discrete selection then changes; the golden
    characterization lock detects exactly that.
    """

    if not 0.0 < duty_cycle < 1.0:
        raise ValueError("duty_cycle must be between 0 and 1")
    if switching_frequency_hz <= 0 or capacitance_f <= 0 or inductance_h <= 0:
        raise ValueError("switching frequency, capacitance and inductance must be positive")
    vout = voltage_from_input(vin, duty_cycle)
    iout = vout / load_ohm
    pout = vout * iout
    ripple = iout * ripple_factor(duty_cycle, vin) / (
        switching_frequency_hz * capacitance_f
    )
    delta_i = inductor_ripple_factor(duty_cycle, vin) / (
        switching_frequency_hz * inductance_h
    )
    efficiency = 1.0
    if loss_parameters is not None:
        path = default_switch_path_duty(
            duty_transform(duty_cycle) if duty_transform is not None else duty_cycle,
            switch_voltage_v=switch_voltage_v,
        )
        breakdown = evaluate_loss_model(
            input_voltage_v=vin,
            output_power_w=pout,
            output_voltage_v=vout,
            output_current_a=iout,
            duty=duty_cycle,
            switching_frequency_hz=switching_frequency_hz,
            inductance_h=inductance_h,
            capacitance_f=capacitance_f,
            inductor_ripple_a=delta_i,
            path=path,
            parameters=loss_parameters,
        )
        efficiency = breakdown.efficiency
    # Input current follows the delivered power and the estimated efficiency.
    iin = pout / (vin * efficiency) if efficiency > 0.0 else float("inf")
    return BoostOperatingPoint(
        input_voltage_v=vin,
        output_voltage_v=vout,
        input_current_a=iin,
        output_current_a=iout,
        output_power_w=pout,
        efficiency=efficiency,
        predicted_ripple_mv=ripple * 1000.0,
        inductor_ripple_a=delta_i,
    )


def solve_ideal_boost_dc(
    vin: float,
    parameters: BoostParameters,
    *,
    loss_parameters: LossParameters | None = None,
) -> BoostOperatingPoint:
    return _ideal_stage_dc(
        vin,
        parameters.duty_cycle,
        parameters.inductance_h,
        parameters.capacitance_f,
        parameters.switching_frequency_hz,
        parameters.load_ohm,
        voltage_from_input=lambda rail, duty: rail / (1.0 - duty),
        ripple_factor=lambda duty, rail: duty,
        inductor_ripple_factor=lambda duty, rail: rail * duty,
        loss_parameters=loss_parameters,
        switch_voltage_v=vin / (1.0 - parameters.duty_cycle),
    )


def solve_ideal_buck_dc(
    vin: float,
    parameters: BoostParameters,
    *,
    loss_parameters: LossParameters | None = None,
) -> BoostOperatingPoint:
    return _ideal_stage_dc(
        vin,
        parameters.duty_cycle,
        parameters.inductance_h,
        parameters.capacitance_f,
        parameters.switching_frequency_hz,
        parameters.load_ohm,
        voltage_from_input=lambda rail, duty: rail * duty,
        ripple_factor=lambda duty, rail: 1.0 - duty,
        # Kept as the original (vin - vout) form: `vin * (1 - duty)` is
        # algebraically identical but not bit-identical.
        inductor_ripple_factor=lambda duty, rail: (rail - rail * duty) * duty,
        loss_parameters=loss_parameters,
        switch_voltage_v=vin,
    )


def solve_ideal_sepic_dc(
    vin: float,
    parameters: BoostParameters,
    *,
    loss_parameters: LossParameters | None = None,
) -> BoostOperatingPoint:
    return _ideal_stage_dc(
        vin,
        parameters.duty_cycle,
        parameters.inductance_h,
        parameters.capacitance_f,
        parameters.switching_frequency_hz,
        parameters.load_ohm,
        voltage_from_input=lambda rail, duty: rail * duty / (1.0 - duty),
        ripple_factor=lambda duty, rail: duty,
        inductor_ripple_factor=lambda duty, rail: rail * duty,
        loss_parameters=loss_parameters,
        # A SEPIC switch blocks the input plus the output rail.
        switch_voltage_v=vin + vin * parameters.duty_cycle / (1.0 - parameters.duty_cycle),
    )


def solve_ideal_flyback_dc(
    vin: float,
    parameters: FlybackParameters,
    *,
    loss_parameters: LossParameters | None = None,
) -> BoostOperatingPoint:
    if parameters.turns_ratio <= 0:
        raise ValueError("flyback turns_ratio must be positive")
    return _ideal_stage_dc(
        vin,
        parameters.duty_cycle,
        parameters.inductance_h,
        parameters.capacitance_f,
        parameters.switching_frequency_hz,
        parameters.load_ohm,
        voltage_from_input=lambda rail, duty: parameters.turns_ratio
        * rail
        * duty
        / (1.0 - duty),
        ripple_factor=lambda duty, rail: duty,
        inductor_ripple_factor=lambda duty, rail: rail * duty,
        loss_parameters=loss_parameters,
        # The flyback switch blocks the input plus the reflected output voltage.
        switch_voltage_v=vin
        + vin
        * parameters.turns_ratio
        * parameters.duty_cycle
        / (1.0 - parameters.duty_cycle),
    )


def simulate_ideal_boost(
    vin: float,
    parameters: BoostParameters,
    *,
    cycles: int = 120,
    steps_per_cycle: int = 40,
) -> TransientTrace:
    """Integrate the piecewise-linear CCM equations with explicit Euler."""
    steps_per_cycle = max(8, int(steps_per_cycle))
    cycles = max(1, int(cycles))
    dt = 1.0 / (parameters.switching_frequency_hz * steps_per_cycle)
    total_steps = cycles * steps_per_cycle
    voltage = 0.0
    current = 0.0
    times: list[float] = []
    voltages: list[float] = []
    currents: list[float] = []
    states: list[bool] = []
    on_steps = max(1, min(steps_per_cycle - 1, round(parameters.duty_cycle * steps_per_cycle)))
    for step in range(total_steps):
        on = step % steps_per_cycle < on_steps
        if on:
            d_current = vin / parameters.inductance_h
            d_voltage = -voltage / (parameters.load_ohm * parameters.capacitance_f)
        else:
            d_current = (vin - voltage) / parameters.inductance_h
            d_voltage = (current - voltage / parameters.load_ohm) / parameters.capacitance_f
        current = max(0.0, current + dt * d_current)
        voltage = max(0.0, voltage + dt * d_voltage)
        times.append((step + 1) * dt)
        voltages.append(voltage)
        currents.append(current)
        states.append(on)
    return TransientTrace(tuple(times), tuple(voltages), tuple(currents), tuple(states))


def simulate_ideal_averaged_converter(
    vin: float,
    parameters: BoostParameters,
    dc_solver,
    *,
    cycles: int = 120,
    steps_per_cycle: int = 40,
) -> TransientTrace:
    """Stable averaged startup trace for ideal topology-level candidates."""
    steps_per_cycle = max(8, int(steps_per_cycle))
    cycles = max(1, int(cycles))
    dt = 1.0 / (parameters.switching_frequency_hz * steps_per_cycle)
    target = dc_solver(vin, parameters)
    voltage = 0.0
    current = 0.0
    voltage_tau = max(parameters.load_ohm * parameters.capacitance_f, 10.0 * dt)
    current_tau = max(parameters.inductance_h / max(parameters.load_ohm, 1e-12), 10.0 * dt)
    times: list[float] = []
    voltages: list[float] = []
    currents: list[float] = []
    states: list[bool] = []
    on_steps = max(1, min(steps_per_cycle - 1, round(parameters.duty_cycle * steps_per_cycle)))
    for step in range(cycles * steps_per_cycle):
        voltage += dt * (target.output_voltage_v - voltage) / voltage_tau
        current += dt * (target.input_current_a - current) / current_tau
        times.append((step + 1) * dt)
        voltages.append(max(0.0, voltage))
        currents.append(max(0.0, current))
        states.append(step % steps_per_cycle < on_steps)
    return TransientTrace(tuple(times), tuple(voltages), tuple(currents), tuple(states))


def simulate_ideal_flyback(
    vin: float,
    parameters: FlybackParameters,
    *,
    cycles: int = 120,
    steps_per_cycle: int = 40,
) -> TransientTrace:
    """Piecewise-linear ideal flyback approximation for architecture-level design."""
    steps_per_cycle = max(8, int(steps_per_cycle))
    cycles = max(1, int(cycles))
    dt = 1.0 / (parameters.switching_frequency_hz * steps_per_cycle)
    total_steps = cycles * steps_per_cycle
    voltage = 0.0
    primary_current = 0.0
    times: list[float] = []
    voltages: list[float] = []
    currents: list[float] = []
    states: list[bool] = []
    on_steps = max(1, min(steps_per_cycle - 1, round(parameters.duty_cycle * steps_per_cycle)))
    for step in range(total_steps):
        on = step % steps_per_cycle < on_steps
        if on:
            d_current = vin / parameters.inductance_h
            d_voltage = -voltage / (parameters.load_ohm * parameters.capacitance_f)
        else:
            d_current = -voltage / (parameters.turns_ratio * parameters.inductance_h)
            d_voltage = (
                parameters.turns_ratio * primary_current
                - voltage / parameters.load_ohm
            ) / parameters.capacitance_f
        primary_current = max(0.0, primary_current + dt * d_current)
        voltage = max(0.0, voltage + dt * d_voltage)
        times.append((step + 1) * dt)
        voltages.append(voltage)
        currents.append(primary_current)
        states.append(on)
    return TransientTrace(tuple(times), tuple(voltages), tuple(currents), tuple(states))


class BoostParameterOptimizer:
    def optimize(self, ir: UnifiedIR, candidate: TopologyCandidate) -> BoostOptimizationResult:
        target = ir.primary_target
        vin = float(target.get("input_voltage_v") or ir.operating_point.get("supply_voltage_v") or 0.0)
        vout = float(target.get("output_voltage_v") or 0.0)
        iout = float(target.get("output_current_a") or 0.0)
        if vin <= 0 or vout <= vin or iout <= 0:
            raise ValueError("ideal boost optimization needs positive vin, vout > vin and output current")
        duty_target = 1.0 - vin / vout
        loss_parameters = _loss_model_for(ir)
        design_rail, _rail_name = _design_corner_rail(ir, vin)
        params, dc, run = _optimize_four_parameter_converter(
            ir,
            vin,
            vout,
            iout,
            duty_target,
            solve_ideal_boost_dc,
            "dc_boost",
            exact_duty=lambda rail, out: _exact_duty_for_ratio(out, rail),
            loss_parameters=loss_parameters,
            design_rail_v=design_rail if loss_parameters is not None else None,
        )
        transient = simulate_ideal_boost(vin, params)
        return BoostOptimizationResult(
            candidate=_materialize_candidate(candidate, params, ir),
            parameters=power_stage_parameters(params),
            operating_point=dc,
            transient=transient,
            optimizer_message=run.message,
            objective=float(run.objective_values["power.objective"]),
            optimization_run=run,
            envelope_analysis=_stage_envelope_analysis(
                ir,
                params,
                nominal_vout=vout,
                nominal_iout=iout,
                dc_solver=solve_ideal_boost_dc,
                loss_parameters=loss_parameters,
                duty_schedule=_duty_schedule_for(ir, params, vout=vout, iout=iout),
            ),
        )


class BuckParameterOptimizer:
    def optimize(self, ir: UnifiedIR, candidate: TopologyCandidate) -> BuckOptimizationResult:
        vin, vout, iout = _target_power_values(ir)
        if vout >= vin:
            raise ValueError("ideal buck optimization needs output voltage below input voltage")
        duty_target = vout / vin
        loss_parameters = _loss_model_for(ir)
        design_rail, _rail_name = _design_corner_rail(ir, vin)
        params, dc, result = _optimize_four_parameter_converter(
            ir,
            vin,
            vout,
            iout,
            duty_target,
            solve_ideal_buck_dc,
            "dc_buck",
            # The buck ratio is exact and linear: vout = vin * d.
            exact_duty=lambda rail, out: out / rail,
            loss_parameters=loss_parameters,
            design_rail_v=design_rail if loss_parameters is not None else None,
        )
        transient = simulate_ideal_averaged_converter(vin, params, solve_ideal_buck_dc)
        return BuckOptimizationResult(
            candidate=_materialize_candidate(candidate, params, ir),
            parameters=power_stage_parameters(params),
            operating_point=dc,
            transient=transient,
            optimizer_message=result.message,
            objective=float(result.objective_values["power.objective"]),
            optimization_run=result,
            envelope_analysis=_stage_envelope_analysis(
                ir,
                params,
                nominal_vout=vout,
                nominal_iout=iout,
                dc_solver=solve_ideal_buck_dc,
                loss_parameters=_loss_model_for(ir),
                duty_schedule=_duty_schedule_for(ir, params, vout=vout, iout=iout),
            ),
        )


class SepicParameterOptimizer:
    def optimize(self, ir: UnifiedIR, candidate: TopologyCandidate) -> SepicOptimizationResult:
        vin, vout, iout = _target_power_values(ir)
        duty_target = vout / (vin + vout)
        loss_parameters = _loss_model_for(ir)
        design_rail, _rail_name = _design_corner_rail(ir, vin)
        params, dc, result = _optimize_four_parameter_converter(
            ir,
            vin,
            vout,
            iout,
            duty_target,
            solve_ideal_sepic_dc,
            "dc_sepic",
            # vout = vin * d / (1 - d), i.e. gain = 1 (the `vin` argument is the rail).
            exact_duty=lambda rail, out: _exact_duty_for_ratio(out, rail),
            loss_parameters=loss_parameters,
            design_rail_v=design_rail if loss_parameters is not None else None,
        )
        transient = simulate_ideal_averaged_converter(vin, params, solve_ideal_sepic_dc)
        return SepicOptimizationResult(
            candidate=_materialize_candidate(candidate, params, ir),
            parameters=power_stage_parameters(params),
            operating_point=dc,
            transient=transient,
            optimizer_message=result.message,
            objective=float(result.objective_values["power.objective"]),
            optimization_run=result,
            envelope_analysis=_stage_envelope_analysis(
                ir,
                params,
                nominal_vout=vout,
                nominal_iout=iout,
                dc_solver=solve_ideal_sepic_dc,
                loss_parameters=_loss_model_for(ir),
                duty_schedule=_duty_schedule_for(ir, params, vout=vout, iout=iout),
            ),
        )


class FlybackParameterOptimizer:
    def optimize(self, ir: UnifiedIR, candidate: TopologyCandidate) -> FlybackOptimizationResult:
        target = ir.primary_target
        vin = float(target.get("input_voltage_v") or ir.operating_point.get("supply_voltage_v") or 0.0)
        vout = float(target.get("output_voltage_v") or 0.0)
        iout = float(target.get("output_current_a") or 0.0)
        if vin <= 0 or vout <= 0 or iout <= 0:
            raise ValueError("ideal flyback optimization needs positive vin, vout and output current")
        load = vout / iout
        ranges = ir.constraints.get("parameter_ranges", {})
        ratio_bounds = _range_or(ranges, "turns_ratio", (0.5, 2.0))
        initial_ratio = _bounded_initial(1.0, ratio_bounds)
        duty_target = vout / (initial_ratio * vin + vout)
        bounds = [
            _range_or(ranges, "duty_cycle", (max(0.05, duty_target - 0.25), min(0.95, duty_target + 0.25))),
            _range_or(ranges, "L", (10e-6, 2e-3)),
            _range_or(ranges, "C", (10e-6, 5e-3)),
            _range_or(ranges, "switching_frequency_hz", (20e3, 500e3)),
            ratio_bounds,
        ]
        ripple_limit = float(target.get("ripple_mv") or float("inf"))
        loss_parameters = _loss_model_for(ir)
        bounds = _loss_aware_frequency_bounds(bounds, loss_parameters=loss_parameters)
        # The rail the design is scheduled at: the conversion ratio is worst at
        # low line, so sizing there is what lets the target be reached across the
        # whole input range instead of only at nominal.
        design_rail, _rail_name = _design_corner_rail(ir, vin)
        # With a loss model the conversion ratio is pinned per choice of turns
        # ratio, so the duty cycle is solved for rather than searched.
        fixed_duty = (
            _exact_duty_for_ratio(vout, design_rail, gain=initial_ratio)
            if loss_parameters is not None
            else None
        )
        if fixed_duty is not None:
            bounds[0] = (max(0.02, fixed_duty - 1e-9), min(0.98, fixed_duty + 1e-9))

        def base_objective(values, dc: BoostOperatingPoint) -> float:
            params = flyback_parameters_from_values(values, load)
            voltage_error = abs(dc.output_voltage_v - vout) / vout
            ripple_error = 0.0 if not math.isfinite(ripple_limit) else max(0.0, dc.predicted_ripple_mv / ripple_limit - 1.0)
            size_penalty = 0.002 * (math.log10(params.inductance_h / 100e-6) ** 2 + math.log10(params.capacitance_f / 100e-6) ** 2 + math.log10(params.turns_ratio) ** 2)
            return 100.0 * voltage_error**2 + 5.0 * ripple_error**2 + size_penalty

        objective = _stage_efficiency_objective(ir, base_objective, weight=None)

        def solve_stage(values) -> BoostOperatingPoint:
            params = flyback_parameters_from_values(values, load)
            if loss_parameters is None:
                return solve_ideal_flyback_dc(vin, params)
            # The duty cycle must track the candidate's own turns ratio at the
            # rail the design is scheduled at.
            duty = _exact_duty_for_ratio(vout, design_rail, gain=params.turns_ratio)
            tracked = replace(params, duty_cycle=duty)
            return solve_ideal_flyback_dc(
                design_rail, tracked, loss_parameters=loss_parameters
            )

        def evaluate(values) -> float:
            return objective(values, solve_stage(values))

        run = _solve_power_problem(
            _power_problem(
                "isolated_flyback",
                bounds,
                include_turns_ratio=True,
                initial_values={
                    "duty_cycle": _bounded_initial(
                        fixed_duty if fixed_duty is not None else duty_target, bounds[0]
                    ),
                    "turns_ratio": initial_ratio,
                },
            ),
            evaluate,
            ir.optimization,
        )
        params = flyback_parameters_from_values(run.values, load)
        if loss_parameters is not None:
            params = replace(
                params,
                duty_cycle=_exact_duty_for_ratio(vout, design_rail, gain=params.turns_ratio),
            )
        dc = solve_ideal_flyback_dc(
            vin, params, loss_parameters=loss_parameters
        )
        transient = simulate_ideal_flyback(vin, params)
        components = tuple(_materialize_component(component, params) for component in candidate.components)
        materialized_candidate = _candidate_with_materialized_graph(
            replace(candidate, components=components, graph=None),
            ir,
        )
        return FlybackOptimizationResult(
            candidate=materialized_candidate,
            parameters=power_stage_parameters(params),
            operating_point=dc,
            transient=transient,
            optimizer_message=run.message,
            objective=float(run.objective_values["power.objective"]),
            optimization_run=run,
            envelope_analysis=_stage_envelope_analysis(
                ir,
                params,
                nominal_vout=vout,
                nominal_iout=iout,
                dc_solver=solve_ideal_flyback_dc,
                loss_parameters=loss_parameters,
                duty_schedule=_duty_schedule_for(ir, params, vout=vout, iout=iout),
            ),
        )


@dataclass(frozen=True)
class PowerStageModel:
    """One registered ideal power stage behind the unified parameter contract.

    Everything topology-specific stays in this module: the capability layer and
    simulator backends only see `parameter_type` and a two-argument `dc_solver`
    that both speak `PowerStageParameters`.

    ``conversion_ratio`` is the family's ideal ratio as a function of duty and
    turns ratio.  Everything else about whether the family can serve a request --
    which target ratios are reachable -- is derived from it, so a new family
    states one function rather than a table of ranges that could drift from the
    solver it describes.
    """

    solver_id: str
    family: str
    optimizer_type: type
    dc_solver: Callable[[float, PowerStageParameters], BoostOperatingPoint]
    conversion_ratio: Callable[[float, float], float] | None = None
    #: Duty cycle outside this band is not usable in a real converter: below it
    #: the controller has no resolution, above it there is no off-time to reset
    #: the magnetics.
    duty_limits: tuple[float, float] = (0.05, 0.95)
    #: Turns-ratio range to consider when the family is isolated.
    turns_ratio_limits: tuple[float, float] | None = None

    @property
    def parameter_type(self) -> type:
        return PowerStageParameters

    def reachable_ratio_range(self) -> tuple[float, float] | None:
        """Conversion ratios this family can produce inside its usable duty band.

        ``None`` when the family has not declared a ratio, in which case no
        reachability claim is made about it.
        """

        if self.conversion_ratio is None:
            return None
        low_duty, high_duty = self.duty_limits
        if self.turns_ratio_limits is None:
            ratios = [
                self.conversion_ratio(duty, 1.0)
                for duty in (low_duty, high_duty)
            ]
        else:
            low_ratio, high_ratio = self.turns_ratio_limits
            ratios = [
                self.conversion_ratio(duty, turns)
                for duty in (low_duty, high_duty)
                for turns in (low_ratio, high_ratio)
            ]
        finite = [value for value in ratios if math.isfinite(value) and value > 0.0]
        if not finite:
            return None
        return min(finite), max(finite)

    def can_reach_ratio(self, target_ratio: float) -> bool:
        """Whether *target_ratio* (vout/vin) is inside the reachable band.

        Families with no declared ratio are treated as unconstrained, so an
        undeclared family is never rejected on a claim it did not make.
        """

        bounds = self.reachable_ratio_range()
        if bounds is None:
            return True
        low, high = bounds
        # A small tolerance keeps boundary-exact requests (a SEPIC at d = 0.5 for
        # unity gain, say) from being rejected by floating-point noise.
        return low * (1.0 - 1e-9) <= target_ratio <= high * (1.0 + 1e-9)

    @property
    def optimizer_capability_id(self) -> str:
        return f"power.optimizer.{self.solver_id}"

    @property
    def simulation_capability_id(self) -> str:
        return f"simulation.power.{self.solver_id}"


def _stage_dc_solver(adapter, solver) -> Callable[[float, PowerStageParameters], BoostOperatingPoint]:
    """Bind a unified record to one family equation set through its adapter."""

    def solve(vin: float, parameters: PowerStageParameters) -> BoostOperatingPoint:
        return solver(vin, adapter(parameters.as_dict(), parameters.load_ohm))

    return solve


# The single production list of ideal power stages. Capability registration
# iterates this tuple, so adding a stage is a data change here rather than a
# new family branch in the main flow.
POWER_STAGE_MODELS: tuple[PowerStageModel, ...] = (
    PowerStageModel(
        solver_id="ideal_buck_averaged",
        family="dc_buck",
        optimizer_type=BuckParameterOptimizer,
        dc_solver=_stage_dc_solver(boost_parameters_from_values, solve_ideal_buck_dc),
        # vout = vin * d : step-down only.
        conversion_ratio=lambda duty, turns: duty,
    ),
    PowerStageModel(
        solver_id="ideal_boost_averaged",
        family="dc_boost",
        optimizer_type=BoostParameterOptimizer,
        dc_solver=_stage_dc_solver(boost_parameters_from_values, solve_ideal_boost_dc),
        # vout = vin / (1 - d) : step-up only.
        conversion_ratio=lambda duty, turns: 1.0 / (1.0 - duty),
    ),
    PowerStageModel(
        solver_id="ideal_sepic_averaged",
        family="dc_sepic",
        optimizer_type=SepicParameterOptimizer,
        dc_solver=_stage_dc_solver(boost_parameters_from_values, solve_ideal_sepic_dc),
        # vout = vin * d / (1 - d) : spans step-down and step-up through unity.
        conversion_ratio=lambda duty, turns: duty / (1.0 - duty),
    ),
    PowerStageModel(
        solver_id="ideal_flyback_averaged",
        family="isolated_flyback",
        optimizer_type=FlybackParameterOptimizer,
        dc_solver=_stage_dc_solver(flyback_parameters_from_values, solve_ideal_flyback_dc),
        # vout = n * vin * d / (1 - d).
        conversion_ratio=lambda duty, turns: turns * duty / (1.0 - duty),
        turns_ratio_limits=(0.05, 4.0),
    ),
)


def stage_model_by_solver_id(solver_id: str) -> PowerStageModel | None:
    """Look up a registered stage by the id a knowledge production declares."""

    wanted = str(solver_id)
    for stage in POWER_STAGE_MODELS:
        if stage.solver_id == wanted:
            return stage
    return None


def _target_power_values(ir: UnifiedIR) -> tuple[float, float, float]:
    target = ir.primary_target
    vin = float(target.get("input_voltage_v") or ir.operating_point.get("supply_voltage_v") or 0.0)
    vout = float(target.get("output_voltage_v") or 0.0)
    iout = float(target.get("output_current_a") or 0.0)
    if vin <= 0 or vout <= 0 or iout <= 0:
        raise ValueError("ideal power optimization needs positive input voltage, output voltage and output current")
    return vin, vout, iout


def _exact_duty_for_ratio(vout: float, vin: float, *, offset: float = 0.0, gain: float = 1.0) -> float:
    """Invert an ideal boost-like ratio ``vout = gain * vin * d / (1 - d) + offset``.

    Used to drive the conversion ratio to its target exactly when a loss model
    is active: with a fixed load the output voltage is then pinned, which is
    what makes the loss estimate -- and therefore the efficiency objective --
    physically meaningful instead of a function of a mis-set duty cycle.
    """

    numerator = vout - offset
    denominator = gain * vin + numerator
    if denominator == 0.0:
        raise ValueError("conversion ratio is degenerate for this operating point")
    duty = numerator / denominator
    if not 0.0 < duty < 1.0:
        raise ValueError(f"conversion ratio yields an unrealisable duty cycle {duty!r}")
    return duty


#: Rails a design can be scheduled at.  ``nominal`` reproduces the historical
#: behaviour exactly; the extremes exist because a converter that only meets its
#: target at nominal does not meet it at all in the field.
#:
#: ``geometric_mean`` is the interesting one.  A fixed-ratio design's output
#: scales with input, so scheduling at ``input_min`` makes the target exact at
#: low line and too high everywhere above it.  Centring the ratio instead
#: minimises the *worst* deviation across the range: for a 24-48 V input the
#: extremes then sit at +-41 % in ratio terms rather than -33 %/+50 %.
DESIGN_CORNERS: tuple[str, ...] = ("nominal", "input_min", "input_max", "geometric_mean")


def _design_corner_rail(ir: UnifiedIR, nominal_vin: float) -> tuple[float, str]:
    """The input rail a design is scheduled at, and which rail that is.

    Without a feedback loop no fixed ratio can hold a target across a wide input
    range.  This chooses where to centre the design so the requirement is met as
    well as an open-loop build can meet it, and the envelope analysis reports
    what is left over rather than hiding it.
    """

    options = dict(ir.optimization.get("design_corner", {}) or {})
    raw = options.get("rail", "nominal") if isinstance(options, Mapping) else "nominal"
    rail = str(raw)
    if rail not in DESIGN_CORNERS:
        raise ValueError(
            f"design_corner.rail must be one of {sorted(DESIGN_CORNERS)}, got {rail!r}"
        )
    if rail == "nominal":
        return nominal_vin, rail
    envelope = _envelope_for(ir)
    if envelope is None:
        raise ValueError(
            f"design_corner.rail={rail!r} needs an operating_envelope to take the rail from"
        )
    if rail == "input_min":
        return envelope.low_input_voltage_v, rail
    if rail == "input_max":
        return envelope.high_input_voltage_v, rail
    return math.sqrt(envelope.low_input_voltage_v * envelope.high_input_voltage_v), rail


def _stage_efficiency_objective(
    ir: UnifiedIR,
    base: Callable[[Any, BoostOperatingPoint], float],
    *,
    weight: float | None,
) -> Callable[[Any, BoostOperatingPoint], float]:
    """Add a loss term to a stage objective when the loss model is enabled.

    The weight defaults to the spec's ``weights.efficiency`` so a user asking
    for maximum efficiency gets it without a second knob to discover.
    """

    resolved = weight
    if resolved is None:
        weights = dict(ir.optimization.get("weights", {}) or {})
        raw = weights.get("efficiency")
        resolved = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0.0
    if resolved == 0.0:
        return base

    def objective(values: Any, dc: BoostOperatingPoint) -> float:
        return base(values, dc) + resolved * (1.0 - dc.efficiency)

    return objective


def _loss_model_for(ir: UnifiedIR) -> LossParameters | None:
    """Read ``optimization.loss_model``; absent or disabled means lossless."""

    options = ir.optimization.get("loss_model", {}) or {}
    if not bool(options.get("enabled", False)):
        return None
    return loss_parameters_from_options(options)


#: With only switching loss modelled (no core loss), the optimiser would drive
#: the switching frequency to its lower bound because that reduces transition
#: and capacitance loss for free.  Real magnetics make low frequency expensive:
#: core loss falls with frequency but the core must then be larger.  Until a
#: core model exists, a floor keeps the result in a buildable range.
_LOSS_MODEL_MIN_FREQUENCY_HZ = 50e3


def _loss_aware_frequency_bounds(
    bounds: list[tuple[float, float]],
    *,
    loss_parameters: LossParameters | None,
) -> list[tuple[float, float]]:
    if loss_parameters is None:
        return bounds
    lower, upper = bounds[3]
    floor = min(_LOSS_MODEL_MIN_FREQUENCY_HZ, upper)
    if lower >= floor:
        return bounds
    updated = list(bounds)
    updated[3] = (floor, upper)
    return updated


def _envelope_for(ir: UnifiedIR) -> OperatingEnvelope | None:
    """Rebuild the declared operating envelope from the IR, if any."""

    return OperatingEnvelope.from_dict(ir.operating_envelope)


#: The duty range a real controller can act over.  Below it there is no
#: resolution, above it there is no off-time to reset the magnetics.  Kept in
#: step with ``PowerStageModel.duty_limits``.
_DUTY_BAND: tuple[float, float] = (0.05, 0.95)


def _duty_schedule_for(
    ir: UnifiedIR,
    design: Any,
    *,
    vout: float,
    iout: float,
) -> DutySchedule | None:
    """Compute the per-corner duty a controller would need, when one is declared."""

    envelope = _envelope_for(ir)
    if envelope is None or envelope.is_degenerate:
        return None
    return _achievable_duty_schedule(
        design,
        nominal_vout=vout,
        nominal_iout=iout,
        dc_solver=None,
        loss_parameters=None,
        corners=enumerate_corners(envelope),
        duty_limits=_DUTY_BAND,
    )


def _achievable_duty_schedule(
    design: Any,
    *,
    nominal_vout: float,
    nominal_iout: float,
    dc_solver,
    loss_parameters: LossParameters | None,
    corners: tuple[OperatingCorner, ...],
    duty_limits: tuple[float, float],
) -> DutySchedule | None:
    """What duty cycle each corner would need to hold the target rail.

    This is the control authority the topology has, stated separately from
    whether it is exercised.  The stored design keeps one fixed duty, so it
    cannot hold the rail by itself; but if every corner's *required* duty is
    inside the usable band, then a controller that schedules duty against the
    input can hold the rail, and the design is realisable.

    It proves feasibility and says nothing about stability: whether a loop can
    actually be compensated needs a stability analysis this project does not
    have, and the returned record states that explicitly.
    """

    vout = float(nominal_vout)
    ratio = float(getattr(design, "turns_ratio", 1.0))
    low, high = duty_limits
    entries: list[dict[str, Any]] = []
    for corner in corners:
        vin = float(corner.input_voltage_v)
        # vout = ratio * vin * d / (1 - d)  ->  d = vout / (ratio * vin + vout)
        denominator = ratio * vin + vout
        if denominator == 0.0:
            return None
        duty = vout / denominator
        entries.append(
            {
                "corner_id": corner.corner_id,
                "input_voltage_v": vin,
                "load_fraction": corner.load_fraction,
                "required_duty": duty,
                "inside_usable_band": bool(low <= duty <= high),
                "headroom_to_upper": float(high - duty),
                "headroom_to_lower": float(duty - low),
            }
        )
    _ = (nominal_iout, dc_solver, loss_parameters)  # load does not move the ratio
    return DutySchedule(
        entries=tuple(entries),
        usable_band=(low, high),
        control_effort=(min(e["required_duty"] for e in entries), max(e["required_duty"] for e in entries)),
    )


def _stage_envelope_analysis(
    ir: UnifiedIR,
    design: Any,
    *,
    nominal_vout: float,
    nominal_iout: float,
    dc_solver,
    loss_parameters: LossParameters | None,
    duty_schedule: DutySchedule | None = None,
) -> WorstCaseSummary | None:
    """Evaluate one chosen design at every corner of its declared envelope.

    The design keeps its component values and its *nominal* duty cycle; only the
    operating point moves.  Efficiency is recomputed at each corner from the loss
    model, so a design that looks good at nominal but poor at full load or at low
    line becomes visible instead of being averaged away.

    Re-solving the duty cycle per corner would describe a different (adaptive)
    design, so it is deliberately not done here: the point is to stress one
    build.  The duty a controller *would* need is reported separately by
    :func:`_achievable_duty_schedule`, so feasibility and regulation stay
    distinct claims.

    Load is treated as a constant-current demand scaled by the corner's load
    fraction, matching how a target declares ``output_current_a``.  The
    resulting output voltage therefore moves with the conversion ratio, which is
    exactly what makes the minimum-input corner the informative one.
    """

    envelope = _envelope_for(ir)
    if envelope is None or envelope.is_degenerate:
        return None

    duty = float(design.duty_cycle)
    inductance_h = float(design.inductance_h)
    capacitance_f = float(design.capacitance_f)
    switching_frequency_hz = float(design.switching_frequency_hz)
    turns_ratio = float(getattr(design, "turns_ratio", 1.0))

    def evaluate(corner: OperatingCorner) -> dict[str, float]:
        vin = corner.input_voltage_v
        iout = nominal_iout * corner.load_fraction
        load_ohm = nominal_vout / iout
        params = _flyback_parameters(duty, inductance_h, capacitance_f, switching_frequency_hz, load_ohm, turns_ratio)
        dc = (
            dc_solver(vin, params, loss_parameters=loss_parameters)
            if loss_parameters is not None
            else dc_solver(vin, params)
        )
        loss_w = (
            dc.output_power_w * (1.0 / dc.efficiency - 1.0)
            if dc.efficiency > 0.0
            else 0.0
        )
        return {
            "efficiency": dc.efficiency,
            "loss_w": loss_w,
            "input_current_a": dc.input_current_a,
            "duty": duty,
            "predicted_ripple_mv": dc.predicted_ripple_mv,
            "output_voltage_v": dc.output_voltage_v,
            "output_current_a": dc.output_current_a,
            "input_voltage_v": vin,
            "load_fraction": corner.load_fraction,
        }

    return worst_case_summary(
        enumerate_corners(envelope),
        evaluate,
        target_output_voltage_v=nominal_vout,
        tolerance_fraction=_output_tolerance_fraction(ir),
        duty_schedule=duty_schedule,
    )


def _output_tolerance_fraction(ir: UnifiedIR) -> float:
    """Relative band the delivered rail may move within across the envelope.

    Defaults to 5 %, the usual starting point for a regulated supply.  A spec
    can widen or tighten it with
    ``optimization.worst_case.output_tolerance_fraction``.
    """

    options = dict(ir.optimization.get("worst_case", {}) or {})
    raw = options.get("output_tolerance_fraction", 0.05)
    value = float(raw)
    if not 0.0 <= value < 1.0:
        raise ValueError("output_tolerance_fraction must be within [0, 1)")
    return value


def _flyback_parameters(
    duty_cycle: float,
    inductance_h: float,
    capacitance_f: float,
    switching_frequency_hz: float,
    load_ohm: float,
    turns_ratio: float,
) -> FlybackParameters:
    """The superset parameter record; the non-isolated solvers ignore turns_ratio."""

    return FlybackParameters(
        duty_cycle,
        inductance_h,
        capacitance_f,
        switching_frequency_hz,
        load_ohm,
        turns_ratio,
    )


def _optimize_four_parameter_converter(
    ir: UnifiedIR,
    vin: float,
    vout: float,
    iout: float,
    duty_target: float,
    dc_solver,
    family: str,
    *,
    exact_duty: Callable[[float, float], float] | None = None,
    loss_parameters: LossParameters | None = None,
    design_rail_v: float | None = None,
):
    load = vout / iout
    # The rail the design is scheduled at.  It only differs from the nominal
    # rail when the spec asks for it, and it only moves the duty cycle: the
    # stored load, and therefore the power level, stay nominal.
    rail = float(design_rail_v) if design_rail_v is not None else vin
    ranges = ir.constraints.get("parameter_ranges", {})
    bounds = _loss_aware_frequency_bounds(
        [
            _range_or(ranges, "duty_cycle", (max(0.05, duty_target - 0.25), min(0.95, duty_target + 0.25))),
            _range_or(ranges, "L", (10e-6, 2e-3)),
            _range_or(ranges, "C", (10e-6, 5e-3)),
            _range_or(ranges, "switching_frequency_hz", (20e3, 500e3)),
        ],
        loss_parameters=loss_parameters,
    )
    ripple_limit = float(ir.primary_target.get("ripple_mv") or float("inf"))
    # With a loss model the conversion ratio is pinned by construction, so the
    # duty cycle is no longer a free variable to be penalised into place.
    solve_stage = (
        (lambda values, params: dc_solver(rail, params, loss_parameters=loss_parameters))
        if loss_parameters is not None
        else (lambda values, params: dc_solver(vin, params))
    )
    scheduled_duty = (
        exact_duty(rail, vout) if (loss_parameters is not None and exact_duty is not None) else None
    )
    if scheduled_duty is not None:
        bounds[0] = (max(0.02, scheduled_duty - 1e-9), min(0.98, scheduled_duty + 1e-9))

    def base_objective(values, dc: BoostOperatingPoint) -> float:
        params = boost_parameters_from_values(values, load)
        voltage_error = abs(dc.output_voltage_v - vout) / vout
        ripple_error = (
            0.0
            if not math.isfinite(ripple_limit)
            else max(0.0, dc.predicted_ripple_mv / ripple_limit - 1.0)
        )
        size_penalty = 0.002 * (
            math.log10(params.inductance_h / 100e-6) ** 2
            + math.log10(params.capacitance_f / 100e-6) ** 2
        )
        return 100.0 * voltage_error**2 + 5.0 * ripple_error**2 + size_penalty

    objective = _stage_efficiency_objective(ir, base_objective, weight=None)

    def evaluate(values) -> float:
        params = boost_parameters_from_values(values, load)
        return objective(values, solve_stage(values, params))

    result = _solve_power_problem(
        _power_problem(
            family,
            bounds,
            initial_values={"duty_cycle": _bounded_initial(
                scheduled_duty if scheduled_duty is not None else duty_target,
                bounds[0],
            )},
        ),
        evaluate,
        ir.optimization,
    )
    params = boost_parameters_from_values(result.values, load)
    return params, solve_stage(result.values, params), result


def _power_problem(
    family: str,
    bounds: list[tuple[float, float]],
    *,
    include_turns_ratio: bool = False,
    initial_values: Mapping[str, float] | None = None,
) -> OptimizationProblem:
    """Build the shared ideal-power-stage variable schema.

    Every stage declares `duty_cycle`, `inductance_h`, `capacitance_f` and
    `switching_frequency_hz` with the same units, scales and order. Isolated
    stages additionally declare a free `turns_ratio`; non-isolated stages omit
    it entirely so their request identity and stored artifacts are unchanged,
    and their unified parameter view carries the neutral identity value 1.0.
    """
    definitions = list(POWER_STAGE_VARIABLE_DEFINITIONS)
    if include_turns_ratio:
        definitions.append(POWER_STAGE_TURNS_RATIO_DEFINITION)
    if len(bounds) != len(definitions):
        raise ValueError("power optimization bounds do not match variable definitions")
    variables = tuple(
        OptimizationVariable(
            variable_id=name,
            kind=VariableKind.CONTINUOUS,
            unit=unit,
            lower=lower,
            upper=upper,
            scale=scale,
            initial=(initial_values or {}).get(name),
            source_path=f"power.{family}.{name}",
        )
        for (name, unit, scale), (lower, upper) in zip(definitions, bounds)
    )
    return OptimizationProblem(
        problem_id=f"power.{family}",
        variables=variables,
        objective_ids=("power.objective",),
        metadata={"family": family, "fidelity": "ideal_averaged"},
    )


def power_stage_optimization_problem(
    family: str,
    *,
    duty_cycle: tuple[float, float],
    inductance_h: tuple[float, float],
    capacitance_f: tuple[float, float],
    switching_frequency_hz: tuple[float, float],
    turns_ratio: tuple[float, float] | None = None,
    initial_values: Mapping[str, float] | None = None,
) -> OptimizationProblem:
    """Public entry point for the shared ideal-power-stage variable schema.

    Pass `turns_ratio` to declare the isolated-stage variable; omit it for the
    non-isolated stages.
    """
    bounds = [duty_cycle, inductance_h, capacitance_f, switching_frequency_hz]
    if turns_ratio is not None:
        bounds.append(turns_ratio)
    return _power_problem(
        family,
        bounds,
        include_turns_ratio=turns_ratio is not None,
        initial_values=initial_values,
    )


def _bounded_initial(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return min(max(value, lower), upper)


def _solve_power_problem(
    problem: OptimizationProblem,
    objective: Callable[[Mapping[str, Any]], float],
    options: Mapping[str, Any],
) -> OptimizationRunResult:
    settings = dict(options)
    nested = settings.get("optimizer")
    if isinstance(nested, dict):
        de = nested.get("differential_evolution")
        if isinstance(de, dict):
            settings.update(de)
    config = DifferentialEvolutionConfig(
        max_iterations=_option_int(settings, "max_iterations", 18, minimum=0),
        popsize=_option_int(settings, "popsize", 8, minimum=1),
        tol=_option_float(settings, "tol", 1e-7, minimum=0.0),
        polish=_option_bool(settings, "polish", True),
        updating=_option_choice(settings, "updating", "immediate", {"immediate", "deferred"}),
        workers=_option_int(settings, "workers", 1, allow_negative=True),
        seed=_option_int(settings, "seed", 7),
        cache=_option_bool(settings, "cache", True),
        budget=EvaluationBudget(
            _option_optional_int(settings, "max_evaluations", minimum=1),
            _option_optional_float(settings, "max_time_s", minimum=0.0, exclusive=True),
        ),
    )
    return DifferentialEvolutionDriver().solve(
        problem,
        objective,
        config=config,
        truth_evaluator=objective,
    )


def _materialize_candidate(
    candidate: TopologyCandidate,
    parameters: BoostParameters,
    ir: UnifiedIR,
) -> TopologyCandidate:
    materialized = replace(
        candidate,
        components=tuple(
            _materialize_component(component, parameters)
            for component in candidate.components
        ),
        graph=None,
    )
    return _candidate_with_materialized_graph(materialized, ir)


def _candidate_with_materialized_graph(
    candidate: TopologyCandidate,
    ir: UnifiedIR,
) -> TopologyCandidate:
    from .graph import topology_candidate_to_graph

    return replace(candidate, graph=topology_candidate_to_graph(candidate, ir))


def _range_or(ranges: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = ranges.get(key)
    if value is None:
        value = ranges.get(key.lower())
    if value is None:
        return default
    lo, hi = float(value[0]), float(value[1])
    if lo <= 0 or hi <= lo:
        raise ValueError(f"invalid parameter range for {key}: {value}")
    return lo, hi


def _option_int(
    options: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    allow_negative: bool = False,
) -> int:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"optimization.{key} must be an integer")
    if not allow_negative and value < 0:
        raise ValueError(f"optimization.{key} must not be negative")
    if minimum is not None and value < minimum:
        raise ValueError(f"optimization.{key} must be at least {minimum}")
    return value


def _option_optional_int(
    options: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int | None:
    if key not in options or options[key] is None:
        return None
    return _option_int(options, key, 0, minimum=minimum)


def _option_float(
    options: Mapping[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    exclusive: bool = False,
) -> float:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"optimization.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"optimization.{key} must be finite")
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        comparison = "greater than" if exclusive else "at least"
        raise ValueError(f"optimization.{key} must be {comparison} {minimum}")
    return result


def _option_optional_float(
    options: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float | None:
    if key not in options or options[key] is None:
        return None
    return _option_float(options, key, 0.0, minimum=minimum, exclusive=exclusive)


def _option_bool(options: Mapping[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"optimization.{key} must be a boolean")
    return value


def _option_choice(
    options: Mapping[str, Any],
    key: str,
    default: str,
    choices: set[str],
) -> str:
    value = options.get(key, default)
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"optimization.{key} must be one of: {allowed}")
    return value


def _materialize_component(component: IRComponent, parameters: BoostParameters | FlybackParameters) -> IRComponent:
    if component.kind == "L":
        return IRComponent(component.name, component.kind, component.nodes, {"value": parameters.inductance_h}, component.attributes)
    if component.kind == "C":
        return IRComponent(component.name, component.kind, component.nodes, {"value": parameters.capacitance_f}, component.attributes)
    if component.kind == "R":
        return IRComponent(component.name, component.kind, component.nodes, {"value": parameters.load_ohm}, component.attributes)
    if component.kind == "ideal_switch":
        return IRComponent(component.name, component.kind, component.nodes, {"duty_cycle": parameters.duty_cycle, "frequency_hz": parameters.switching_frequency_hz}, component.attributes)
    if component.kind == "ideal_transformer" and isinstance(parameters, FlybackParameters):
        return IRComponent(component.name, component.kind, component.nodes, {"turns_ratio": parameters.turns_ratio}, component.attributes)
    return component
