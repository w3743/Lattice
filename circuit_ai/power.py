"""Deterministic ideal power-stage solver and parameter optimizer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, ClassVar, Mapping

import numpy as np

from .experts import TopologyCandidate
from .ir import IRComponent, UnifiedIR
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


@dataclass(frozen=True)
class BuckOptimizationResult:
    candidate: TopologyCandidate
    parameters: PowerStageParameters
    operating_point: BoostOperatingPoint
    transient: TransientTrace
    optimizer_message: str
    objective: float
    optimization_run: OptimizationRunResult | None = None


@dataclass(frozen=True)
class SepicOptimizationResult:
    candidate: TopologyCandidate
    parameters: PowerStageParameters
    operating_point: BoostOperatingPoint
    transient: TransientTrace
    optimizer_message: str
    objective: float
    optimization_run: OptimizationRunResult | None = None


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


def solve_ideal_boost_dc(vin: float, parameters: BoostParameters) -> BoostOperatingPoint:
    if not 0.0 < parameters.duty_cycle < 1.0:
        raise ValueError("duty_cycle must be between 0 and 1")
    vout = vin / (1.0 - parameters.duty_cycle)
    iout = vout / parameters.load_ohm
    pout = vout * iout
    iin = pout / vin
    ripple = iout * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.capacitance_f
    )
    delta_i = vin * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.inductance_h
    )
    return BoostOperatingPoint(
        input_voltage_v=vin,
        output_voltage_v=vout,
        input_current_a=iin,
        output_current_a=iout,
        output_power_w=pout,
        efficiency=1.0,
        predicted_ripple_mv=ripple * 1000.0,
        inductor_ripple_a=delta_i,
    )


def solve_ideal_buck_dc(vin: float, parameters: BoostParameters) -> BoostOperatingPoint:
    if not 0.0 < parameters.duty_cycle < 1.0:
        raise ValueError("duty_cycle must be between 0 and 1")
    vout = vin * parameters.duty_cycle
    iout = vout / parameters.load_ohm
    pout = vout * iout
    iin = pout / vin
    ripple = iout * (1.0 - parameters.duty_cycle) / (
        parameters.switching_frequency_hz * parameters.capacitance_f
    )
    delta_i = (vin - vout) * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.inductance_h
    )
    return BoostOperatingPoint(
        input_voltage_v=vin,
        output_voltage_v=vout,
        input_current_a=iin,
        output_current_a=iout,
        output_power_w=pout,
        efficiency=1.0,
        predicted_ripple_mv=ripple * 1000.0,
        inductor_ripple_a=delta_i,
    )


def solve_ideal_sepic_dc(vin: float, parameters: BoostParameters) -> BoostOperatingPoint:
    if not 0.0 < parameters.duty_cycle < 1.0:
        raise ValueError("duty_cycle must be between 0 and 1")
    vout = vin * parameters.duty_cycle / (1.0 - parameters.duty_cycle)
    iout = vout / parameters.load_ohm
    pout = vout * iout
    iin = pout / vin
    ripple = iout * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.capacitance_f
    )
    delta_i = vin * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.inductance_h
    )
    return BoostOperatingPoint(
        input_voltage_v=vin,
        output_voltage_v=vout,
        input_current_a=iin,
        output_current_a=iout,
        output_power_w=pout,
        efficiency=1.0,
        predicted_ripple_mv=ripple * 1000.0,
        inductor_ripple_a=delta_i,
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


def solve_ideal_flyback_dc(vin: float, parameters: FlybackParameters) -> BoostOperatingPoint:
    if not 0.0 < parameters.duty_cycle < 1.0 or parameters.turns_ratio <= 0:
        raise ValueError("flyback duty_cycle must be between 0 and 1 and turns_ratio must be positive")
    vout = parameters.turns_ratio * vin * parameters.duty_cycle / (1.0 - parameters.duty_cycle)
    iout = vout / parameters.load_ohm
    pout = vout * iout
    iin = pout / vin
    ripple = iout * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.capacitance_f
    )
    delta_i = vin * parameters.duty_cycle / (
        parameters.switching_frequency_hz * parameters.inductance_h
    )
    return BoostOperatingPoint(
        input_voltage_v=vin,
        output_voltage_v=vout,
        input_current_a=iin,
        output_current_a=iout,
        output_power_w=pout,
        efficiency=1.0,
        predicted_ripple_mv=ripple * 1000.0,
        inductor_ripple_a=delta_i,
    )


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
        load = vout / iout
        duty_target = 1.0 - vin / vout
        ranges = ir.constraints.get("parameter_ranges", {})
        bounds = [
            _range_or(ranges, "duty_cycle", (max(0.05, duty_target - 0.25), min(0.95, duty_target + 0.25))),
            _range_or(ranges, "L", (10e-6, 2e-3)),
            _range_or(ranges, "C", (10e-6, 5e-3)),
            _range_or(ranges, "switching_frequency_hz", (20e3, 500e3)),
        ]
        ripple_limit = float(target.get("ripple_mv") or float("inf"))

        def objective(values) -> float:
            params = boost_parameters_from_values(values, load)
            dc = solve_ideal_boost_dc(vin, params)
            voltage_error = abs(dc.output_voltage_v - vout) / vout
            ripple_error = 0.0 if not math.isfinite(ripple_limit) else max(0.0, dc.predicted_ripple_mv / ripple_limit - 1.0)
            # Mild preferences keep the result practical without overriding
            # the declared electrical target.
            size_penalty = 0.002 * (math.log10(params.inductance_h / 100e-6) ** 2 + math.log10(params.capacitance_f / 100e-6) ** 2)
            return 100.0 * voltage_error**2 + 5.0 * ripple_error**2 + size_penalty

        run = _solve_power_problem(
            _power_problem(
                "dc_boost",
                bounds,
                initial_values={"duty_cycle": _bounded_initial(duty_target, bounds[0])},
            ),
            objective,
            ir.optimization,
        )
        params = boost_parameters_from_values(run.values, load)
        dc = solve_ideal_boost_dc(vin, params)
        transient = simulate_ideal_boost(vin, params)
        components = tuple(_materialize_component(component, params) for component in candidate.components)
        materialized_candidate = _candidate_with_materialized_graph(
            replace(candidate, components=components, graph=None),
            ir,
        )
        return BoostOptimizationResult(
            candidate=materialized_candidate,
            parameters=power_stage_parameters(params),
            operating_point=dc,
            transient=transient,
            optimizer_message=run.message,
            objective=float(run.objective_values["power.objective"]),
            optimization_run=run,
        )


class BuckParameterOptimizer:
    def optimize(self, ir: UnifiedIR, candidate: TopologyCandidate) -> BuckOptimizationResult:
        target = ir.primary_target
        vin, vout, iout = _target_power_values(ir)
        if vout >= vin:
            raise ValueError("ideal buck optimization needs output voltage below input voltage")
        duty_target = vout / vin
        params, dc, result = _optimize_four_parameter_converter(
            ir,
            vin,
            vout,
            iout,
            duty_target,
            solve_ideal_buck_dc,
            "dc_buck",
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
        )


class SepicParameterOptimizer:
    def optimize(self, ir: UnifiedIR, candidate: TopologyCandidate) -> SepicOptimizationResult:
        vin, vout, iout = _target_power_values(ir)
        duty_target = vout / (vin + vout)
        params, dc, result = _optimize_four_parameter_converter(
            ir,
            vin,
            vout,
            iout,
            duty_target,
            solve_ideal_sepic_dc,
            "dc_sepic",
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

        def objective(values) -> float:
            params = flyback_parameters_from_values(values, load)
            dc = solve_ideal_flyback_dc(vin, params)
            voltage_error = abs(dc.output_voltage_v - vout) / vout
            ripple_error = 0.0 if not math.isfinite(ripple_limit) else max(0.0, dc.predicted_ripple_mv / ripple_limit - 1.0)
            size_penalty = 0.002 * (math.log10(params.inductance_h / 100e-6) ** 2 + math.log10(params.capacitance_f / 100e-6) ** 2 + math.log10(params.turns_ratio) ** 2)
            return 100.0 * voltage_error**2 + 5.0 * ripple_error**2 + size_penalty

        run = _solve_power_problem(
            _power_problem(
                "isolated_flyback",
                bounds,
                include_turns_ratio=True,
                initial_values={
                    "duty_cycle": _bounded_initial(duty_target, bounds[0]),
                    "turns_ratio": initial_ratio,
                },
            ),
            objective,
            ir.optimization,
        )
        params = flyback_parameters_from_values(run.values, load)
        dc = solve_ideal_flyback_dc(vin, params)
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
        )


@dataclass(frozen=True)
class PowerStageModel:
    """One registered ideal power stage behind the unified parameter contract.

    Everything topology-specific stays in this module: the capability layer and
    simulator backends only see `parameter_type` and a two-argument `dc_solver`
    that both speak `PowerStageParameters`.
    """

    solver_id: str
    family: str
    optimizer_type: type
    dc_solver: Callable[[float, PowerStageParameters], BoostOperatingPoint]

    @property
    def parameter_type(self) -> type:
        return PowerStageParameters

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
    ),
    PowerStageModel(
        solver_id="ideal_boost_averaged",
        family="dc_boost",
        optimizer_type=BoostParameterOptimizer,
        dc_solver=_stage_dc_solver(boost_parameters_from_values, solve_ideal_boost_dc),
    ),
    PowerStageModel(
        solver_id="ideal_sepic_averaged",
        family="dc_sepic",
        optimizer_type=SepicParameterOptimizer,
        dc_solver=_stage_dc_solver(boost_parameters_from_values, solve_ideal_sepic_dc),
    ),
    PowerStageModel(
        solver_id="ideal_flyback_averaged",
        family="isolated_flyback",
        optimizer_type=FlybackParameterOptimizer,
        dc_solver=_stage_dc_solver(flyback_parameters_from_values, solve_ideal_flyback_dc),
    ),
)


def _target_power_values(ir: UnifiedIR) -> tuple[float, float, float]:
    target = ir.primary_target
    vin = float(target.get("input_voltage_v") or ir.operating_point.get("supply_voltage_v") or 0.0)
    vout = float(target.get("output_voltage_v") or 0.0)
    iout = float(target.get("output_current_a") or 0.0)
    if vin <= 0 or vout <= 0 or iout <= 0:
        raise ValueError("ideal power optimization needs positive input voltage, output voltage and output current")
    return vin, vout, iout


def _optimize_four_parameter_converter(
    ir: UnifiedIR,
    vin: float,
    vout: float,
    iout: float,
    duty_target: float,
    dc_solver,
    family: str,
):
    load = vout / iout
    ranges = ir.constraints.get("parameter_ranges", {})
    bounds = [
        _range_or(ranges, "duty_cycle", (max(0.05, duty_target - 0.25), min(0.95, duty_target + 0.25))),
        _range_or(ranges, "L", (10e-6, 2e-3)),
        _range_or(ranges, "C", (10e-6, 5e-3)),
        _range_or(ranges, "switching_frequency_hz", (20e3, 500e3)),
    ]
    ripple_limit = float(ir.primary_target.get("ripple_mv") or float("inf"))

    def objective(values) -> float:
        params = boost_parameters_from_values(values, load)
        dc = dc_solver(vin, params)
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

    result = _solve_power_problem(
        _power_problem(
            family,
            bounds,
            initial_values={"duty_cycle": _bounded_initial(duty_target, bounds[0])},
        ),
        objective,
        ir.optimization,
    )
    params = boost_parameters_from_values(result.values, load)
    return params, dc_solver(vin, params), result


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
