"""Unified power-stage parameter contract across the optimizer and backends.

These tests defend the phase 5 requirement that the main flow depends on one
parameter schema instead of per-family parameter classes: the same
`OptimizationProblem` variable schema must be consumable by the generic
differential-evolution driver and by every registered ideal power stage, and
the capability layer must expose only the topology-neutral parameter record.
"""

from __future__ import annotations

import pytest

from circuit_ai.capabilities import build_default_registry
from circuit_ai.capabilities.contracts import CapabilityRole
from circuit_ai.optimization import (
    DifferentialEvolutionConfig,
    DifferentialEvolutionDriver,
)
from circuit_ai.power import (
    POWER_STAGE_MODELS,
    PowerStageParameters,
    boost_parameters_from_values,
    flyback_parameters_from_values,
    power_stage_optimization_problem,
    power_stage_parameters,
    solve_ideal_boost_dc,
    solve_ideal_flyback_dc,
)

NON_ISOLATED_RANGE = (0.05, 0.95)
INDUCTANCE_RANGE = (10e-6, 2e-3)
CAPACITANCE_RANGE = (10e-6, 5e-3)
FREQUENCY_RANGE = (20e3, 500e3)
TURNS_RATIO_RANGE = (0.5, 2.0)


def _stage_problem(family: str, *, isolated: bool):
    """Build the stage problem through the public shared-schema entry point."""
    return power_stage_optimization_problem(
        family,
        duty_cycle=NON_ISOLATED_RANGE,
        inductance_h=INDUCTANCE_RANGE,
        capacitance_f=CAPACITANCE_RANGE,
        switching_frequency_hz=FREQUENCY_RANGE,
        turns_ratio=TURNS_RATIO_RANGE if isolated else None,
    )


@pytest.mark.parametrize("stage", POWER_STAGE_MODELS, ids=lambda s: s.solver_id)
def test_shared_variable_schema_is_consumed_by_the_generic_de_driver(stage) -> None:
    isolated = stage.family == "isolated_flyback"
    problem = _stage_problem(stage.family, isolated=isolated)

    def objective(values) -> float:
        reference = values["duty_cycle"]
        return (reference - 0.5) ** 2

    run = DifferentialEvolutionDriver().solve(
        problem,
        objective,
        config=DifferentialEvolutionConfig(max_iterations=3, popsize=4, seed=3),
        truth_evaluator=objective,
    )

    # Every declared variable of the one shared schema decodes back to a value.
    assert set(run.values) == {variable.variable_id for variable in problem.variables}
    assert run.truth_evaluated
    # Decoded values stay inside the declared physical bounds of the schema.
    for variable in problem.variables:
        value = float(run.values[variable.variable_id])
        assert variable.lower - 1e-12 <= value <= variable.upper + 1e-12


@pytest.mark.parametrize("stage", POWER_STAGE_MODELS, ids=lambda s: s.solver_id)
def test_one_parameter_record_is_accepted_by_every_registered_stage(stage) -> None:
    """One decoded record must be a valid input for all four stage solvers."""
    problem = _stage_problem(stage.family, isolated=stage.family == "isolated_flyback")
    coordinates = problem.initial_vector
    values = problem.decode(coordinates)

    record = PowerStageParameters(
        duty_cycle=float(values["duty_cycle"]),
        inductance_h=float(values["inductance_h"]),
        capacitance_f=float(values["capacitance_f"]),
        switching_frequency_hz=float(values["switching_frequency_hz"]),
        load_ohm=100.0,
        turns_ratio=float(values["turns_ratio"]) if "turns_ratio" in values else None,
    )

    operating_point = stage.dc_solver(12.0, record)

    assert operating_point.input_voltage_v == 12.0
    assert operating_point.output_voltage_v > 0.0
    assert operating_point.efficiency == 1.0


def test_registered_stages_only_expose_the_unified_parameter_record() -> None:
    registry = build_default_registry()
    parameter_types = {}
    for registration in registry.registrations():
        if registration.target.role is not CapabilityRole.SIMULATION_BACKEND:
            continue
        implementation = registration.implementation
        if not hasattr(implementation, "parameter_type"):
            continue
        capability_id = registration.target.capability_id
        if capability_id.startswith("simulation.power."):
            parameter_types[capability_id] = implementation.parameter_type

    assert parameter_types, "expected registered ideal power simulation adapters"
    assert {item.__name__ for item in parameter_types.values()} == {
        PowerStageParameters.__name__
    }
    # The family classes stay inside circuit_ai.power; the capability boundary
    # must not leak them to consumers.
    assert all(item is PowerStageParameters for item in parameter_types.values())


def test_registered_stage_registry_matches_the_four_registered_capabilities() -> None:
    registry = build_default_registry()
    registered_ids = {
        registration.target.capability_id
        for registration in registry.registrations()
        if registration.target.role is CapabilityRole.PARAMETER_OPTIMIZER
    }

    assert registered_ids == {
        stage.optimizer_capability_id for stage in POWER_STAGE_MODELS
    }
    # Family and solver literals are derived from the registry, not duplicated.
    families = {
        registration.target.families
        for registration in registry.registrations()
        if registration.target.role is CapabilityRole.PARAMETER_OPTIMIZER
    }
    assert families == {frozenset({stage.family}) for stage in POWER_STAGE_MODELS}


def test_adapters_round_trip_every_physical_parameter_value() -> None:
    non_isolated = boost_parameters_from_values(
        {
            "duty_cycle": 0.4,
            "inductance_h": 220e-6,
            "capacitance_f": 47e-6,
            "switching_frequency_hz": 250e3,
        },
        load_ohm=50.0,
    )
    isolated = flyback_parameters_from_values(
        {
            "duty_cycle": 0.35,
            "inductance_h": 150e-6,
            "capacitance_f": 33e-6,
            "switching_frequency_hz": 120e3,
            "turns_ratio": 1.75,
        },
        load_ohm=20.0,
    )

    non_isolated_view = power_stage_parameters(non_isolated)
    isolated_view = power_stage_parameters(isolated)

    assert non_isolated_view.duty_cycle == 0.4
    assert non_isolated_view.inductance_h == 220e-6
    assert non_isolated_view.capacitance_f == 47e-6
    assert non_isolated_view.switching_frequency_hz == 250e3
    assert non_isolated_view.load_ohm == 50.0
    # A stage without a transformer does not invent a turns ratio.
    assert non_isolated_view.turns_ratio is None

    assert isolated_view.turns_ratio == 1.75
    assert isolated_view.load_ohm == 20.0

    # The unified view drives the same physics as the family parameter object.
    assert solve_ideal_boost_dc(10.0, non_isolated).output_voltage_v == pytest.approx(
        10.0 / (1.0 - non_isolated_view.duty_cycle)
    )
    assert solve_ideal_flyback_dc(10.0, isolated).output_voltage_v == pytest.approx(
        isolated_view.turns_ratio * 10.0 * isolated_view.duty_cycle / (1.0 - isolated_view.duty_cycle)
    )


def test_turns_ratio_is_used_only_when_the_stage_declares_it() -> None:
    """The optional transformer variable changes physics only for isolated stages."""
    base = {
        "duty_cycle": 0.3,
        "inductance_h": 100e-6,
        "capacitance_f": 100e-6,
        "switching_frequency_hz": 100e3,
        "load_ohm": 10.0,
    }
    flyback = next(
        stage for stage in POWER_STAGE_MODELS if stage.family == "isolated_flyback"
    )
    boost = next(stage for stage in POWER_STAGE_MODELS if stage.family == "dc_boost")

    record = PowerStageParameters(**base, turns_ratio=2.0)

    isolated_vout = flyback.dc_solver(5.0, record).output_voltage_v
    non_isolated_vout = boost.dc_solver(5.0, record).output_voltage_v

    assert isolated_vout == pytest.approx(2.0 * 5.0 * 0.3 / 0.7)
    # The boost stage ignores the declared turns ratio entirely.
    assert non_isolated_vout == pytest.approx(5.0 / 0.7)
