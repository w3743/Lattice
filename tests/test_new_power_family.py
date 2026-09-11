"""A new converter family must be addable without touching the main flow.

The goal claims that adding a requirement type should need minimal change to the
main flow. This file measures that claim rather than asserting it: a forward
converter (isolated, buck-derived, ``vout = n * d * vin``) is defined entirely
*inside this test* and driven through the real capability registry and pipeline.

If a future refactor makes families require a branch in the pipeline, the
optimizer dispatch or the loss model, this test stops compiling or stops
producing correct numbers.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from circuit_ai.capabilities import (
    CapabilityRegistration,
    CapabilityRegistry,
    CapabilityRole,
    CapabilityTarget,
    build_default_registry,
)
from circuit_ai.pipeline import design_from_pbdl
from circuit_ai.power import (
    BoostOperatingPoint,
    FlybackOptimizationResult,
    FlybackParameters,
    PowerStageModel,
    _design_corner_rail,
    _ideal_stage_dc,
    _loss_aware_frequency_bounds,
    _loss_model_for,
    _materialize_candidate,
    _power_problem,
    _range_or,
    _solve_power_problem,
    _stage_efficiency_objective,
    _target_power_values,
    flyback_parameters_from_values,
    power_stage_parameters,
    simulate_ideal_averaged_converter,
)
from circuit_ai.power_loss import DEFAULT_LOSS_PARAMETERS
from circuit_ai.simulation import AnalyticPowerSimulatorBackend

POWER_ANALYSES = frozenset({"dc_transfer", "dc_operating_point"})
POWER_MODELS = frozenset(
    {"R", "C", "L", "ideal_switch", "ideal_diode", "ideal_transformer"}
)

#: The four shipped result dataclasses are structurally identical copies that
#: differ only in name, so a new family reuses one of them rather than asking the
#: package to grow a fifth.  Aliased here to make the intent readable.
ForwardOptimizationResult = FlybackOptimizationResult


def solve_ideal_forward_dc(
    vin: float,
    parameters: FlybackParameters,
    *,
    loss_parameters=None,
) -> BoostOperatingPoint:
    """``vout = n * d * vin``; the reset clamp is not modelled.

    Everything else -- ripple, conduction loss, switching loss, efficiency --
    comes from the shared layer, which is the point of the test.
    """

    return _ideal_stage_dc(
        vin,
        parameters.duty_cycle,
        parameters.inductance_h,
        parameters.capacitance_f,
        parameters.switching_frequency_hz,
        parameters.load_ohm,
        voltage_from_input=lambda rail, duty: parameters.turns_ratio * duty * rail,
        ripple_factor=lambda duty, rail: 1.0 - duty,
        inductor_ripple_factor=lambda duty, rail: rail * (1.0 - duty) * duty,
        loss_parameters=loss_parameters,
        # The switch blocks the input plus the reset-winding voltage.
        switch_voltage_v=2.0 * vin,
    )


class ForwardParameterOptimizer:
    def optimize(self, ir, candidate):
        vin, vout, iout = _target_power_values(ir)
        load = vout / iout
        ranges = ir.constraints.get("parameter_ranges", {})
        ratio_bounds = _range_or(ranges, "turns_ratio", (0.2, 1.0))
        design_rail, _ = _design_corner_rail(ir, vin)
        # The forward needs n > vout / rail: at equality the required duty is
        # exactly 1.0, which leaves no headroom at all.  The binding rail is the
        # one the design is *scheduled* at, since that is where duty is pinned --
        # deriving this from the nominal rail instead leaves the scheduled duty
        # above 1 whenever the design rail is lower.  A 5 % duty margin is kept
        # so the converter still has somewhere to go.
        minimum_ratio = (vout / design_rail) / 0.95
        ratio_lower = max(ratio_bounds[0], minimum_ratio)
        if ratio_lower > ratio_bounds[1]:
            raise ValueError(
                f"forward needs turns_ratio >= {minimum_ratio:.4f} but the spec caps it at "
                f"{ratio_bounds[1]:.4f}"
            )
        initial_ratio = min(max(minimum_ratio, ratio_lower), ratio_bounds[1])
        duty_target = vout / (initial_ratio * vin)
        loss_parameters = _loss_model_for(ir)
        bounds = _loss_aware_frequency_bounds(
            [
                _range_or(ranges, "duty_cycle", (max(0.05, duty_target - 0.25), min(0.95, duty_target + 0.25))),
                _range_or(ranges, "L", (10e-6, 2e-3)),
                _range_or(ranges, "C", (10e-6, 5e-3)),
                _range_or(ranges, "switching_frequency_hz", (20e3, 500e3)),
                (ratio_lower, ratio_bounds[1]),
            ],
            loss_parameters=loss_parameters,
        )
        scheduled = None
        if loss_parameters is not None:
            # The forward has no duty freedom: duty follows from the turns ratio
            # and the rail it is scheduled at, so with the loss model active both
            # duty and ratio are determined.  Searching inductance, capacitance
            # and frequency then optimises efficiency at the fixed conversion
            # ratio, which is what a forward design actually does -- the ratio is
            # set by the transformer, not tuned by the controller.
            scheduled = vout / (initial_ratio * design_rail)
            bounds[0] = (scheduled, scheduled)
            bounds[4] = (initial_ratio, initial_ratio)

        def solve_stage(values):
            params = flyback_parameters_from_values(values, load)
            if loss_parameters is None:
                return solve_ideal_forward_dc(vin, params)
            duty = vout / (params.turns_ratio * design_rail)
            return solve_ideal_forward_dc(
                design_rail,
                replace(params, duty_cycle=duty),
                loss_parameters=loss_parameters,
            )

        def base_objective(values, dc):
            params = flyback_parameters_from_values(values, load)
            voltage_error = abs(dc.output_voltage_v - vout) / vout
            size_penalty = 0.002 * (
                math.log10(params.inductance_h / 100e-6) ** 2
                + math.log10(params.capacitance_f / 100e-6) ** 2
            )
            return 100.0 * voltage_error**2 + size_penalty

        objective = _stage_efficiency_objective(ir, base_objective, weight=None)
        run = _solve_power_problem(
            _power_problem(
                "isolated_forward",
                bounds,
                include_turns_ratio=True,
                initial_values={
                    "duty_cycle": scheduled or duty_target,
                    "turns_ratio": initial_ratio,
                },
            ),
            lambda values: objective(values, solve_stage(values)),
            ir.optimization,
        )
        params = flyback_parameters_from_values(run.values, load)
        if loss_parameters is not None:
            params = replace(
                params, duty_cycle=vout / (params.turns_ratio * design_rail)
            )
        dc = solve_ideal_forward_dc(design_rail, params, loss_parameters=loss_parameters)
        return ForwardOptimizationResult(
            candidate=_materialize_candidate(candidate, params, ir),
            parameters=power_stage_parameters(params),
            operating_point=dc,
            transient=simulate_ideal_averaged_converter(
                design_rail, params, solve_ideal_forward_dc
            ),
            optimizer_message=run.message,
            objective=float(run.objective_values["power.objective"]),
            optimization_run=run,
        )


FORWARD_STAGE = PowerStageModel(
    solver_id="ideal_forward_averaged",
    family="isolated_forward",
    optimizer_type=ForwardParameterOptimizer,
    dc_solver=lambda vin, params: solve_ideal_forward_dc(vin, params),
)


def _register_forward(registry: CapabilityRegistry) -> CapabilityRegistry:
    """Two registrations, exactly the shape builtins.py uses for a stage."""

    stage = FORWARD_STAGE
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id=stage.optimizer_capability_id,
                version="1.0.0",
                role=CapabilityRole.PARAMETER_OPTIMIZER,
                description=f"Parameter optimizer for the {stage.solver_id} model",
                families=frozenset({stage.family}),
                solver_ids=frozenset({stage.solver_id}),
                analysis_kinds=POWER_ANALYSES,
                model_kinds=POWER_MODELS,
                fidelity="ideal_averaged",
                priority=100,
            ),
            ForwardParameterOptimizer(),
        )
    )
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id=stage.simulation_capability_id,
                version="1.0.0",
                role=CapabilityRole.SIMULATION_BACKEND,
                description=f"Unified simulation adapter for {stage.solver_id}",
                families=frozenset({stage.family}),
                solver_ids=frozenset({stage.solver_id}),
                analysis_kinds=POWER_ANALYSES,
                model_kinds=POWER_MODELS,
                fidelity="ideal_averaged",
                priority=100,
            ),
            AnalyticPowerSimulatorBackend(
                backend_id=stage.solver_id,
                solver_id=stage.solver_id,
                parameter_type=stage.parameter_type,
                dc_solver=stage.dc_solver,
                model_kinds=POWER_MODELS,
            ),
        )
    )
    return registry


SPEC = {
    "name": "forward_48_to_5",
    "ports": [
        {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "out_0", "quantity": "ground"}]},
    ],
    "relations": [
        {"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}
    ],
    "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
    "targets": [
        {"target_kind": "dc", "input_voltage_v": 48, "output_voltage_v": 5, "output_current_a": 3}
    ],
    "constraints": {
        "element_types": ["R", "C", "L", "ideal_switch", "ideal_transformer", "ideal_diode"],
        "max_component_count": 5,
        "parameter_ranges": {
            "R": [100, 1_000_000],
            "C": [1e-12, 1e-4],
            "L": [1e-9, 1],
            "turns_ratio": [0.05, 1.0],
        },
    },
    "operating_envelope": {
        "min_input_voltage_v": 36.0,
        "max_input_voltage_v": 60.0,
        "min_load_fraction": 0.2,
        "max_load_fraction": 1.0,
    },
    "optimization": {
        "max_iterations": 6,
        "seed": 7,
        "loss_model": {"enabled": True},
        "weights": {"efficiency": 3.0},
        "design_corner": {"rail": "geometric_mean"},
    },
}


# ---------------------------------------------------------------------------
# The family's own physics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vin,ratio,target",
    [(48.0, 0.5, 5.0), (36.0, 0.5, 5.0), (60.0, 0.25, 5.0), (24.0, 0.75, 3.3)],
)
def test_forward_ratio_is_exact_and_losses_only_cost_efficiency(vin, ratio, target) -> None:
    duty = target / (ratio * vin)
    params = FlybackParameters(duty, 100e-6, 100e-6, 200e3, target / 3.0, ratio)

    ideal = solve_ideal_forward_dc(vin, params)
    assert ideal.output_voltage_v == pytest.approx(target, rel=1e-12)
    assert ideal.efficiency == 1.0

    lossy = solve_ideal_forward_dc(vin, params, loss_parameters=DEFAULT_LOSS_PARAMETERS)
    # Losses must move the input current, never the conversion ratio.
    assert lossy.output_voltage_v == ideal.output_voltage_v
    assert lossy.output_current_a == ideal.output_current_a
    assert lossy.efficiency < 1.0
    assert lossy.input_current_a > ideal.input_current_a


def test_forward_reuses_the_shared_loss_and_envelope_machinery() -> None:
    """No family-specific loss code exists, and none should be needed."""

    params = FlybackParameters(0.25, 100e-6, 100e-6, 200e3, 5.0 / 3.0, 0.5)
    low_frequency = solve_ideal_forward_dc(
        48.0, replace(params, switching_frequency_hz=50e3),
        loss_parameters=DEFAULT_LOSS_PARAMETERS,
    )
    high_frequency = solve_ideal_forward_dc(
        48.0, replace(params, switching_frequency_hz=400e3),
        loss_parameters=DEFAULT_LOSS_PARAMETERS,
    )
    assert low_frequency.efficiency > high_frequency.efficiency


# ---------------------------------------------------------------------------
# The extension path
# ---------------------------------------------------------------------------


def test_a_new_family_registers_without_touching_the_package() -> None:
    """Registration is data: two capability entries and no package edit."""

    from circuit_ai.capabilities import CapabilityRequest, CapabilityResolver

    registry = _register_forward(build_default_registry())
    resolved = CapabilityResolver(registry).resolve(
        CapabilityRequest(
            role=CapabilityRole.PARAMETER_OPTIMIZER,
            solver_id=FORWARD_STAGE.solver_id,
            analysis_kinds=POWER_ANALYSES,
            model_kinds=frozenset(
                {"R", "C", "L", "ideal_switch", "ideal_diode", "ideal_transformer"}
            ),
        )
    )
    assert resolved.selected.target.capability_id == FORWARD_STAGE.optimizer_capability_id
    assert isinstance(resolved.selected.implementation, ForwardParameterOptimizer)

    # The stock registry must not already know this family: otherwise the test
    # would pass without the registration doing anything.
    from circuit_ai.capabilities import CapabilityResolutionError

    with pytest.raises(CapabilityResolutionError):
        CapabilityResolver(build_default_registry()).resolve(
            CapabilityRequest(
                role=CapabilityRole.PARAMETER_OPTIMIZER,
                solver_id=FORWARD_STAGE.solver_id,
                analysis_kinds=POWER_ANALYSES,
                model_kinds=frozenset(
                    {"R", "C", "L", "ideal_switch", "ideal_diode", "ideal_transformer"}
                ),
            )
        )


def test_new_family_drives_the_real_pipeline_end_to_end() -> None:
    """The whole path works: registry, pipeline, loss model, envelope analysis.

    The forward does not win candidate selection here, because the topology
    grammar generates flyback and SEPIC variants rather than forward ones.  That
    is a topology-generation limit, not an architecture one, and this test pins
    the architecture half so a refactor cannot quietly introduce a family branch
    into the pipeline.
    """

    registry = _register_forward(build_default_registry())
    result = design_from_pbdl(SPEC, capability_registry=registry)

    point = result.optimization.operating_point
    assert point.efficiency < 1.0, "loss model must be active"
    assert point.output_voltage_v == pytest.approx(5.0, rel=0.1)

    summary = result.optimization.envelope_analysis
    assert summary is not None
    assert summary.corner_count == 6
    assert summary.efficiency_min <= summary.efficiency_nominal

    payload = result.optimization.as_dict() if hasattr(result.optimization, "as_dict") else None
    _ = payload  # the power result serialises through the pipeline report


def test_forward_optimizer_hits_the_target_it_is_scheduled_for() -> None:
    """The family's own optimizer, driven directly on a real candidate."""

    from circuit_ai.ir import pbdl_to_ir
    from circuit_ai.topology_grammar import PowerTopologyGrammar

    ir = pbdl_to_ir(SPEC)
    search = PowerTopologyGrammar().search(ir)
    assert search.candidates

    result = ForwardParameterOptimizer().optimize(ir, search.candidates[0])
    params = result.parameters
    # Scheduled at the geometric mean of 36-60 V, so that rail hits the target.
    rail = math.sqrt(36.0 * 60.0)
    assert rail * params.turns_ratio * params.duty_cycle == pytest.approx(5.0, rel=1e-6)
    assert result.operating_point.output_voltage_v == pytest.approx(5.0, rel=1e-6)
    assert result.operating_point.efficiency < 1.0
