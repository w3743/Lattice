"""Loss model and its opt-in wiring into the power stage optimizers.

The ideal averaged stages report ``efficiency == 1.0`` because they model no
loss mechanism, which makes "maximise efficiency" a meaningless objective.  The
loss model is opt-in (``optimization.loss_model.enabled``) so the existing
characterization lock on the lossless families keeps holding; these tests cover
the model itself and the behaviour it unlocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuit_ai.ir import pbdl_to_ir
from circuit_ai.power import (
    BoostParameters,
    FlybackParameters,
    solve_ideal_boost_dc,
    solve_ideal_buck_dc,
    solve_ideal_flyback_dc,
    solve_ideal_sepic_dc,
)
from circuit_ai.power_loss import (
    DEFAULT_LOSS_PARAMETERS,
    LossParameters,
    default_switch_path_duty,
    evaluate_loss_model,
    loss_parameters_from_options,
)


BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


def _breakdown(
    *,
    vin: float = 36.0,
    vout: float = 5.0,
    iout: float = 2.0,
    fsw: float = 100e3,
    inductance_h: float = 100e-6,
    capacitance_f: float = 100e-6,
    stress: float = 41.0,
    parameters: LossParameters = DEFAULT_LOSS_PARAMETERS,
    duty: float = 0.5,
):
    return evaluate_loss_model(
        input_voltage_v=vin,
        output_power_w=vout * iout,
        output_voltage_v=vout,
        output_current_a=iout,
        duty=duty,
        switching_frequency_hz=fsw,
        inductance_h=inductance_h,
        capacitance_f=capacitance_f,
        inductor_ripple_a=vin * duty / (fsw * inductance_h),
        path=default_switch_path_duty(duty, switch_voltage_v=stress),
        parameters=parameters,
    )


def test_loss_model_is_bounded_and_conserves_power() -> None:
    breakdown = _breakdown()
    assert 0.0 < breakdown.efficiency < 1.0
    assert breakdown.input_power_w == pytest.approx(
        breakdown.output_power_w + breakdown.total_loss_w
    )
    assert breakdown.efficiency == pytest.approx(
        breakdown.output_power_w / breakdown.input_power_w
    )
    assert breakdown.total_loss_w == pytest.approx(sum(breakdown.mechanisms.values()))


def test_efficiency_peaks_then_falls_as_output_power_rises() -> None:
    """The property the ideal model could not express.

    Efficiency is not monotonic in load: at low load the fixed switching loss
    dominates, at high load conduction loss does.  A correct loss model must
    show that crossover, so this pins the shape rather than a single direction.
    """

    currents = (0.25, 0.5, 2.0, 5.0, 10.0)
    efficiencies = [_breakdown(iout=current).efficiency for current in currents]
    peak = max(efficiencies)
    assert efficiencies[-1] == min(efficiencies), efficiencies
    assert efficiencies.index(peak) not in (0, len(efficiencies) - 1), efficiencies
    # Falling side is strictly monotonic in load.
    falling = efficiencies[efficiencies.index(peak) :]
    assert falling == sorted(falling, reverse=True), falling


def test_efficiency_falls_as_switching_frequency_rises() -> None:
    """Switching loss is the frequency-dependent term."""

    low = _breakdown(fsw=50e3).efficiency
    high = _breakdown(fsw=400e3).efficiency
    assert low > high


def test_lower_switched_voltage_is_more_efficient() -> None:
    """A family that blocks less voltage loses less: this is why turn ratio matters."""

    low_stress = _breakdown(stress=10.0).efficiency
    high_stress = _breakdown(stress=100.0).efficiency
    assert low_stress > high_stress


def test_lossless_parameters_degenerate_to_unity_efficiency() -> None:
    """All-zero coefficients must recover the ideal answer exactly."""

    ideal = LossParameters(
        switch_resistance_ohm=0.0,
        inductor_resistance_ohm=0.0,
        rectifier_forward_v=0.0,
        rectifier_resistance_ohm=0.0,
        switching_transition_s=0.0,
        switch_output_capacitance_f=0.0,
        output_capacitor_esr_ohm=0.0,
    )
    breakdown = _breakdown(parameters=ideal)
    assert breakdown.total_loss_w == pytest.approx(0.0, abs=1e-15)
    assert breakdown.efficiency == pytest.approx(1.0)
    assert breakdown.dominant_mechanism is None


def test_dominant_mechanism_is_reported() -> None:
    """A design decision needs to know which loss to attack."""

    heavy_conduction = _breakdown(iout=20.0, fsw=30e3)
    assert heavy_conduction.dominant_mechanism in set(heavy_conduction.mechanisms)
    assert heavy_conduction.mechanisms[heavy_conduction.dominant_mechanism] == max(
        heavy_conduction.mechanisms.values()
    )


def test_breakdown_serialises_with_its_evidence() -> None:
    payload = _breakdown().as_dict()
    assert payload["schema"] == "circuit_ai.power_loss_model"
    assert payload["schema_version"] == 1
    assert payload["extensions"] == {}
    assert set(payload["losses"]) == set(_breakdown().mechanisms)
    json.dumps(payload)  # must be JSON-serialisable for report.json


def test_loss_parameters_reject_negative_coefficients() -> None:
    with pytest.raises(ValueError):
        LossParameters(switch_resistance_ohm=-1.0)


def test_loss_model_rejects_degenerate_operating_points() -> None:
    with pytest.raises(ValueError):
        evaluate_loss_model(
            input_voltage_v=0.0,
            output_power_w=1.0,
            output_voltage_v=1.0,
            output_current_a=1.0,
            duty=0.5,
            switching_frequency_hz=1e5,
            inductance_h=1e-4,
            capacitance_f=1e-4,
            inductor_ripple_a=0.1,
            path=default_switch_path_duty(0.5, switch_voltage_v=1.0),
        )


def test_options_absent_means_defaults_and_present_keys_override() -> None:
    assert loss_parameters_from_options(None) == DEFAULT_LOSS_PARAMETERS
    assert loss_parameters_from_options({}) == DEFAULT_LOSS_PARAMETERS
    tuned = loss_parameters_from_options({"switch_resistance_ohm": 0.01})
    assert tuned.switch_resistance_ohm == 0.01
    assert tuned.inductor_resistance_ohm == DEFAULT_LOSS_PARAMETERS.inductor_resistance_ohm


def test_loss_model_is_not_claimed_to_include_core_loss() -> None:
    """Guard against reading the efficiency number as more than it is."""

    assert DEFAULT_LOSS_PARAMETERS.core_loss_modelled is False
    assert DEFAULT_LOSS_PARAMETERS.as_dict()["core_loss_modelled"] is False


# ---------------------------------------------------------------------------
# Opt-in wiring
# ---------------------------------------------------------------------------


def test_dc_solvers_default_to_lossless() -> None:
    """The default path must stay bit-identical to the characterization lock."""

    boost = solve_ideal_boost_dc(5.0, BoostParameters(0.5, 100e-6, 100e-6, 300e3, 40.0))
    buck = solve_ideal_buck_dc(12.0, BoostParameters(0.4, 100e-6, 100e-6, 300e3, 5.0))
    sepic = solve_ideal_sepic_dc(12.0, BoostParameters(0.4, 100e-6, 100e-6, 300e3, 18.0))
    flyback = solve_ideal_flyback_dc(
        5.0, FlybackParameters(0.5, 100e-6, 100e-6, 300e3, 12.0, 1.0)
    )
    for point in (boost, buck, sepic, flyback):
        assert point.efficiency == 1.0
        assert point.input_voltage_v * point.input_current_a == pytest.approx(
            point.output_power_w
        )


def test_dc_solvers_apply_losses_only_when_asked() -> None:
    parameters = FlybackParameters(0.5, 100e-6, 100e-6, 300e3, 12.0, 1.0)
    lossless = solve_ideal_flyback_dc(5.0, parameters)
    lossy = solve_ideal_flyback_dc(
        5.0, parameters, loss_parameters=DEFAULT_LOSS_PARAMETERS
    )
    # Losses must not move the conversion ratio, only the input current.
    assert lossy.output_voltage_v == lossless.output_voltage_v
    assert lossy.output_current_a == lossless.output_current_a
    assert lossy.predicted_ripple_mv == lossless.predicted_ripple_mv
    assert lossy.efficiency < 1.0
    assert lossy.input_current_a > lossless.input_current_a
    assert lossy.output_power_w / (lossy.input_voltage_v * lossy.input_current_a) == pytest.approx(
        lossy.efficiency
    )


def test_spec_gate_reads_loss_model_and_is_off_by_default() -> None:
    from circuit_ai.power import _loss_model_for

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    assert _loss_model_for(pbdl_to_ir(data)) is None

    data.setdefault("optimization", {})["loss_model"] = {"enabled": False}
    assert _loss_model_for(pbdl_to_ir(data)) is None

    data["optimization"]["loss_model"] = {"enabled": True}
    assert _loss_model_for(pbdl_to_ir(data)) == DEFAULT_LOSS_PARAMETERS

    data["optimization"]["loss_model"] = {"enabled": True, "switch_resistance_ohm": 0.02}
    resolved = _loss_model_for(pbdl_to_ir(data))
    assert resolved is not None
    assert resolved.switch_resistance_ohm == 0.02


def test_power_optimization_reports_real_efficiency_when_enabled() -> None:
    """End-to-end: the loss model changes the design and the reported efficiency."""

    from circuit_ai.pipeline import design_from_pbdl

    base = {
        "name": "loss_wiring",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "out_0", "quantity": "ground"}]},
        ],
        "relations": [
            {"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}
        ],
        "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
        "targets": [
            {"target_kind": "dc", "input_voltage_v": 24, "output_voltage_v": 5, "output_current_a": 3}
        ],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_transformer", "ideal_diode"],
            "max_component_count": 5,
            "parameter_ranges": {
                "R": [100, 1e6],
                "C": [1e-12, 1e-4],
                "L": [1e-9, 1],
                "turns_ratio": [0.3, 2.0],
            },
        },
    }

    ideal = design_from_pbdl(
        {**base, "optimization": {"max_iterations": 8, "seed": 3}}
    )
    assert ideal.optimization.operating_point.efficiency == 1.0

    lossy = design_from_pbdl(
        {
            **base,
            "optimization": {
                "max_iterations": 8,
                "seed": 3,
                "loss_model": {"enabled": True},
                "weights": {"efficiency": 3.0},
            },
        }
    )
    point = lossy.optimization.operating_point
    assert 0.5 < point.efficiency < 1.0
    assert point.input_current_a > point.output_power_w / 24.0
    # The conversion ratio is pinned when losses are modelled.
    assert point.output_voltage_v == pytest.approx(5.0, rel=1e-6)


def test_efficiency_weight_comes_from_spec_weights() -> None:
    """A user asking for efficiency gets it without discovering a second knob."""

    from circuit_ai.power import _stage_efficiency_objective

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    data.setdefault("optimization", {})["weights"] = {"efficiency": 7.0}
    ir = pbdl_to_ir(data)

    def base(values, point):
        return 0.0

    objective = _stage_efficiency_objective(ir, base, weight=None)

    class _Point:
        efficiency = 0.9

    assert objective(None, _Point()) == pytest.approx(7.0 * 0.1)


def test_zero_efficiency_weight_leaves_objective_untouched() -> None:
    from circuit_ai.power import _stage_efficiency_objective

    data = json.loads(BOOST_SPEC.read_text(encoding="utf-8"))
    ir = pbdl_to_ir(data)

    def base(values, point):
        return 0.25

    assert _stage_efficiency_objective(ir, base, weight=None) is base
    assert _stage_efficiency_objective(ir, base, weight=0.0) is base
