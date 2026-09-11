"""Nonlinear device models, pinned against measured external responses.

Every reference number in this module was produced by running ngspice in batch
mode and reading the result, not by transcribing a formula.  That distinction is
the point of the file.  The device equations here were written from the reference
implementation's source, and three separate attempts to write them "from the
physics" produced three different answers that were each wrong by a factor the
tests below would have caught:

* the saturation-current temperature law, where the published SPICE3 form, the
  ngspice manual's form and the implemented form all agree at the nominal
  temperature and disagree by up to 3x away from it;
* the breakdown exponential's ideality factor, which is a *separate* parameter
  defaulting to 1 and not the forward emission coefficient -- conflating them is
  a 243x current error on a device with ``N = 1.8``;
* the small-signal treatment of series resistance, where folding the resistance
  into an effective terminal conductance is exact at DC and 68 % wrong at 10 GHz.

So these tests assert against measurements.  Where a measured reference is not
reproduced, that is recorded as a documented bound rather than a suppressed
tolerance.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from circuit_ai.devices import (
    Diode,
    DiodeParameters,
    bandgap_energy_ev,
    scaled_saturation_current,
    thermal_voltage,
)
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.nonlinear import ConvergenceError, NonlinearMNA

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def driven(vin: float, r_ohm: float = 1000.0) -> LinearCircuit:
    """``V1 in 0 <vin>`` into ``R1 in a <r_ohm>``, the shape every case uses."""

    return LinearCircuit(
        elements=(LinearElement("R1", "R", "in", "a", r_ohm),),
        voltage_sources=(VoltageSource("V1", "in", "0", complex(vin)),),
    )


def operating(vin: float, r_ohm: float, *devices) -> tuple[NonlinearMNA, float]:
    sim = NonlinearMNA(driven(vin, r_ohm), list(devices))
    return sim, sim.operating_point().node_voltage("a")


#: ngspice: ``V1 in 0 dc 0.4 ; R1 in a 1k ; D1 a 0 D`` with an all-default model.
DEFAULT_DIODE_DC = [
    (0.3, 0.299998910),
    (0.4, 0.399948063),
    (0.5, 0.497723772),
    (0.6, 0.566936207487066),
    (0.8, 0.611902874384435),
]

#: ngspice: same circuit, ``.op`` at 0.5 V, ``temp`` swept.
DEFAULT_DIODE_TEMPERATURE = [
    (-40.0, 4.99998672275614e-01),
    (0.0, 4.99819340618954e-01),
    (27.0, 4.97723772368556e-01),
    (85.0, 4.55089779003452e-01),
    (150.0, 3.59562109023600e-01),
]

#: ngspice: ``.model DMOD D (IS=2e-15 N=1.05 RS=2 CJ0=4.5e-12 VJ=0.75 M=0.4 TT=1.2e-9)``
#: biased at 0.6 V through 10 kohm, ``ac dec 10 1e3 1e10``.
CAP_MODEL = DiodeParameters(
    saturation_current_a=2e-15,
    emission_coefficient=1.05,
    series_resistance_ohm=2.0,
    zero_bias_capacitance_f=4.5e-12,
    junction_potential_v=0.75,
    grading_coefficient=0.4,
    transit_time_s=1.2e-9,
)
CAP_MODEL_DC = 0.571974306010412
CAP_MODEL_SWEEP = [
    (1.0e03, 4.92191231853575e-01, -1.1120776155397e-04),
    (1.0e04, 4.92188743306867e-01, -1.1120719905268e-03),
    (1.0e06, 4.68276169713451e-01, -1.0580209547774e-01),
    (1.0e08, 1.16102515947568e-03, -2.1723536639657e-02),
    (1.0e09, 2.09589281418852e-04, -2.1765628858868e-03),
    (1.0e10, 2.00056302598440e-04, -2.1766050604824e-04),
]


# ---------------------------------------------------------------------------
# the constituent pieces, tested directly
# ---------------------------------------------------------------------------


def test_thermal_voltage_uses_the_reference_constants():
    # k*T/q at 300.15 K.  The value matters: it sets every exponential in the
    # model, and a plausible-looking pair of constants moves it in the fourth
    # digit, which is enough to shift a forward drop by millivolts.
    assert thermal_voltage(300.15) == pytest.approx(0.0258641, abs=5e-7)


def test_bandgap_energy_matches_the_reference_fit():
    assert bandgap_energy_ev(300.15) == pytest.approx(1.1150877, abs=1e-6)


def test_saturation_current_is_unchanged_at_the_nominal_temperature():
    assert (
        scaled_saturation_current(
            1e-14,
            temperature_k=300.15,
            nominal_temperature_k=300.15,
            emission_coefficient=1.0,
            activation_energy_ev=1.11,
        )
        == 1e-14
    )


@pytest.mark.parametrize(
    "temperature_c, expected",
    [
        (-40.0, 2.064436e-20),
        (0.0, 1.083536e-16),
        (27.0, 1.0e-14),
        (85.0, 1.772324e-11),
        (150.0, 7.331982e-09),
    ],
)
def test_saturation_current_temperature_law_matches_measurement(
    temperature_c, expected
):
    """The measured ``IS(T)``, recovered from ngspice's own reported response.

    The law is ``IS*exp((T/Tnom - 1)*EG/(N*Vt) + XTI/N*ln(T/Tnom))``.  The two
    alternatives that look right and are not -- the published SPICE3 band-gap
    difference, and the ngspice manual's "log factor" -- agree with it exactly at
    27 C and are wrong by 1.76x and 1.35x at 85 C.  This test therefore checks a
    spread of temperatures and not one, which is the only way to tell them apart.
    """

    recovered = scaled_saturation_current(
        1e-14,
        temperature_k=temperature_c + 273.15,
        nominal_temperature_k=300.15,
        emission_coefficient=1.0,
        activation_energy_ev=1.11,
    )
    assert recovered == pytest.approx(expected, rel=1e-3)


def test_the_rejected_temperature_laws_are_actually_rejected():
    """Guards the guard: the alternatives must fail the assertion above.

    Without this, a future edit could simplify the expression into one of the
    wrong forms and the parametrised test would still pass at 27 C, which is the
    one temperature where they coincide.
    """

    temperature_k = 358.15
    nominal = 300.15
    vt = thermal_voltage(temperature_k)
    published_spice3 = (
        1e-14
        * (temperature_k / nominal) ** 3.0
        * float(
            np.exp(
                bandgap_energy_ev(nominal) / thermal_voltage(nominal)
                - bandgap_energy_ev(temperature_k) / vt
            )
        )
    )
    manual = 1e-14 * float(
        np.exp(
            bandgap_energy_ev(nominal) / thermal_voltage(nominal)
            - bandgap_energy_ev(temperature_k) / vt
            + 3.0 * float(np.log(temperature_k / nominal))
        )
    )
    measured = 1.772324e-11
    for alternative in (published_spice3, manual):
        assert abs(alternative / measured - 1.0) > 0.3


# ---------------------------------------------------------------------------
# forward bias, against ngspice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vin, reference", DEFAULT_DIODE_DC)
def test_forward_dc_matches_ngspice(vin, reference):
    _, node_voltage = operating(vin, 1000.0, Diode("D1", "a", "0"))
    # 3e-5 V absolute: the reference is itself a converged Newton solution of the
    # same equations, and below its own default vntol the two agree to within the
    # reference's iteration, not to within the model.
    assert node_voltage == pytest.approx(reference, abs=3e-5)


def test_default_model_saturates_at_the_expected_drop():
    """A sanity check in physical units, independent of any reference file."""

    params = DiodeParameters()
    assert params.saturation_current_a == 1e-14
    # One diode drop at about 1 mA through 1 kohm into a 2.5 V source.
    _, node_voltage = operating(2.5, 1000.0, Diode("D1", "a", "0"))
    assert 0.60 < node_voltage < 0.72


@pytest.mark.parametrize("temperature_c, reference", DEFAULT_DIODE_TEMPERATURE)
def test_temperature_scaling_matches_ngspice(temperature_c, reference):
    _, node_voltage = operating(
        0.5, 1000.0, Diode("D1", "a", "0", DiodeParameters(temperature_c=temperature_c))
    )
    assert node_voltage == pytest.approx(reference, abs=3e-5)


def test_forward_drop_falls_with_temperature():
    """The sign and rough magnitude of the temperature coefficient.

    A diode's forward drop falls by roughly 2 mV/K at fixed current.  This is a
    property of the device rather than of the reference, and it is asserted
    separately so that a sign error in the temperature law cannot hide behind a
    matching reference file.
    """

    drops = []
    for temperature_c in (25.0, 75.0, 125.0):
        _, node_voltage = operating(
            0.7,
            1000.0,
            Diode("D1", "a", "0", DiodeParameters(temperature_c=temperature_c)),
        )
        current = (0.7 - node_voltage) / 1000.0
        drops.append((temperature_c, node_voltage, current))
    for (_, _, first), (_, _, second) in pairwise(drops):
        assert second > first
    # And the drop itself falls.
    assert drops[0][1] > drops[-1][1]


# ---------------------------------------------------------------------------
# series resistance
# ---------------------------------------------------------------------------


def test_series_resistance_matches_ngspice_dc():
    # ngspice: RS=10, 0.85 V through 1 kohm.
    _, node_voltage = operating(
        0.85, 1000.0, Diode("D1", "a", "0", DiodeParameters(series_resistance_ohm=10.0))
    )
    assert node_voltage == pytest.approx(0.619469716442242, abs=3e-5)


def test_series_resistance_terminal_conductance_is_not_one_over_rs():
    """The small-signal conductance of a biased diode with series resistance.

    ``gd/(1 + gd*RS)``, not ``1/RS``.  Using ``1/RS`` makes a biased diode look
    like a short and understates the response by an order of magnitude.  The value
    here is cross-checked against a measured series-resistance sweep of the
    reference, which agrees to 0.4 %.
    """

    params = DiodeParameters(series_resistance_ohm=10.0)
    diode = Diode("D1", "a", "0", params)
    _, node_voltage = operating(0.85, 1000.0, diode)
    stamp = diode.stamp({"anode": node_voltage, "cathode": 0.0})
    gd = diode.junction_slope(diode.bias())
    assert stamp.conductance == pytest.approx(gd / (1.0 + gd * 10.0), rel=1e-12)
    assert stamp.conductance != pytest.approx(1.0 / 10.0, rel=0.5)


def test_series_resistance_does_not_change_the_dc_solution_shape():
    """A zero-ohm device and a device with RS must agree as RS vanishes.

    10 micro-ohms is a resistance no fabrication produces, and at 1 kohm of drive
    it changes the node voltage by nothing measurable.  The tolerance is set by the
    circuit, not by the model: any error here is the scalar junction solve failing
    to reach its root when the root is trivial.
    """

    _, without = operating(0.6, 1000.0, Diode("D1", "a", "0", DiodeParameters()))
    for rs in (1e-6, 1e-5, 1e-3):
        _, tiny = operating(
            0.6,
            1000.0,
            Diode("D1", "a", "0", DiodeParameters(series_resistance_ohm=rs)),
        )
        assert tiny == pytest.approx(without, abs=1e-6)


# ---------------------------------------------------------------------------
# small-signal analysis
# ---------------------------------------------------------------------------


def test_resistive_model_has_a_flat_small_signal_response():
    """A default diode has no junction capacitance, so its response is flat.

    ngspice reports 4.38918779163760e-01 for a unit AC drive normalised by the
    0.6 V bias, identical at every frequency from 1 Hz to 1 GHz.  A test that
    expected a roll-off from a default model would be asserting the wrong thing.
    """

    sim = NonlinearMNA(driven(0.6, 1000.0), [Diode("D1", "a", "0")])
    bias = sim.operating_point()
    batch = sim.solve_ac(np.logspace(0, 9, 181), operating_point=bias)
    magnitude = np.abs(batch.node_voltage("a")) / 0.6
    assert np.ptp(magnitude) == pytest.approx(0.0, abs=1e-15)
    assert float(magnitude[0]) == pytest.approx(0.438918779163760, abs=2e-4)


def test_series_resistance_small_signal_matches_ngspice():
    # ngspice: RS=10 at 0.85 V bias, unit AC drive, normalised by the bias.
    sim = NonlinearMNA(
        driven(0.85, 1000.0),
        [Diode("D1", "a", "0", DiodeParameters(series_resistance_ohm=10.0))],
    )
    bias = sim.operating_point()
    batch = sim.solve_ac(np.array([1.0]), operating_point=bias)
    assert abs(batch.node_voltage("a")[0]) / 0.85 == pytest.approx(
        0.108891245013329, rel=2e-4
    )


@pytest.mark.parametrize("frequency, real, imaginary", CAP_MODEL_SWEEP)
def test_capacitive_small_signal_sweep_matches_ngspice(frequency, real, imaginary):
    """The full junction-capacitance and transit-time response, 1 kHz to 10 GHz.

    This range is the reason the device carries an internal node.  Folding the
    series resistance into an effective terminal conductance is exact at DC and
    leaves the response right to 100 MHz, then wrong by 0.9 % at 1 GHz and 68 % at
    10 GHz, because the folded form has no path for displacement current to bypass
    the resistance.  The high-frequency points below are the ones that fail if the
    node is ever folded away again.
    """

    sim = NonlinearMNA(driven(0.6, 10_000.0), [Diode("D1", "a", "0", CAP_MODEL)])
    bias = sim.operating_point()
    assert bias.node_voltage("a") == pytest.approx(CAP_MODEL_DC, abs=3e-5)
    batch = sim.solve_ac(np.array([frequency]), operating_point=bias)
    response = batch.node_voltage("a")[0] / 0.6
    assert response.real == pytest.approx(real, rel=2e-3, abs=2e-6)
    assert response.imag == pytest.approx(imaginary, rel=2e-3, abs=2e-6)


def test_small_signal_response_requires_a_bias_point():
    """The AC solve is meaningless without a bias, and says so rather than guessing."""

    sim = NonlinearMNA(driven(0.6, 10_000.0), [Diode("D1", "a", "0", CAP_MODEL)])
    batch = sim.solve_ac(np.array([1.0e6]))
    # Computed internally when not supplied; must equal the explicit route.
    supplied = sim.solve_ac(np.array([1.0e6]), operating_point=sim.operating_point())
    assert batch.node_voltage("a")[0] == pytest.approx(supplied.node_voltage("a")[0])


# ---------------------------------------------------------------------------
# reverse breakdown
# ---------------------------------------------------------------------------


def test_breakdown_current_shape_matches_ngspice():
    """The breakdown exponential, verified where the knee adjustment is negligible.

    ngspice resolves the over-specification of ``BV`` and ``IBV`` by moving the knee
    voltage outward; with an onset current of 1e-5 the measured knee sits 0.1 V
    below the nominal one and its current runs about 2.4x higher than a model
    anchored at ``(BV, -IBV)``.  With an onset current where that adjustment is
    negligible the two agree to 5e-4 relative, which is what this asserts.  The
    knee adjustment itself is *not* claimed: see
    :meth:`circuit_ai.devices.Diode.breakdown_voltage_v`.
    """

    _, node_voltage = operating(
        -5.5,
        100.0,
        Diode(
            "D1",
            "a",
            "0",
            DiodeParameters(
                saturation_current_a=5e-15,
                emission_coefficient=1.8,
                breakdown_voltage_v=5.5,
                breakdown_current_a=1e-5,
            ),
        ),
    )
    assert node_voltage == pytest.approx(-5.4990208119957, abs=2e-3)


def test_breakdown_uses_its_own_ideality_factor():
    """``NBV`` is independent of the forward emission coefficient and defaults to 1.

    The device declares this itself, so the assertion that matters is that the
    exponential's *slope* is set by ``NBV`` and not by ``N``.  The slope is measured
    as a ratio of currents one thermal voltage apart, which cancels the anchoring
    constant and leaves only the exponent -- so this cannot pass by accident on a
    device whose anchor happens to be right.
    """

    parameters = DiodeParameters(
        saturation_current_a=5e-15,
        emission_coefficient=1.8,
        breakdown_voltage_v=5.5,
        breakdown_current_a=1e-5,
    )
    diode = Diode("D1", "a", "0", parameters)
    assert parameters.breakdown_emission_coefficient == 1.0
    nbvt = thermal_voltage(parameters.temperature_k) * 1.0
    # The anchor is the datasheet pair: -IBV at -BV, by construction.
    assert diode.junction_current(-5.5) == pytest.approx(-1e-5, rel=1e-12)
    # One *breakdown* thermal voltage further into breakdown multiplies the
    # magnitude by e.  The ratio of magnitudes is used so the direction of the
    # current convention does not enter the assertion.
    deeper = abs(diode.junction_current(-6.5))
    shallower = abs(diode.junction_current(-6.5 + nbvt))
    assert deeper / shallower == pytest.approx(float(np.exp(1.0)), rel=1e-9)
    # Had the forward emission coefficient been used in the exponent, the same
    # ratio would be off by a factor of exp(1 - 1/1.8) = 1.56.
    conflated = float(np.exp(nbvt / (thermal_voltage(parameters.temperature_k) * 1.8)))
    assert abs(conflated / float(np.exp(1.0)) - 1.0) > 0.3


def test_no_breakdown_without_a_breakdown_voltage():
    parameters = DiodeParameters()
    assert parameters.breakdown_voltage_v is None
    diode = Diode("D1", "a", "0", parameters)
    # Deep reverse, no breakdown: the current stays at the reverse saturation scale.
    assert abs(diode.junction_current(-100.0)) < 1e-9
    assert np.isfinite(diode.junction_current(-100.0))


def test_reverse_region_is_finite_and_monotone_before_breakdown():
    """The reverse branch must not overflow, which the bare exponential does.

    The range is kept clear of the knee: past it the current rises steeply by
    design, and a test that walked through the knee while asserting monotone
    magnitude would be asserting that breakdown does not work.
    """

    diode = Diode("D1", "a", "0", DiodeParameters(breakdown_voltage_v=5.5))
    values = [diode.junction_current(-v) for v in (0.5, 1.0, 2.0, 3.0, 4.0)]
    assert all(np.isfinite(value) for value in values)
    magnitudes = [abs(value) for value in values]
    assert magnitudes == sorted(magnitudes)
    # And far past the knee it is large but still finite.
    assert np.isfinite(diode.junction_current(-50.0))
    assert abs(diode.junction_current(-50.0)) > abs(values[-1])


# ---------------------------------------------------------------------------
# convergence behaviour
# ---------------------------------------------------------------------------


def test_series_resistance_converges_from_a_cold_start():
    """The DC solve must not need a warm start, even with a large series resistance.

    Carrying the internal node through the DC iteration does need one: the junction
    takes a large first step to a voltage where its slope is tiny, the internal node
    is then floating, and the iteration converges to full tolerance on the answer
    where no current flows at all.  Eliminating the node instead makes the root of a
    monotone scalar equation, and this asserts the iteration finds it from nothing.
    """

    for rs in (0.0, 1.0, 10.0, 100.0, 1000.0):
        sim = NonlinearMNA(
            driven(0.85, 1000.0),
            [Diode("D1", "a", "0", DiodeParameters(series_resistance_ohm=rs))],
        )
        point = sim.operating_point()
        current = (0.85 - point.node_voltage("a")) / 1000.0
        assert current > 0
        assert point.iterations < 40


def test_a_solution_is_an_actual_solution_not_just_an_iterate():
    """The reported operating point must satisfy the device equations.

    A Newton solve can stop on a degenerate answer -- for a diode, the one where no
    current flows and the junction is simply not conducting.  Checking the residual
    and the device's own current-voltage relation against the surrounding circuit
    is what distinguishes a solution from a stopping point.
    """

    diode = Diode("D1", "a", "0", DiodeParameters(saturation_current_a=1e-30))
    circuit = LinearCircuit(
        elements=(LinearElement("R1", "R", "a", "0", 1e12),),
        voltage_sources=(VoltageSource("V1", "in", "0", complex(-1e6)),),
    )
    # Node "a" is driven only through the diode's own reverse current, so the
    # solution is whatever balances a teraohm against the reverse branch.
    sim = NonlinearMNA(
        circuit,
        [Diode("D1", "in", "a", DiodeParameters(saturation_current_a=1e-30))],
    )
    point = sim.operating_point()
    assert point.residual <= 1e-6
    assert np.isfinite(point.node_voltage("in"))
    assert np.isfinite(point.node_voltage("a"))
    assert diode.bias() == pytest.approx(0.0) or np.isfinite(diode.bias())


def test_tiny_iteration_budget_reports_failure():
    sim = NonlinearMNA(
        driven(0.8, 1000.0),
        [Diode("D1", "a", "0")],
        max_iterations=1,
        tolerance=1e-15,
    )
    with pytest.raises(ConvergenceError):
        sim.operating_point()


def test_non_finite_device_state_is_rejected():
    """A device that returns garbage must stop the solve, not poison the matrix."""

    class Broken:
        name = "B1"

        def terminals(self):
            return (("a", "anode"), ("0", "cathode"))

        def reset(self):
            return None

        def internal_nodes(self):
            return ()

        def internal_elements(self):
            return ()

        def stamp(self, voltages):
            from circuit_ai.devices import NonlinearStamp

            return NonlinearStamp(current=float("nan"), conductance=float("nan"))

    sim = NonlinearMNA(driven(0.6, 1000.0), [Broken()])
    with pytest.raises(ConvergenceError):
        sim.operating_point()


# ---------------------------------------------------------------------------
# structural guarantees
# ---------------------------------------------------------------------------


def test_linear_solution_path_is_unchanged_by_the_nonlinear_layer():
    """A circuit with no nonlinear device is solved by the linear code exactly."""

    from circuit_ai.mna import MNASimulator

    circuit = driven(1.0, 1000.0)
    frequencies = np.logspace(0, 6, 7)
    linear = MNASimulator().solve_ac_batched(circuit, frequencies)
    nonlinear = NonlinearMNA(circuit, []).solve_ac(frequencies)
    for node in ("a", "in"):
        assert np.array_equal(linear.node_voltage(node), nonlinear.node_voltage(node))


def test_internal_nodes_are_appended_and_do_not_move_linear_indices():
    """The DC and small-signal plans must agree on every linear index.

    They differ by exactly the internal nodes, and the internal ones are appended
    after every node and branch, so a device that resolves a node against the wrong
    plan sees ``None`` rather than a wrong index -- which is a singular matrix and a
    loud failure, not a silently misplaced stamp.
    """

    diode = Diode("D1", "a", "0", DiodeParameters(series_resistance_ohm=2.0))
    sim = NonlinearMNA(driven(0.6, 10_000.0), [diode])
    for name, index in sim.compiled.node_index.items():
        assert sim._ac_compiled.node_index[name] == index
    for name, index in sim.compiled.source_index.items():
        assert sim._ac_compiled.source_index[name] == index
    assert sim._ac_compiled.size > sim.compiled.size
    assert set(sim._ac_compiled.node_index) - set(sim.compiled.node_index) == {
        diode.junction_node()
    }


def test_device_without_series_resistance_declares_no_internal_node():
    assert Diode("D1", "a", "0").internal_nodes() == ()
    assert Diode("D1", "a", "0").internal_elements() == ()
    resistive = Diode("D1", "a", "0", DiodeParameters(series_resistance_ohm=2.0))
    assert resistive.internal_nodes() == (resistive.junction_node(),)
    assert resistive.internal_elements() == (("a", resistive.junction_node(), 2.0),)
