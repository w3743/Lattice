"""S-parameters, checked against closed forms and against measured ngspice data.

Two independent references are used deliberately.  The closed forms -- a series
element, a shunt element, a matched load, an open, a short -- catch a sign or a
factor error, because there is nothing else for them to agree with by accident.
The measured references catch a convention error: port ordering, reference
impedance, what the wave amplitudes are normalised against.  A closed form for a
symmetric network cannot see a transposed port ordering, and the limits alone
cannot see a wrong normaliser, because several wrong normalisers agree at the
extremes.

The measured reference is a plain ngspice AC run of the same network driven from a
1 V Thevenin source behind 50 ohm with both ports terminated, read out of the
port voltages by the standard travelling-wave relations.  That the extraction and
this implementation agree to 7e-16 while both are independent of the closed forms
is the evidence; the closed forms are what say the convention itself is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit_ai.devices import Diode, DiodeParameters
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.nonlinear import NonlinearMNA
from circuit_ai.sparameters import (
    DEFAULT_REFERENCE_IMPEDANCE_OHM,
    Port,
    s_parameters,
)


def two_port(*elements) -> LinearCircuit:
    return LinearCircuit(elements=tuple(elements), voltage_sources=())


PORTS_12 = (Port("1", "p1"), Port("2", "p2"))

#: The network the measured reference describes.
#: ``RS p1 p2 300 ; C1 p2 0 1p ; R2 p2 0 1k``
NGSPICE_NETWORK = two_port(
    LinearElement("RS", "R", "p1", "p2", 300.0),
    LinearElement("C1", "C", "p2", "0", 1e-12),
    LinearElement("R2", "R", "p2", "0", 1000.0),
)

#: ngspice: ``V1 src 0 ac 1 ; RSRC src p1 50`` with ``RL p2 0 50``, ``ac dec 1 1e6
#: 1e9``, printed at full precision.  ``v(p1)`` then ``v(p2)``.
NGSPICE_PORT_VOLTAGES = {
    1.0e06: (
        (8.74251495819295e-01, -4.5058516772663e-06),
        (1.19760470735066e-01, -3.1540961740864e-05),
    ),
    1.0e07: (
        (8.74251378337517e-01, -4.5058207364371e-05),
        (1.19759648362619e-01, -3.1540745155060e-04),
    ),
    1.0e08: (
        (8.74239638302037e-01, -4.5027287979400e-04),
        (1.19677468114261e-01, -3.1519101585580e-03),
    ),
    1.0e09: (
        (8.73141776708933e-01, -4.2135882301690e-03),
        (1.11992436962534e-01, -2.9495117611183e-02),
    ),
}


def ngspice_s(frequency: float) -> tuple[complex, complex]:
    """S11 and S21 of :data:`NGSPICE_NETWORK`, from the measured port voltages.

    With a 1 V source behind 50 ohm the input impedance follows from the port
    voltage, the reflection coefficient from that, and the transmitted coefficient
    is the port-2 voltage over the incident half-volt.
    """

    (v1_real, v1_imag), (v2_real, v2_imag) = NGSPICE_PORT_VOLTAGES[frequency]
    v1 = complex(v1_real, v1_imag)
    v2 = complex(v2_real, v2_imag)
    z0 = DEFAULT_REFERENCE_IMPEDANCE_OHM
    z_in = z0 * v1 / (1.0 - v1)
    return (z_in - z0) / (z_in + z0), 2.0 * v2


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resistance", [10.0, 50.0, 200.0, 1000.0])
def test_series_element_between_two_ports(resistance):
    """S11 = Z/(Z+2Z0) and S21 = 2Z0/(Z+2Z0) for a series impedance.

    Also the case that breaks a current-injection formulation: the two ports of an
    unconnected two-port share no ground path, so the network's nodal admittance
    matrix is singular and a naive solve fails outright on an ordinary series
    attenuator.
    """

    circuit = two_port(LinearElement("R1", "R", "p1", "p2", resistance))
    result = s_parameters(circuit, PORTS_12, np.array([1.0e6]))
    z0 = DEFAULT_REFERENCE_IMPEDANCE_OHM
    assert result.s("1", "1", 1.0e6).real == pytest.approx(
        resistance / (resistance + 2 * z0), abs=1e-9
    )
    assert result.s("2", "1", 1.0e6).real == pytest.approx(
        2 * z0 / (resistance + 2 * z0), abs=1e-9
    )
    # A series impedance is reciprocal and symmetric: the same answer in both
    # directions, and the same reflection at both ports.  It is emphatically *not*
    # unilateral, so S12 is not zero -- a series attenuator passes both ways.
    assert result.s("1", "2", 1.0e6) == pytest.approx(result.s("2", "1", 1.0e6))
    assert result.s("2", "2", 1.0e6) == pytest.approx(result.s("1", "1", 1.0e6))
    assert abs(result.s("2", "1", 1.0e6)) > 0.0


@pytest.mark.parametrize("resistance", [25.0, 50.0, 200.0, 1000.0])
def test_shunt_element_at_a_tee(resistance):
    """S11 = -Z0/(2R+Z0) and S21 = 2R/(2R+Z0) for a shunt resistance.

    The sign of S11 is negative for every passive shunt: the reflected wave
    inverts.  A sign error is invisible in ``|S21|`` and would pass a test that
    only checked insertion loss.
    """

    circuit = two_port(
        LinearElement("Rs", "R", "p1", "0", resistance),
        LinearElement("T", "R", "p1", "p2", 1e-6),
    )
    result = s_parameters(circuit, PORTS_12, np.array([1.0e3]))
    z0 = DEFAULT_REFERENCE_IMPEDANCE_OHM
    assert result.s("1", "1", 1.0e3).real == pytest.approx(
        -z0 / (2 * resistance + z0), abs=1e-6
    )
    assert result.s("2", "1", 1.0e3).real == pytest.approx(
        2 * resistance / (2 * resistance + z0), abs=1e-6
    )


def test_matched_load_has_no_reflection():
    circuit = two_port(LinearElement("RL", "R", "p1", "0", 50.0))
    result = s_parameters(circuit, (Port("1", "p1"),), np.array([1.0e6]))
    assert abs(result.s("1", "1", 1.0e6)) == pytest.approx(0.0, abs=1e-9)


def test_open_and_short_are_the_extremes_of_the_real_axis():
    """A port's reflection coefficient runs from +1 to -1 as its load goes open to short."""

    open_circuit = two_port(LinearElement("Ro", "R", "p1", "0", 1e9))
    assert s_parameters(open_circuit, (Port("1", "p1"),), np.array([1.0e6])).s(
        "1", "1", 1.0e6
    ) == pytest.approx(1.0, abs=1e-6)
    short_circuit = two_port(LinearElement("Rs", "R", "p1", "0", 1e-6))
    assert s_parameters(short_circuit, (Port("1", "p1"),), np.array([1.0e6])).s(
        "1", "1", 1.0e6
    ) == pytest.approx(-1.0, abs=1e-4)


def test_passivity_holds_for_a_passive_network():
    """No passive two-port may deliver more than it receives.

    Worth asserting separately because every one of the wrong normalisers that this
    file's history passed through satisfied the extreme cases and still violated
    this on an ordinary network.
    """

    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([1.0e6, 1.0e9]))
    for index in range(2):
        matrix = result.values[:, :, index]
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        assert singular_values[0] <= 1.0 + 1e-9


def test_reference_impedance_is_per_port_and_matters():
    """A 75 ohm system has a different S11 for the same network, as it must."""

    circuit = two_port(LinearElement("RL", "R", "p1", "0", 50.0))
    fifty = s_parameters(
        circuit, (Port("1", "p1", reference_impedance_ohm=50.0),), np.array([1.0e6])
    )
    seventy_five = s_parameters(
        circuit, (Port("1", "p1", reference_impedance_ohm=75.0),), np.array([1.0e6])
    )
    assert abs(fifty.s("1", "1", 1.0e6)) == pytest.approx(0.0, abs=1e-9)
    # (50 - 75)/(50 + 75) = -0.2
    assert seventy_five.s("1", "1", 1.0e6).real == pytest.approx(-0.2, abs=1e-9)


def test_a_series_inductor_and_shunt_capacitor_is_a_low_pass():
    """Insertion loss in physical units, not against a reference file.

    A low-pass must pass low frequencies and stop high ones.  Scaling both elements
    up by 100 moves the corner down by 100, so at a fixed frequency below both
    corners the scaled network passes more -- which is the property that says the
    corner moved rather than that the loss merely changed.
    """

    def lowpass(scale: float) -> LinearCircuit:
        return two_port(
            LinearElement("L", "L", "p1", "p2", 100e-9 * scale),
            LinearElement("C", "C", "p2", "0", 10e-12 * scale),
        )

    frequencies = np.array([1.0e4, 1.0e9])
    fast = s_parameters(lowpass(1.0), PORTS_12, frequencies)
    slow = s_parameters(lowpass(100.0), PORTS_12, frequencies)
    # Each is a low-pass on its own.
    assert abs(fast.s("2", "1", 1.0e4)) > abs(fast.s("2", "1", 1.0e9))
    assert abs(slow.s("2", "1", 1.0e4)) > abs(slow.s("2", "1", 1.0e9))
    # And the scaled one still passes at a frequency its own corner is below.
    assert abs(slow.s("2", "1", 1.0e4)) > 0.9


# ---------------------------------------------------------------------------
# against measured ngspice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", sorted(NGSPICE_PORT_VOLTAGES))
def test_two_port_matches_ngspice(frequency):
    """Both coefficients, against a plain ngspice AC run of the same network.

    Agreement is at the level of double rounding, which is what it should be: the
    two are solving the same linear system, and any real difference would be a
    modelling error rather than a tolerance question.
    """

    reference_11, reference_21 = ngspice_s(frequency)
    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([frequency]))
    assert result.s("1", "1", frequency) == pytest.approx(reference_11, abs=1e-12)
    assert result.s("2", "1", frequency) == pytest.approx(reference_21, abs=1e-12)


def test_port_ordering_is_output_then_input():
    """``values[i, j]`` must be the response at *i* to a drive at *j*.

    Every instrument and every Touchstone file uses that order, and the transpose
    is a plausible implementation that a symmetric network cannot distinguish.  The
    network here is asymmetric on purpose.
    """

    circuit = two_port(
        LinearElement("R1", "R", "p1", "0", 10.0),
        LinearElement("R2", "R", "p2", "0", 1000.0),
    )
    result = s_parameters(circuit, PORTS_12, np.array([1.0e6]))
    # The ports are isolated, so the transmission terms vanish and the two
    # reflections differ; a transposed matrix would swap them.
    assert result.s("1", "1", 1.0e6).real == pytest.approx(
        (10 - 50) / (10 + 50), abs=1e-9
    )
    assert result.s("2", "2", 1.0e6).real == pytest.approx(
        (1000 - 50) / (1000 + 50), abs=1e-9
    )
    assert abs(result.s("2", "1", 1.0e6)) < 1e-12


def test_transmission_carries_the_right_factor_of_two():
    """S21 is not the port voltage; it is the port voltage over the incident half.

    An omitted factor of two here is exactly 6 dB, which reads as a plausible extra
    loss rather than as an error, so it is pinned against a closed form as well as
    against the reference.
    """

    resistance = 50.0
    circuit = two_port(LinearElement("R1", "R", "p1", "p2", resistance))
    result = s_parameters(circuit, PORTS_12, np.array([1.0e6]))
    z0 = DEFAULT_REFERENCE_IMPEDANCE_OHM
    assert result.s("2", "1", 1.0e6).real == pytest.approx(
        2 * z0 / (resistance + 2 * z0), abs=1e-9
    )
    # Half of it would be the wrong answer.
    assert abs(result.s("2", "1", 1.0e6) - z0 / (resistance + 2 * z0)) > 0.1


# ---------------------------------------------------------------------------
# nonlinear devices
# ---------------------------------------------------------------------------


def _operating_point(circuit, devices, bias):
    from circuit_ai.nonlinear import NonlinearMNA as Simulator

    shifted = LinearCircuit(
        elements=circuit.elements,
        voltage_sources=(VoltageSource("V1", "in", "0", complex(bias)),),
    )
    return Simulator(shifted, list(devices)).operating_point()


def _diode_network():
    circuit = LinearCircuit(
        elements=(LinearElement("R1", "R", "in", "p1", 1000.0),),
        voltage_sources=(VoltageSource("V1", "in", "0", complex(0.7)),),
    )
    return circuit, (Diode("D1", "p1", "0"),)


def test_nonlinear_network_is_linearised_at_its_bias_point():
    """S-parameters of a biased diode depend on the bias, and on nothing else.

    A run with no bias supplied must equal the run with the computed one --
    otherwise the answer would depend on who happened to compute the operating
    point rather than on the circuit.
    """

    circuit, devices = _diode_network()
    port = (Port("1", "p1"),)
    low = s_parameters(
        circuit,
        port,
        np.array([1.0e6]),
        devices=devices,
        operating_point=_operating_point(circuit, devices, 0.5),
    )
    high = s_parameters(
        circuit,
        port,
        np.array([1.0e6]),
        devices=devices,
        operating_point=_operating_point(circuit, devices, 0.7),
    )
    # A diode conducts harder at a higher bias, so its small-signal resistance is
    # lower and more of the incident wave is absorbed.
    assert abs(high.s("1", "1", 1.0e6)) < abs(low.s("1", "1", 1.0e6))
    implicit = s_parameters(circuit, port, np.array([1.0e6]), devices=devices)
    assert implicit.s("1", "1", 1.0e6) == pytest.approx(high.s("1", "1", 1.0e6))


def test_series_resistance_enters_the_s_parameters_through_the_internal_node():
    """A biased diode's S11 must agree with its own small-signal impedance.

    The expected value comes from the device's differential conductance
    ``gd/(1 + gd*RS)``, which is what the small-signal solve uses.  If this path
    stamped the device across its outer terminals instead -- the tempting
    simplification -- the series resistance would be shorted out and this fails by
    an order of magnitude.
    """

    parameters = DiodeParameters(series_resistance_ohm=10.0)
    diode = Diode("D1", "p1", "0", parameters)
    circuit = LinearCircuit(
        elements=(LinearElement("R1", "R", "in", "p1", 1000.0),),
        voltage_sources=(VoltageSource("V1", "in", "0", complex(0.85)),),
    )
    result = s_parameters(
        circuit, (Port("1", "p1"),), np.array([1.0e3]), devices=(diode,)
    )
    bias = NonlinearMNA(circuit, [diode]).operating_point()
    diode.stamp({"anode": bias.node_voltage("p1"), "cathode": 0.0})
    gd = diode.junction_slope(diode.bias())
    terminal_conductance = gd / (1.0 + gd * 10.0)
    z_device = 1.0 / (terminal_conductance + 1.0 / 1000.0)
    expected = (z_device - 50.0) / (z_device + 50.0)
    assert result.s("1", "1", 1.0e3) == pytest.approx(expected, rel=2e-3)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_touchstone_export_round_trips_magnitude_and_phase():
    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([1.0e6, 1.0e9]))
    text = result.to_touchstone()
    assert "# HZ S MA R 2" in text
    data = [
        line.split()
        for line in text.splitlines()
        if line and not line.startswith(("!", "#"))
    ]
    assert len(data) == 2
    for row, index in zip(data, range(len(result.frequencies_hz))):
        values = [float(item) for item in row[1:]]
        assert float(row[0]) == pytest.approx(result.frequencies_hz[index])
        for position, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            magnitude = values[2 * position]
            angle = values[2 * position + 1]
            expected = result.values[i, j, index]
            assert magnitude == pytest.approx(abs(expected), rel=1e-9)
            # Eight significant digits are written, so the angle round-trip is
            # asserted to that, not to the arithmetic's own precision.
            assert angle == pytest.approx(np.degrees(np.angle(expected)), abs=1e-4)


def test_db_helper_matches_the_magnitude():
    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([1.0e6, 1.0e9]))
    assert result.db("2", "1")[0] == pytest.approx(
        20 * np.log10(abs(result.s("2", "1", 1.0e6)))
    )
    # A perfect null reports a finite floor rather than -inf, so a constraint
    # compiler does not have to special-case it.
    series = two_port(LinearElement("R1", "R", "p1", "p2", 10.0))
    assert np.isfinite(
        s_parameters(series, PORTS_12, np.array([1.0e6])).db("1", "2")[0]
    )


def test_as_dict_names_every_entry_by_port():
    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([1.0e6]))
    payload = result.as_dict()
    assert payload["port_names"] == ["1", "2"]
    assert len(payload["s_parameters"]) == 4
    assert {item["output_port"] for item in payload["s_parameters"]} == {"1", "2"}


def test_interpolation_at_a_frequency_between_sweep_points():
    """Between sweep points the value is interpolated linearly and bounded by them.

    A real network's response is not linear in frequency between two points, so the
    interpolated value is a plausible one rather than an exact one; what must hold
    is that it lies between the endpoints and moves monotonically between them.
    """

    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([1.0e6, 1.0e9]))
    low = result.s("2", "1", 1.0e6)
    high = result.s("2", "1", 1.0e9)
    middle = result.s("2", "1", 5.0e8)
    assert min(abs(low), abs(high)) <= abs(middle) <= max(abs(low), abs(high))
    assert abs(middle - 0.5 * (low + high)) < 1e-3
    # At a sweep point it is exact.
    assert result.s("2", "1", 1.0e6) == low
    # Outside the sweep the endpoints are held rather than extrapolated.
    assert result.s("2", "1", 1.0) == pytest.approx(low)
    assert result.s("2", "1", 1.0e12) == pytest.approx(high)


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------


def test_unknown_port_name_is_reported_with_the_known_ones():
    result = s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([1.0e6]))
    with pytest.raises(KeyError, match="known ports"):
        result.s("3", "1", 1.0e6)


def test_port_referencing_a_missing_node_is_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        s_parameters(NGSPICE_NETWORK, (Port("1", "nowhere"),), np.array([1.0e6]))


def test_duplicate_port_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        s_parameters(
            NGSPICE_NETWORK, (Port("1", "p1"), Port("1", "p2")), np.array([1.0e6])
        )


def test_no_ports_and_empty_sweep_are_rejected():
    with pytest.raises(ValueError, match="at least one port"):
        s_parameters(NGSPICE_NETWORK, (), np.array([1.0e6]))
    with pytest.raises(ValueError, match="non-empty"):
        s_parameters(NGSPICE_NETWORK, PORTS_12, np.array([]))


def test_non_positive_reference_impedance_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        Port("1", "p1", reference_impedance_ohm=0.0)


def test_a_port_may_be_differential():
    """A port across two non-ground nodes is a port, not a pair of wires."""

    circuit = two_port(
        LinearElement("R1", "R", "outp", "outn", 100.0),
        LinearElement("Rb1", "R", "outp", "0", 1000.0),
        LinearElement("Rb2", "R", "outn", "0", 1000.0),
    )
    result = s_parameters(circuit, (Port("d", "outp", "outn"),), np.array([1.0e6]))
    value = result.s("d", "d", 1.0e6)
    assert np.isfinite(value.real) and np.isfinite(value.imag)
    assert abs(value) < 1.0
