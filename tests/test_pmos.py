"""Channel-only PMOS: polarity, independent roots, derivatives and live SPICE.

The p-channel device is not the n-channel one with a minus sign.  Its gate drive is
measured from the *higher* of its two channel terminals, so a model that references
the gate to the low terminal still produces a smooth, plausible-looking transfer
curve that is simply the wrong device.  The polarity tests below exist because that
error is invisible in any check that only asks "does the current increase
somewhere".
"""

import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from circuit_ai.devices import (
    NMOS,
    PMOS,
    ConvergenceError,
    NMOSParameters,
    PMOSParameters,
)
from circuit_ai.graph import (
    ComponentInstance,
    ParameterBinding,
    TerminalConnection,
    linear_circuit_to_graph,
    primitive_model_ref,
)
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.nonlinear import NonlinearMNA
from circuit_ai.simulation import (
    AnalysisSpec,
    Excitation,
    NonlinearMNASimulatorBackend,
    ObservableSpec,
    OperatingConditions,
    SimulationExecutor,
    SimulationRequest,
    SimulationResultCache,
    SimulationStatus,
    SweepSpec,
)
from circuit_ai.spice_discovery import find_executable

SOURCE_RAIL = 5.0


def clamped(vd=-1.3, vg=0.0, vs=SOURCE_RAIL):
    """Every terminal held by a source; the source rail is the PMOS source."""

    return LinearCircuit(
        elements=(),
        voltage_sources=(
            VoltageSource("VD", "d", "0", complex(vd)),
            VoltageSource("VG", "g", "0", complex(vg)),
            VoltageSource("VS", "s", "0", complex(vs)),
        ),
    )


def loaded(vg=2.5, supply=SOURCE_RAIL, resistance=1000.0):
    """PMOS common source: source on the rail, drain pulled down by *resistance*."""

    return LinearCircuit(
        elements=(LinearElement("RD", "R", "d", "0", resistance),),
        voltage_sources=(
            VoltageSource("VD", "vdd", "0", complex(supply)),
            VoltageSource("VG", "g", "0", complex(vg)),
        ),
    )


def parameters(lam=0.02):
    return PMOSParameters(
        threshold_voltage_v=1.0,
        transconductance_a_per_v2=1e-4,
        width_m=10e-6,
        length_m=1e-6,
        channel_length_modulation_per_v=lam,
    )


# ---------------------------------------------------------------------------
# polarity: the failure mode this device actually has
# ---------------------------------------------------------------------------


def test_conduction_needs_the_gate_below_the_source():
    """A PMOS is on when its gate sits below its source, and off when it does not.

    Both halves are asserted because only one of them is satisfied by a model that
    references the gate to the wrong terminal.
    """

    device = PMOS("M1", "d", "g", "s", parameters(lam=0.0))
    on = device.stamp({"drain": 0.0, "gate": 0.0, "source": SOURCE_RAIL})
    off = device.stamp({"drain": 0.0, "gate": SOURCE_RAIL, "source": SOURCE_RAIL})
    assert on.terminal_currents["source"] == pytest.approx(0.008, rel=1e-12)
    assert on.terminal_currents["drain"] == pytest.approx(-0.008, rel=1e-12)
    assert off.terminal_currents["source"] == 0.0
    assert off.terminal_currents["drain"] == 0.0


def test_transfer_curve_falls_as_the_gate_rises():
    """Drain current magnitude decreases monotonically with rising gate voltage."""

    device = PMOS("M1", "d", "g", "s", parameters())
    magnitudes = [
        abs(device.stamp({"drain": 0.0, "gate": vg, "source": SOURCE_RAIL})
            .terminal_currents["drain"])
        for vg in (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0)
    ]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert magnitudes[0] > 1e-3
    assert magnitudes[-1] == 0.0


def test_referencing_the_gate_to_the_low_terminal_would_be_a_different_device():
    """Pins the quantity that decides conduction, so a refactor cannot quietly drop it.

    With the source on the rail and the gate at the rail the device is off.  The
    *low*-referenced quantity ``vg - v_drain`` would be strongly positive there, so a
    model that used it would conduct at full strength instead of not at all -- the
    comparison below is what that error looks like.
    """

    device = PMOS("M1", "d", "g", "s", parameters(lam=0.0))
    bias = {"drain": 0.0, "gate": SOURCE_RAIL, "source": SOURCE_RAIL}
    assert device.stamp(bias).terminal_currents["source"] == 0.0

    # The n-channel rule would read this bias as a deeply overdriven device.
    low_referenced = bias["gate"] - bias["drain"] - 1.0
    assert low_referenced > 0.0
    n_channel_lookalike = 0.5 * 1e-3 * low_referenced**2
    assert n_channel_lookalike == pytest.approx(0.008, rel=1e-12)
    # ... and the p-channel answer is zero current, so the two are not the same
    # device described two ways.
    assert n_channel_lookalike > 0.0


def test_parameters_reject_a_negative_threshold():
    """PMOS stores |VTO|; a caller passing the SPICE sign is told, not silently used."""

    with pytest.raises(ValueError, match="magnitude"):
        PMOSParameters(threshold_voltage_v=-1.0)


# ---------------------------------------------------------------------------
# KCL, independent roots, derivatives
# ---------------------------------------------------------------------------


def test_clamped_branch_current_obeys_kcl_and_preserves_gate_isolation():
    device = PMOS("M1", "d", "g", "s", parameters())
    point = NonlinearMNA(clamped(), [device]).operating_point()
    # Saturation at vds = 6.3, overdrive = 4: 0.5*1e-3*16*(1+0.02*6.3).
    expected = 0.5 * 1e-3 * 16.0 * (1.0 + 0.02 * 6.3)
    # The two channel ends carry the same current in opposite senses; the individual
    # signs belong to the source branches, the invariant belongs to the device.
    assert point.branch_current("VS") == pytest.approx(-expected, abs=1e-12)
    assert point.branch_current("VD") == pytest.approx(expected, abs=1e-12)
    assert point.branch_current("VS") + point.branch_current("VD") == pytest.approx(0.0, abs=1e-15)
    assert point.branch_current("VG") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("vg", [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
def test_resistive_load_matches_independent_scalar_root(vg):
    """Load line solved independently of the device and of the MNA helpers."""

    def drain_current(vd):
        overdrive = max(SOURCE_RAIL - vg - 1.0, 0.0)
        vds = SOURCE_RAIL - vd
        if overdrive <= 0.0 or vds <= 0.0:
            return 0.0
        if vds < overdrive:
            channel = 1e-3 * (overdrive * vds - 0.5 * vds**2)
        else:
            channel = 0.5e-3 * overdrive**2
        return channel * (1.0 + 0.02 * vds)

    # KCL at the drain: the channel delivers drain_current, the resistor takes vd/R.
    expected = brentq(lambda vd: drain_current(vd) - vd / 1000.0, 0.0, SOURCE_RAIL)
    point = NonlinearMNA(loaded(vg), [PMOS("M1", "d", "g", "vdd", parameters())]).operating_point()
    assert point.node_voltage("d") == pytest.approx(expected, abs=1e-8)
    assert point.branch_current("VD") == pytest.approx(-drain_current(expected), abs=1e-11)
    if vg == 2.5:
        # Hand root: vd = 1000*(1.2375e-3 - 2.25e-5*vd) -> vd = 1.2375/1.0225.
        assert point.node_voltage("d") == pytest.approx(1.2375 / 1.0225, abs=1e-9)


@pytest.mark.parametrize(
    "vd,vg,vs",
    [(0.0, 0.0, 5.0), (5.0, 0.0, 5.0), (2.0, 2.5, 5.0), (-1.3, 3.0, 5.0),
     (6.7, 0.0, 5.0), (5.0, 1.5, 5.0), (1.0, 4.5, 5.0)],
)
def test_jacobian_matches_finite_difference_and_conserves_charge(vd, vg, vs):
    device = PMOS("M1", "d", "g", "s", parameters())
    voltages = {"drain": vd, "gate": vg, "source": vs}
    stamp = device.stamp(voltages)
    assert sum(stamp.terminal_currents.values()) == pytest.approx(0.0, abs=1e-15)

    step = 1e-6
    for terminal in ("drain", "gate", "source"):
        shifted = dict(voltages)
        shifted[terminal] += step
        upper = device.stamp(shifted).terminal_currents
        shifted[terminal] -= 2 * step
        lower = device.stamp(shifted).terminal_currents
        for current_terminal in ("drain", "source"):
            numeric = (upper[current_terminal] - lower[current_terminal]) / (2 * step)
            analytic = stamp.jacobian[(current_terminal, terminal)]
            assert numeric == pytest.approx(analytic, rel=1e-5, abs=1e-9)


def test_reverse_operation_mirrors_the_forward_characteristic():
    """Swapping the two channel ends mirrors the current, as the physics requires."""

    forward = PMOS("M1", "d", "g", "s", parameters())
    reversed_ = PMOS("M2", "d", "g", "s", parameters())
    a = forward.stamp({"drain": -1.3, "gate": 0.0, "source": SOURCE_RAIL})
    b = reversed_.stamp({"drain": SOURCE_RAIL, "gate": 0.0, "source": -1.3})
    # Same channel, ends exchanged: what entered at the source of the first now enters
    # at the drain of the second, so the two currents agree at opposite terminals and
    # the magnitudes match everywhere.
    assert a.terminal_currents["drain"] == pytest.approx(
        b.terminal_currents["source"], rel=1e-12
    )
    assert abs(a.terminal_currents["drain"]) == pytest.approx(
        abs(b.terminal_currents["drain"]), rel=1e-12
    )


def test_non_finite_voltages_are_refused():
    device = PMOS("M1", "d", "g", "s", parameters())
    with pytest.raises(ConvergenceError, match="non-finite"):
        device.stamp({"drain": float("nan"), "gate": 0.0, "source": SOURCE_RAIL})


def test_pmos_and_nmos_are_distinct_devices_with_the_same_interface():
    """The two channel models share a protocol, so the backend can treat them alike."""

    n = NMOS("N", "d", "g", "s", NMOSParameters())
    p = PMOS("P", "d", "g", "s", PMOSParameters())
    assert n.terminals() == p.terminals()
    # Saturated n-channel at vg above the low end; saturated p-channel with the gate
    # below the high end.  Both are "on" in their own sense and both conduct.
    assert n.stamp({"drain": 5.0, "gate": 3.5, "source": 0.0}).terminal_currents["drain"] > 0
    assert p.stamp({"drain": 0.0, "gate": 0.0, "source": 5.0}).terminal_currents["drain"] < 0


# ---------------------------------------------------------------------------
# live ngspice
# ---------------------------------------------------------------------------


def _ngspice(deck, files):
    executable = find_executable()
    if executable is None:
        pytest.skip("ngspice unavailable; independent analytic tests still run")
    with tempfile.TemporaryDirectory(prefix="lattice_pmos_") as directory:
        root = Path(directory)
        netlist = root / "model.cir"
        netlist.write_text(deck, encoding="ascii")
        result = subprocess.run(
            [str(executable), "-n", "-b", "-o", str(root / "run.log"), str(netlist)],
            cwd=root, capture_output=True, text=True, timeout=60, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log = (root / "run.log").read_text(errors="replace")
        assert result.returncode == 0, log
        assert "error:" not in log.lower(), log
        for file in files:
            assert (root / file).is_file(), log
        return {file: np.loadtxt(root / file) for file in files}


@pytest.mark.parametrize("lam", [0.0, 0.02])
def test_channel_dc_sweep_against_live_ngspice(lam):
    """196 points per channel-length parameter, forward and reversed.

    The bulk is parked above every swept potential so both body junctions stay
    reverse biased.  Without that, ngspice's own MOS1 conducts through the
    drain-body diode once the drain rises above the source, and its answer runs away
    -- it reaches ``3.5e14 A`` on the last point of an unclamped sweep.  That is a
    model difference, not a solver error: this device deliberately has no body diode,
    and the comparison is only meaningful where the diode is off.
    """

    gates = [0.0, 1.0, 2.0, 3.0, 3.5, 4.5]
    commands = "\n".join(
        f"alter VG = {vg}\ndc VD -1.3 6.7 0.25\nwrdata sweep_{i}.dat v(d) i(VSENSE)"
        for i, vg in enumerate(gates)
    )
    files = [f"sweep_{i}.dat" for i in range(len(gates))]
    reference = _ngspice(
        f"""PMOS channel reference, LAMBDA={lam}, body diodes reverse biased
VD drive 0 5
VSENSE drive d 0
VG g 0 3.5
VS s 0 5
VB b 0 8
M1 d g s b MOD W=10u L=1u
.model MOD PMOS (LEVEL=1 VTO=-1 KP=1e-4 LAMBDA={lam} GAMMA=0 PHI=0.6)
.options temp=27 tnom=27 reltol=1e-9 abstol=1e-14 vntol=1e-10
.control
set numdgt=15
{commands}
quit
.endc
.end
""",
        files,
    )
    device = PMOS("M1", "d", "g", "s", parameters(lam))
    max_error = 0.0
    for index, vg in enumerate(gates):
        rows = reference[files[index]]
        assert rows.shape == (33, 4)
        for _, vd, _, measured in rows:
            circuit = LinearCircuit(
                elements=(),
                voltage_sources=(
                    VoltageSource("VD", "drive", "0", complex(vd)),
                    VoltageSource("VSENSE", "drive", "d", complex(0.0)),
                    VoltageSource("VS", "s", "0", complex(SOURCE_RAIL)),
                    VoltageSource("VG", "g", "0", complex(vg)),
                ),
            )
            point = NonlinearMNA(circuit, [device]).operating_point()
            simulated = point.branch_current("VSENSE")
            max_error = max(max_error, abs(simulated - measured))
            assert simulated == pytest.approx(measured, rel=1e-6, abs=2e-10)
            # The channel is a two-terminal path: whatever enters the drain leaves
            # the source, and the gate draws nothing.
            assert point.branch_current("VD") == pytest.approx(-simulated, abs=1e-12)
            assert point.branch_current("VG") == pytest.approx(0.0, abs=1e-12)
    print(f"PMOS channel DC LAMBDA={lam}: {len(gates) * 33} points, "
          f"max current error={max_error:.3e} A")
    assert max_error < 1e-9


def test_loaded_bias_and_ac_against_live_ngspice():
    data = _ngspice(
        """PMOS common-source amplifier, channel only
VD vdd 0 DC 5 AC 0
VG g 0 DC 2.5 AC 1
RD d 0 1k
CLOAD d 0 1n
M1 d g vdd vdd MOD W=10u L=1u
.model MOD PMOS (LEVEL=1 VTO=-1 KP=1e-4 LAMBDA=0.02 GAMMA=0 PHI=0.6)
.options temp=27 tnom=27 reltol=1e-9 abstol=1e-14 vntol=1e-10
.control
set numdgt=15
op
wrdata bias.dat v(d) i(VD)
ac dec 2 1 1e6
wrdata ac.dat v(d) i(VD)
quit
.endc
.end
""",
        ["bias.dat", "ac.dat"],
    )
    device = PMOS("M1", "d", "g", "vdd", parameters(0.02))
    circuit = replace(
        loaded(2.5),
        elements=loaded(2.5).elements + (LinearElement("CL", "C", "d", "0", 1e-9),),
    )
    bias = NonlinearMNA(circuit, [device]).operating_point()
    assert bias.node_voltage("d") == pytest.approx(data["bias.dat"][1], abs=1e-7)
    assert bias.branch_current("VD") == pytest.approx(data["bias.dat"][3], abs=2e-10)

    ac_circuit = replace(
        circuit,
        voltage_sources=(
            VoltageSource("VD", "vdd", "0", 0.0),
            VoltageSource("VG", "g", "0", 1.0),
        ),
    )
    reference = data["ac.dat"]
    assert reference.shape == (13, 6)
    ac = NonlinearMNA(ac_circuit, [device]).solve_ac(reference[:, 0], operating_point=bias)
    expected = reference[:, 1] + 1j * reference[:, 2]
    assert ac.node_voltage("d") == pytest.approx(expected, abs=1e-7)
    assert ac.branch_current("VD") == pytest.approx(
        reference[:, 4] + 1j * reference[:, 5], abs=2e-10
    )
    max_error = max(abs(ac.node_voltage("d") - expected))
    print(f"PMOS AC: 13 points, max voltage error={max_error:.3e} V")
    assert max_error < 1e-6

    # The same amplifier through the actual graph/resolver/executor path.
    graph, request = pmos_graph(ac=True, frequencies=tuple(reference[:, 0]),
                                load_capacitance_f=1e-9)
    unified = execute_graph(graph, request)
    assert unified.status == SimulationStatus.PASSED, unified.diagnostics
    assert unified.waveforms["out"].values == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# the two channel models composed: a CMOS inverter
# ---------------------------------------------------------------------------


def inverter_circuit(vin):
    """Rail, input drive and a 1 pF load; the load only gives ``out`` a node.

    A DC solve needs the output node to exist in the linear circuit, and a
    capacitor is open at DC, so it changes no bias.  Its 1 pF is also what a real
    gate input presents.
    """

    return LinearCircuit(
        elements=(LinearElement("CL", "C", "out", "0", 1e-12),),
        voltage_sources=(
            VoltageSource("VDD", "vdd", "0", complex(SOURCE_RAIL)),
            VoltageSource("VIN", "in", "0", complex(vin)),
        ),
    )


def inverter_devices(lam=0.02):
    return [
        NMOS("MN", "out", "in", "0",
             NMOSParameters(channel_length_modulation_per_v=lam)),
        PMOS("MP", "out", "in", "vdd",
             PMOSParameters(channel_length_modulation_per_v=lam)),
    ]


def test_cmos_inverter_restores_both_rails():
    """Logic levels: the output sits at a rail for either input rail.

    This is the property that makes the pair worth more than either device alone,
    and it is a direct consequence of the PMOS polarity: with the gate reference
    reversed, the p-channel device would conduct at the wrong time and the output
    would not reach a rail at all.
    """

    low = NonlinearMNA(inverter_circuit(0.0), inverter_devices()).operating_point()
    high = NonlinearMNA(inverter_circuit(SOURCE_RAIL), inverter_devices()).operating_point()
    assert low.node_voltage("out") == pytest.approx(SOURCE_RAIL, abs=1e-6)
    assert high.node_voltage("out") == pytest.approx(0.0, abs=1e-6)
    # Neither rail draws supply current: exactly one device is off at each extreme.
    assert abs(low.branch_current("VDD")) < 1e-9
    assert abs(high.branch_current("VDD")) < 1e-9


def test_cmos_inverter_switching_threshold_is_half_the_rail():
    """Matched devices put the switching point at VDD/2, and the gain is large there.

    The symmetry is a check on both models at once: an error in either threshold or
    in the p-channel gate reference moves the crossover away from half the rail.  The
    gain is then compared with ``(gm_n + gm_p)/(gds_n + gds_p)`` from the two devices'
    own stamps, so the slope is checked against the small-signal model rather than
    against a threshold picked to fit.
    """

    devices = inverter_devices()
    bias = NonlinearMNA(inverter_circuit(SOURCE_RAIL / 2), devices).operating_point()
    assert bias.node_voltage("out") == pytest.approx(SOURCE_RAIL / 2, abs=1e-6)

    step = 1e-3
    below = NonlinearMNA(inverter_circuit(SOURCE_RAIL / 2 - step), devices).operating_point()
    above = NonlinearMNA(inverter_circuit(SOURCE_RAIL / 2 + step), devices).operating_point()
    slope = abs(above.node_voltage("out") - below.node_voltage("out")) / (2 * step)

    transconductance = 0.0
    output_conductance = 0.0
    for device in devices:
        stamp = device.stamp(
            {
                "drain": bias.node_voltage(device.drain),
                "gate": bias.node_voltage(device.gate),
                "source": bias.node_voltage(device.source),
            }
        )
        transconductance += abs(stamp.jacobian[("drain", "gate")])
        output_conductance += abs(stamp.jacobian[("drain", "drain")])
    expected_gain = transconductance / output_conductance
    assert expected_gain > 50  # the trip point is a high-gain region
    assert slope == pytest.approx(expected_gain, rel=0.02)


def test_cmos_inverter_transfer_matches_live_ngspice():
    """51 bias points, both devices active at once, against an external run.

    Both body junctions stay reverse biased across 0..VDD for a CMOS pair -- the
    n-channel bulk at ground never forward biases against a non-negative drain, and
    the p-channel bulk at the rail never forward biases against a drain below it --
    so this is a fair comparison for a model that deliberately omits body diodes.
    """

    data = _ngspice(
        """CMOS inverter transfer characteristic
VDD vdd 0 5
VIN in 0 0
MP out in vdd vdd PM W=10u L=1u
MN out in 0 0 NM W=10u L=1u
.model NM NMOS (LEVEL=1 VTO=1 KP=1e-4 LAMBDA=0.02 GAMMA=0 PHI=0.6)
.model PM PMOS (LEVEL=1 VTO=-1 KP=1e-4 LAMBDA=0.02 GAMMA=0 PHI=0.6)
.options temp=27 tnom=27 reltol=1e-9 abstol=1e-14 vntol=1e-10
.control
set numdgt=15
dc VIN 0 5 0.1
wrdata inv.dat v(in) v(out) i(VDD)
quit
.endc
.end
""",
        ["inv.dat"],
    )["inv.dat"]
    assert data.shape == (51, 6)
    # wrdata writes (scale, value) per requested vector, so the circuit quantities
    # sit in the odd columns.
    inputs = data[:, 0]
    reference = data[:, 3]
    supply = data[:, 5]
    max_voltage_error = 0.0
    max_current_error = 0.0
    previous = None
    for vin, expected, expected_supply in zip(inputs, reference, supply):
        point = NonlinearMNA(
            inverter_circuit(vin), inverter_devices()
        ).operating_point()
        simulated = point.node_voltage("out")
        max_voltage_error = max(max_voltage_error, abs(simulated - expected))
        max_current_error = max(
            max_current_error, abs(point.branch_current("VDD") - expected_supply)
        )
        assert simulated == pytest.approx(expected, abs=1e-7)
        # The characteristic is monotonically falling: this is an inverter.
        if previous is not None:
            assert simulated <= previous + 1e-9
        previous = simulated
    print(f"CMOS inverter: 51 points, max |vout| error={max_voltage_error:.3e} V, "
          f"max |i(VDD)| error={max_current_error:.3e} A")
    assert max_voltage_error < 1e-7
    assert max_current_error < 1e-9


# ---------------------------------------------------------------------------
# unified graph -> resolver -> executor integration
# ---------------------------------------------------------------------------


def pmos_graph(*, ac=False, frequencies=(1000.0, 10000.0), load_capacitance_f=0.0):
    graph = linear_circuit_to_graph(
        loaded(2.5), name="pmos_common_source", input_node="g", output_node="d"
    )
    pmos = ComponentInstance(
        "pmos",
        "M1",
        primitive_model_ref("pmos_channel"),
        (
            TerminalConnection("drain", "d"),
            TerminalConnection("gate", "g"),
            TerminalConnection("source", "vdd"),
        ),
        (
            ParameterBinding("width_m", "um", 10.0),
            ParameterBinding("channel_length_modulation_per_v", "1/V", 0.02),
        ),
    )
    components = tuple(
        replace(component, instance_id="gate_bias")
        if component.reference == "VG"
        else component
        for component in graph.components
    ) + (pmos,)
    if load_capacitance_f:
        components += (
            ComponentInstance(
                "load_cap",
                "CL",
                primitive_model_ref("C"),
                (TerminalConnection("p", "d"), TerminalConnection("n", "0")),
                (ParameterBinding("value", "nF", load_capacitance_f * 1e9),),
            ),
        )
    graph = replace(
        graph,
        components=components,
        family="arbitrary_name",
        preferred_solver_ids=("nonlinear_mna",),
    )
    request = SimulationRequest(
        request_id="unified_pmos",
        graph_id=graph.graph_id,
        analysis=(
            AnalysisSpec(
                "small_signal_ac",
                (SweepSpec("frequency", "Hz", values=frequencies),),
            )
            if ac
            else AnalysisSpec("dc_operating_point")
        ),
        conditions=OperatingConditions(temperature_c=27.0),
        fidelity="nonlinear_device",
        excitations=(
            (Excitation("signal", "ac_voltage", "component", "gate_bias", {"magnitude": 1.0}),)
            if ac
            else ()
        ),
        requested_observables=(ObservableSpec("out", "voltage", "V", "port", "output"),),
    )
    return graph, request


def execute_graph(graph, request):
    return SimulationExecutor(cache=SimulationResultCache()).execute(
        NonlinearMNASimulatorBackend(), graph, [request]
    )[0]


def test_graph_backend_runs_a_pmos_bias_and_marks_the_model_as_channel_only():
    graph, request = pmos_graph()
    result = execute_graph(graph, request)
    assert result.status == SimulationStatus.PASSED, result.diagnostics
    assert result.scalars["out"].value == pytest.approx(1.2375 / 1.0225, abs=1e-6)
    assert any(model.kind == "pmos_channel" for model in result.model_manifest)
    assert result.fidelity == "nonlinear_device"
    assert result.statistics["operating_point"]["residual"] < 1e-8
    # The channel-only limitation must travel with the result, not stay in a doc.
    assert any(diagnostic.code == "channel_only_model" for diagnostic in result.diagnostics)


def test_graph_backend_rejects_a_temperature_the_channel_model_cannot_scale():
    graph, request = pmos_graph()
    rejected = execute_graph(
        graph, replace(request, conditions=OperatingConditions(temperature_c=85.0))
    )
    # Refused as unsupported rather than answered with an unmodelled assumption.
    assert rejected.status in (SimulationStatus.FAILED, SimulationStatus.UNSUPPORTED)
    assert "27 C" in rejected.diagnostics[0].message


def test_graph_backend_refuses_a_series_mosfet_pair():
    """A missing node is a request error, not a silently floating device."""

    graph, request = pmos_graph()
    broken = replace(
        graph,
        components=tuple(
            replace(component, connections=(TerminalConnection("drain", "d"),
                                            TerminalConnection("gate", "g"),
                                            TerminalConnection("source", "nowhere")))
            if component.reference == "M1"
            else component
            for component in graph.components
        ),
    )
    result = execute_graph(broken, request)
    assert result.status == SimulationStatus.FAILED


def test_ac_without_a_supplied_bias_computes_the_same_bias():
    """``solve_ac`` derives the operating point when not given one.

    The small-signal model is the Jacobian at the bias, so an AC sweep is only
    meaningful about a bias; the two routes must agree or one of them is using a
    different operating point than it reports.
    """

    device = PMOS("M1", "d", "g", "vdd", parameters())
    circuit = replace(
        loaded(2.5),
        elements=loaded(2.5).elements + (LinearElement("CL", "C", "d", "0", 1e-9),),
    )
    frequencies = np.array([1.0, 1e3, 1e6])
    implicit = NonlinearMNA(circuit, [device]).solve_ac(frequencies)
    bias = NonlinearMNA(circuit, [device]).operating_point()
    explicit = NonlinearMNA(circuit, [device]).solve_ac(frequencies, operating_point=bias)
    assert implicit.node_voltage("d") == pytest.approx(explicit.node_voltage("d"), abs=1e-12)
    # And the response rolls off, so the sweep is actually doing something: the pole
    # is the 1 kohm pull-down against the 1 nF load, near 160 kHz.
    assert abs(implicit.node_voltage("d")[0]) > 2 * abs(implicit.node_voltage("d")[-1])
