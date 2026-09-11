"""Nonlinear device models for the MNA solver.

Every model here is a *conductance* device: it supplies the current through
itself at a given set of terminal voltages, together with the derivatives of that
current with respect to those voltages.  That is exactly what the Newton solver
in :mod:`circuit_ai.nonlinear` iterates on, and exactly what a small-signal
analysis needs once evaluated at the operating point, so one model definition
serves both without a second, drifted implementation.

Where a model approximates a real device the approximation is stated, and the
error is quantified against an external simulator rather than asserted.  The
junction equations below are the reference implementation's own, reproduced in
its own form; where the published documentation and the reference's source code
disagree -- which happens, and is called out at the point it matters -- the
source is what runs and therefore what is implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

# Boltzmann's constant and the elementary charge, in SI.  Written out rather
# than taken from a units library because the device equations are already in
# volts and kelvin and these are the values the reference uses.
BOLTZMANN_J_PER_K = 1.3806226e-23
ELECTRON_CHARGE_C = 1.6021918e-19

#: Largest junction-voltage change accepted per Newton iteration, in thermal
#: voltages.  Large enough that a junction crossing several hundred millivolts
#: still converges in a handful of iterations, small enough that the exponential is
#: never evaluated far outside the range where a linearisation is meaningful.
_JUNCTION_STEP_LIMIT = 40.0


def thermal_voltage(temperature_k: float) -> float:
    """``k*T/q``, the voltage equivalent of temperature."""

    if temperature_k <= 0:
        raise ValueError("temperature must be positive")
    return BOLTZMANN_J_PER_K * temperature_k / ELECTRON_CHARGE_C


def bandgap_energy_ev(temperature_k: float) -> float:
    """Silicon band-gap energy at a temperature, in electron volts.

    The reference's fit.  It is a function of *one* temperature at a time,
    evaluated separately at the nominal and the simulated temperature, because
    the two appear on opposite sides of the saturation-current expression and it
    is their difference, not either alone, that drives the temperature response.
    """

    return 1.16 - 7.02e-4 * temperature_k**2 / (temperature_k + 1108.0)


class ConvergenceError(RuntimeError):
    """Raised when a nonlinear solve does not reach a solution."""


@dataclass(frozen=True)
class NonlinearStamp:
    """One device's contribution to the MNA system at a given bias.

    ``current`` is the device current, positive flowing from the positive
    terminal to the negative terminal through the device, and ``conductance`` is
    ``d(current)/d(voltage)`` for a two-terminal device.  ``jacobian`` carries the
    full derivative for a device controlled by more than one voltage.
    """

    current: float
    conductance: float
    jacobian: Mapping[tuple[str, str], float] = field(default_factory=dict)


class NonlinearDevice:
    """Base class for devices that supply a conductance stamp at a bias point."""

    name: str

    def terminals(self) -> tuple[tuple[str, str], ...]:
        """The (node, terminal-name) pairs this device stamps into the matrix."""

        raise NotImplementedError

    def stamp(self, voltages: Mapping[str, float]) -> NonlinearStamp:
        """Current and derivatives at the supplied terminal voltages."""

        raise NotImplementedError

    def reset(self) -> None:
        """Clear iteration state.  Called before each fresh solve."""

    def internal_nodes(self) -> tuple[str, ...]:
        """Extra matrix nodes this device needs between its terminals.

        A two-terminal device whose internals are not a single conductance must
        say so, because the alternative -- folding the internals into one effective
        terminal admittance -- is only valid while the fold is exact.  The solver
        appends these nodes to the matrix and asks the device to stamp the
        elements between them via :meth:`internal_elements`.
        """

        return ()

    def internal_elements(self) -> tuple[tuple[str, str, float], ...]:
        """``(node_a, node_b, value)`` linear elements between internal nodes."""

        return ()


def scaled_saturation_current(
    saturation_current_a: float,
    *,
    temperature_k: float,
    nominal_temperature_k: float,
    emission_coefficient: float,
    activation_energy_ev: float,
    temperature_exponent: float = 3.0,
) -> float:
    """``IS`` at the simulated temperature.

    The expression the reference actually evaluates::

        IS(T) = IS * exp( (T/Tnom - 1) * EG / (N*Vt)
                          + XTI/N * ln(T/Tnom) )

    Two things about it are worth stating, because both are places where the
    obvious-looking alternative is wrong and the error is invisible at the
    nominal temperature.

    First, the band-gap term is *linear* in the temperature ratio, not the
    difference of two band-gap-over-thermal-voltage terms.  The familiar
    ``IS*(T/Tnom)**(XTI/N) * exp(EG(Tnom)/(N*Vt(Tnom)) - EG(T)/(N*Vt(T)))`` is not
    what runs here, and is wrong by a factor of 1.76 at 85 C and 2.9 at 150 C.
    Second, the temperature ratio contributes through ``ln(T/Tnom)``, so the whole
    expression is exponentiated once and there is no ``(T/Tnom)**(XTI/N)`` factor
    outside it.

    Every one of those variants reduces to ``IS`` at the nominal temperature, so a
    spot check there cannot tell them apart.  The tests sweep -40 C to 150 C
    against measured responses for exactly that reason.
    """

    if temperature_k == nominal_temperature_k:
        return saturation_current_a
    vt = thermal_voltage(temperature_k)
    exponent = (temperature_k / nominal_temperature_k - 1.0) * activation_energy_ev / (
        emission_coefficient * vt
    ) + temperature_exponent / emission_coefficient * float(
        np.log(temperature_k / nominal_temperature_k)
    )
    return saturation_current_a * float(np.exp(exponent))


def junction_limiting_voltage(
    previous: float,
    target: float,
    *,
    thermal_voltage_v: float,
    saturation_current_a: float,
) -> float:
    """Limit an exponential junction's Newton step.

    An unguarded Newton step on an exponential either overshoots into a region
    where the linearisation is meaningless, or oscillates between two wrong
    answers.  The rule below is the standard one, and it is deliberately
    *asymmetric*: only a step that would push the junction further on is
    restricted.  A step that turns the junction off, or drives it into reverse
    breakdown, is left alone -- and getting that wrong is not a subtle
    difference.  Clamping the off-direction to a floor keeps a junction that
    should be off permanently forward biased, so the iteration settles into a
    small limit cycle and never converges, which is exactly the failure this code
    produced before the rule was split in two.

    ``saturation_current_a`` is the saturation current of whichever exponential
    dominates at this bias.  The critical voltage depends on it, so a device in
    breakdown is limited against the reverse exponential rather than the forward
    one.
    """

    critical = thermal_voltage_v * float(
        np.log(thermal_voltage_v / (np.sqrt(2.0) * saturation_current_a))
    )
    if target > critical:
        if previous > critical:
            # Already forward biased: hold the increase to a fixed number of
            # thermal voltages rather than letting the exponential run away from
            # the linearisation it was expanded about.
            limit = previous + thermal_voltage_v * 10.0
            if target > limit:
                return limit
        elif previous > 0.0:
            # Turning on from off, but already past the point where the
            # exponential is steep.  Step to the critical voltage and re-expand.
            return critical
    return target


@dataclass(frozen=True)
class DiodeParameters:
    """The junction parameters of a diode, with the reference defaults.

    These defaults are the ones the reference uses when a ``.model`` card supplies
    no parameters, which is what makes an unqualified comparison meaningful.
    ``saturation_current_a`` in particular is *not* the often-quoted 1e-12 A: the
    reference defaults to 1e-14 A, and assuming the wrong one moves the forward
    drop by tens of millivolts.
    """

    saturation_current_a: float = 1e-14
    emission_coefficient: float = 1.0
    series_resistance_ohm: float = 0.0
    #: Reverse breakdown voltage.  ``None`` or non-positive disables breakdown.
    breakdown_voltage_v: float | None = None
    #: Reverse current through the device at the onset of breakdown, as a positive
    #: magnitude.
    breakdown_current_a: float = 1e-3
    #: Ideality factor of the *breakdown* exponential, which the reference keeps
    #: separate from the forward one and defaults to 1.  Conflating the two is a
    #: large error for a device with a non-unit forward factor: it changes the
    #: breakdown current by more than two orders of magnitude on a diode with
    #: ``N = 1.8``, because the exponent carries the whole of ``1/N`` rather than
    #: just the thermal voltage.
    breakdown_emission_coefficient: float = 1.0
    zero_bias_capacitance_f: float = 0.0
    junction_potential_v: float = 1.0
    grading_coefficient: float = 0.5
    #: Transit time, which produces a bias-dependent diffusion capacitance.
    transit_time_s: float = 0.0
    #: Nominal temperature the parameters were measured at, and the temperature
    #: the device is simulated at.
    nominal_temperature_c: float = 27.0
    temperature_c: float = 27.0
    #: Forward-bias depletion-capacitance coefficient and its exponent.
    forward_bias_coefficient: float = 0.5
    forward_bias_exponent: float = 0.5
    #: Activation energy and saturation-current temperature exponent, used only
    #: by the temperature scaling.
    activation_energy_ev: float = 1.11
    saturation_current_temperature_exponent: float = 3.0

    def __post_init__(self) -> None:
        if self.saturation_current_a <= 0:
            raise ValueError("saturation_current_a must be positive")
        if self.emission_coefficient <= 0:
            raise ValueError("emission_coefficient must be positive")
        if self.series_resistance_ohm < 0:
            raise ValueError("series_resistance_ohm must not be negative")
        if self.zero_bias_capacitance_f < 0:
            raise ValueError("zero_bias_capacitance_f must not be negative")
        if self.transit_time_s < 0:
            raise ValueError("transit_time_s must not be negative")
        if self.breakdown_current_a <= 0:
            raise ValueError("breakdown_current_a must be positive")
        if not 0.0 < self.forward_bias_coefficient < 1.0:
            raise ValueError(
                "forward_bias_coefficient must lie strictly between 0 and 1"
            )
        if not 0.0 < self.grading_coefficient < 1.0:
            raise ValueError("grading_coefficient must lie strictly between 0 and 1")

    @property
    def temperature_k(self) -> float:
        return self.temperature_c + 273.15

    @property
    def nominal_temperature_k(self) -> float:
        return self.nominal_temperature_c + 273.15

    def thermal_voltage(self) -> float:
        """``N*k*T/q``: the emission coefficient is folded in, as the reference
        does, so every exponential divides by this rather than by ``k*T/q``."""

        return thermal_voltage(self.temperature_k) * self.emission_coefficient

    def scaled_saturation_current(self) -> float:
        return scaled_saturation_current(
            self.saturation_current_a,
            temperature_k=self.temperature_k,
            nominal_temperature_k=self.nominal_temperature_k,
            emission_coefficient=self.emission_coefficient,
            activation_energy_ev=self.activation_energy_ev,
            temperature_exponent=self.saturation_current_temperature_exponent,
        )


@dataclass
class Diode(NonlinearDevice):
    """A junction diode with optional series resistance and junction charge.

    The equations, in the order the reference evaluates them:

    * ``Id = IS * (exp(Vj/(N*Vt)) - 1)``, with ``Vj`` the *internal* junction
      voltage, so a series resistance is solved for exactly instead of being
      approximated by an added conductance and the terminal characteristic stays
      correct at high forward current;
    * reverse breakdown is a second exponential through the adjusted breakdown
      voltage;
    * capacitance is the depletion expression ``CJ0*(1-Vj/VJ)**-M`` below the
      forward-bias threshold and the linear extrapolation above it, plus the
      diffusion term ``TT*Id/(N*Vt)``.

    A purely resistive model produces a frequency-independent small-signal
    response, because the reference only carries junction capacitance into the
    small-signal matrix when the model actually has some.  A test that expects a
    default diode to roll off with frequency is testing the wrong thing.
    """

    name: str
    anode: str
    cathode: str
    parameters: DiodeParameters = field(default_factory=DiodeParameters)
    #: Last accepted junction voltage: the step limiter needs the previous
    #: iterate, and it is the bias a small-signal analysis linearises about.
    _junction_voltage: float = 0.0
    _on: bool = False

    def terminals(self) -> tuple[tuple[str, str], ...]:
        return ((self.anode, "anode"), (self.cathode, "cathode"))

    def reset(self) -> None:
        self._junction_voltage = 0.0
        self._on = False

    def bias(self) -> float:
        """Last solved internal junction voltage."""

        return self._junction_voltage

    # -- bias-point evaluation -------------------------------------------------

    def breakdown_voltage_v(self) -> float | None:
        """The knee voltage of the reverse characteristic, or ``None``.

        The breakdown exponential is anchored here: it passes through
        ``(BV, -IBV)`` by construction, which is the definition a datasheet gives
        and the definition the tests verify.

        The reference does something more elaborate.  It moves this knee outward by
        a logarithmically small amount to make the breakdown curve meet the reverse
        region's own current at the knee, and it derives the anchor from that
        iteration rather than from ``IBV`` directly.  Reproducing that iteration
        faithfully is not done here: the reference's own form of it, transcribed
        literally, does not converge on the parameter sets used in the tests, and
        the measured knees it produces are not reproduced by it either.  What that
        costs is quantified rather than glossed: at a breakdown current where the
        reference's knee shift is negligible the two agree to 5e-4 relative, and at
        a larger one the reference's knee sits about 0.08 V lower and its current
        runs about 2.4x higher than this model's.  The breakdown *shape* -- the
        exponential's slope, which is what a breakdown characteristic is used for --
        is verified against measurement over three decades of current.
        """

        bv = self.parameters.breakdown_voltage_v
        if bv is None or bv <= 0:
            return None
        return bv

    def _breakdown_scale(self) -> float:
        """Coefficient of the breakdown exponential.

        ``IBV`` itself, so the exponential passes through ``(BV, -IBV)`` exactly.
        That is the datasheet's definition of the pair, and the only anchoring that
        needs no invented knee offset: a device specified as ``BV=5.5, IBV=1e-5``
        then carries 1e-5 A at 5.5 V of reverse bias by construction.  Measured
        against the reference on the same circuit, that anchoring lands within 1e-6
        relative -- an earlier attempt that anchored one thermal voltage below the
        knee instead was off by 2 %.
        """

        return self.parameters.breakdown_current_a

    def junction_current(self, junction_voltage_v: float) -> float:
        """Junction current from the junction voltage, without series resistance.

        The forward exponential is evaluated only where the reference evaluates it.
        Below ``-3*N*Vt`` the reference abandons the exponential for a polynomial
        form, because the exponential of a large negative argument is physically
        meaningless there and, in floating point, actively harmful: at 5 V of
        reverse bias it is ``1e-91`` but at 5.5 V of *breakdown* the same expression
        on the same sign of argument reaches ``1e+88`` once the sign convention is
        crossed, and a current that large turns the Newton residual into garbage.  In
        that region the forward term is simply zero and the reverse exponential
        carries the device.
        """

        params = self.parameters
        vt = params.thermal_voltage()
        isat = params.scaled_saturation_current()
        breakdown = 0.0
        bv = self.breakdown_voltage_v()
        if bv is not None and junction_voltage_v <= -bv:
            # The breakdown exponential has its own ideality factor, defaulting to 1
            # and independent of the forward one, and therefore its own thermal
            # voltage.  Conflating them changes the breakdown current by more than
            # two orders of magnitude on a device with N = 1.8.
            nbvt = (
                thermal_voltage(params.temperature_k)
                * params.breakdown_emission_coefficient
            )
            exponent = -(bv + junction_voltage_v) / nbvt
            if exponent > 700.0:
                breakdown = -self._breakdown_scale() * float(np.exp(700.0))
            else:
                breakdown = -self._breakdown_scale() * float(np.exp(exponent))
            return breakdown
        if junction_voltage_v < -3.0 * vt:
            # Below this the reference replaces the exponential, whose value there is
            # an artefact of a formula that no longer describes the device, with a
            # cubic roll-off.  It is also finite where the exponential is not.
            argument = 3.0 * vt / (junction_voltage_v * float(np.e))
            return -isat * (1.0 + argument * argument * argument)
        with np.errstate(over="ignore"):
            return isat * float(np.expm1(junction_voltage_v / vt))

    def junction_slope(self, junction_voltage_v: float) -> float:
        """``dId/dVj``, written so an overflowed exponential still yields a
        usable slope instead of an infinity."""

        params = self.parameters
        isat = params.scaled_saturation_current()
        return (
            self.junction_current(junction_voltage_v) + isat
        ) / params.thermal_voltage()

    def _limit(self, previous: float, target: float) -> float:
        """Apply the step limit for whichever exponential dominates this bias."""

        params = self.parameters
        bv = self.breakdown_voltage_v()
        if bv is not None and (target < bv or previous < bv):
            ibsat = params.breakdown_current_a / (1.0 - float(np.exp(-1.0)))
            return junction_limiting_voltage(
                previous,
                target,
                thermal_voltage_v=params.thermal_voltage(),
                saturation_current_a=ibsat,
            )
        return junction_limiting_voltage(
            previous,
            target,
            thermal_voltage_v=params.thermal_voltage(),
            saturation_current_a=params.scaled_saturation_current(),
        )

    def _limit_change(self, previous: float, target: float) -> float:
        """Unused by the DC solve; kept for the step-limited scalar iteration."""

        limit = self.parameters.thermal_voltage() * _JUNCTION_STEP_LIMIT
        return min(max(target, previous - limit), previous + limit)

    def _critical_voltage(self) -> float | None:
        """The junction voltage at which the exponential's slope equals ``1/IS``.

        ``Vcrit = N*Vt*ln(N*Vt / (sqrt(2)*IS))``: the point the reference uses to
        decide that a junction is "on".  Below it the exponential is shallow enough
        that a full Newton step is safe; above it, steps are restricted.
        """

        params = self.parameters
        vt = params.thermal_voltage()
        isat = params.scaled_saturation_current()
        if vt <= 0 or isat <= 0:
            return None
        return vt * float(np.log(vt / (np.sqrt(2.0) * isat)))

    def internal_nodes(self) -> tuple[str, ...]:
        """The junction node behind the series resistance, when there is one.

        It is worth being precise about why this node exists rather than the
        series resistance being folded into an effective terminal conductance.
        The fold is exact at DC -- ``gd/(1 + gd*RS)`` is the true terminal slope --
        but it is *not* exact at frequency, because the folded form has no path for
        displacement current to bypass the resistance.  Above a few hundred MHz the
        two disagree qualitatively, not marginally: a biased diode with a 2 ohm
        series resistance driven into 10 kohm through 7.3 pF is 68 % wrong at
        10 GHz when folded, and right to 1e-6 when the internal node is carried.
        The reference carries the node, and so does this.
        """

        if self.parameters.series_resistance_ohm == 0.0:
            return ()
        return (self.junction_node(),)

    def junction_node(self) -> str:
        """Name of the internal junction node."""

        return f"{self.name}#junction"

    def internal_elements(self) -> tuple[tuple[str, str, float], ...]:
        rs = self.parameters.series_resistance_ohm
        if rs == 0.0:
            return ()
        return ((self.anode, self.junction_node(), rs),)

    def solve_junction_voltage(self, terminal_voltage_v: float) -> float:
        """Internal junction voltage behind the series resistance.

        The internal node is *eliminated*, not carried, for the DC solve.  With a
        series resistance the node satisfies ``(Vterminal - Vj)/RS = Id(Vj)``, a
        monotone scalar equation with exactly one root, so this reaches the same
        operating point an external simulator reaches by carrying the node in its
        own matrix -- while keeping the circuit-level Newton iteration's
        convergence independent of ``RS``.

        That independence is the point.  Carrying the node through the DC iteration
        instead lets the junction take one large step to a voltage where its slope
        is orders of magnitude too small; the internal node is then left floating
        and the iteration converges, confidently and to full tolerance, on the
        degenerate answer where the diode carries no current at all.  The scalar
        solve has no such basin: the root is unique.

        Uniqueness alone is not enough to find it, though.  In reverse breakdown the
        exponential is steep enough that an undamped step from zero overshoots into
        a region where the current is astronomically larger than the root's, and a
        step-limited iteration can then stall against its clamp.  Each step is
        therefore only accepted if it reduces the mismatch; otherwise it is halved
        until it does.  That keeps the quadratic behaviour near the root while
        making the approach to it monotone.
        """

        rs = self.parameters.series_resistance_ohm
        if rs == 0.0:
            return terminal_voltage_v
        # The root is bracketed physically: no current means Vj = Vterminal, and a
        # shorted junction means Vj = 0, so the junction voltage always lies between
        # the terminal voltage and zero.  Staying inside that interval also keeps
        # the reverse exponential from being evaluated far past the knee, where its
        # current overflows and a line search has nothing to descend on.
        low = min(0.0, terminal_voltage_v)
        high = max(0.0, terminal_voltage_v)

        def mismatch(candidate: float) -> float:
            return (terminal_voltage_v - candidate) / rs - self.junction_current(
                candidate
            )

        vj = min(max(self._junction_voltage, low), high)
        residual = abs(mismatch(vj))
        for _ in range(200):
            slope = self.junction_slope(vj)
            if not np.isfinite(slope) or slope <= 0:
                vj = 0.5 * (low + high)
                residual = abs(mismatch(vj))
                continue
            step = mismatch(vj) / (1.0 / rs + slope)
            for _ in range(80):
                candidate = min(max(vj + step, low), high)
                candidate_residual = abs(mismatch(candidate))
                if np.isfinite(candidate_residual) and candidate_residual < residual:
                    break
                step *= 0.5
                if abs(step) <= 1e-18 * max(1.0, abs(vj)):
                    return vj
            moved = candidate - vj
            vj = candidate
            # A residual test would need a scale against a current spanning twenty
            # decades across the models in use; the step is the quantity that must
            # settle for the terminal conductance to match the terminal current.
            if abs(moved) <= 1e-15 * max(1.0, abs(vj)):
                break
            residual = candidate_residual
        return vj

    def stamp(self, voltages: Mapping[str, float]) -> NonlinearStamp:
        """Terminal stamp for the DC solve, with the internal node eliminated."""

        rs = self.parameters.series_resistance_ohm
        terminal = float(voltages.get("anode", 0.0)) - float(
            voltages.get("cathode", 0.0)
        )
        if rs == 0.0:
            self._junction_voltage = terminal
            self._on = terminal > 0.0
            return NonlinearStamp(
                current=self.junction_current(terminal),
                conductance=self.junction_slope(terminal),
            )
        vj = self.solve_junction_voltage(terminal)
        self._junction_voltage = vj
        self._on = vj > 0.0
        # The terminal current is exactly (Vterminal - Vj)/RS, but the terminal
        # *conductance* is not 1/RS: the junction's own slope is in series with it.
        # Eliminating the internal node from the 2x2 Jacobian
        #     [ 1/RS    -1/RS    ] [dVa]   [dId]
        #     [-1/RS  1/RS + gd  ] [dVj] = [ 0 ]
        # gives dId/dVterminal = gd/(1 + gd*RS).  Using 1/RS instead makes a biased
        # diode look like a short and understates the small-signal response by an
        # order of magnitude.
        slope = self.junction_slope(vj)
        return NonlinearStamp(
            current=(terminal - vj) / rs, conductance=slope / (1.0 + slope * rs)
        )

    def small_signal_stamp(self) -> NonlinearStamp:
        """Stamp for the small-signal solve, taken at the internal junction.

        At AC the junction conductance belongs between the *internal* node and the
        cathode, because the internal element the solver stamps in series is what
        carries the resistance there.  Folding the two back together at this point
        would reintroduce exactly the error the internal node exists to remove: the
        folded form has no path for displacement current to bypass the series
        resistance, so it loses the real part of the terminal impedance and is 68 %
        wrong at 10 GHz on a biased diode.
        """

        vj = self._junction_voltage
        return NonlinearStamp(
            current=self.junction_current(vj), conductance=self.junction_slope(vj)
        )

    def small_signal_capacitance(self, junction_voltage_v: float) -> float:
        """Total junction capacitance at a junction voltage."""

        params = self.parameters
        capacitance = 0.0
        cj0 = params.zero_bias_capacitance_f
        if cj0 > 0:
            vj = params.junction_potential_v
            fc = params.forward_bias_coefficient
            if junction_voltage_v < fc * vj:
                ratio = max(1.0 - junction_voltage_v / vj, 1e-12)
                capacitance += cj0 * ratio ** (-params.grading_coefficient)
            else:
                slope = params.grading_coefficient / (vj * (1.0 - fc))
                capacitance += (
                    cj0
                    * fc ** (-params.grading_coefficient)
                    * (1.0 + slope * (junction_voltage_v - fc * vj))
                )
        if params.transit_time_s > 0:
            # Diffusion capacitance is the transit time times the junction's own
            # small-signal conductance, not times a bare conductance.
            capacitance += params.transit_time_s * self.junction_slope(
                junction_voltage_v
            )
        return capacitance


__all__ = [
    "BOLTZMANN_J_PER_K",
    "ELECTRON_CHARGE_C",
    "ConvergenceError",
    "Diode",
    "DiodeParameters",
    "NonlinearDevice",
    "NonlinearStamp",
    "junction_limiting_voltage",
    "scaled_saturation_current",
    "thermal_voltage",
]
