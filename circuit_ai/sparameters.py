"""S-parameters from the already-linearised small-signal network.

S-parameters are how every radio-frequency datasheet, every filter specification
and every matching network is written, and they are the one part of the RF
vocabulary that needs no new numerical machinery.  The small-signal solve in
:mod:`circuit_ai.nonlinear` already produces a linear network at a bias point, and
S-parameters are a change of basis on its port admittance matrix.

The route taken is the admittance one, because it is the one the solver can
actually supply.  For each port a unit current is injected with every other port
open-circuited; the resulting port voltages give the corresponding column of the
short-circuit admittance matrix ``Y``, and

    ``S = (I - Z0*Y) @ inv(I + Z0*Y)``

converts it.  The alternative -- driving travelling waves directly -- would need
the solver to impose matched terminations inside the network, which is a change to
the solve rather than a use of it.

Reference impedance is per port and defaults to 50 ohm, because that is what the
word means unless someone says otherwise.  Assuming it wrongly is the most common
way a port measurement is silently wrong by a large factor, so it is a parameter
and not a constant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .devices import Diode, NonlinearDevice
from .mna import GROUND_NAMES, CompiledLinearMNA, LinearCircuit, assemble_system
from .nonlinear import NonlinearMNA, OperatingPoint

#: The reference impedance a port has when none is given.
DEFAULT_REFERENCE_IMPEDANCE_OHM = 50.0


@dataclass(frozen=True)
class Port:
    """A pair of nodes across which a port is defined.

    A port is not a wire: it is a two-terminal measurement between two nodes,
    either of which may be ground.  Carrying both nodes is what lets a port sit
    across a differential pair rather than only from a node to ground.
    """

    name: str
    positive: str
    negative: str = "0"
    reference_impedance_ohm: float = DEFAULT_REFERENCE_IMPEDANCE_OHM

    def __post_init__(self) -> None:
        if self.reference_impedance_ohm <= 0:
            raise ValueError(
                f"port {self.name!r} reference impedance must be positive, got "
                f"{self.reference_impedance_ohm!r}"
            )


@dataclass(frozen=True)
class SParameterResult:
    """A sweep of S-parameter matrices, one per frequency.

    ``values[i, j, k]`` is ``S_ij`` at frequency ``k``: the response at port *i* to
    a wave incident at port *j*, which is the order every network analyser and
    Touchstone file uses.  This class and :meth:`to_touchstone` therefore speak the
    convention the instruments do rather than a transposed internal one that
    readers would have to remember to flip.
    """

    frequencies_hz: np.ndarray
    values: np.ndarray
    port_names: tuple[str, ...]
    reference_impedance_ohm: tuple[float, ...]
    operating_point: OperatingPoint | None = None

    @property
    def port_count(self) -> int:
        return len(self.port_names)

    def s(self, output_port: str, input_port: str, frequency_hz: float) -> complex:
        """One entry, looked up by name and interpolated at a frequency.

        Linear interpolation between adjacent sweep points, which is what a
        network analyser's marker does and is exact at any frequency the sweep
        actually lands on.
        """

        values = self.values[
            self._port_index(output_port), self._port_index(input_port), :
        ]
        if frequency_hz <= self.frequencies_hz[0]:
            return complex(values[0])
        if frequency_hz >= self.frequencies_hz[-1]:
            return complex(values[-1])
        return complex(np.interp(frequency_hz, self.frequencies_hz, values))

    def _port_index(self, name: str) -> int:
        try:
            return self.port_names.index(name)
        except ValueError as error:
            raise KeyError(
                f"unknown port {name!r}; known ports are {self.port_names}"
            ) from error

    def db(self, output_port: str, input_port: str) -> np.ndarray:
        """``|S|`` in dB, the way a plot or a specification states it.

        Floored at -300 dB rather than allowed to reach ``-inf``, so a plot or a
        constraint compiler gets a finite number for a perfect null instead of a
        value it has to special-case.
        """

        values = self.values[
            self._port_index(output_port), self._port_index(input_port), :
        ]
        return 20.0 * np.log10(np.maximum(np.abs(values), 1e-300))

    def as_dict(self) -> dict[str, object]:
        return {
            "frequencies_hz": self.frequencies_hz.tolist(),
            "port_names": list(self.port_names),
            "reference_impedance_ohm": list(self.reference_impedance_ohm),
            "s_parameters": [
                {
                    "output_port": self.port_names[i],
                    "input_port": self.port_names[j],
                    "real": self.values[i, j, :].real.tolist(),
                    "imag": self.values[i, j, :].imag.tolist(),
                }
                for i in range(self.port_count)
                for j in range(self.port_count)
            ],
        }

    def to_touchstone(self) -> str:
        """A Touchstone 1.0 file, magnitude-angle in degrees at Hz.

        Written because a result no other tool can read is not a deliverable.  The
        frequency unit and the data format are both stated in the option line,
        because assuming either is how a Touchstone file ends up silently wrong by
        a factor of a thousand or by a sign.
        """

        lines = [
            f"! Lattice S-parameter export, {self.port_count}-port",
            f"! ports: {', '.join(self.port_names)}",
            f"# HZ S MA R {self.port_count}",
        ]
        for index, frequency in enumerate(self.frequencies_hz):
            entries: list[str] = []
            for i in range(self.port_count):
                for j in range(self.port_count):
                    value = self.values[i, j, index]
                    entries.append(
                        f"{abs(value):.10e} {np.degrees(np.angle(value)):.6e}"
                    )
            lines.append(f"{frequency:.10e} " + " ".join(entries))
        return "\n".join(lines) + "\n"


def s_parameters(
    circuit: LinearCircuit,
    ports: Iterable[Port],
    frequencies_hz: np.ndarray,
    *,
    devices: Iterable[NonlinearDevice] = (),
    operating_point: OperatingPoint | None = None,
    reference_impedance_ohm: float = DEFAULT_REFERENCE_IMPEDANCE_OHM,
) -> SParameterResult:
    """Port S-parameters of *circuit* about its operating point.

    With nonlinear ``devices`` the network is linearised at the operating point
    first, exactly as a small-signal AC sweep would be; without them the circuit is
    already linear and no bias is involved.

    ``reference_impedance_ohm`` supplies a default for ports that do not state
    their own, so a plain 50 ohm measurement needs no ceremony and a 75 ohm one
    states it once.
    """

    port_list = tuple(
        port
        if port.reference_impedance_ohm != DEFAULT_REFERENCE_IMPEDANCE_OHM
        else Port(port.name, port.positive, port.negative, reference_impedance_ohm)
        for port in ports
    )
    if not port_list:
        raise ValueError("at least one port is required")
    names = [port.name for port in port_list]
    if len(set(names)) != len(names):
        raise ValueError(f"port names must be unique, got {names}")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")

    devices = tuple(devices)
    sim: NonlinearMNA | None = None
    if devices:
        sim = NonlinearMNA(circuit, devices)
        # S-parameters of an unbiassed nonlinear circuit are not a meaningful thing
        # to ask for, so the bias is computed rather than assumed.
        if operating_point is None:
            operating_point = sim.operating_point()

    plan: CompiledLinearMNA = (
        sim._ac_compiled if sim is not None else CompiledLinearMNA.compile(circuit)
    )
    _require_nodes(plan.node_index, port_list)

    z0 = np.array([port.reference_impedance_ohm for port in port_list], dtype=float)
    port_count = len(port_list)
    values = np.zeros((port_count, port_count, frequencies.size), dtype=np.complex128)

    for index, frequency in enumerate(frequencies):
        matrix = _small_signal_network(
            circuit, devices, sim, operating_point, frequency, plan
        )
        size = matrix.shape[0]
        # Each port in turn is driven by a Norton source -- ``1/Z0`` of current in
        # parallel with the port's own reference impedance, which is exactly a one
        # volt Thevenin source behind Z0 -- while every other port is terminated in
        # its reference impedance.
        #
        # Both halves of the drive are essential and leaving either out produces an
        # answer that is wrong in a way that looks plausible.  Driving without the
        # port's own termination measures the network in a configuration that is not
        # the one the reflection coefficient is defined against; terminating without
        # the drive leaves nothing to measure.  Terminating the *other* ports is what
        # "terminated in Z0" means, and it is also what makes the system nonsingular:
        # a nodal admittance matrix is singular whenever a node has no path to
        # ground, which is the normal state of a series attenuator, and a reference
        # termination is a physically meaningful path rather than a leak conductance
        # chosen to be small compared with the answer.
        for j, port in enumerate(port_list):
            p_j = _node(plan.node_index, port.positive)
            m_j = _node(plan.node_index, port.negative)
            rhs = np.zeros(size, dtype=np.complex128)
            _add_termination(matrix, plan.node_index, port)
            drive = 1.0 / port.reference_impedance_ohm
            if p_j is not None:
                rhs[p_j] += drive
            if m_j is not None:
                rhs[m_j] -= drive
            for k, other in enumerate(port_list):
                if k != j:
                    _add_termination(matrix, plan.node_index, other)
            solution = np.linalg.solve(matrix, rhs)
            # The network current at the driven port is the injected current minus
            # what its reference branch takes, ``1/Z0 - v/Z0``.  With the wave
            # definitions ``a = (v + Z0*I)/2`` and ``b = (v - Z0*I)/2``, the incident
            # wave is exactly one half and the reflected wave is ``v - 1/2``, so the
            # reflection coefficient is ``2v - 1``.  A port terminated in its own
            # reference impedance carries no incident wave of its own, so its voltage
            # *is* its transmitted wave and ``S_i1 = 2*v_i``.
            #
            # The limits pin the constants down: a shorted port reads zero volts and
            # reflects ``-1``, a matched port reads half a volt and reflects nothing,
            # and an open port reads one volt and reflects ``+1``.  The factor of two
            # on the transmitted term is easy to drop and costs exactly 6 dB, which
            # looks like a plausible extra loss rather than an error.
            for i, port_i in enumerate(port_list):
                voltage = _port_voltage(solution, plan.node_index, port_i)
                values[i, j, index] = 2.0 * voltage - 1.0 if i == j else 2.0 * voltage
            _remove_termination(matrix, plan.node_index, port)
            for k, other in enumerate(port_list):
                if k != j:
                    _remove_termination(matrix, plan.node_index, other)

    return SParameterResult(
        frequencies_hz=frequencies,
        values=values,
        port_names=tuple(names),
        reference_impedance_ohm=tuple(float(item) for item in z0),
        operating_point=operating_point,
    )


def _small_signal_network(
    circuit: LinearCircuit,
    devices: tuple[NonlinearDevice, ...],
    sim: NonlinearMNA | None,
    operating_point: OperatingPoint | None,
    frequency: float,
    plan: CompiledLinearMNA,
) -> np.ndarray:
    """The network's nodal admittance matrix at one frequency, devices linearised.

    Nonlinear devices are stamped the way the AC solve stamps them: at their
    *internal* node, on the plan that carries those nodes.  Stamping them across
    their outer terminals here would short out any series resistance and make the
    S-parameters of a biased diode disagree with its own small-signal response,
    which is the same error the AC path exists to avoid.
    """

    s = np.array([2j * np.pi * frequency])
    matrices, _ = assemble_system(
        circuit,
        plan.node_index,
        plan.source_index,
        plan.controlled_source_index,
        s,
        size=plan.size,
    )
    matrix = matrices[0].copy()
    if sim is not None and operating_point is not None:
        sim._stamp_internal_elements(matrix, s, plan.node_index)
        for device in devices:
            p, m = sim._device_terminals(
                device, internal=True, node_index=plan.node_index
            )
            stamp = device.small_signal_stamp()
            capacitance = (
                device.small_signal_capacitance(device.bias())
                if isinstance(device, Diode)
                else 0.0
            )
            admittance = float(stamp.conductance) + 2j * np.pi * frequency * capacitance
            if p is not None:
                matrix[p, p] += admittance
            if m is not None:
                matrix[m, m] += admittance
            if p is not None and m is not None:
                matrix[p, m] -= admittance
                matrix[m, p] -= admittance
    return matrix


def _node(node_index: Mapping[str, int], name: str) -> int | None:
    if name in GROUND_NAMES:
        return None
    return node_index.get(name)


#: Conductance from every node to ground while port parameters are extracted.  A
#: nodal admittance matrix is defined only up to a common-mode reference, so a
#: network with no ground path -- an unconnected two-port, a series attenuator --
#: has a singular matrix.  This removes that one degree of freedom and is far below
#: any admittance a real network presents.
_FLOATING_NODE_CONDUCTANCE = 1e-12

#: Extra conductance from the driven port's return node to ground.  Same gauge
#: choice, applied where the driving source needs a definite return.
_DRIVEN_RETURN_CONDUCTANCE = 1e-9


def _add_termination(
    matrix: np.ndarray, node_index: Mapping[str, int], port: Port
) -> None:
    """Connect *port* to ground through its own reference impedance."""

    admittance = 1.0 / port.reference_impedance_ohm
    p = _node(node_index, port.positive)
    m = _node(node_index, port.negative)
    if p is None and m is None:
        return
    if p is None:
        matrix[m, m] += admittance
        return
    if m is None:
        matrix[p, p] += admittance
        return
    matrix[p, p] += admittance
    matrix[m, m] += admittance
    matrix[p, m] -= admittance
    matrix[m, p] -= admittance


def _remove_termination(
    matrix: np.ndarray, node_index: Mapping[str, int], port: Port
) -> None:
    """Undo :func:`_add_termination` so the next port sees the bare network."""

    admittance = 1.0 / port.reference_impedance_ohm
    p = _node(node_index, port.positive)
    m = _node(node_index, port.negative)
    if p is None and m is None:
        return
    if p is None:
        matrix[m, m] -= admittance
        return
    if m is None:
        matrix[p, p] -= admittance
        return
    matrix[p, p] -= admittance
    matrix[m, m] -= admittance
    matrix[p, m] += admittance
    matrix[m, p] += admittance


def _port_voltage(
    node_voltages: np.ndarray, node_index: Mapping[str, int], port: Port
) -> complex:
    p = _node(node_index, port.positive)
    m = _node(node_index, port.negative)
    voltage = 0.0 + 0.0j
    if p is not None:
        voltage += node_voltages[p]
    if m is not None:
        voltage -= node_voltages[m]
    return complex(voltage)


def _require_nodes(node_index: Mapping[str, int], ports: tuple[Port, ...]) -> None:
    known = set(node_index) | set(GROUND_NAMES)
    for port in ports:
        for node in (port.positive, port.negative):
            if node not in known:
                raise ValueError(
                    f"port {port.name!r} references node {node!r}, which does not "
                    "exist in the circuit"
                )


__all__ = [
    "DEFAULT_REFERENCE_IMPEDANCE_OHM",
    "Port",
    "SParameterResult",
    "s_parameters",
]
