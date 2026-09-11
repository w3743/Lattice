"""Nonlinear MNA: DC operating point by Newton, then small-signal AC about it.

The linear solver in :mod:`circuit_ai.mna` answers "what does this network of
resistors, capacitors and inductors do".  That vocabulary cannot express a
diode, a transistor, or anything else whose current is not proportional to the
voltage across it -- which is to say it cannot express any circuit that does
useful work.  This module adds the missing half.

Two analyses, and they are not independent:

* :meth:`NonlinearMNA.operating_point` solves the resistive network with each
  nonlinear device replaced by its linearisation, iterating until the solution
  stops moving.  Capacitors are open and inductors are shorts, which is what the
  DC limit means rather than a special case in the code.
* :meth:`NonlinearMNA.solve_ac` linearises each nonlinear device *at that
  operating point* and then solves the resulting linear network at each
  frequency.

The small-signal analysis is therefore only as good as the operating point, and
neither is meaningful without the other: an AC sweep of an unbiassed diode is a
sweep of an open circuit.

Convergence is not assumed.  A Newton iteration on an exponential can diverge,
cycle, or stall, and a solver that returns whatever the last iterate happened to
be would be worse than one that refuses.  When the iteration does not meet its
residual tolerance within its budget, this raises :class:`ConvergenceError` and
reports the residual it reached; it never returns a non-solution as a solution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from .devices import ConvergenceError, Diode, NonlinearDevice
from .mna import (
    GROUND_NAMES,
    CompiledLinearMNA,
    LinearCircuit,
    MNASolutionBatch,
    assemble_system,
)

#: Upper bound on Newton iterations.  A well-behaved circuit converges in well
#: under twenty; a circuit that has not converged by a few hundred is not going
#: to, and continuing would only disguise the failure as slowness.
DEFAULT_MAX_ITERATIONS = 200

#: Convergence is declared when both the largest node-voltage change and the
#: largest residual, relative to the circuit's current scale, fall below this.
DEFAULT_TOLERANCE = 1e-9


def _index(node_index: Mapping[str, int], node: str) -> int | None:
    if node in GROUND_NAMES:
        return None
    return node_index.get(node)


def _node_voltage(
    node_index: Mapping[str, int], values: np.ndarray, node: str
) -> float:
    index = _index(node_index, node)
    return 0.0 if index is None else float(values[index].real)


@dataclass(frozen=True)
class OperatingPoint:
    """A solved DC bias.

    ``values`` is the raw MNA solution vector, so the node and branch index maps
    in :attr:`node_index` / :attr:`source_index` read it the same way an AC
    solution is read.
    """

    values: np.ndarray
    node_index: Mapping[str, int]
    source_index: Mapping[str, int]
    controlled_source_index: Mapping[str, int]
    iterations: int
    residual: float
    #: Internal junction voltages, by device name.  A terminal voltage is not
    #: enough to describe a device with series resistance, and the small-signal
    #: model needs the internal one.
    junction_voltages: Mapping[str, float] = field(default_factory=dict)

    def node_voltage(self, node: str) -> float:
        if node in GROUND_NAMES:
            return 0.0
        index = self.node_index.get(node)
        if index is None:
            raise KeyError(f"unknown node {node!r} in operating point")
        return float(self.values[index].real)

    def branch_current(self, source_name: str) -> float:
        index = self.source_index.get(source_name)
        if index is None:
            index = self.controlled_source_index.get(source_name)
        if index is None:
            raise KeyError(f"unknown branch {source_name!r} in operating point")
        return float(self.values[index].real)

    def as_dict(self) -> dict[str, float]:
        node_voltages = {
            node: float(self.values[index].real)
            for node, index in self.node_index.items()
        }
        return {
            "node_voltages_v": node_voltages,
            "branch_currents_a": {
                name: float(self.values[index].real)
                for name, index in self.source_index.items()
            },
            "junction_voltages_v": dict(self.junction_voltages),
            "iterations": self.iterations,
            "residual": self.residual,
        }


class NonlinearMNA:
    """Solves a :class:`LinearCircuit` plus nonlinear devices.

    The linear part is carried unchanged, so a circuit that happens to contain no
    nonlinear device is solved by exactly the linear stamping code, and the two
    paths cannot disagree about what a resistor is.
    """

    def __init__(
        self,
        circuit: LinearCircuit,
        devices: Iterable[NonlinearDevice] = (),
        *,
        compiled: CompiledLinearMNA | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        self.circuit = circuit
        self.devices = tuple(devices)
        base = compiled or CompiledLinearMNA.compile(circuit)
        # Two plans, deliberately.  The DC solve eliminates each device's internal
        # node -- see Diode.solve_junction_voltage for why carrying it through a
        # Newton iteration is unsafe -- so it works on the plain linear topology.
        # The small-signal solve is a single linear system, where the internal node
        # is both safe and necessary, so it works on a plan with those nodes
        # appended.  The appended indices come after every existing node and branch,
        # so the linear indices are identical in both plans.
        self.compiled = base
        self._internal_nodes = tuple(
            node for device in self.devices for node in device.internal_nodes()
        )
        self._ac_compiled = base.with_internal_nodes(self._internal_nodes)
        self._internal_elements = tuple(
            element for device in self.devices for element in device.internal_elements()
        )
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self._validate_terminals()
        self._validate_internal_elements()

    def _validate_terminals(self) -> None:
        known = set(self.compiled.node_index) | set(GROUND_NAMES)
        for device in self.devices:
            for node, _ in device.terminals():
                if node not in known:
                    raise ValueError(
                        f"nonlinear device {device.name!r} references node {node!r}, "
                        "which does not exist in the linear circuit"
                    )

    def _validate_internal_elements(self) -> None:
        known = set(self._ac_compiled.node_index) | set(GROUND_NAMES)
        for node_a, node_b, value in self._internal_elements:
            for node in (node_a, node_b):
                if node not in known:
                    raise ValueError(
                        f"internal element references node {node!r}, which does not "
                        "exist in the compiled plan"
                    )
            if value <= 0:
                raise ValueError(
                    f"internal element between {node_a!r} and {node_b!r} has a "
                    f"non-positive value {value!r}"
                )

    # -- DC operating point ---------------------------------------------------

    def operating_point(
        self, *, initial: Mapping[str, float] | None = None
    ) -> OperatingPoint:
        """Solve the DC bias point by Newton iteration on the conductance stamps.

        ``initial`` may supply starting node voltages; a caller that already has a
        nearby solution (a sweep, or a resumed run) converges faster and more
        reliably by supplying it.  Absent that, the iteration starts from a
        zero-bias linearisation, which is the same cold start an external
        simulator uses.
        """

        for device in self.devices:
            device.reset()
        size = self.compiled.size
        node_index = self.compiled.node_index
        # s = 0 is the resistive case: capacitors contribute zero admittance and
        # inductors become shorts, so the DC limit is assembled, not special-cased.
        base_matrix, base_rhs = assemble_system(
            self.circuit,
            node_index,
            self.compiled.source_index,
            self.compiled.controlled_source_index,
            np.zeros(1, dtype=np.complex128),
            size=size,
        )
        base = base_matrix[0]
        rhs = base_rhs[0]

        values = np.zeros(size, dtype=np.complex128)
        if initial:
            for node, voltage in initial.items():
                index = node_index.get(node)
                if index is not None:
                    values[index] = complex(voltage)
        elif size:
            values = np.linalg.solve(base, rhs).astype(np.complex128)

        network_scale = max(1.0, float(np.max(np.abs(rhs))) if size else 1.0)
        # A tiny conductance to ground on each nonlinear terminal keeps the matrix
        # non-singular when a device is the only path from a node, which happens
        # for a floating junction.  It is far below any conductance the circuit can
        # physically express and does not move a converged solution.
        gmin = 1e-12 * network_scale

        change = np.full(size, np.inf)
        residual = np.zeros(size, dtype=np.complex128)
        for iteration in range(1, self.max_iterations + 1):
            matrix = base.copy()
            equivalent = np.zeros(size, dtype=np.complex128)
            for device in self.devices:
                # The device keeps its own junction voltage across iterations:
                # its step limiter needs the previous iterate, and a stamp that
                # discarded it would turn a limited convergence into a divergent
                # one for a large series resistance.
                stamp = device.stamp(self._terminal_voltages(device, values))
                self._stamp_device(matrix, equivalent, gmin, device, stamp, values)
            # Residual of the *unlinearised* network at the current iterate, so
            # convergence is tested against the circuit's own equations rather
            # than against the companion model that was just built from them.
            residual = rhs - matrix @ values + equivalent
            try:
                candidate = np.linalg.solve(matrix, rhs + equivalent)
            except np.linalg.LinAlgError as error:  # pragma: no cover - defensive
                raise ConvergenceError(
                    f"singular MNA matrix at Newton iteration {iteration}: {error}"
                ) from error
            change = candidate - values
            values = candidate
            residual_norm = float(np.max(np.abs(residual))) if size else 0.0
            if (
                float(np.max(np.abs(change))) <= self.tolerance * network_scale
                and residual_norm <= self.tolerance * network_scale
            ):
                break
        else:
            raise ConvergenceError(
                "DC operating point did not converge in "
                f"{self.max_iterations} iterations; largest node change "
                f"{float(np.max(np.abs(change))) if size else 0.0:.3e}, "
                f"residual {residual_norm:.3e}, scale {network_scale:.3e}"
            )

        return OperatingPoint(
            values=values,
            node_index=node_index,
            source_index=self.compiled.source_index,
            controlled_source_index=self.compiled.controlled_source_index,
            iterations=iteration,
            residual=residual_norm,
            junction_voltages={
                device.name: self._device_junction_voltage(device)
                for device in self.devices
                if isinstance(device, Diode)
            },
        )

    def _stamp_internal_elements(
        self,
        matrices: np.ndarray,
        s: np.ndarray,
        node_index: Mapping[str, int] | None = None,
    ) -> None:
        """Stamp the linear elements a device needs between its internal nodes.

        These are ordinary resistances, so they are stamped exactly the way the
        linear solver stamps any other.  ``node_index`` must be the plan the
        matrices were sized against: the internal element's nodes exist only in the
        small-signal plan, and resolving them against the DC plan yields ``None``
        for both ends, stamping nothing at all -- a silently floating node rather
        than an error.
        """

        index = self._ac_compiled.node_index if node_index is None else node_index
        for node_a, node_b, value in self._internal_elements:
            # A scalar broadcasts against a single matrix and against a frequency
            # batch alike, so one code path serves the DC and AC callers.
            admittance = np.complex128(1.0 / value)
            a = _index(index, node_a)
            b = _index(index, node_b)
            if matrices.ndim == 3:
                if a is not None:
                    matrices[:, a, a] += admittance
                if b is not None:
                    matrices[:, b, b] += admittance
                if a is not None and b is not None:
                    matrices[:, a, b] -= admittance
                    matrices[:, b, a] -= admittance
                continue
            if a is not None:
                matrices[a, a] += admittance
            if b is not None:
                matrices[b, b] += admittance
            if a is not None and b is not None:
                matrices[a, b] -= admittance
                matrices[b, a] -= admittance

    def _device_junction_voltage(self, device: NonlinearDevice) -> float:
        if isinstance(device, Diode):
            return device.bias()
        return 0.0

    def _terminal_voltages(
        self, device: NonlinearDevice, values: np.ndarray
    ) -> dict[str, float]:
        """Every voltage *device* needs, by terminal name.

        Terminal names come from the device's own declaration; an internal node
        whose name matches a keyword the device asks for is resolved by attribute
        so that a device never has to be told about node naming.
        """

        voltages: dict[str, float] = {}
        for node, terminal in device.terminals():
            voltages[terminal] = _node_voltage(self.compiled.node_index, values, node)
        return voltages

    def _device_terminals(
        self,
        device: NonlinearDevice,
        *,
        internal: bool = False,
        node_index: Mapping[str, int] | None = None,
    ) -> tuple[int | None, int | None]:
        """Matrix indices the device's conductance stamp spans.

        For the DC solve a device is two-terminal: it has already eliminated its own
        internals, so its conductance goes across the terminals it presents to the
        circuit.

        For the small-signal solve the same device is stamped at its internal
        junction instead, because the internal element carrying the series
        resistance is present in that matrix.  Stamping the junction conductance
        across the outer terminals there would short out that resistance and
        reintroduce the error the internal node exists to remove.

        ``node_index`` must be the plan being stamped into.  The two plans differ by
        exactly the internal nodes, so resolving an internal node against the DC
        plan yields ``None`` -- a silently floating node and a singular matrix,
        rather than a diagnostic.
        """

        index = self.compiled.node_index if node_index is None else node_index
        nodes = [node for node, _ in device.terminals()]
        negative = _index(index, nodes[1]) if len(nodes) > 1 else None
        if internal and device.internal_nodes():
            return _index(index, device.junction_node()), negative
        return _index(index, nodes[0]), negative

    def _stamp_device(
        self,
        matrix: np.ndarray,
        equivalent: np.ndarray,
        gmin: float,
        device: NonlinearDevice,
        stamp,
        values: np.ndarray,
    ) -> None:
        """Stamp a device companion model.

        The device is represented by ``I(v)`` approximated as
        ``I(v_k) + G_k * (v - v_k)``: ``G_k`` goes into the matrix and the
        equivalent current ``I(v_k) - G_k * (v_p - v_m)`` into the right-hand
        side.  At a converged iterate the two cancel, which is what makes a
        Newton solution satisfy the original equations exactly rather than only
        to the tolerance of a linearised substitute.
        """

        p, m = self._device_terminals(device)
        conductance = float(stamp.conductance)
        current = float(stamp.current)
        if not np.isfinite(conductance) or not np.isfinite(current):
            raise ConvergenceError(
                f"device {device.name!r} produced a non-finite stamp at the "
                f"current bias (conductance {conductance!r}, current {current!r})"
            )
        vp = 0.0 if p is None else float(values[p].real)
        vm = 0.0 if m is None else float(values[m].real)
        equivalent_current = current - conductance * (vp - vm)
        if p is not None:
            matrix[p, p] += conductance + gmin
            equivalent[p] -= equivalent_current
        if m is not None:
            matrix[m, m] += conductance + gmin
            equivalent[m] += equivalent_current
        if p is not None and m is not None:
            matrix[p, m] -= conductance
            matrix[m, p] -= conductance

    # -- small-signal AC -----------------------------------------------------

    def solve_ac(
        self,
        frequencies_hz: np.ndarray,
        *,
        operating_point: OperatingPoint | None = None,
    ) -> MNASolutionBatch:
        """Linearise each device at the bias point and sweep.

        The bias point is computed if the caller does not supply one.  Every
        reactive parasitic that a device contributes is a function of that bias,
        so computing it here rather than accepting a caller-supplied guess keeps
        the linearisation consistent with the solution it came from.
        """

        bias = (
            operating_point if operating_point is not None else self.operating_point()
        )
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
        s = 2j * np.pi * frequencies
        # The small-signal network is the linear circuit plus every device's
        # internal elements, on a plan that has the internal nodes.
        ac = self._ac_compiled
        matrices, rhs = assemble_system(
            self.circuit,
            ac.node_index,
            ac.source_index,
            ac.controlled_source_index,
            s,
            size=ac.size,
        )
        self._stamp_internal_elements(matrices, s, ac.node_index)
        for device in self.devices:
            self._stamp_small_signal(matrices, device, bias, s)
        values = np.linalg.solve(matrices, rhs[..., None])[..., 0]
        return MNASolutionBatch(
            values=values,
            node_index=ac.node_index,
            source_index=ac.source_index,
            controlled_source_index=ac.controlled_source_index,
        )

    def _stamp_small_signal(
        self,
        matrices: np.ndarray,
        device: NonlinearDevice,
        bias: OperatingPoint,
        s: np.ndarray,
    ) -> None:
        """Stamp the device's linearised admittance at the bias point.

        Conductive and reactive parts are stamped together because they are in
        parallel across the same two nodes, and separating them would mean two
        passes over the same indices for no gain.
        """

        p, m = self._device_terminals(
            device, internal=True, node_index=self._ac_compiled.node_index
        )
        stamp = (
            device.small_signal_stamp()
            if hasattr(device, "small_signal_stamp")
            else device.stamp(self._terminal_voltages(device, bias.values))
        )
        conductance = float(stamp.conductance)
        capacitance = 0.0
        if isinstance(device, Diode):
            capacitance = device.small_signal_capacitance(device.bias())
        admittance = conductance + s * capacitance
        if p is not None:
            matrices[:, p, p] += admittance
        if m is not None:
            matrices[:, m, m] += admittance
        if p is not None and m is not None:
            matrices[:, p, m] -= admittance
            matrices[:, m, p] -= admittance


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "ConvergenceError",
    "NonlinearMNA",
    "OperatingPoint",
]
