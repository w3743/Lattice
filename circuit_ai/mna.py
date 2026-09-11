from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


GROUND_NAMES = {"0", "gnd", "GND"}


@dataclass(frozen=True)
class LinearElement:
    name: str
    kind: str
    n1: str
    n2: str
    value: float


@dataclass(frozen=True)
class VoltageSource:
    name: str
    n_plus: str
    n_minus: str
    value: complex


@dataclass(frozen=True)
class CurrentSource:
    name: str
    n_plus: str
    n_minus: str
    value: complex


@dataclass(frozen=True)
class VoltageControlledVoltageSource:
    name: str
    n_plus: str
    n_minus: str
    control_plus: str
    control_minus: str
    gain: complex


@dataclass(frozen=True)
class LinearCircuit:
    elements: tuple[LinearElement, ...]
    voltage_sources: tuple[VoltageSource, ...]
    current_sources: tuple[CurrentSource, ...] = ()
    controlled_voltage_sources: tuple[VoltageControlledVoltageSource, ...] = ()

    def nodes(self) -> tuple[str, ...]:
        seen: set[str] = set()
        for element in self.elements:
            seen.update([element.n1, element.n2])
        for source in self.voltage_sources:
            seen.update([source.n_plus, source.n_minus])
        for source in self.current_sources:
            seen.update([source.n_plus, source.n_minus])
        for source in self.controlled_voltage_sources:
            seen.update([source.n_plus, source.n_minus, source.control_plus, source.control_minus])
        return tuple(sorted(node for node in seen if node not in GROUND_NAMES))


@dataclass(frozen=True)
class MNASolutionBatch:
    """Batched MNA solutions for one circuit and many frequencies.

    ``values`` has shape ``(frequency, equation)``.  Keeping the index maps
    beside the raw solution lets the analysis layer extract several
    observables without solving the same matrix again.
    """

    values: np.ndarray
    node_index: Mapping[str, int]
    source_index: Mapping[str, int]
    controlled_source_index: Mapping[str, int]

    def node_voltage(self, node: str) -> np.ndarray:
        if node in GROUND_NAMES:
            return np.zeros(self.values.shape[0], dtype=np.complex128)
        index = self.node_index.get(node)
        if index is None:
            return np.zeros(self.values.shape[0], dtype=np.complex128)
        return self.values[:, index]

    def branch_current(self, source_name: str) -> np.ndarray:
        if source_name in self.source_index:
            return self.values[:, self.source_index[source_name]]
        return self.values[:, self.controlled_source_index[source_name]]


def linear_topology_signature(circuit: LinearCircuit) -> tuple[tuple[str, ...], ...]:
    """The identity a compiled MNA plan is only valid for.

    Element and source names are part of the signature, so two circuits that
    differ only in component naming do NOT share a plan.  Callers that cache
    plans must key on this value (or include it), because keying on a purely
    structural graph hash is not sufficient: such hashes deliberately ignore
    instance ids and reference designators.
    """

    signature: list[tuple[str, ...]] = [
        ("element", item.name, item.kind.upper(), item.n1, item.n2)
        for item in circuit.elements
    ]
    signature.extend(
        ("voltage_source", item.name, item.n_plus, item.n_minus)
        for item in circuit.voltage_sources
    )
    signature.extend(
        ("current_source", item.name, item.n_plus, item.n_minus)
        for item in circuit.current_sources
    )
    signature.extend(
        (
            "vcvs",
            item.name,
            item.n_plus,
            item.n_minus,
            item.control_plus,
            item.control_minus,
        )
        for item in circuit.controlled_voltage_sources
    )
    return tuple(signature)


@dataclass(frozen=True)
class CompiledLinearMNA:
    """Reusable topology/stamp plan for a linear MNA circuit.

    Element values are deliberately excluded from the compiled signature.  A
    request can therefore provide new bound parameter values while reusing
    the same node/branch indexing and batched stamp plan.
    """

    node_index: Mapping[str, int]
    source_index: Mapping[str, int]
    controlled_source_index: Mapping[str, int]
    size: int
    topology_signature: tuple[tuple[str, ...], ...]

    @classmethod
    def compile(cls, circuit: LinearCircuit) -> "CompiledLinearMNA":
        nodes = circuit.nodes()
        node_index = {node: idx for idx, node in enumerate(nodes)}
        source_index = {
            source.name: len(nodes) + idx
            for idx, source in enumerate(circuit.voltage_sources)
        }
        controlled_source_index = {
            source.name: len(nodes) + len(circuit.voltage_sources) + idx
            for idx, source in enumerate(circuit.controlled_voltage_sources)
        }
        return cls(
            node_index=node_index,
            source_index=source_index,
            controlled_source_index=controlled_source_index,
            size=len(nodes)
            + len(circuit.voltage_sources)
            + len(circuit.controlled_voltage_sources),
            topology_signature=linear_topology_signature(circuit),
        )

    def solve_ac(
        self,
        circuit: LinearCircuit,
        frequencies_hz: np.ndarray,
    ) -> MNASolutionBatch:
        if self._signature(circuit) != self.topology_signature:
            raise ValueError("circuit topology does not match compiled MNA plan")
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a non-empty one-dimensional array")

        s = 2j * np.pi * frequencies
        matrices = np.zeros((len(frequencies), self.size, self.size), dtype=np.complex128)
        rhs = np.zeros((len(frequencies), self.size), dtype=np.complex128)

        for element in circuit.elements:
            y = _admittance_vector(element, s)
            a = _node(self.node_index, element.n1)
            b = _node(self.node_index, element.n2)
            if a is not None:
                matrices[:, a, a] += y
            if b is not None:
                matrices[:, b, b] += y
            if a is not None and b is not None:
                matrices[:, a, b] -= y
                matrices[:, b, a] -= y

        for source in circuit.voltage_sources:
            row = self.source_index[source.name]
            _stamp_voltage_branch_batch(
                matrices, self.node_index, row, source.n_plus, source.n_minus
            )
            rhs[:, row] = source.value

        for source in circuit.current_sources:
            p = _node(self.node_index, source.n_plus)
            m = _node(self.node_index, source.n_minus)
            if p is not None:
                rhs[:, p] -= source.value
            if m is not None:
                rhs[:, m] += source.value

        for source in circuit.controlled_voltage_sources:
            row = self.controlled_source_index[source.name]
            _stamp_voltage_branch_batch(
                matrices, self.node_index, row, source.n_plus, source.n_minus
            )
            cp = _node(self.node_index, source.control_plus)
            cm = _node(self.node_index, source.control_minus)
            if cp is not None:
                matrices[:, row, cp] -= source.gain
            if cm is not None:
                matrices[:, row, cm] += source.gain

        # NumPy >= 2 treats ``b`` as a stack of core-(m, n) matrices, so a bare
        # ``(n_freq, size)`` RHS is misread as ``(m, n) = (n_freq, size)`` and
        # rejected unless ``n_freq == size``.  An explicit trailing axis keeps
        # the batched semantics and the ``(n_freq, size)`` return contract.
        values = np.linalg.solve(matrices, rhs[..., None])[..., 0]
        return MNASolutionBatch(
            values=values,
            node_index=self.node_index,
            source_index=self.source_index,
            controlled_source_index=self.controlled_source_index,
        )

    def _signature(self, circuit: LinearCircuit) -> tuple[tuple[str, ...], ...]:
        return linear_topology_signature(circuit)


def admittance(element: LinearElement, s: complex) -> complex:
    kind = element.kind.upper()
    if element.value <= 0:
        raise ValueError(f"{element.name} value must be positive")
    if kind == "R":
        return 1.0 / element.value
    if kind == "C":
        return s * element.value
    if kind == "L":
        return 1.0 / (s * element.value)
    raise ValueError(f"unsupported linear element kind {element.kind!r}")


def _admittance_vector(element: LinearElement, s: np.ndarray) -> np.ndarray:
    kind = element.kind.upper()
    if element.value <= 0:
        raise ValueError(f"{element.name} value must be positive")
    if kind == "R":
        return np.full(s.shape, 1.0 / element.value, dtype=np.complex128)
    if kind == "C":
        return s * element.value
    if kind == "L":
        with np.errstate(divide="raise", invalid="raise"):
            return 1.0 / (s * element.value)
    raise ValueError(f"unsupported linear element kind {element.kind!r}")


def _node(node_index: Mapping[str, int], name: str) -> int | None:
    if name in GROUND_NAMES:
        return None
    return node_index[name]


def _stamp_voltage_branch_batch(
    matrices: np.ndarray,
    node_index: Mapping[str, int],
    row: int,
    n_plus: str,
    n_minus: str,
) -> None:
    p = _node(node_index, n_plus)
    m = _node(node_index, n_minus)
    if p is not None:
        matrices[:, p, row] += 1.0
        matrices[:, row, p] += 1.0
    if m is not None:
        matrices[:, m, row] -= 1.0
        matrices[:, row, m] -= 1.0


class MNASimulator:
    """Small frequency-domain modified nodal analysis simulator.

    This is intentionally compact but real: it stamps R/C/L elements and ideal
    independent voltage sources into the MNA matrix and solves the linear system.
    Future work can add controlled sources, op-amps, nonlinear Newton iterations,
    and differentiable JAX versions behind the same conceptual interface.
    """

    def solve_ac(
        self,
        circuit: LinearCircuit,
        frequencies_hz: np.ndarray,
    ) -> list[dict[str, complex]]:
        batch = self.solve_ac_batched(circuit, frequencies_hz)
        solutions: list[dict[str, complex]] = []
        for row in batch.values:
            solution = {node: row[idx] for node, idx in batch.node_index.items()}
            for source, idx in batch.source_index.items():
                solution[f"I({source})"] = row[idx]
            for source, idx in batch.controlled_source_index.items():
                solution[f"I({source})"] = row[idx]
            solutions.append(solution)
        return solutions

    def compile(self, circuit: LinearCircuit) -> CompiledLinearMNA:
        return CompiledLinearMNA.compile(circuit)

    def solve_ac_batched(
        self,
        circuit: LinearCircuit,
        frequencies_hz: np.ndarray,
        *,
        compiled: CompiledLinearMNA | None = None,
    ) -> MNASolutionBatch:
        return (compiled or self.compile(circuit)).solve_ac(circuit, frequencies_hz)

    def transfer(
        self,
        circuit: LinearCircuit,
        output_node: str,
        source_name: str,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        source = next(src for src in circuit.voltage_sources if src.name == source_name)
        if source.value == 0:
            raise ValueError("source value must be nonzero for transfer calculation")
        solutions = self.solve_ac(circuit, frequencies_hz)
        return np.asarray([solution.get(output_node, 0.0) / source.value for solution in solutions])

    @staticmethod
    def _node(node_index: dict[str, int], name: str) -> int | None:
        if name in GROUND_NAMES:
            return None
        return node_index[name]

    def _stamp_admittance(
        self,
        matrix: np.ndarray,
        node_index: dict[str, int],
        n1: str,
        n2: str,
        y: complex,
    ) -> None:
        a = self._node(node_index, n1)
        b = self._node(node_index, n2)
        if a is not None:
            matrix[a, a] += y
        if b is not None:
            matrix[b, b] += y
        if a is not None and b is not None:
            matrix[a, b] -= y
            matrix[b, a] -= y

    def _stamp_voltage_source(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        node_index: dict[str, int],
        row: int,
        source: VoltageSource,
    ) -> None:
        self._stamp_voltage_branch(matrix, node_index, row, source.n_plus, source.n_minus)
        rhs[row] = source.value

    def _stamp_current_source(
        self,
        rhs: np.ndarray,
        node_index: dict[str, int],
        source: CurrentSource,
    ) -> None:
        p = self._node(node_index, source.n_plus)
        m = self._node(node_index, source.n_minus)
        if p is not None:
            rhs[p] -= source.value
        if m is not None:
            rhs[m] += source.value

    def _stamp_vcvs(
        self,
        matrix: np.ndarray,
        node_index: dict[str, int],
        row: int,
        source: VoltageControlledVoltageSource,
    ) -> None:
        self._stamp_voltage_branch(matrix, node_index, row, source.n_plus, source.n_minus)
        cp = self._node(node_index, source.control_plus)
        cm = self._node(node_index, source.control_minus)
        if cp is not None:
            matrix[row, cp] -= source.gain
        if cm is not None:
            matrix[row, cm] += source.gain

    def _stamp_voltage_branch(
        self,
        matrix: np.ndarray,
        node_index: dict[str, int],
        row: int,
        n_plus: str,
        n_minus: str,
    ) -> None:
        p = self._node(node_index, n_plus)
        m = self._node(node_index, n_minus)
        if p is not None:
            matrix[p, row] += 1.0
            matrix[row, p] += 1.0
        if m is not None:
            matrix[m, row] -= 1.0
            matrix[row, m] -= 1.0
