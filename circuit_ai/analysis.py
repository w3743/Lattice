from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mna import (
    CompiledLinearMNA,
    CurrentSource,
    LinearCircuit,
    MNASimulator,
    MNASolutionBatch,
    VoltageSource,
)


ANALYSIS_ALIASES = {
    "transfer": "voltage_transfer",
    "vout_over_vin": "voltage_transfer",
    "voltage_gain": "voltage_transfer",
    "voltage_transfer": "voltage_transfer",
    "transimpedance": "transimpedance",
    "tia": "transimpedance",
    "vout_over_iin": "transimpedance",
    "voltage_over_current": "transimpedance",
    "zin": "input_impedance",
    "input_impedance": "input_impedance",
    "zout": "output_impedance",
    "output_impedance": "output_impedance",
}


@dataclass(frozen=True)
class AnalysisRequest:
    """A circuit behavior measurement requested from a simulator.

    The synthesis MVP still optimizes voltage transfer by default, but this
    request object separates "what to measure" from "what topology generated
    the circuit".  That is the hook for impedance, transconductance, S-parameter,
    noise, and operating-point goals.
    """

    kind: str = "voltage_transfer"
    source_name: str = "Vin"
    output_node: str = "out"
    input_node: str = "in"
    reference_node: str = "0"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str | None) -> "AnalysisRequest":
        if data is None:
            return cls()
        if isinstance(data, str):
            data = {"kind": data}

        kind = _normalize_analysis_kind(str(data.get("kind", "voltage_transfer")))
        if kind in {"input_impedance", "output_impedance"}:
            default_source = "Vin" if kind == "input_impedance" else "Itest"
            return cls(
                kind=kind,
                source_name=str(data.get("source_name", data.get("source", default_source))),
                output_node=str(data.get("output_node", "out")),
                input_node=str(data.get("input_node", data.get("port_node", "in"))),
                reference_node=str(data.get("reference_node", data.get("reference", "0"))),
            )
        if kind in {"voltage_transfer", "transimpedance"}:
            return cls(
                kind=kind,
                source_name=str(data.get("source_name", data.get("source", "Vin" if kind == "voltage_transfer" else "Iin"))),
                output_node=str(data.get("output_node", data.get("output", "out"))),
                input_node=str(data.get("input_node", "in")),
                reference_node=str(data.get("reference_node", data.get("reference", "0"))),
            )
        raise ValueError(f"unsupported analysis kind: {kind!r}")

    @classmethod
    def voltage_transfer(
        cls,
        output_node: str = "out",
        source_name: str = "Vin",
        reference_node: str = "0",
    ) -> "AnalysisRequest":
        return cls(
            kind="voltage_transfer",
            source_name=source_name,
            output_node=output_node,
            reference_node=reference_node,
        )

    @classmethod
    def input_impedance(
        cls,
        input_node: str = "in",
        source_name: str = "Vin",
        reference_node: str = "0",
    ) -> "AnalysisRequest":
        return cls(
            kind="input_impedance",
            source_name=source_name,
            input_node=input_node,
            reference_node=reference_node,
        )

    @classmethod
    def output_impedance(
        cls,
        output_node: str = "out",
        source_name: str = "Itest",
        reference_node: str = "0",
    ) -> "AnalysisRequest":
        return cls(
            kind="output_impedance",
            source_name=source_name,
            output_node=output_node,
            reference_node=reference_node,
        )

    @classmethod
    def transimpedance(
        cls,
        output_node: str = "out",
        source_name: str = "Iin",
        input_node: str = "in",
        reference_node: str = "0",
    ) -> "AnalysisRequest":
        return cls(
            kind="transimpedance",
            source_name=source_name,
            output_node=output_node,
            input_node=input_node,
            reference_node=reference_node,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "source_name": self.source_name,
            "output_node": self.output_node,
            "input_node": self.input_node,
            "reference_node": self.reference_node,
        }


@dataclass(frozen=True)
class ACAnalysisResult:
    request: AnalysisRequest
    frequencies_hz: np.ndarray
    values: np.ndarray


class LinearACAnalyzer:
    def __init__(self, simulator: MNASimulator | None = None):
        self.simulator = simulator or MNASimulator()
        self.last_statistics: dict[str, int] = {}

    def analyze(
        self,
        circuit: LinearCircuit,
        request: AnalysisRequest,
        frequencies_hz: np.ndarray,
    ) -> ACAnalysisResult:
        return self.analyze_many(circuit, (request,), frequencies_hz)[0]

    def analyze_many(
        self,
        circuit: LinearCircuit,
        requests: tuple[AnalysisRequest, ...] | list[AnalysisRequest],
        frequencies_hz: np.ndarray,
        *,
        compiled_mna: CompiledLinearMNA | None = None,
    ) -> tuple[ACAnalysisResult, ...]:
        """Analyze multiple observables while reusing compatible solves."""

        requests = tuple(requests)
        if not requests:
            self.last_statistics = {"solve_groups": 0, "observable_extractions": 0}
            return ()
        frequencies = np.asarray(frequencies_hz, dtype=float)
        results: dict[int, ACAnalysisResult] = {}
        solve_groups = 0

        shared_requests = tuple(
            request for request in requests if request.kind != "output_impedance"
        )
        if shared_requests:
            batch = self.simulator.solve_ac_batched(
                circuit,
                frequencies,
                compiled=compiled_mna,
            )
            solve_groups += 1
            for index, request in enumerate(requests):
                if request.kind == "output_impedance":
                    continue
                values = self._extract_values(circuit, request, batch)
                results[index] = ACAnalysisResult(request, frequencies, values)

        output_indices: dict[str, list[int]] = {}
        for index, request in enumerate(requests):
            if request.kind == "output_impedance":
                output_indices.setdefault(request.source_name, []).append(index)
        for source_name, indices in output_indices.items():
            request = requests[indices[0]]
            test_circuit = self._output_impedance_circuit(circuit, request)
            batch = self.simulator.solve_ac_batched(test_circuit, frequencies)
            solve_groups += 1
            for index in indices:
                current_request = requests[index]
                values = (
                    batch.node_voltage(current_request.output_node)
                    - batch.node_voltage(current_request.reference_node)
                )
                results[index] = ACAnalysisResult(
                    current_request,
                    frequencies,
                    values.astype(np.complex128),
                )

        self.last_statistics = {
            "solve_groups": solve_groups,
            "observable_extractions": len(requests),
        }
        return tuple(results[index] for index in range(len(requests)))

    def voltage_transfer(
        self,
        circuit: LinearCircuit,
        request: AnalysisRequest,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        return self.analyze_many(circuit, (request,), frequencies_hz)[0].values

    def input_impedance(
        self,
        circuit: LinearCircuit,
        request: AnalysisRequest,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        return self.analyze_many(circuit, (request,), frequencies_hz)[0].values

    def output_impedance(
        self,
        circuit: LinearCircuit,
        request: AnalysisRequest,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        return self.analyze_many(circuit, (request,), frequencies_hz)[0].values

    def transimpedance(
        self,
        circuit: LinearCircuit,
        request: AnalysisRequest,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        return self.analyze_many(circuit, (request,), frequencies_hz)[0].values

    def _extract_values(
        self,
        circuit: LinearCircuit,
        request: AnalysisRequest,
        batch: MNASolutionBatch,
    ) -> np.ndarray:
        if request.kind == "voltage_transfer":
            source = _find_source(circuit, request.source_name)
            if source.value == 0:
                raise ValueError("source value must be nonzero for transfer calculation")
            output = batch.node_voltage(request.output_node) - batch.node_voltage(
                request.reference_node
            )
            return (output / source.value).astype(np.complex128)
        if request.kind == "input_impedance":
            source = _find_source(circuit, request.source_name)
            if {source.n_plus, source.n_minus} != {
                request.input_node,
                request.reference_node,
            }:
                raise ValueError(
                    "input impedance analysis requires the named source to connect "
                    "between input_node and reference_node"
                )
            input_current = -batch.branch_current(source.name)
            with np.errstate(divide="ignore", invalid="ignore"):
                impedance = source.value / input_current
            return impedance.astype(np.complex128)
        if request.kind == "transimpedance":
            source = _find_current_source(circuit, request.source_name)
            if source.value == 0:
                raise ValueError(
                    "current source value must be nonzero for transimpedance calculation"
                )
            output = batch.node_voltage(request.output_node) - batch.node_voltage(
                request.reference_node
            )
            return (output / source.value).astype(np.complex128)
        raise ValueError(f"unsupported analysis kind: {request.kind!r}")

    @staticmethod
    def _output_impedance_circuit(
        circuit: LinearCircuit,
        request: AnalysisRequest,
    ) -> LinearCircuit:
        return LinearCircuit(
            elements=circuit.elements,
            voltage_sources=tuple(
                VoltageSource(source.name, source.n_plus, source.n_minus, 0.0)
                for source in circuit.voltage_sources
            ),
            current_sources=(
                *(
                    CurrentSource(source.name, source.n_plus, source.n_minus, 0.0)
                    for source in circuit.current_sources
                ),
                CurrentSource(
                    request.source_name,
                    request.reference_node,
                    request.output_node,
                    1.0,
                ),
            ),
            controlled_voltage_sources=circuit.controlled_voltage_sources,
        )


def _normalize_analysis_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace("-", "_")
    if normalized not in ANALYSIS_ALIASES:
        raise ValueError(f"unsupported analysis kind: {kind!r}")
    return ANALYSIS_ALIASES[normalized]


def _find_source(circuit: LinearCircuit, source_name: str) -> VoltageSource:
    for source in circuit.voltage_sources:
        if source.name == source_name:
            return source
    raise ValueError(f"voltage source {source_name!r} not found")


def _find_current_source(circuit: LinearCircuit, source_name: str) -> CurrentSource:
    for source in circuit.current_sources:
        if source.name == source_name:
            return source
    raise ValueError(f"current source {source_name!r} not found")


def _node_voltage(solution: dict[str, complex], node: str) -> complex:
    if node in {"0", "gnd", "GND"}:
        return 0.0
    return solution.get(node, 0.0)
