from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .analysis import AnalysisRequest, LinearACAnalyzer
from .formatting import eng
from .mna import (
    CurrentSource,
    LinearCircuit,
    LinearElement,
    VoltageControlledVoltageSource,
    VoltageSource,
)
from .spec import LibrarySpec, normalize_element

if TYPE_CHECKING:
    from .graph import CircuitGraph


OPAMP_OPEN_LOOP_GAIN = 1_000_000.0


@dataclass(frozen=True)
class ParamSpec:
    name: str
    element_type: str
    min_value: float | None = None
    max_value: float | None = None


class CircuitTemplate(ABC):
    name: str
    description: str
    required_elements: frozenset[str]
    supported_kinds: frozenset[str]
    component_count: int
    params: tuple[ParamSpec, ...]
    output_node: str = "out"
    source_name: str = "Vin"

    @abstractmethod
    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        """Fast template response used during optimization."""

    @abstractmethod
    def netlist(self, values: dict[str, float], title: str = "generated") -> str:
        """Export a SPICE-style netlist for the template."""

    @abstractmethod
    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        """Export the template as the unified linear circuit representation."""

    def analyze(
        self,
        values: dict[str, float],
        frequencies_hz: np.ndarray,
        request: AnalysisRequest | None = None,
    ) -> np.ndarray:
        request = request or AnalysisRequest.voltage_transfer(
            output_node=self.output_node,
            source_name=self.source_name,
        )
        if (
            request.kind == "voltage_transfer"
            and request.output_node == self.output_node
            and request.source_name == self.source_name
            and request.reference_node in {"0", "gnd", "GND"}
        ):
            return self.response(values, frequencies_hz)
        return LinearACAnalyzer().analyze(self.to_circuit(values), request, frequencies_hz).values

    def analyze_many(
        self,
        values_batch: Sequence[dict[str, float]],
        frequencies_hz: np.ndarray,
        request: AnalysisRequest | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a deterministic parameter batch in input order.

        Templates may override this hook with a vectorized or compiled MNA
        implementation.  The compatibility implementation keeps the public
        batch contract available without changing existing template formulas.
        """

        return tuple(
            self.analyze(values, frequencies_hz, request)
            for values in values_batch
        )

    def mna_response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        request = AnalysisRequest.voltage_transfer(
            output_node=self.output_node,
            source_name=self.source_name,
        )
        return LinearACAnalyzer().analyze(
            self.to_circuit(values),
            request,
            frequencies_hz,
        ).values

    def to_graph(self, values: dict[str, float]) -> "CircuitGraph":
        """Materialize this template as the solver-independent circuit graph."""

        from .graph import linear_circuit_to_graph

        return linear_circuit_to_graph(
            self.to_circuit(values),
            name=self.name,
            graph_id=self.name,
            output_node=self.output_node,
            source_name=self.source_name,
        )

    def summary(self, values: dict[str, float]) -> str:
        pairs = ", ".join(f"{key}={eng(value)}" for key, value in values.items())
        return f"{self.name}: {pairs}"

    def parameter_bounds(self, library: LibrarySpec) -> list[tuple[float, float]]:
        bounds = []
        for param in self.params:
            if param.min_value is not None and param.max_value is not None:
                bounds.append((param.min_value, param.max_value))
            else:
                bounds.append(library.range_for(param.element_type))
        return bounds

    def is_compatible(self, library: LibrarySpec, max_components: int, behavior_kind: str) -> bool:
        if self.component_count > max_components:
            return False
        if behavior_kind not in self.supported_kinds and "*" not in self.supported_kinds:
            return False
        return (
            self.required_elements.issubset(library.allowed)
            and library.required.issubset(self.required_elements)
            and all(library.supports_real_component_family(element) for element in self.required_elements)
        )

    def component_inventory(self) -> dict[str, int]:
        inventory: dict[str, int] = {}
        for param in self.params:
            element_type = normalize_element(param.element_type)
            inventory[element_type] = inventory.get(element_type, 0) + 1
        remaining = self.component_count - sum(inventory.values())
        if remaining > 0:
            if "opamp" in self.required_elements or "E" in self.required_elements:
                inventory["opamp"] = inventory.get("opamp", 0) + remaining
            else:
                inventory["other"] = inventory.get("other", 0) + remaining
        return inventory


class RCLowPass(CircuitTemplate):
    name = "rc_lowpass"
    description = "One-pole passive RC low-pass, output across capacitor."
    required_elements = frozenset({"R", "C"})
    supported_kinds = frozenset({"lowpass", "samples", "zpk"})
    component_count = 2
    params = (ParamSpec("R", "R"), ParamSpec("C", "C"))

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        return 1.0 / (1.0 + s * values["R"] * values["C"])

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "out", values["R"]),
                LinearElement("C1", "C", "out", "0", values["C"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        )

    def netlist(self, values: dict[str, float], title: str = "rc_lowpass") -> str:
        return "\n".join(
            [
                f"* {title}: passive RC low-pass",
                "Vin in 0 AC 1",
                f"R1 in out {eng(values['R'])}",
                f"C1 out 0 {eng(values['C'])}",
                ".ac dec 80 1 100Meg",
                ".end",
            ]
        )


class RCHighPass(CircuitTemplate):
    name = "rc_highpass"
    description = "One-pole passive RC high-pass, output across resistor."
    required_elements = frozenset({"R", "C"})
    supported_kinds = frozenset({"highpass", "samples", "zpk"})
    component_count = 2
    params = (ParamSpec("R", "R"), ParamSpec("C", "C"))

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        tau = values["R"] * values["C"]
        return (s * tau) / (1.0 + s * tau)

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(
                LinearElement("C1", "C", "in", "out", values["C"]),
                LinearElement("R1", "R", "out", "0", values["R"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        )

    def netlist(self, values: dict[str, float], title: str = "rc_highpass") -> str:
        return "\n".join(
            [
                f"* {title}: passive RC high-pass",
                "Vin in 0 AC 1",
                f"C1 in out {eng(values['C'])}",
                f"R1 out 0 {eng(values['R'])}",
                ".ac dec 80 1 100Meg",
                ".end",
            ]
        )


class RLCBandPass(CircuitTemplate):
    name = "rlc_bandpass"
    description = "Series RLC band-pass, output across resistor."
    required_elements = frozenset({"R", "L", "C"})
    supported_kinds = frozenset({"bandpass", "samples", "zpk"})
    component_count = 3
    params = (ParamSpec("R", "R"), ParamSpec("L", "L"), ParamSpec("C", "C"))

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        r, l, c = values["R"], values["L"], values["C"]
        return (s * r * c) / (s**2 * l * c + s * r * c + 1.0)

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(
                LinearElement("C1", "C", "in", "n1", values["C"]),
                LinearElement("L1", "L", "n1", "out", values["L"]),
                LinearElement("R1", "R", "out", "0", values["R"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        )

    def netlist(self, values: dict[str, float], title: str = "rlc_bandpass") -> str:
        return "\n".join(
            [
                f"* {title}: series RLC band-pass, V(out) across R1",
                "Vin in 0 AC 1",
                f"C1 in n1 {eng(values['C'])}",
                f"L1 n1 out {eng(values['L'])}",
                f"R1 out 0 {eng(values['R'])}",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


class ShuntResistorImpedance(CircuitTemplate):
    name = "shunt_resistor_impedance"
    description = "One-resistor input impedance target from input to ground."
    required_elements = frozenset({"R"})
    supported_kinds = frozenset({"impedance", "constant_impedance"})
    component_count = 1
    params = (ParamSpec("R", "R"),)
    output_node = "in"

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        return np.ones_like(frequencies_hz, dtype=np.complex128)

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(LinearElement("R1", "R", "in", "0", values["R"]),),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        )

    def netlist(self, values: dict[str, float], title: str = "shunt_resistor_impedance") -> str:
        return "\n".join(
            [
                f"* {title}: one-resistor input impedance",
                "Vin in 0 AC 1",
                f"R1 in 0 {eng(values['R'])}",
                ".ac dec 80 1 100Meg",
                ".end",
            ]
        )


class OutputResistorImpedance(CircuitTemplate):
    name = "output_resistor_impedance"
    description = "One-resistor output impedance target from output to reference."
    required_elements = frozenset({"R"})
    supported_kinds = frozenset({"output_impedance", "constant_output_impedance"})
    component_count = 1
    params = (ParamSpec("R", "R"),)

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        return np.full_like(frequencies_hz, values["R"], dtype=np.complex128)

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(LinearElement("Rout", "R", "out", "0", values["R"]),),
            voltage_sources=(),
        )

    def mna_response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        return LinearACAnalyzer().analyze(
            self.to_circuit(values),
            AnalysisRequest.output_impedance(output_node="out"),
            frequencies_hz,
        ).values

    def netlist(self, values: dict[str, float], title: str = "output_resistor_impedance") -> str:
        return "\n".join(
            [
                f"* {title}: one-resistor output impedance",
                f"Rout out 0 {eng(values['R'])}",
                "* Output impedance is measured by injecting AC test current into out",
                ".ac dec 80 1 100Meg",
                ".end",
            ]
        )


class BufferedCascadeRCLowPass(CircuitTemplate):
    name = "buffered_cascade_rc_lowpass"
    description = "Two buffered RC low-pass stages for second-order responses."
    required_elements = frozenset({"R", "C", "opamp"})
    supported_kinds = frozenset({"lowpass", "samples", "zpk"})
    component_count = 5
    params = (
        ParamSpec("R1", "R"),
        ParamSpec("C1", "C"),
        ParamSpec("R2", "R"),
        ParamSpec("C2", "C"),
    )

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        h1 = 1.0 / (1.0 + s * values["R1"] * values["C1"])
        h2 = 1.0 / (1.0 + s * values["R2"] * values["C2"])
        return h1 * h2

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "n1", values["R1"]),
                LinearElement("C1", "C", "n1", "0", values["C1"]),
                LinearElement("R2", "R", "nbuf", "out", values["R2"]),
                LinearElement("C2", "C", "out", "0", values["C2"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Ebuf", "nbuf", "0", "n1", "0", 1.0),
            ),
        )

    def netlist(self, values: dict[str, float], title: str = "buffered_cascade_rc_lowpass") -> str:
        return "\n".join(
            [
                f"* {title}: two RC low-pass stages with ideal unity buffer",
                "Vin in 0 AC 1",
                f"R1 in n1 {eng(values['R1'])}",
                f"C1 n1 0 {eng(values['C1'])}",
                "Ebuf nbuf 0 n1 0 1",
                f"R2 nbuf out {eng(values['R2'])}",
                f"C2 out 0 {eng(values['C2'])}",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


class SallenKeyLowPass(CircuitTemplate):
    name = "sallen_key_lowpass"
    description = "Unity-gain Sallen-Key second-order active RC low-pass."
    required_elements = frozenset({"R", "C", "opamp"})
    supported_kinds = frozenset({"lowpass", "samples", "zpk"})
    component_count = 5
    params = (
        ParamSpec("R1", "R"),
        ParamSpec("R2", "R"),
        ParamSpec("C1", "C"),
        ParamSpec("C2", "C"),
    )

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        r1, r2 = values["R1"], values["R2"]
        c1, c2 = values["C1"], values["C2"]
        denominator = 1.0 + s * c2 * (r1 + r2) + (s**2) * r1 * r2 * c1 * c2
        return 1.0 / denominator

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "n1", values["R1"]),
                LinearElement("R2", "R", "n1", "n2", values["R2"]),
                LinearElement("C1", "C", "n1", "out", values["C1"]),
                LinearElement("C2", "C", "n2", "0", values["C2"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Ebuf", "out", "0", "n2", "0", 1.0),
            ),
        )

    def netlist(self, values: dict[str, float], title: str = "sallen_key_lowpass") -> str:
        return "\n".join(
            [
                f"* {title}: unity-gain Sallen-Key second-order low-pass",
                "Vin in 0 AC 1",
                f"R1 in n1 {eng(values['R1'])}",
                f"R2 n1 n2 {eng(values['R2'])}",
                f"C1 n1 out {eng(values['C1'])}",
                f"C2 n2 0 {eng(values['C2'])}",
                "Ebuf out 0 n2 0 1",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


class SallenKeyGainLowPass(CircuitTemplate):
    name = "sallen_key_gain_lowpass"
    description = "Gain Sallen-Key second-order active RC low-pass."
    required_elements = frozenset({"R", "C", "opamp"})
    supported_kinds = frozenset({"lowpass", "samples", "zpk"})
    component_count = 7
    params = (
        ParamSpec("R1", "R"),
        ParamSpec("R2", "R"),
        ParamSpec("C1", "C"),
        ParamSpec("C2", "C"),
        ParamSpec("Rg", "R"),
        ParamSpec("Rf", "R"),
    )

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        r1, r2 = values["R1"], values["R2"]
        c1, c2 = values["C1"], values["C2"]
        gain = 1.0 + values["Rf"] / values["Rg"]
        denominator = (
            1.0
            + s * (r2 * c2 + r1 * c2 + r1 * c1 * (1.0 - gain))
            + (s**2) * r1 * r2 * c1 * c2
        )
        return gain / denominator

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        gain = 1.0 + values["Rf"] / values["Rg"]
        return LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "n1", values["R1"]),
                LinearElement("R2", "R", "n1", "n2", values["R2"]),
                LinearElement("C1", "C", "n1", "out", values["C1"]),
                LinearElement("C2", "C", "n2", "0", values["C2"]),
                LinearElement("Rg", "R", "neg", "0", values["Rg"]),
                LinearElement("Rf", "R", "out", "neg", values["Rf"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Eop", "out", "0", "n2", "0", gain),
            ),
        )

    def netlist(self, values: dict[str, float], title: str = "sallen_key_gain_lowpass") -> str:
        gain = 1.0 + values["Rf"] / values["Rg"]
        return "\n".join(
            [
                f"* {title}: gain Sallen-Key second-order low-pass",
                "Vin in 0 AC 1",
                f"R1 in n1 {eng(values['R1'])}",
                f"R2 n1 n2 {eng(values['R2'])}",
                f"C1 n1 out {eng(values['C1'])}",
                f"C2 n2 0 {eng(values['C2'])}",
                f"* Conceptual non-inverting gain network: gain = 1 + Rf/Rg = {gain:.8g}",
                f"Rg neg 0 {eng(values['Rg'])}",
                f"Rf out neg {eng(values['Rf'])}",
                f"Eop out 0 n2 0 {gain:.12g}",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


class GainRCLowPass(CircuitTemplate):
    name = "gain_rc_lowpass"
    description = "One-pole RC low-pass followed by ideal non-inverting gain."
    required_elements = frozenset({"R", "C", "opamp"})
    supported_kinds = frozenset({"lowpass", "samples", "zpk"})
    component_count = 5
    params = (
        ParamSpec("R", "R"),
        ParamSpec("C", "C"),
        ParamSpec("Rg", "R"),
        ParamSpec("Rf", "R"),
    )

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        gain = 1.0 + values["Rf"] / values["Rg"]
        return gain / (1.0 + s * values["R"] * values["C"])

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        gain = 1.0 + values["Rf"] / values["Rg"]
        return LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "n1", values["R"]),
                LinearElement("C1", "C", "n1", "0", values["C"]),
                LinearElement("Rg", "R", "neg", "0", values["Rg"]),
                LinearElement("Rf", "R", "out", "neg", values["Rf"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Eop", "out", "0", "n1", "0", gain),
            ),
        )

    def netlist(self, values: dict[str, float], title: str = "gain_rc_lowpass") -> str:
        gain = 1.0 + values["Rf"] / values["Rg"]
        return "\n".join(
            [
                f"* {title}: RC low-pass plus ideal op-amp gain",
                "Vin in 0 AC 1",
                f"R1 in n1 {eng(values['R'])}",
                f"C1 n1 0 {eng(values['C'])}",
                f"* Conceptual non-inverting gain network: gain = 1 + Rf/Rg = {gain:.8g}",
                f"Rg neg 0 {eng(values['Rg'])}",
                f"Rf out neg {eng(values['Rf'])}",
                f"Eop out 0 n1 0 {gain:.12g}",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


class GainRCHighPass(CircuitTemplate):
    name = "gain_rc_highpass"
    description = "One-pole RC high-pass followed by ideal non-inverting gain."
    required_elements = frozenset({"R", "C", "opamp"})
    supported_kinds = frozenset({"highpass", "samples", "zpk"})
    component_count = 5
    params = (
        ParamSpec("R", "R"),
        ParamSpec("C", "C"),
        ParamSpec("Rg", "R"),
        ParamSpec("Rf", "R"),
    )

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        tau = values["R"] * values["C"]
        gain = 1.0 + values["Rf"] / values["Rg"]
        return gain * (s * tau) / (1.0 + s * tau)

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        gain = 1.0 + values["Rf"] / values["Rg"]
        return LinearCircuit(
            elements=(
                LinearElement("C1", "C", "in", "n1", values["C"]),
                LinearElement("R1", "R", "n1", "0", values["R"]),
                LinearElement("Rg", "R", "neg", "0", values["Rg"]),
                LinearElement("Rf", "R", "out", "neg", values["Rf"]),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Eop", "out", "0", "n1", "0", gain),
            ),
        )

    def netlist(self, values: dict[str, float], title: str = "gain_rc_highpass") -> str:
        gain = 1.0 + values["Rf"] / values["Rg"]
        return "\n".join(
            [
                f"* {title}: RC high-pass plus ideal op-amp gain",
                "Vin in 0 AC 1",
                f"C1 in n1 {eng(values['C'])}",
                f"R1 n1 0 {eng(values['R'])}",
                f"* Conceptual non-inverting gain network: gain = 1 + Rf/Rg = {gain:.8g}",
                f"Rg neg 0 {eng(values['Rg'])}",
                f"Rf out neg {eng(values['Rf'])}",
                f"Eop out 0 n1 0 {gain:.12g}",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


class TransimpedanceAmplifier(CircuitTemplate):
    name = "transimpedance_amplifier"
    description = "Ideal op-amp transimpedance amplifier with feedback R and C."
    required_elements = frozenset({"R", "C", "opamp"})
    supported_kinds = frozenset({"transimpedance", "tia", "samples"})
    component_count = 3
    params = (
        ParamSpec("Rf", "R"),
        ParamSpec("Cf", "C"),
    )
    source_name = "Iin"

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        feedback_impedance = values["Rf"] / (1.0 + s * values["Rf"] * values["Cf"])
        closed_loop_factor = OPAMP_OPEN_LOOP_GAIN / (1.0 + OPAMP_OPEN_LOOP_GAIN)
        return -closed_loop_factor * feedback_impedance

    def analyze(
        self,
        values: dict[str, float],
        frequencies_hz: np.ndarray,
        request: AnalysisRequest | None = None,
    ) -> np.ndarray:
        request = request or AnalysisRequest.transimpedance(
            output_node=self.output_node,
            source_name=self.source_name,
        )
        if (
            request.kind == "transimpedance"
            and request.output_node == self.output_node
            and request.source_name == self.source_name
            and request.reference_node in {"0", "gnd", "GND"}
        ):
            return self.response(values, frequencies_hz)
        return LinearACAnalyzer().analyze(self.to_circuit(values), request, frequencies_hz).values

    def mna_response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        return LinearACAnalyzer().analyze(
            self.to_circuit(values),
            AnalysisRequest.transimpedance(output_node=self.output_node, source_name=self.source_name),
            frequencies_hz,
        ).values

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return LinearCircuit(
            elements=(
                LinearElement("Rf", "R", "out", "in", values["Rf"]),
                LinearElement("Cf", "C", "out", "in", values["Cf"]),
            ),
            voltage_sources=(),
            current_sources=(CurrentSource("Iin", "0", "in", 1.0),),
            controlled_voltage_sources=(
                VoltageControlledVoltageSource("Eop", "out", "0", "0", "in", OPAMP_OPEN_LOOP_GAIN),
            ),
        )

    def netlist(self, values: dict[str, float], title: str = "transimpedance_amplifier") -> str:
        return "\n".join(
            [
                f"* {title}: ideal op-amp transimpedance amplifier",
                "Iin 0 in AC 1",
                f"Rf out in {eng(values['Rf'])}",
                f"Cf out in {eng(values['Cf'])}",
                f"Eop out 0 0 in {OPAMP_OPEN_LOOP_GAIN:.12g}",
                ".ac dec 100 1 100Meg",
                ".end",
            ]
        )


def default_templates() -> list[CircuitTemplate]:
    return [
        RCLowPass(),
        RCHighPass(),
        RLCBandPass(),
        ShuntResistorImpedance(),
        OutputResistorImpedance(),
        BufferedCascadeRCLowPass(),
        SallenKeyLowPass(),
        SallenKeyGainLowPass(),
        GainRCLowPass(),
        GainRCHighPass(),
        TransimpedanceAmplifier(),
    ]
