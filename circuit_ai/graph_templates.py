from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .analysis import AnalysisRequest, LinearACAnalyzer
from .formatting import eng
from .mna import LinearCircuit, LinearElement, VoltageSource
from .graph import CircuitGraph, graph_to_linear_circuit
from .spec import LibrarySpec
from .templates import CircuitTemplate, ParamSpec


@dataclass(frozen=True)
class ElementSlot:
    name: str
    kind: str
    n1: str
    n2: str

    def normalized_edge(self) -> tuple[str, str]:
        return tuple(sorted((self.n1, self.n2)))


class GraphCircuitTemplate(CircuitTemplate):
    """A generated linear R/C/L candidate evaluated by MNA.

    This class is the bridge from fixed hand-written templates to arbitrary
    generated circuit graphs.  Each slot fixes an element kind and two nodes;
    synthesis optimizes the numeric values.
    """

    supported_kinds = frozenset(
        {"lowpass", "highpass", "bandpass", "samples", "zpk", "impedance", "constant_impedance"}
    )

    def __init__(self, slots: tuple[ElementSlot, ...], output_node: str = "out", source_name: str = "Vin"):
        self.slots = slots
        self.output_node = output_node
        self.source_name = source_name
        self.component_count = len(slots)
        self.required_elements = frozenset(slot.kind for slot in slots)
        self.params = tuple(ParamSpec(slot.name, slot.kind) for slot in slots)
        self.name = "graph_" + "__".join(
            f"{slot.kind}{slot.n1.replace('0', 'g')}_{slot.n2.replace('0', 'g')}" for slot in slots
        )
        self.description = "Generated linear graph: " + ", ".join(
            f"{slot.name}:{slot.kind}({slot.n1},{slot.n2})" for slot in slots
        )
        self._analyzer = LinearACAnalyzer()

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        try:
            return self._analyzer.analyze(
                self.to_circuit(values),
                AnalysisRequest.voltage_transfer(output_node=self.output_node, source_name=self.source_name),
                frequencies_hz,
            ).values
        except (np.linalg.LinAlgError, ValueError, StopIteration):
            return np.full(frequencies_hz.shape, np.nan + 1j * np.nan, dtype=np.complex128)

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        elements = tuple(
            LinearElement(slot.name, slot.kind, slot.n1, slot.n2, values[slot.name])
            for slot in self.slots
        )
        return LinearCircuit(
            elements=elements,
            voltage_sources=(VoltageSource(self.source_name, "in", "0", 1.0),),
        )

    def netlist(self, values: dict[str, float], title: str = "generated_graph") -> str:
        lines = [
            f"* {title}: generated R/C/L graph candidate",
            f"* topology: {self.description}",
            "Vin in 0 AC 1",
        ]
        for slot in self.slots:
            lines.append(f"{slot.kind}{slot.name[1:]} {slot.n1} {slot.n2} {eng(values[slot.name])}")
        lines.extend([".ac dec 100 1 100Meg", ".end"])
        return "\n".join(lines)

    def summary(self, values: dict[str, float]) -> str:
        topology = ", ".join(f"{slot.kind}({slot.n1},{slot.n2})" for slot in self.slots)
        pairs = ", ".join(f"{key}={eng(value)}" for key, value in values.items())
        return f"{self.name}: {topology}; {pairs}"

    def is_compatible(self, library: LibrarySpec, max_components: int, behavior_kind: str) -> bool:
        return (
            self.component_count <= max_components
            and self.required_elements.issubset(library.allowed)
            and library.required.issubset(self.required_elements)
            and all(library.supports_real_component_family(element) for element in self.required_elements)
        )

    def structural_report(self):
        from .structure import analyze_slots

        return analyze_slots(self.slots)


class NativeCircuitGraphTemplate(CircuitTemplate):
    """Evaluate a native :class:`CircuitGraph` through the legacy optimizer API.

    The graph remains the structural artifact.  Only variable bindings are
    materialized into fixed parameter values for one optimizer evaluation;
    the exported graph and all later backends therefore share the same
    topology and terminal connections.
    """

    supported_kinds = frozenset({"*"})

    def __init__(self, graph: CircuitGraph):
        graph.require_valid()
        self.graph = graph
        self.name = f"{graph.name}.{graph.topology_hash[:10]}"
        self.description = graph.description or "Native CircuitGraph candidate"
        self.component_count = sum(
            component.model.kind not in {"voltage_source", "current_source"}
            for component in graph.components
        )
        self.required_elements = frozenset(
            _inventory_kind(component.model.kind)
            for component in graph.components
            if component.model.kind not in {"voltage_source", "current_source"}
        )
        variables = {item.variable_id: item for item in graph.variables}
        self.params = tuple(
            ParamSpec(
                variable.variable_id,
                _inventory_kind(_variable_component_kind(graph, variable.variable_id)),
                variable.lower,
                variable.upper,
            )
            for variable in sorted(variables.values(), key=lambda item: item.variable_id)
            if _variable_component_kind(graph, variable.variable_id) not in {"voltage_source", "current_source"}
        )
        self.output_node = _graph_port_node(graph, "output", fallback="out")
        self.source_name = _graph_source_name(graph)
        self._analyzer = LinearACAnalyzer()

    def response(self, values: dict[str, float], frequencies_hz: np.ndarray) -> np.ndarray:
        try:
            request = AnalysisRequest.voltage_transfer(
                output_node=self.output_node,
                source_name=self.source_name,
            )
            return self._analyzer.analyze(
                self.to_circuit(values), request, frequencies_hz
            ).values
        except (np.linalg.LinAlgError, ValueError, StopIteration, KeyError):
            return np.full(frequencies_hz.shape, np.nan + 1j * np.nan, dtype=np.complex128)

    def to_graph(self, values: dict[str, float]) -> CircuitGraph:
        variables = {item.variable_id: item for item in self.graph.variables}
        unknown = set(values) - set(variables)
        if unknown:
            raise ValueError(f"unknown native graph parameters: {sorted(unknown)}")
        components = []
        for component in self.graph.components:
            parameters = []
            for parameter in component.parameters:
                if parameter.variable_id is None:
                    parameters.append(parameter)
                    continue
                if parameter.variable_id not in values:
                    raise ValueError(f"missing native graph parameter {parameter.variable_id!r}")
                parameters.append(
                    replace(
                        parameter,
                        value=values[parameter.variable_id],
                        variable_id=None,
                    )
                )
            components.append(replace(component, parameters=tuple(parameters)))
        return replace(self.graph, components=tuple(components), variables=())

    def to_circuit(self, values: dict[str, float]) -> LinearCircuit:
        return graph_to_linear_circuit(self.to_graph(values))

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
        return self._analyzer.analyze(self.to_circuit(values), request, frequencies_hz).values

    def netlist(self, values: dict[str, float], title: str = "native_graph") -> str:
        circuit = self.to_circuit(values)
        lines = [f"* {title}: {self.description}"]
        for source in circuit.voltage_sources:
            lines.append(f"{source.name} {source.n_plus} {source.n_minus} AC {_format_scalar(source.value)}")
        for element in circuit.elements:
            lines.append(f"{element.kind}{element.name[1:]} {element.n1} {element.n2} {eng(element.value)}")
        for source in circuit.current_sources:
            lines.append(f"{source.name} {source.n_plus} {source.n_minus} {_format_scalar(source.value)}")
        lines.extend([".ac dec 100 1 100Meg", ".end"])
        return "\n".join(lines)

    def parameter_bounds(self, library: LibrarySpec) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        variables = {item.variable_id: item for item in self.graph.variables}
        for parameter in self.params:
            variable = variables[parameter.name]
            if variable.lower is not None and variable.upper is not None:
                bounds.append((variable.lower, variable.upper))
            else:
                bounds.append(library.range_for(parameter.element_type))
        return bounds

    def component_inventory(self) -> dict[str, int]:
        inventory: dict[str, int] = {}
        for component in self.graph.components:
            kind = _inventory_kind(component.model.kind)
            if kind in {"voltage_source", "current_source"}:
                continue
            inventory[kind] = inventory.get(kind, 0) + 1
        return inventory


def _inventory_kind(kind: str) -> str:
    return {"vcvs": "opamp", "E": "opamp"}.get(kind, kind)


def _variable_component_kind(graph: CircuitGraph, variable_id: str) -> str:
    for component in graph.components:
        if any(parameter.variable_id == variable_id for parameter in component.parameters):
            return component.model.kind
    return "other"


def _graph_port_node(graph: CircuitGraph, direction: str, *, fallback: str) -> str:
    port = next((item for item in graph.ports if item.direction == direction), None)
    if port is None or not port.terminals:
        return fallback
    net_names = {item.net_id: item.name for item in graph.nets}
    return net_names.get(port.terminals[0].net_id, port.terminals[0].net_id)


def _graph_source_name(graph: CircuitGraph) -> str:
    source = next(
        (item for item in graph.components if item.model.kind == "voltage_source"),
        None,
    )
    return source.reference if source is not None else "Vin"


def _format_scalar(value) -> str:
    if isinstance(value, complex):
        if abs(value.imag) < 1e-18:
            value = value.real
        else:
            return f"({value.real:g},{value.imag:g})"
    return eng(float(value))
