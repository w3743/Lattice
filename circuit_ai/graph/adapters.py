"""Loss-aware adapters between legacy circuit containers and CircuitGraph v1."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from ..experts import TopologyCandidate
from ..ir import IRComponent, IRPort, UnifiedIR
from ..mna import (
    CurrentSource,
    GROUND_NAMES,
    LinearCircuit,
    LinearElement,
    VoltageControlledVoltageSource,
    VoltageSource,
)
from .model import (
    CircuitGraph,
    ComponentInstance,
    GraphDomain,
    GraphPort,
    Net,
    ParameterBinding,
    ParameterVariable,
    PortTerminal,
    ProvenanceRecord,
    TerminalConnection,
    encode_graph_value,
)
from .primitives import expected_parameters, parameter_unit, primitive_model_ref


def linear_circuit_to_graph(
    circuit: LinearCircuit,
    *,
    name: str = "linear_circuit",
    graph_id: str | None = None,
    input_node: str = "in",
    output_node: str = "out",
    reference_node: str = "0",
    source_name: str = "Vin",
) -> CircuitGraph:
    nodes = set(circuit.nodes())
    nodes.add(reference_node)
    domain_id = "electrical"
    nets = tuple(
        Net(
            net_id=node,
            name=node,
            domain_id=domain_id,
            kind="reference" if node == reference_node else "signal",
            is_reference=node == reference_node,
        )
        for node in sorted(nodes)
    )
    components: list[ComponentInstance] = []
    for element in circuit.elements:
        model = primitive_model_ref(element.kind, 2)
        components.append(
            ComponentInstance(
                instance_id=element.name,
                reference=element.name,
                model=model,
                connections=(
                    TerminalConnection(
                        model.terminals[0].name,
                        _normalize_reference_node(element.n1, reference_node),
                    ),
                    TerminalConnection(
                        model.terminals[1].name,
                        _normalize_reference_node(element.n2, reference_node),
                    ),
                ),
                parameters=(
                    ParameterBinding("value", parameter_unit(element.kind, "value"), element.value),
                ),
            )
        )
    for source in circuit.voltage_sources:
        components.append(
            _independent_source_instance(source, "voltage_source", reference_node)
        )
    for source in circuit.current_sources:
        components.append(
            _independent_source_instance(source, "current_source", reference_node)
        )
    for source in circuit.controlled_voltage_sources:
        model = primitive_model_ref("vcvs")
        components.append(
            ComponentInstance(
                instance_id=source.name,
                reference=source.name,
                model=model,
                connections=tuple(
                    TerminalConnection(terminal.name, node)
                    for terminal, node in zip(
                        model.terminals,
                        (
                            _normalize_reference_node(source.n_plus, reference_node),
                            _normalize_reference_node(source.n_minus, reference_node),
                            _normalize_reference_node(source.control_plus, reference_node),
                            _normalize_reference_node(source.control_minus, reference_node),
                        ),
                    )
                ),
                parameters=(ParameterBinding("gain", "1", source.gain),),
            )
        )
    ports: list[GraphPort] = []
    if input_node in nodes:
        ports.append(_two_terminal_port("input", "input", domain_id, input_node, reference_node))
    if output_node in nodes:
        ports.append(_two_terminal_port("output", "output", domain_id, output_node, reference_node))
    graph = CircuitGraph(
        graph_id=graph_id or name,
        name=name,
        description="LinearCircuit adapter output",
        family="linear_ac",
        preferred_solver_ids=("linear_mna",),
        domains=(GraphDomain(domain_id, reference_net_id=reference_node),),
        ports=tuple(ports),
        nets=nets,
        components=tuple(components),
        provenance=(
            ProvenanceRecord(
                source="LinearCircuit",
                source_id=name,
                details={"source_name": source_name},
            ),
        ),
    )
    return graph.require_valid()


def graph_to_linear_circuit(
    graph: CircuitGraph,
    parameter_values: Mapping[str, Any] | None = None,
) -> LinearCircuit:
    """Materialize a linear graph using the request's parameter bindings.

    A compiled graph is structural; variable-backed parameter values belong to
    the simulation request and must be resolved at execution time.  Fixed
    graphs continue to work with the original one-argument call, while a
    supplied key that is not actually bound is rejected instead of being
    silently ignored.
    """

    graph.require_valid()
    values = dict(parameter_values or {})
    bound_variables = {
        parameter.variable_id
        for component in graph.components
        for parameter in component.parameters
        if parameter.variable_id is not None
    }
    unknown = set(values) - bound_variables
    if unknown:
        raise ValueError(
            f"unknown or unused linear graph parameters: {sorted(unknown)}"
        )
    node_names = {item.net_id: item.name for item in graph.nets}
    elements: list[LinearElement] = []
    voltage_sources: list[VoltageSource] = []
    current_sources: list[CurrentSource] = []
    controlled_sources: list[VoltageControlledVoltageSource] = []
    for component in graph.components:
        kind = component.model.kind
        nodes = _connected_nodes(component, node_names)
        if kind in {"R", "C", "L"}:
            elements.append(
                LinearElement(
                    component.reference,
                    kind,
                    nodes["p"],
                    nodes["n"],
                    float(_parameter_value(component, "value", values)),
                )
            )
        elif kind == "voltage_source":
            voltage_sources.append(
                VoltageSource(
                    component.reference,
                    nodes["p"],
                    nodes["n"],
                    complex(_parameter_value(component, "value", values)),
                )
            )
        elif kind == "current_source":
            current_sources.append(
                CurrentSource(
                    component.reference,
                    nodes["p"],
                    nodes["n"],
                    complex(_parameter_value(component, "value", values)),
                )
            )
        elif kind == "vcvs":
            controlled_sources.append(
                VoltageControlledVoltageSource(
                    component.reference,
                    nodes["p"],
                    nodes["n"],
                    nodes["control_p"],
                    nodes["control_n"],
                    complex(_parameter_value(component, "gain", values)),
                )
            )
        else:
            raise ValueError(
                f"CircuitGraph component {component.reference!r} uses unsupported linear model {kind!r}"
            )
    return LinearCircuit(
        elements=tuple(elements),
        voltage_sources=tuple(voltage_sources),
        current_sources=tuple(current_sources),
        controlled_voltage_sources=tuple(controlled_sources),
    )


def graph_parameter_defaults(graph: CircuitGraph) -> dict[str, Any]:
    """Return deterministic compile-time values for variable-backed graphs.

    These values are used only to build a stamp plan.  Simulation execution
    still requires the request to provide every bound variable explicitly.
    """

    graph.require_valid()
    used = {
        parameter.variable_id
        for component in graph.components
        for parameter in component.parameters
        if parameter.variable_id is not None
    }
    variables = {item.variable_id: item for item in graph.variables}
    defaults: dict[str, Any] = {}
    for variable_id in sorted(used):
        variable = variables[variable_id]
        value = variable.initial
        if value is None:
            value = variable.lower if variable.lower is not None else variable.upper
        if value is None:
            value = 1.0
        if isinstance(value, (int, float)) and float(value) <= 0.0:
            value = 1.0
        defaults[variable_id] = value
    return defaults


def topology_candidate_to_graph(candidate: TopologyCandidate, ir: UnifiedIR) -> CircuitGraph:
    domains, node_domains = _candidate_domains(candidate, ir)
    all_nodes = set(node_domains)
    for component in candidate.components:
        all_nodes.update(component.nodes)
    reference_nets = {
        domain.reference_net_id for domain in domains if domain.reference_net_id is not None
    }
    nets = tuple(
        Net(
            net_id=node,
            name=node,
            domain_id=node_domains[node],
            kind="reference" if node in reference_nets else "signal",
            is_reference=node in reference_nets,
        )
        for node in sorted(all_nodes)
    )
    variables: list[ParameterVariable] = []
    components = tuple(
        _ir_component_to_instance(component, ir, variables)
        for component in candidate.components
    )
    input_port = _port_by_role(ir.ports, "input", ir.ports[0])
    output_port = _port_by_role(ir.ports, "output", ir.ports[-1])
    source_port_name = str(ir.primary_analysis.get("source_port") or input_port.name)
    output_port_name = str(
        ir.primary_analysis.get("output_port")
        or ir.primary_analysis.get("response_port")
        or output_port.name
    )
    ports = tuple(
        _ir_port_to_graph(
            port,
            node_domains,
            direction_override=(
                "input"
                if port.name == source_port_name
                else "output"
                if port.name == output_port_name
                else None
            ),
        )
        for port in ir.ports
    )
    metadata = dict(candidate.metadata)
    solver_id = str(getattr(candidate, "solver_id", None) or metadata.get("solver") or "")
    provenance = [
        ProvenanceRecord(
            source=str(metadata.get("origin", "TopologyCandidate")),
            source_id=str(metadata.get("knowledge_id", candidate.name)),
            version=str(metadata.get("knowledge_version", "")),
            details={
                "derivation": metadata.get("derivation", ()),
                "knowledge_sources": metadata.get("knowledge_sources", ()),
            },
        )
    ]
    graph = CircuitGraph(
        graph_id=candidate.name,
        name=candidate.name,
        description=candidate.rationale,
        family=candidate.family,
        preferred_solver_ids=(solver_id,) if solver_id else (),
        domains=domains,
        ports=ports,
        nets=nets,
        components=components,
        variables=tuple(variables),
        provenance=tuple(provenance),
        extensions={
            "legacy_candidate_metadata": metadata,
            "source_ir_name": ir.name,
        },
    )
    return graph.require_valid()


def graph_to_topology_candidate(graph: CircuitGraph) -> TopologyCandidate:
    graph.require_valid()
    node_names = {item.net_id: item.name for item in graph.nets}
    components = tuple(
        IRComponent(
            name=component.reference,
            kind=component.model.kind,
            nodes=tuple(
                node_names[connection.net_id]
                for connection in _model_ordered_connections(component)
            ),
            parameters={
                parameter.name: parameter.value
                for parameter in component.parameters
                if parameter.value is not None
            },
            attributes=dict(encode_graph_value(component.attributes)),
        )
        for component in graph.components
    )
    extensions = encode_graph_value(graph.extensions)
    metadata = dict(extensions.get("legacy_candidate_metadata", {}))
    if graph.preferred_solver_ids:
        metadata["solver"] = graph.preferred_solver_ids[0]
    metadata["graph_hash"] = graph.graph_hash
    metadata["topology_hash"] = graph.topology_hash
    metadata["isolated"] = any(domain.isolated_from for domain in graph.domains)
    if graph.provenance:
        metadata.setdefault("origin", graph.provenance[0].source)
    for port in graph.ports:
        if not port.terminals:
            continue
        prefix = "input" if port.direction == "input" else "output" if port.direction == "output" else None
        if prefix is None:
            continue
        metadata[f"{prefix}_port"] = port.name
        metadata[f"{prefix}_node"] = node_names[port.terminals[0].net_id]
        if len(port.terminals) > 1:
            metadata[f"{prefix}_reference_node"] = node_names[port.terminals[1].net_id]
    return TopologyCandidate(
        name=graph.name,
        family=graph.family or "unspecified",
        rationale=graph.description,
        components=components,
        metadata=metadata,
        solver_id=graph.preferred_solver_ids[0] if graph.preferred_solver_ids else None,
        graph=graph,
    )


def _candidate_domains(
    candidate: TopologyCandidate,
    ir: UnifiedIR,
) -> tuple[tuple[GraphDomain, ...], dict[str, str]]:
    isolated = bool(candidate.metadata.get("isolated")) or any(
        str(item.get("kind", "")).casefold() in {"galvanic_isolation", "isolated"}
        for item in ir.relations
    )
    nodes = {terminal for port in ir.ports for terminal in port.terminals}
    for component in candidate.components:
        nodes.update(component.nodes)
    input_port = _port_by_role(ir.ports, "input", ir.ports[0])
    output_port = _port_by_role(ir.ports, "output", ir.ports[-1])
    if not isolated:
        domain_id = "electrical"
        reference = input_port.negative
        return (
            (GraphDomain(domain_id, reference_net_id=reference),),
            {node: domain_id for node in nodes},
        )

    input_domain = "input_domain"
    output_domain = "output_domain"
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for component in candidate.components:
        groups = (
            (component.nodes[:2], component.nodes[2:])
            if component.kind == "ideal_transformer" and len(component.nodes) == 4
            else (component.nodes,)
        )
        for group in groups:
            for first in group:
                adjacency.setdefault(first, set()).update(node for node in group if node != first)
    labels: dict[str, str] = {}
    _label_connected(adjacency, input_port.terminals, input_domain, labels)
    _label_connected(adjacency, output_port.terminals, output_domain, labels)
    missing = nodes - set(labels)
    if missing:
        raise ValueError(f"isolated candidate has unclassified nets: {sorted(missing)}")
    return (
        (
            GraphDomain(
                input_domain,
                reference_net_id=input_port.negative,
                isolated_from=(output_domain,),
            ),
            GraphDomain(
                output_domain,
                reference_net_id=output_port.negative,
                isolated_from=(input_domain,),
            ),
        ),
        labels,
    )


def _label_connected(
    adjacency: Mapping[str, set[str]],
    seeds: Iterable[str],
    label: str,
    labels: dict[str, str],
) -> None:
    pending = list(seeds)
    while pending:
        node = pending.pop()
        existing = labels.get(node)
        if existing is not None:
            if existing != label:
                raise ValueError(f"net {node!r} bridges isolated domains {existing!r} and {label!r}")
            continue
        labels[node] = label
        pending.extend(adjacency.get(node, ()))


def _ir_port_to_graph(
    port: IRPort,
    node_domains: Mapping[str, str],
    *,
    direction_override: str | None = None,
) -> GraphPort:
    terminal_names = (
        ("positive", "reference")
        if len(port.terminals) == 2
        else tuple(f"terminal_{index + 1}" for index in range(len(port.terminals)))
    )
    direction = direction_override or (
        port.role if port.role in {"input", "output", "bidirectional"} else "bidirectional"
    )
    domain_ids = {node_domains[node] for node in port.terminals}
    if len(domain_ids) != 1:
        raise ValueError(f"port {port.name!r} spans multiple graph domains")
    return GraphPort(
        port_id=port.name,
        name=port.name,
        direction=direction,
        domain_id=next(iter(domain_ids)),
        terminals=tuple(
            PortTerminal(
                name=terminal_name,
                net_id=node,
                quantity="voltage" if index == 0 else "ground" if index == 1 else "unspecified",
                role="potential" if index == 0 else "reference" if index == 1 else "unspecified",
            )
            for index, (terminal_name, node) in enumerate(zip(terminal_names, port.terminals))
        ),
        attributes={
            "pbdl_domain": port.domain,
            "variables": port.variables,
            "constraints": port.constraints,
            "excitation": port.excitation,
        },
    )


def _ir_component_to_instance(
    component: IRComponent,
    ir: UnifiedIR,
    variables: list[ParameterVariable],
) -> ComponentInstance:
    model = primitive_model_ref(component.kind, len(component.nodes))
    parameters: list[ParameterBinding] = []
    for name, value in component.parameters.items():
        parameters.append(
            ParameterBinding(name, parameter_unit(component.kind, name), value=value)
        )
    existing = {item.name for item in parameters}
    for parameter_name in expected_parameters(component.kind):
        if parameter_name in existing:
            continue
        unit = parameter_unit(component.kind, parameter_name)
        if component.attributes.get("role") == "external_load" and parameter_name == "value":
            parameters.append(
                ParameterBinding(
                    parameter_name,
                    unit,
                    expression="target.output_voltage_v / target.output_current_a",
                )
            )
            continue
        variable_id = f"{component.name}.{parameter_name}"
        lower, upper = _parameter_range(ir, component.kind, parameter_name)
        variables.append(
            ParameterVariable(
                variable_id=variable_id,
                value_type="continuous",
                unit=unit,
                lower=lower,
                upper=upper,
                scale="log" if component.kind in {"R", "C", "L"} else "linear",
            )
        )
        parameters.append(
            ParameterBinding(parameter_name, unit, variable_id=variable_id)
        )
    return ComponentInstance(
        instance_id=component.name,
        reference=component.name,
        model=model,
        connections=tuple(
            TerminalConnection(terminal.name, node)
            for terminal, node in zip(model.terminals, component.nodes)
        ),
        parameters=tuple(parameters),
        attributes=component.attributes,
    )


def _parameter_range(
    ir: UnifiedIR,
    component_kind: str,
    parameter_name: str,
) -> tuple[float | None, float | None]:
    ranges = ir.constraints.get("parameter_ranges", {})
    keys = [component_kind, component_kind.casefold(), parameter_name]
    if parameter_name == "frequency_hz":
        keys.append("switching_frequency_hz")
    for key in keys:
        value = ranges.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
    return None, None


def _independent_source_instance(
    source,
    kind: str,
    reference_node: str,
) -> ComponentInstance:
    model = primitive_model_ref(kind)
    return ComponentInstance(
        instance_id=source.name,
        reference=source.name,
        model=model,
        connections=(
            TerminalConnection("p", _normalize_reference_node(source.n_plus, reference_node)),
            TerminalConnection("n", _normalize_reference_node(source.n_minus, reference_node)),
        ),
        parameters=(ParameterBinding("value", parameter_unit(kind, "value"), source.value),),
    )


def _normalize_reference_node(node: str, reference_node: str) -> str:
    return reference_node if node in GROUND_NAMES else node


def _two_terminal_port(
    name: str,
    direction: str,
    domain_id: str,
    positive: str,
    reference: str,
) -> GraphPort:
    return GraphPort(
        port_id=name,
        name=name,
        direction=direction,
        domain_id=domain_id,
        terminals=(
            PortTerminal("positive", positive, "voltage", "potential"),
            PortTerminal("reference", reference, "ground", "reference"),
        ),
    )


def _connected_nodes(component: ComponentInstance, node_names: Mapping[str, str]) -> dict[str, str]:
    return {
        connection.terminal: node_names[connection.net_id]
        for connection in component.connections
    }


def _fixed_parameter(component: ComponentInstance, name: str) -> Any:
    for parameter in component.parameters:
        if parameter.name == name:
            if parameter.value is None:
                raise ValueError(
                    f"component {component.reference!r} parameter {name!r} is not fixed"
                )
            return parameter.value
    raise ValueError(f"component {component.reference!r} has no parameter {name!r}")


def _parameter_value(
    component: ComponentInstance,
    name: str,
    parameter_values: Mapping[str, Any],
) -> Any:
    for parameter in component.parameters:
        if parameter.name != name:
            continue
        if parameter.value is not None:
            return parameter.value
        if parameter.variable_id is not None:
            if parameter.variable_id not in parameter_values:
                raise ValueError(
                    f"missing parameter value for variable {parameter.variable_id!r}"
                )
            return parameter_values[parameter.variable_id]
        if parameter.expression is not None:
            raise ValueError(
                f"parameter expression {parameter.expression!r} on "
                f"component {component.reference!r} is unsupported by linear_mna"
            )
    raise ValueError(f"component {component.reference!r} has no parameter {name!r}")


def _model_ordered_connections(component: ComponentInstance) -> tuple[TerminalConnection, ...]:
    by_terminal = {item.terminal: item for item in component.connections}
    return tuple(by_terminal[item.name] for item in component.model.terminals)


def _port_by_role(ports: tuple[IRPort, ...], role: str, fallback: IRPort) -> IRPort:
    for port in ports:
        if port.role == role:
            return port
    return fallback
