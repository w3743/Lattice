"""Typed, immutable graph-construction actions for open topology search.

The existing :mod:`circuit_ai.graph` schema remains the source of truth.  An
action returns a new graph plus the search state's open terminals; it never
mutates a graph that may already be present in a cache or a replay record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .model import (
    CircuitGraph,
    ComponentInstance,
    GraphPort,
    Net,
    ParameterBinding,
    ParameterVariable,
    PortTerminal,
    TerminalConnection,
    encode_graph_value,
)
from .primitives import expected_parameters, parameter_unit, primitive_model_ref


class GraphActionError(ValueError):
    """A typed action was rejected before it entered the search frontier."""

    def __init__(self, code: str, message: str, *, action: object | None = None):
        self.code = code
        self.action = action
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class OpenTerminal:
    component_id: str
    terminal_name: str
    net_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "component_id": self.component_id,
            "terminal_name": self.terminal_name,
            "net_id": self.net_id,
        }


@dataclass(frozen=True)
class AddComponent:
    model: Any
    instance_id: str | None = None
    connections: Mapping[str, str | None] = field(default_factory=dict)
    parameters: tuple[ParameterBinding, ...] | Mapping[str, Any] = ()
    ratings: tuple[Any, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    action_type: str = field(default="AddComponent", init=False)


@dataclass(frozen=True)
class AddFragment:
    fragment: CircuitGraph
    prefix: str = ""

    action_type: str = field(default="AddFragment", init=False)


@dataclass(frozen=True)
class ConnectTerminal:
    terminal: Any
    net: str

    action_type: str = field(default="ConnectTerminal", init=False)


@dataclass(frozen=True)
class SplitNet:
    net: str
    new_net: str
    terminals: tuple[Any, ...] = ()

    action_type: str = field(default="SplitNet", init=False)


@dataclass(frozen=True)
class AddFeedback:
    source: Any
    destination: Any

    action_type: str = field(default="AddFeedback", init=False)


@dataclass(frozen=True)
class TerminatePort:
    port: str

    action_type: str = field(default="TerminatePort", init=False)


@dataclass(frozen=True)
class ReplaceModel:
    instance: str
    model: Any

    action_type: str = field(default="ReplaceModel", init=False)


@dataclass(frozen=True)
class RemoveComponent:
    instance: str

    action_type: str = field(default="RemoveComponent", init=False)


GraphAction = (
    AddComponent
    | AddFragment
    | ConnectTerminal
    | SplitNet
    | AddFeedback
    | TerminatePort
    | ReplaceModel
    | RemoveComponent
)


def apply_graph_action(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: GraphAction,
    *,
    policy: Any | None = None,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    """Apply one action and immediately validate the resulting graph."""

    graph.require_valid()
    if isinstance(action, AddComponent):
        result = _add_component(graph, open_terminals, action, policy)
    elif isinstance(action, AddFragment):
        result = _add_fragment(graph, open_terminals, action)
    elif isinstance(action, ConnectTerminal):
        result = _connect_terminal(graph, open_terminals, action)
    elif isinstance(action, SplitNet):
        result = _split_net(graph, open_terminals, action)
    elif isinstance(action, AddFeedback):
        result = _add_feedback(graph, open_terminals, action)
    elif isinstance(action, TerminatePort):
        result = _terminate_port(graph, open_terminals, action)
    elif isinstance(action, ReplaceModel):
        result = _replace_model(graph, open_terminals, action)
    elif isinstance(action, RemoveComponent):
        result = _remove_component(graph, open_terminals, action)
    else:  # pragma: no cover - protected by the type alias
        raise GraphActionError("unknown_action", type(action).__name__, action=action)

    updated_graph, updated_open = result
    _validate_typed_connections(updated_graph, action=action)
    updated_graph.require_valid()
    _check_policy(updated_graph, policy, action)
    return updated_graph, tuple(sorted(set(updated_open), key=_open_key))


def action_payload(action: GraphAction) -> dict[str, Any]:
    """Serialize an action for a search certificate or replay episode."""

    if isinstance(action, AddComponent):
        model = _model_ref(action.model)
        parameters = action.parameters
        if isinstance(parameters, Mapping):
            parameter_payload = encode_graph_value(parameters)
        else:
            parameter_payload = [item.as_dict() for item in parameters]
        return {
            "type": action.action_type,
            "model": model.as_dict(),
            "instance_id": action.instance_id,
            "connections": dict(sorted(action.connections.items())),
            "parameters": parameter_payload,
        }
    if isinstance(action, AddFragment):
        return {"type": action.action_type, "graph_id": action.fragment.graph_id, "prefix": action.prefix}
    if isinstance(action, ConnectTerminal):
        return {"type": action.action_type, "terminal": _endpoint_payload(action.terminal), "net": action.net}
    if isinstance(action, SplitNet):
        return {
            "type": action.action_type,
            "net": action.net,
            "new_net": action.new_net,
            "terminals": [_endpoint_payload(item) for item in action.terminals],
        }
    if isinstance(action, AddFeedback):
        return {
            "type": action.action_type,
            "source": _endpoint_payload(action.source),
            "destination": _endpoint_payload(action.destination),
        }
    if isinstance(action, TerminatePort):
        return {"type": action.action_type, "port": action.port}
    if isinstance(action, ReplaceModel):
        return {"type": action.action_type, "instance": action.instance, "model": _model_ref(action.model).as_dict()}
    if isinstance(action, RemoveComponent):
        return {"type": action.action_type, "instance": action.instance}
    raise GraphActionError("unknown_action", type(action).__name__, action=action)


def _add_component(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: AddComponent,
    policy: Any | None,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    model = _model_ref(action.model)
    instance_id = action.instance_id or _next_id(graph.components, model.kind)
    if any(item.instance_id == instance_id for item in graph.components):
        raise GraphActionError("duplicate_component", f"component {instance_id!r} already exists", action=action)

    provided = dict(action.connections)
    unknown_terminals = set(provided) - {item.name for item in model.terminals}
    if unknown_terminals:
        raise GraphActionError("unknown_terminal", f"unknown terminals {sorted(unknown_terminals)}", action=action)
    nets = list(graph.nets)
    new_open = list(open_terminals)
    connections: list[TerminalConnection] = []
    for terminal in model.terminals:
        net_id = provided.get(terminal.name)
        if net_id is None:
            net_id = _next_net_id(nets, f"{instance_id}_{terminal.name}")
            domain_id = _domain_for_terminal(graph, terminal.domain)
            nets.append(
                Net(
                    net_id=net_id,
                    name=net_id,
                    domain_id=domain_id,
                    kind="reference" if terminal.name in {"n", "reference"} else "signal",
                    is_reference=False,
                )
            )
            new_open.append(OpenTerminal(instance_id, terminal.name, net_id))
        elif not any(item.net_id == net_id for item in nets):
            raise GraphActionError("missing_net", f"net {net_id!r} does not exist", action=action)
        connections.append(TerminalConnection(terminal.name, str(net_id)))

    parameters, variables = _component_parameters(
        model,
        instance_id,
        action.parameters,
        policy,
    )
    component = ComponentInstance(
        instance_id=instance_id,
        reference=instance_id,
        model=model,
        connections=tuple(connections),
        parameters=parameters,
        ratings=tuple(action.ratings),
        attributes=dict(action.attributes),
    )
    updated = replace(
        graph,
        nets=tuple(nets),
        components=graph.components + (component,),
        variables=graph.variables + tuple(variables),
    )
    return updated, tuple(new_open)


def _add_fragment(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: AddFragment,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    fragment = action.fragment.require_valid()
    prefix = action.prefix.strip()
    if not prefix:
        fragment = fragment
    else:
        fragment = _prefix_graph(fragment, prefix)
    if graph.graph_id == fragment.graph_id:
        raise GraphActionError("graph_id_conflict", f"fragment graph id {fragment.graph_id!r} conflicts", action=action)
    if set(item.domain_id for item in graph.domains) & set(item.domain_id for item in fragment.domains):
        raise GraphActionError("domain_conflict", "fragment domain ids conflict", action=action)
    if set(item.net_id for item in graph.nets) & set(item.net_id for item in fragment.nets):
        raise GraphActionError("net_conflict", "fragment net ids conflict", action=action)
    if set(item.instance_id for item in graph.components) & set(item.instance_id for item in fragment.components):
        raise GraphActionError("component_conflict", "fragment component ids conflict", action=action)
    if set(item.port_id for item in graph.ports) & set(item.port_id for item in fragment.ports):
        raise GraphActionError("port_conflict", "fragment port ids conflict", action=action)
    if set(item.variable_id for item in graph.variables) & set(item.variable_id for item in fragment.variables):
        raise GraphActionError("variable_conflict", "fragment variable ids conflict", action=action)
    domains = graph.domains + fragment.domains
    merged = replace(
        graph,
        domains=domains,
        nets=graph.nets + fragment.nets,
        components=graph.components + fragment.components,
        ports=graph.ports + fragment.ports,
        variables=graph.variables + fragment.variables,
        provenance=graph.provenance + fragment.provenance,
        extensions={**dict(graph.extensions), "fragments": tuple(
            dict(graph.extensions).get("fragments", ())
        ) + (fragment.graph_id,)},
    )
    return merged, open_terminals


def _connect_terminal(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: ConnectTerminal,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    component_id, terminal_name = _component_terminal(action.terminal)
    component = _find_component(graph, component_id)
    if not any(item.net_id == action.net for item in graph.nets):
        raise GraphActionError("missing_net", f"net {action.net!r} does not exist", action=action)
    if terminal_name not in {item.name for item in component.model.terminals}:
        raise GraphActionError("unknown_terminal", f"terminal {terminal_name!r} does not exist", action=action)
    updated_component = replace(
        component,
        connections=tuple(
            replace(connection, net_id=action.net)
            if connection.terminal == terminal_name
            else connection
            for connection in component.connections
        ),
    )
    updated = replace(
        graph,
        components=tuple(
            updated_component if item.instance_id == component_id else item
            for item in graph.components
        ),
    )
    remaining = tuple(
        item
        for item in open_terminals
        if (item.component_id, item.terminal_name) != (component_id, terminal_name)
    )
    return _cleanup_orphan_nets(updated), remaining


def _split_net(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: SplitNet,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    source = next((item for item in graph.nets if item.net_id == action.net), None)
    if source is None:
        raise GraphActionError("missing_net", f"net {action.net!r} does not exist", action=action)
    if any(item.net_id == action.new_net for item in graph.nets):
        raise GraphActionError("duplicate_net", f"net {action.new_net!r} already exists", action=action)
    if not action.terminals:
        raise GraphActionError("empty_split", "SplitNet needs at least one terminal", action=action)

    component_targets: set[tuple[str, str]] = set()
    port_targets: set[tuple[str, str]] = set()
    for target in action.terminals:
        if isinstance(target, tuple) and len(target) == 3 and target[0] == "port":
            port_targets.add((str(target[1]), str(target[2])))
        else:
            component_targets.add(_component_terminal(target))

    components: list[ComponentInstance] = []
    moved = 0
    for component in graph.components:
        connections = []
        for connection in component.connections:
            if (
                (component.instance_id, connection.terminal) in component_targets
                and connection.net_id == action.net
            ):
                connections.append(replace(connection, net_id=action.new_net))
                moved += 1
            else:
                connections.append(connection)
        components.append(replace(component, connections=tuple(connections)))

    ports: list[GraphPort] = []
    for port in graph.ports:
        terminals = []
        for terminal in port.terminals:
            if (port.port_id, terminal.name) in port_targets and terminal.net_id == action.net:
                terminals.append(replace(terminal, net_id=action.new_net))
                moved += 1
            else:
                terminals.append(terminal)
        ports.append(replace(port, terminals=tuple(terminals)))
    if moved == 0:
        raise GraphActionError("terminal_not_on_net", "no selected terminal belongs to the source net", action=action)
    updated = replace(
        graph,
        nets=graph.nets + (replace(source, net_id=action.new_net, name=action.new_net),),
        components=tuple(components),
        ports=tuple(ports),
    )
    return updated, open_terminals


def _add_feedback(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: AddFeedback,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    feedback = tuple(dict(graph.extensions).get("feedback_edges", ()))
    edge = {"source": _endpoint_payload(action.source), "destination": _endpoint_payload(action.destination)}
    if edge not in feedback:
        feedback += (edge,)
    return replace(graph, extensions={**dict(graph.extensions), "feedback_edges": feedback}), open_terminals


def _terminate_port(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: TerminatePort,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    port = next((item for item in graph.ports if item.port_id == action.port or item.name == action.port), None)
    if port is None:
        raise GraphActionError("missing_port", f"port {action.port!r} does not exist", action=action)
    terminated = tuple(dict(graph.extensions).get("terminated_ports", ()))
    if port.port_id not in terminated:
        terminated += (port.port_id,)
    return replace(graph, extensions={**dict(graph.extensions), "terminated_ports": terminated}), open_terminals


def _replace_model(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: ReplaceModel,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    component = _find_component(graph, action.instance)
    model = _model_ref(action.model)
    old_names = {item.name for item in component.model.terminals}
    new_names = {item.name for item in model.terminals}
    if old_names != new_names:
        raise GraphActionError(
            "terminal_contract",
            f"replacement changes terminals from {sorted(old_names)} to {sorted(new_names)}",
            action=action,
        )
    updated = replace(
        graph,
        components=tuple(
            replace(item, model=model) if item.instance_id == action.instance else item
            for item in graph.components
        ),
    )
    return updated, open_terminals


def _remove_component(
    graph: CircuitGraph,
    open_terminals: tuple[OpenTerminal, ...],
    action: RemoveComponent,
) -> tuple[CircuitGraph, tuple[OpenTerminal, ...]]:
    if not any(item.instance_id == action.instance for item in graph.components):
        raise GraphActionError("missing_component", f"component {action.instance!r} does not exist", action=action)
    removed = next(item for item in graph.components if item.instance_id == action.instance)
    variable_ids = {item.variable_id for item in removed.parameters if item.variable_id is not None}
    remaining_components = tuple(item for item in graph.components if item.instance_id != action.instance)
    used_variables = {
        parameter.variable_id
        for component in remaining_components
        for parameter in component.parameters
        if parameter.variable_id is not None
    }
    updated = replace(
        graph,
        components=remaining_components,
        variables=tuple(item for item in graph.variables if item.variable_id not in variable_ids - used_variables),
    )
    remaining_open = tuple(item for item in open_terminals if item.component_id != action.instance)
    return _cleanup_orphan_nets(updated), remaining_open


def _component_parameters(
    model,
    instance_id: str,
    provided: tuple[ParameterBinding, ...] | Mapping[str, Any],
    policy: Any | None,
) -> tuple[tuple[ParameterBinding, ...], tuple[ParameterVariable, ...]]:
    if isinstance(provided, Mapping):
        bindings = tuple(
            ParameterBinding(str(name), parameter_unit(model.kind, str(name)), value=value)
            for name, value in sorted(provided.items())
        )
    else:
        bindings = tuple(provided)
    by_name = {item.name: item for item in bindings}
    variables: list[ParameterVariable] = []
    completed: list[ParameterBinding] = []
    for name in expected_parameters(model.kind):
        binding = by_name.get(name)
        if binding is not None:
            completed.append(binding)
            if binding.variable_id is not None:
                continue
            continue
        unit = parameter_unit(model.kind, name)
        if model.kind in {"R", "C", "L"} and _policy_value(policy, "variable_parameters", True):
            variable_id = f"{instance_id}.{name}"
            lower, upper = _parameter_range(policy, model.kind)
            variables.append(
                ParameterVariable(
                    variable_id=variable_id,
                    value_type="continuous",
                    unit=unit,
                    lower=lower,
                    upper=upper,
                    initial=(lower * upper) ** 0.5,
                    scale="log10",
                )
            )
            completed.append(ParameterBinding(name, unit, variable_id=variable_id))
        else:
            default = 1.0 if model.kind in {"R", "C", "L", "vcvs", "voltage_source"} else 0.001
            completed.append(ParameterBinding(name, unit, value=default))
    expected = set(expected_parameters(model.kind))
    completed.extend(item for item in bindings if item.name not in expected)
    return tuple(completed), tuple(variables)


def _parameter_range(policy: Any | None, kind: str) -> tuple[float, float]:
    configured = _policy_value(policy, "parameter_ranges", {})
    if isinstance(configured, Mapping) and kind in configured:
        value = configured[kind]
        return float(value[0]), float(value[1])
    defaults = {"R": (1.0, 1e7), "C": (1e-12, 1e-2), "L": (1e-9, 10.0)}
    return defaults.get(kind, (1e-9, 1e9))


def _validate_typed_connections(graph: CircuitGraph, *, action: object) -> None:
    domains = {item.domain_id: item for item in graph.domains}
    nets = {item.net_id: item for item in graph.nets}
    port_quantities: dict[str, set[str]] = {}
    for port in graph.ports:
        for terminal in port.terminals:
            port_quantities.setdefault(terminal.net_id, set()).add(terminal.quantity)
    for component in graph.components:
        connection_nets = [item.net_id for item in component.connections]
        if len(connection_nets) == 2 and connection_nets[0] == connection_nets[1]:
            raise GraphActionError(
                "invalid_self_loop",
                f"{component.reference} connects both terminals to {connection_nets[0]!r}",
                action=action,
            )
        connection_domains = {nets[item].domain_id for item in connection_nets}
        for domain_id in connection_domains:
            domain = domains[domain_id]
            if any(other in connection_domains for other in domain.isolated_from):
                raise GraphActionError(
                    "isolation_short",
                    f"{component.reference} bridges isolated graph domains",
                    action=action,
                )
        for connection in component.connections:
            terminal = next(item for item in component.model.terminals if item.name == connection.terminal)
            net = nets[connection.net_id]
            domain = domains[net.domain_id]
            if terminal.domain not in {domain.domain_id, domain.kind}:
                raise GraphActionError(
                    "terminal_domain_mismatch",
                    f"{component.reference}.{terminal.name} domain {terminal.domain!r} cannot connect to {domain.domain_id!r}",
                    action=action,
                )
            quantities = port_quantities.get(connection.net_id, set()) - {"", "unspecified"}
            compatible_quantities = {
                "voltage",
                "ground",
            }
            if quantities and not (
                terminal.quantity in quantities
                or terminal.quantity in compatible_quantities
                and quantities.issubset(compatible_quantities)
            ):
                raise GraphActionError(
                    "terminal_quantity_mismatch",
                    f"{component.reference}.{terminal.name} quantity {terminal.quantity!r} cannot connect to {sorted(quantities)}",
                    action=action,
                )


def _check_policy(graph: CircuitGraph, policy: Any | None, action: object) -> None:
    if policy is None:
        return
    max_components = _policy_value(policy, "max_components", None)
    design_components = sum(
        item.model.kind not in {"voltage_source", "current_source"}
        for item in graph.components
    )
    if max_components is not None and design_components > int(max_components):
        raise GraphActionError("component_budget", f"component budget {max_components} exceeded", action=action)
    max_nodes = _policy_value(policy, "max_nodes", None)
    if max_nodes is not None and len(graph.nets) > int(max_nodes):
        raise GraphActionError("node_budget", f"node budget {max_nodes} exceeded", action=action)


def _cleanup_orphan_nets(graph: CircuitGraph) -> CircuitGraph:
    referenced = {
        connection.net_id
        for component in graph.components
        for connection in component.connections
    }
    referenced.update(
        terminal.net_id for port in graph.ports for terminal in port.terminals
    )
    referenced.update(
        domain.reference_net_id
        for domain in graph.domains
        if domain.reference_net_id is not None
    )
    return replace(graph, nets=tuple(item for item in graph.nets if item.net_id in referenced))


def _prefix_graph(graph: CircuitGraph, prefix: str) -> CircuitGraph:
    def domain_id(value: str) -> str:
        return f"{prefix}{value}"

    def net_id(value: str) -> str:
        return f"{prefix}{value}"

    domains = tuple(
        replace(
            item,
            domain_id=domain_id(item.domain_id),
            reference_net_id=net_id(item.reference_net_id) if item.reference_net_id else None,
            isolated_from=tuple(domain_id(other) for other in item.isolated_from),
        )
        for item in graph.domains
    )
    nets = tuple(replace(item, net_id=net_id(item.net_id), domain_id=domain_id(item.domain_id)) for item in graph.nets)
    ports = tuple(
        replace(
            item,
            port_id=f"{prefix}{item.port_id}",
            terminals=tuple(replace(terminal, net_id=net_id(terminal.net_id)) for terminal in item.terminals),
            domain_id=domain_id(item.domain_id),
        )
        for item in graph.ports
    )
    variables = tuple(replace(item, variable_id=f"{prefix}{item.variable_id}") for item in graph.variables)
    components = tuple(
        replace(
            item,
            instance_id=f"{prefix}{item.instance_id}",
            reference=f"{prefix}{item.reference}",
            connections=tuple(replace(connection, net_id=net_id(connection.net_id)) for connection in item.connections),
            parameters=tuple(
                replace(parameter, variable_id=f"{prefix}{parameter.variable_id}")
                if parameter.variable_id
                else parameter
                for parameter in item.parameters
            ),
        )
        for item in graph.components
    )
    return replace(
        graph,
        graph_id=f"{prefix}{graph.graph_id}",
        name=f"{prefix}{graph.name}",
        domains=domains,
        nets=nets,
        ports=ports,
        variables=variables,
        components=components,
    )


def _model_ref(model: Any):
    if hasattr(model, "model_id") and hasattr(model, "terminals"):
        return model
    return primitive_model_ref(str(model))


def _domain_for_terminal(graph: CircuitGraph, terminal_domain: str) -> str:
    for domain in graph.domains:
        if domain.domain_id == terminal_domain or domain.kind == terminal_domain:
            return domain.domain_id
    raise GraphActionError("missing_domain", f"no graph domain matches terminal domain {terminal_domain!r}")


def _next_id(items, kind: str) -> str:
    prefix = {"R": "R", "C": "C", "L": "L"}.get(kind, kind[:1].upper() or "X")
    used = {item.instance_id for item in items}
    index = 1
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _next_net_id(nets: list[Net], base: str) -> str:
    used = {item.net_id for item in nets}
    candidate = base
    index = 1
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _find_component(graph: CircuitGraph, instance_id: str) -> ComponentInstance:
    for component in graph.components:
        if component.instance_id == instance_id or component.reference == instance_id:
            return component
    raise GraphActionError("missing_component", f"component {instance_id!r} does not exist")


def _component_terminal(value: Any) -> tuple[str, str]:
    if isinstance(value, OpenTerminal):
        return value.component_id, value.terminal_name
    if isinstance(value, tuple) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, str) and "." in value:
        component, terminal = value.split(".", 1)
        return component, terminal
    raise GraphActionError("invalid_terminal", f"expected component terminal, got {value!r}")


def _endpoint_payload(value: Any) -> Any:
    if isinstance(value, OpenTerminal):
        return value.as_dict()
    if isinstance(value, tuple):
        return list(value)
    return value


def _open_key(item: OpenTerminal) -> tuple[str, str, str]:
    return item.component_id, item.terminal_name, item.net_id


def _policy_value(policy: Any | None, name: str, default: Any) -> Any:
    if policy is None:
        return default
    return getattr(policy, name, default)
