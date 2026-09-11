"""Deterministic ordering and label-invariant circuit graph fingerprints."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import permutations
import json
from typing import Any

from .model import CircuitGraph, ComponentInstance, ModelRef, encode_graph_value


def canonicalize_graph(graph: CircuitGraph) -> CircuitGraph:
    domains = tuple(
        replace(item, isolated_from=tuple(sorted(item.isolated_from)))
        for item in sorted(graph.domains, key=lambda value: value.domain_id)
    )
    ports = tuple(
        replace(item, terminals=tuple(sorted(item.terminals, key=lambda value: value.name)))
        for item in sorted(graph.ports, key=lambda value: value.port_id)
    )
    components = tuple(
        _canonical_component(item)
        for item in sorted(graph.components, key=lambda value: value.instance_id)
    )
    return replace(
        graph,
        domains=domains,
        ports=ports,
        nets=tuple(sorted(graph.nets, key=lambda value: value.net_id)),
        components=components,
        variables=tuple(sorted(graph.variables, key=lambda value: value.variable_id)),
        provenance=tuple(
            sorted(graph.provenance, key=lambda value: (value.source, value.source_id, value.version))
        ),
        preferred_solver_ids=tuple(sorted(graph.preferred_solver_ids)),
    )


def document_hash(graph: CircuitGraph) -> str:
    payload = canonicalize_graph(graph).as_dict()
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def structural_hash(graph: CircuitGraph, *, include_parameters: bool) -> str:
    """Return a label-invariant WL fingerprint, not an isomorphism proof."""

    labels: dict[str, str] = {}
    adjacency: dict[str, set[str]] = {}

    def vertex(key: str, payload: Any) -> None:
        labels[key] = _canonical_json(payload)
        adjacency.setdefault(key, set())

    def edge(first: str, second: str) -> None:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    for domain in graph.domains:
        vertex(
            f"domain:{domain.domain_id}",
            {"type": "domain", "kind": domain.kind, "attributes": encode_graph_value(domain.attributes)},
        )
    isolation_pairs: set[tuple[str, str]] = set()
    for domain in graph.domains:
        for other in domain.isolated_from:
            pair = tuple(sorted((domain.domain_id, other)))
            if pair in isolation_pairs or other not in {item.domain_id for item in graph.domains}:
                continue
            isolation_pairs.add(pair)
            relation_key = f"isolation:{pair[0]}:{pair[1]}"
            vertex(relation_key, {"type": "relation", "kind": "galvanic_isolation"})
            edge(relation_key, f"domain:{pair[0]}")
            edge(relation_key, f"domain:{pair[1]}")

    for net in graph.nets:
        key = f"net:{net.net_id}"
        vertex(
            key,
            {
                "type": "net",
                "kind": net.kind,
                "is_reference": net.is_reference,
                "attributes": encode_graph_value(net.attributes),
            },
        )
        if net.domain_id in {item.domain_id for item in graph.domains}:
            edge(key, f"domain:{net.domain_id}")

    for port in graph.ports:
        port_key = f"port:{port.port_id}"
        vertex(
            port_key,
            {
                "type": "port",
                "name": port.name,
                "direction": port.direction,
                "attributes": encode_graph_value(port.attributes),
            },
        )
        if port.domain_id in {item.domain_id for item in graph.domains}:
            edge(port_key, f"domain:{port.domain_id}")
        for index, terminal in enumerate(port.terminals):
            terminal_key = f"port-terminal:{port.port_id}:{index}"
            vertex(
                terminal_key,
                {
                    "type": "port_terminal",
                    "name": terminal.name,
                    "quantity": terminal.quantity,
                    "role": terminal.role,
                },
            )
            edge(port_key, terminal_key)
            if terminal.net_id in {item.net_id for item in graph.nets}:
                edge(terminal_key, f"net:{terminal.net_id}")

    variables = {item.variable_id: item for item in graph.variables}
    for component in graph.components:
        component_key = f"component:{component.instance_id}"
        component_payload: dict[str, Any] = {
            "type": "component",
            "model": _model_payload(component.model),
            "attributes": encode_graph_value(component.attributes),
        }
        if include_parameters:
            component_payload["ratings"] = [
                item.as_dict()
                for item in sorted(component.ratings, key=lambda value: (value.quantity, value.unit))
            ]
        vertex(component_key, component_payload)
        for index, connection in enumerate(component.connections):
            connection_key = f"connection:{component.instance_id}:{index}"
            vertex(connection_key, {"type": "terminal", "name": connection.terminal})
            edge(component_key, connection_key)
            if connection.net_id in {item.net_id for item in graph.nets}:
                edge(connection_key, f"net:{connection.net_id}")
        if include_parameters:
            for index, parameter in enumerate(component.parameters):
                parameter_key = f"parameter:{component.instance_id}:{index}"
                payload = {
                    "type": "parameter",
                    "name": parameter.name,
                    "unit": parameter.unit,
                    "value": encode_graph_value(parameter.value),
                    "expression": parameter.expression,
                }
                if parameter.variable_id and parameter.variable_id in variables:
                    payload["variable"] = variables[parameter.variable_id].as_dict()
                vertex(parameter_key, payload)
                edge(component_key, parameter_key)

    colors = {key: _digest(label) for key, label in labels.items()}
    iterations = max(3, min(len(colors), 16))
    for _ in range(iterations):
        colors = {
            key: _digest(
                colors[key] + "|" + "|".join(sorted(colors[neighbor] for neighbor in adjacency[key]))
            )
            for key in colors
        }
    edge_colors = sorted(
        tuple(sorted((colors[first], colors[second])))
        for first, neighbors in adjacency.items()
        for second in neighbors
        if first < second
    )
    payload = {
        "schema_version": graph.schema_version,
        "vertices": sorted(colors.values()),
        "edges": edge_colors,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def exact_isomorphism_key(graph: CircuitGraph, *, max_internal_nets: int = 8) -> str:
    """Return an exact small-graph isomorphism key.

    ``structural_hash`` is a Weisfeiler-Lehman fingerprint and is useful for
    inexpensive bucketing, but it is deliberately not an isomorphism proof.
    This function canonicalizes the boundary nets and exhaustively tries all
    labels for the remaining nets.  Circuit search states are intentionally
    kept small, so the bounded exhaustive step is a good deterministic second
    stage.  Larger graphs use a stable conservative fallback and must still
    be treated as a bucket collision by callers that require a proof.
    """

    graph.require_valid()
    ports = tuple(sorted(graph.ports, key=lambda item: (item.name, item.port_id)))
    port_net_descriptors: dict[str, list[tuple[Any, ...]]] = {}
    for port in ports:
        for terminal in port.terminals:
            port_net_descriptors.setdefault(terminal.net_id, []).append(
                (
                    port.name,
                    terminal.name,
                    terminal.quantity,
                    terminal.role,
                    port.direction,
                )
            )

    domain_map = {item.domain_id: item for item in graph.domains}
    external_nets = set(port_net_descriptors)
    for domain in graph.domains:
        if domain.reference_net_id is not None:
            external_nets.add(domain.reference_net_id)

    net_descriptors: dict[str, tuple[Any, ...]] = {}
    for net in graph.nets:
        domain = domain_map.get(net.domain_id)
        net_descriptors[net.net_id] = (
            tuple(sorted(port_net_descriptors.get(net.net_id, ()))),
            bool(net.is_reference),
            net.kind,
            domain.kind if domain is not None else "",
            encode_graph_value(net.attributes),
        )

    external_labels = {
        net_id: "e:" + _canonical_json(net_descriptors[net_id])
        for net_id in sorted(external_nets)
        if net_id in net_descriptors
    }
    internal_ids = tuple(
        sorted(net_id for net_id in net_descriptors if net_id not in external_labels)
    )

    if len(internal_ids) > max_internal_nets:
        # This remains deterministic, but is explicitly a conservative
        # fallback rather than a claim that the large graph was proven.
        payload = _exact_payload(graph, external_labels, {
            net_id: f"i:{index}" for index, net_id in enumerate(internal_ids)
        })
        return "fallback:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    best: str | None = None
    for labels in permutations(tuple(f"i:{index}" for index in range(len(internal_ids)))):
        internal_labels = dict(zip(internal_ids, labels))
        payload = _exact_payload(graph, external_labels, internal_labels)
        candidate = _canonical_json(payload)
        if best is None or candidate < best:
            best = candidate
    encoded = best or _canonical_json(_exact_payload(graph, external_labels, {}))
    return "exact:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _exact_payload(
    graph: CircuitGraph,
    external_labels: dict[str, str],
    internal_labels: dict[str, str],
) -> dict[str, Any]:
    labels = dict(external_labels)
    labels.update(internal_labels)
    domains = {item.domain_id: item for item in graph.domains}

    def net_label(net_id: str) -> str:
        return labels.get(net_id, "missing:" + net_id)

    domain_payload = []
    for domain in graph.domains:
        domain_payload.append(
            {
                "kind": domain.kind,
                "attributes": encode_graph_value(domain.attributes),
                "reference": net_label(domain.reference_net_id)
                if domain.reference_net_id is not None
                else None,
                "isolated_from": sorted(
                    domains[item].kind
                    for item in domain.isolated_from
                    if item in domains
                ),
            }
        )

    components = []
    for component in graph.components:
        components.append(
            {
                "model": _model_payload(component.model),
                "connections": sorted(
                    (connection.terminal, net_label(connection.net_id))
                    for connection in component.connections
                ),
                "parameters": sorted(
                    (parameter.name, parameter.unit)
                    for parameter in component.parameters
                ),
                "ratings": sorted(
                    (rating.quantity, rating.unit, rating.minimum, rating.maximum)
                    for rating in component.ratings
                ),
                "attributes": encode_graph_value(component.attributes),
            }
        )

    ports = []
    for port in graph.ports:
        ports.append(
            {
                "name": port.name,
                "direction": port.direction,
                "domain": domains.get(port.domain_id).kind
                if port.domain_id in domains
                else port.domain_id,
                "terminals": sorted(
                    (
                        terminal.name,
                        net_label(terminal.net_id),
                        terminal.quantity,
                        terminal.role,
                    )
                    for terminal in port.terminals
                ),
                "attributes": encode_graph_value(port.attributes),
            }
        )

    extensions = encode_graph_value(graph.extensions)
    if isinstance(extensions, dict):
        extensions = {
            key: value
            for key, value in extensions.items()
            if key not in {"legacy_candidate_metadata", "search_trace"}
        }
    return {
        "schema_version": graph.schema_version,
        "family": graph.family,
        "domains": sorted(domain_payload, key=_canonical_json),
        "nets": sorted(
            (
                net_label(net.net_id),
                net.kind,
                net.is_reference,
                net.domain_id in domains and domains[net.domain_id].kind,
                encode_graph_value(net.attributes),
            )
            for net in graph.nets
        ),
        "ports": sorted(ports, key=_canonical_json),
        "components": sorted(components, key=_canonical_json),
        "extensions": extensions,
    }


def _canonical_component(component: ComponentInstance) -> ComponentInstance:
    model = replace(
        component.model,
        terminals=tuple(sorted(component.model.terminals, key=lambda value: value.name)),
    )
    return replace(
        component,
        model=model,
        connections=tuple(sorted(component.connections, key=lambda value: value.terminal)),
        parameters=tuple(sorted(component.parameters, key=lambda value: value.name)),
        ratings=tuple(sorted(component.ratings, key=lambda value: (value.quantity, value.unit))),
    )


def _model_payload(model: ModelRef) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "kind": model.kind,
        "version": model.version,
        "source": model.source,
        "terminals": [
            item.as_dict() for item in sorted(model.terminals, key=lambda value: value.name)
        ],
    }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
