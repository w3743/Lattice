"""Data-driven functional blocks and topology productions.

The schema follows the functional-block/slot idea used by open analog
generators, while remaining native to Circuit AI's UnifiedIR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .experts import TopologyCandidate
from .graph import (
    CircuitGraph,
    ComponentInstance,
    GraphDomain,
    GraphPort,
    Net,
    PortTerminal,
    ProvenanceRecord,
    TerminalConnection,
    primitive_model_ref,
)
from .ir import IRComponent


DEFAULT_POWER_KNOWLEDGE = Path(__file__).parent / "knowledge" / "power_topologies.yaml"


@dataclass(frozen=True)
class ModuleComponent:
    ref: str
    kind: str
    nodes: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FunctionalModule:
    name: str
    category: str
    ports: tuple[str, ...]
    graph: CircuitGraph

    @property
    def components(self) -> tuple[ModuleComponent, ...]:
        net_names = {item.net_id: item.name for item in self.graph.nets}
        return tuple(
            ModuleComponent(
                ref=component.reference,
                kind=component.model.kind,
                nodes=tuple(
                    net_names[connection.net_id]
                    for connection in _ordered_connections(component)
                ),
                attributes=dict(component.attributes),
            )
            for component in self.graph.components
        )


@dataclass(frozen=True)
class ProductionSlot:
    name: str
    module: str
    bindings: dict[str, str]
    references: dict[str, str]


@dataclass(frozen=True)
class TopologyProduction:
    name: str
    family: str
    solver: str
    rationale: str
    applicability: dict[str, str]
    slots: tuple[ProductionSlot, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_applicable(self, *, isolated: bool, vin: float, vout: float) -> bool:
        isolation = self.applicability.get("isolation", "any")
        if isolation == "required" and not isolated:
            return False
        if isolation == "forbidden" and isolated:
            return False
        relation = self.applicability.get("voltage_relation", "any")
        if relation == "step_up" and not vout > vin:
            return False
        if relation == "step_down" and not vout < vin:
            return False
        if relation == "equal" and not vout == vin:
            return False
        return True


@dataclass(frozen=True)
class TopologyKnowledgeBase:
    schema_version: int
    knowledge_id: str
    sources: tuple[dict[str, str], ...]
    modules: dict[str, FunctionalModule]
    productions: tuple[TopologyProduction, ...]

    def instantiate(self, production: TopologyProduction, context: dict[str, Any]) -> TopologyCandidate:
        node_tokens = {
            "$input": context["input_port"].positive,
            "$output": context["output_port"].positive,
            "$input_ref": context["input_port"].negative,
            "$output_ref": context["output_port"].negative,
        }
        components: list[IRComponent] = []
        derivation: list[str] = []
        for slot in production.slots:
            module = self.modules[slot.module]
            _validate_slot_bindings(slot, module)
            for component in module.components:
                reference = slot.references.get(component.ref)
                if not reference:
                    reference = f"{slot.name}_{component.ref}"
                nodes = tuple(
                    _resolve_module_node(node, slot, module, node_tokens)
                    for node in component.nodes
                )
                components.append(
                    IRComponent(
                        reference,
                        component.kind,
                        nodes,
                        attributes=dict(component.attributes),
                    )
                )
            derivation.append(
                f"slot:{slot.name}:module={module.name}:bindings={sorted(slot.bindings.items())}"
            )

        input_port = context["input_port"]
        output_port = context["output_port"]
        metadata = {
            "origin": "functional_block_knowledge_base",
            "knowledge_id": self.knowledge_id,
            "solver": production.solver,
            "input_port": input_port.name,
            "output_port": output_port.name,
            "input_node": input_port.positive,
            "output_node": output_port.positive,
            "input_reference_node": input_port.negative,
            "output_reference_node": output_port.negative,
            "reference_node": input_port.negative,
            "derivation": derivation,
            "knowledge_sources": [dict(item) for item in self.sources],
            **production.metadata,
        }
        candidate = TopologyCandidate(
            name=production.name,
            family=production.family,
            rationale=production.rationale,
            components=tuple(components),
            metadata=metadata,
            solver_id=production.solver,
        )
        from .graph import topology_candidate_to_graph

        return replace(candidate, graph=topology_candidate_to_graph(candidate, context["ir"]))


def load_topology_knowledge(path: str | Path | None = None) -> TopologyKnowledgeBase:
    source_path = Path(path) if path is not None else DEFAULT_POWER_KNOWLEDGE
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"topology knowledge must be a mapping: {source_path}")
    schema_version = int(data.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"unsupported topology knowledge schema_version={schema_version}")

    modules: dict[str, FunctionalModule] = {}
    for entry in _list_of_mappings(data, "modules"):
        name = _required_text(entry, "name")
        if name in modules:
            raise ValueError(f"duplicate functional module {name!r}")
        ports = tuple(str(item) for item in entry.get("ports", []))
        if not ports or len(set(ports)) != len(ports):
            raise ValueError(f"module {name!r} must declare unique ports")
        components = []
        for component in _list_of_mappings(entry, "components"):
            nodes = tuple(str(item) for item in component.get("nodes", []))
            if len(nodes) < 2:
                raise ValueError(f"component in module {name!r} needs at least two nodes")
            components.append(
                ModuleComponent(
                    ref=_required_text(component, "ref"),
                    kind=_required_text(component, "kind"),
                    nodes=nodes,
                    attributes=dict(component.get("attributes", {})),
                )
            )
        if not components:
            raise ValueError(f"module {name!r} has no components")
        modules[name] = FunctionalModule(
            name=name,
            category=str(entry.get("category", name)),
            ports=ports,
            graph=_module_fragment_graph(
                name=name,
                category=str(entry.get("category", name)),
                ports=ports,
                components=tuple(components),
                source_path=source_path,
            ),
        )

    productions: list[TopologyProduction] = []
    seen_productions: set[str] = set()
    for entry in _list_of_mappings(data, "productions"):
        name = _required_text(entry, "name")
        if name in seen_productions:
            raise ValueError(f"duplicate topology production {name!r}")
        seen_productions.add(name)
        slots = []
        for slot in _list_of_mappings(entry, "slots"):
            module_name = _required_text(slot, "module")
            if module_name not in modules:
                raise ValueError(f"production {name!r} references unknown module {module_name!r}")
            production_slot = ProductionSlot(
                name=_required_text(slot, "name"),
                module=module_name,
                bindings={str(k): str(v) for k, v in dict(slot.get("bindings", {})).items()},
                references={str(k): str(v) for k, v in dict(slot.get("references", {})).items()},
            )
            _validate_slot_bindings(production_slot, modules[module_name])
            component_refs = {component.ref for component in modules[module_name].components}
            unknown_refs = set(production_slot.references) - component_refs
            if unknown_refs:
                raise ValueError(
                    f"slot {production_slot.name!r} has references for unknown components: "
                    f"{sorted(unknown_refs)}"
                )
            slots.append(production_slot)
        if not slots:
            raise ValueError(f"production {name!r} has no slots")
        productions.append(
            TopologyProduction(
                name=name,
                family=_required_text(entry, "family"),
                solver=_required_text(entry, "solver"),
                rationale=_required_text(entry, "rationale"),
                applicability={str(k): str(v) for k, v in dict(entry.get("applicability", {})).items()},
                slots=tuple(slots),
                metadata=dict(entry.get("metadata", {})),
            )
        )

    if not productions:
        raise ValueError("topology knowledge has no productions")
    return TopologyKnowledgeBase(
        schema_version=schema_version,
        knowledge_id=str(data.get("knowledge_id", source_path.stem)),
        sources=tuple(dict(item) for item in data.get("sources", [])),
        modules=modules,
        productions=tuple(productions),
    )


def _resolve_module_node(
    node: str,
    slot: ProductionSlot,
    module: FunctionalModule,
    node_tokens: dict[str, str],
) -> str:
    if node in module.ports:
        bound = slot.bindings[node]
        return node_tokens.get(bound, bound)
    return f"{slot.name}__{node}"


def _validate_slot_bindings(slot: ProductionSlot, module: FunctionalModule) -> None:
    missing = set(module.ports) - set(slot.bindings)
    extra = set(slot.bindings) - set(module.ports)
    if missing or extra:
        raise ValueError(
            f"slot {slot.name!r} bindings do not match module {module.name!r}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _list_of_mappings(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return value


def _required_text(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required text field {key!r}")
    return value


def _module_fragment_graph(
    *,
    name: str,
    category: str,
    ports: tuple[str, ...],
    components: tuple[ModuleComponent, ...],
    source_path: Path,
) -> CircuitGraph:
    domain_id = "module_domain"
    node_names = set(ports)
    for component in components:
        node_names.update(component.nodes)
    graph_components = []
    for component in components:
        model = primitive_model_ref(component.kind, len(component.nodes))
        graph_components.append(
            ComponentInstance(
                instance_id=component.ref,
                reference=component.ref,
                model=model,
                connections=tuple(
                    TerminalConnection(terminal.name, node)
                    for terminal, node in zip(model.terminals, component.nodes)
                ),
                attributes=component.attributes,
            )
        )
    graph = CircuitGraph(
        graph_id=f"module:{name}",
        name=name,
        description=f"Functional module fragment: {category}",
        family=category,
        domains=(GraphDomain(domain_id),),
        ports=tuple(
            GraphPort(
                port_id=port,
                name=port,
                direction="bidirectional",
                domain_id=domain_id,
                terminals=(PortTerminal("terminal", port, "unspecified", "unspecified"),),
            )
            for port in ports
        ),
        nets=tuple(
            Net(node, node, domain_id, kind="module_port" if node in ports else "internal")
            for node in sorted(node_names)
        ),
        components=tuple(graph_components),
        provenance=(
            ProvenanceRecord(
                source="topology_knowledge_yaml",
                source_id=str(source_path),
                details={"module": name},
            ),
        ),
    )
    return graph.require_valid()


def _ordered_connections(component: ComponentInstance) -> tuple[TerminalConnection, ...]:
    by_name = {item.terminal: item for item in component.connections}
    return tuple(by_name[terminal.name] for terminal in component.model.terminals)
