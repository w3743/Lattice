"""Unified intermediate representation for PBDL-driven circuit design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pbdl_boundary import load_pbdl_dict


@dataclass(frozen=True)
class IRPort:
    name: str
    terminals: tuple[str, ...]
    role: str
    domain: str
    port_id: str | None = None
    variables: tuple[dict[str, Any], ...] = ()
    constraints: tuple[dict[str, Any], ...] = ()
    excitation: dict[str, Any] | None = None

    @property
    def positive(self) -> str:
        return self.terminals[0]

    @property
    def negative(self) -> str:
        return self.terminals[1] if len(self.terminals) > 1 else "0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id or self.name,
            "name": self.name,
            "terminals": list(self.terminals),
            "role": self.role,
            "domain": self.domain,
            "variables": [dict(item) for item in self.variables],
            "constraints": [dict(item) for item in self.constraints],
            "excitation": dict(self.excitation) if self.excitation is not None else None,
        }


@dataclass(frozen=True)
class IRComponent:
    name: str
    kind: str
    nodes: tuple[str, ...]
    parameters: dict[str, float] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "nodes": list(self.nodes),
            "parameters": dict(self.parameters),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class UnifiedIR:
    """Stable contract consumed by experts, solvers, optimizers and renderers."""

    name: str
    description: str
    ports: tuple[IRPort, ...]
    relations: tuple[dict[str, Any], ...]
    analyses: tuple[dict[str, Any], ...]
    targets: tuple[dict[str, Any], ...]
    constraints: dict[str, Any]
    operating_point: dict[str, Any]
    optimization: dict[str, Any]
    components: tuple[IRComponent, ...] = ()
    functions: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_analysis(self) -> dict[str, Any]:
        return self.analyses[0] if self.analyses else {}

    @property
    def primary_target(self) -> dict[str, Any]:
        return self.targets[0] if self.targets else {}

    @property
    def intent_kind(self) -> str:
        if self.primary_target.get("target_kind") == "dc":
            return "dc"
        return str(self.primary_analysis.get("kind", "unknown"))

    @property
    def node_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for port in self.ports:
            names.update(port.terminals)
        for component in self.components:
            names.update(component.nodes)
        return tuple(sorted(names))

    def with_components(self, components: tuple[IRComponent, ...]) -> "UnifiedIR":
        return UnifiedIR(
            name=self.name,
            description=self.description,
            ports=self.ports,
            relations=self.relations,
            analyses=self.analyses,
            targets=self.targets,
            constraints=self.constraints,
            operating_point=self.operating_point,
            optimization=self.optimization,
            components=components,
            functions=self.functions,
            metadata=self.metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "ports": [port.as_dict() for port in self.ports],
            "relations": [dict(item) for item in self.relations],
            "analyses": [dict(item) for item in self.analyses],
            "targets": [dict(item) for item in self.targets],
            "constraints": dict(self.constraints),
            "operating_point": dict(self.operating_point),
            "optimization": dict(self.optimization),
            "components": [component.as_dict() for component in self.components],
            "functions": [dict(item) for item in self.functions],
            "metadata": dict(self.metadata),
        }


def pbdl_to_ir(data: dict[str, Any]) -> UnifiedIR:
    """Parse PBDL once and produce a lossless execution-facing IR."""
    spec = load_pbdl_dict(data)
    ports: list[IRPort] = []
    for port in spec.ports:
        ports.append(
            IRPort(
                name=port.name,
                terminals=tuple(item.name for item in port.terminals),
                role=port.role,
                domain=port.domain,
                port_id=port.stable_id,
                variables=tuple(variable.as_dict() for variable in port.variables),
                constraints=tuple(constraint.as_dict() for constraint in port.variable_constraints),
                excitation=dict(port.excitation) if port.excitation is not None else None,
            )
        )
    analyses = tuple(analysis.as_dict() for analysis in spec.analyses)
    targets = tuple(target.as_dict() for target in spec.targets)
    return UnifiedIR(
        name=spec.name,
        description=spec.description,
        ports=tuple(ports),
        relations=tuple(dict(item) for item in spec.relations),
        analyses=analyses,
        targets=targets,
        constraints=spec.constraints.as_dict(),
        operating_point=spec.operating_point.as_dict(),
        optimization=dict(spec.optimization),
        functions=tuple(item.as_dict() for item in spec.functions),
        metadata={"source": "PBDL"},
    )
