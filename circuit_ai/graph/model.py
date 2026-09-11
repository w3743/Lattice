"""Versioned, solver-independent circuit graph data model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping


CIRCUIT_GRAPH_SCHEMA = "circuit_ai.circuit_graph"
CIRCUIT_GRAPH_VERSION = 1


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def encode_graph_value(value: Any) -> Any:
    if isinstance(value, complex):
        return {"$complex": [float(value.real), float(value.imag)]}
    if isinstance(value, Mapping):
        return {
            str(key): encode_graph_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, frozenset, set)):
        return [encode_graph_value(item) for item in value]
    return value


def decode_graph_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$complex"}:
            real, imag = value["$complex"]
            return complex(float(real), float(imag))
        return {str(key): decode_graph_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(decode_graph_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ModelTerminal:
    name: str
    direction: str = "passive"
    domain: str = "electrical"
    quantity: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "direction": self.direction,
            "domain": self.domain,
            "quantity": self.quantity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelTerminal":
        return cls(
            name=str(data.get("name", "")),
            direction=str(data.get("direction", "passive")),
            domain=str(data.get("domain", "electrical")),
            quantity=str(data.get("quantity", "")),
        )


@dataclass(frozen=True)
class ModelRef:
    model_id: str
    kind: str
    terminals: tuple[ModelTerminal, ...]
    version: str = "1.0.0"
    source: str = "builtin"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "version": self.version,
            "source": self.source,
            "terminals": [item.as_dict() for item in self.terminals],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelRef":
        return cls(
            model_id=str(data.get("model_id", "")),
            kind=str(data.get("kind", "")),
            version=str(data.get("version", "1.0.0")),
            source=str(data.get("source", "builtin")),
            terminals=tuple(
                ModelTerminal.from_dict(item) for item in data.get("terminals", [])
            ),
        )


@dataclass(frozen=True)
class TerminalConnection:
    terminal: str
    net_id: str

    def as_dict(self) -> dict[str, str]:
        return {"terminal": self.terminal, "net_id": self.net_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TerminalConnection":
        return cls(terminal=str(data.get("terminal", "")), net_id=str(data.get("net_id", "")))


@dataclass(frozen=True)
class ParameterBinding:
    name: str
    unit: str = ""
    value: Any = None
    variable_id: str | None = None
    expression: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": encode_graph_value(self.value),
            "variable_id": self.variable_id,
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterBinding":
        return cls(
            name=str(data.get("name", "")),
            unit=str(data.get("unit", "")),
            value=decode_graph_value(data.get("value")),
            variable_id=str(data["variable_id"]) if data.get("variable_id") is not None else None,
            expression=str(data["expression"]) if data.get("expression") is not None else None,
        )


@dataclass(frozen=True)
class Rating:
    quantity: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    conditions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", _freeze(self.conditions))

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "conditions": encode_graph_value(self.conditions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rating":
        return cls(
            quantity=str(data.get("quantity", "")),
            unit=str(data.get("unit", "")),
            minimum=float(data["minimum"]) if data.get("minimum") is not None else None,
            maximum=float(data["maximum"]) if data.get("maximum") is not None else None,
            conditions=decode_graph_value(data.get("conditions", {})),
        )


@dataclass(frozen=True)
class ComponentInstance:
    instance_id: str
    reference: str
    model: ModelRef
    connections: tuple[TerminalConnection, ...]
    parameters: tuple[ParameterBinding, ...] = ()
    ratings: tuple[Rating, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "reference": self.reference,
            "model": self.model.as_dict(),
            "connections": [item.as_dict() for item in self.connections],
            "parameters": [item.as_dict() for item in self.parameters],
            "ratings": [item.as_dict() for item in self.ratings],
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentInstance":
        return cls(
            instance_id=str(data.get("instance_id", "")),
            reference=str(data.get("reference", data.get("instance_id", ""))),
            model=ModelRef.from_dict(data.get("model", {})),
            connections=tuple(
                TerminalConnection.from_dict(item) for item in data.get("connections", [])
            ),
            parameters=tuple(
                ParameterBinding.from_dict(item) for item in data.get("parameters", [])
            ),
            ratings=tuple(Rating.from_dict(item) for item in data.get("ratings", [])),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class Net:
    net_id: str
    name: str
    domain_id: str
    kind: str = "signal"
    is_reference: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "net_id": self.net_id,
            "name": self.name,
            "domain_id": self.domain_id,
            "kind": self.kind,
            "is_reference": self.is_reference,
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Net":
        return cls(
            net_id=str(data.get("net_id", "")),
            name=str(data.get("name", data.get("net_id", ""))),
            domain_id=str(data.get("domain_id", "")),
            kind=str(data.get("kind", "signal")),
            is_reference=bool(data.get("is_reference", False)),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class GraphDomain:
    domain_id: str
    kind: str = "electrical"
    reference_net_id: str | None = None
    isolated_from: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "kind": self.kind,
            "reference_net_id": self.reference_net_id,
            "isolated_from": list(self.isolated_from),
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphDomain":
        return cls(
            domain_id=str(data.get("domain_id", "")),
            kind=str(data.get("kind", "electrical")),
            reference_net_id=(
                str(data["reference_net_id"])
                if data.get("reference_net_id") is not None
                else None
            ),
            isolated_from=tuple(str(item) for item in data.get("isolated_from", [])),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class PortTerminal:
    name: str
    net_id: str
    quantity: str = "voltage"
    role: str = "potential"

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "net_id": self.net_id,
            "quantity": self.quantity,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortTerminal":
        return cls(
            name=str(data.get("name", "")),
            net_id=str(data.get("net_id", "")),
            quantity=str(data.get("quantity", "voltage")),
            role=str(data.get("role", "potential")),
        )


@dataclass(frozen=True)
class GraphPort:
    port_id: str
    name: str
    direction: str
    domain_id: str
    terminals: tuple[PortTerminal, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "name": self.name,
            "direction": self.direction,
            "domain_id": self.domain_id,
            "terminals": [item.as_dict() for item in self.terminals],
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphPort":
        return cls(
            port_id=str(data.get("port_id", data.get("name", ""))),
            name=str(data.get("name", data.get("port_id", ""))),
            direction=str(data.get("direction", "bidirectional")),
            domain_id=str(data.get("domain_id", "")),
            terminals=tuple(PortTerminal.from_dict(item) for item in data.get("terminals", [])),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class ParameterVariable:
    variable_id: str
    value_type: str
    unit: str = ""
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()
    initial: Any = None
    scale: str = "linear"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", tuple(_freeze(item) for item in self.choices))
        object.__setattr__(self, "initial", _freeze(self.initial))
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable_id": self.variable_id,
            "value_type": self.value_type,
            "unit": self.unit,
            "lower": self.lower,
            "upper": self.upper,
            "choices": [encode_graph_value(item) for item in self.choices],
            "initial": encode_graph_value(self.initial),
            "scale": self.scale,
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterVariable":
        return cls(
            variable_id=str(data.get("variable_id", "")),
            value_type=str(data.get("value_type", "continuous")),
            unit=str(data.get("unit", "")),
            lower=float(data["lower"]) if data.get("lower") is not None else None,
            upper=float(data["upper"]) if data.get("upper") is not None else None,
            choices=tuple(decode_graph_value(item) for item in data.get("choices", [])),
            initial=decode_graph_value(data.get("initial")),
            scale=str(data.get("scale", "linear")),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str
    source_id: str = ""
    version: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze(self.details))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "version": self.version,
            "details": encode_graph_value(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceRecord":
        return cls(
            source=str(data.get("source", "")),
            source_id=str(data.get("source_id", "")),
            version=str(data.get("version", "")),
            details=decode_graph_value(data.get("details", {})),
        )


@dataclass(frozen=True)
class CircuitGraph:
    graph_id: str
    name: str
    domains: tuple[GraphDomain, ...]
    nets: tuple[Net, ...]
    components: tuple[ComponentInstance, ...]
    ports: tuple[GraphPort, ...] = ()
    variables: tuple[ParameterVariable, ...] = ()
    description: str = ""
    family: str | None = None
    preferred_solver_ids: tuple[str, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CIRCUIT_GRAPH_SCHEMA
    schema_version: int = CIRCUIT_GRAPH_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", _freeze(self.extensions))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "preferred_solver_ids": list(self.preferred_solver_ids),
            "domains": [item.as_dict() for item in self.domains],
            "ports": [item.as_dict() for item in self.ports],
            "nets": [item.as_dict() for item in self.nets],
            "components": [item.as_dict() for item in self.components],
            "variables": [item.as_dict() for item in self.variables],
            "provenance": [item.as_dict() for item in self.provenance],
            "extensions": encode_graph_value(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitGraph":
        from .migration import migrate_circuit_graph_payload

        data = migrate_circuit_graph_payload(data)
        schema = str(data.get("schema", ""))
        version = int(data.get("schema_version", 0))
        return cls(
            graph_id=str(data.get("graph_id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            family=str(data["family"]) if data.get("family") is not None else None,
            preferred_solver_ids=tuple(str(item) for item in data.get("preferred_solver_ids", [])),
            domains=tuple(GraphDomain.from_dict(item) for item in data.get("domains", [])),
            ports=tuple(GraphPort.from_dict(item) for item in data.get("ports", [])),
            nets=tuple(Net.from_dict(item) for item in data.get("nets", [])),
            components=tuple(
                ComponentInstance.from_dict(item) for item in data.get("components", [])
            ),
            variables=tuple(
                ParameterVariable.from_dict(item) for item in data.get("variables", [])
            ),
            provenance=tuple(
                ProvenanceRecord.from_dict(item) for item in data.get("provenance", [])
            ),
            extensions=decode_graph_value(data.get("extensions", {})),
            schema=schema,
            schema_version=version,
        )

    def validate(self):
        from .validation import validate_circuit_graph

        return validate_circuit_graph(self)

    def require_valid(self) -> "CircuitGraph":
        from .validation import require_valid_graph

        return require_valid_graph(self)

    def canonical(self) -> "CircuitGraph":
        from .canonical import canonicalize_graph

        return canonicalize_graph(self)

    @property
    def document_hash(self) -> str:
        from .canonical import document_hash

        return document_hash(self)

    @property
    def graph_hash(self) -> str:
        from .canonical import structural_hash

        return structural_hash(self, include_parameters=True)

    @property
    def topology_hash(self) -> str:
        from .canonical import structural_hash

        return structural_hash(self, include_parameters=False)


def is_finite_graph_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        return True
    if isinstance(value, complex):
        return math.isfinite(value.real) and math.isfinite(value.imag)
    return math.isfinite(float(value))
