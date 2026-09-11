"""Versioned metric query and evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping

from ..units import normalize_unit, parse_unit


METRIC_SPEC_SCHEMA = "circuit_ai.metric_spec"
METRIC_VALUE_SCHEMA = "circuit_ai.metric_value"
METRIC_SCHEMA_VERSION = 1


class MetricContractError(ValueError):
    pass


class MetricStatus(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    provider_id: str
    source: str
    quantity: str
    unit: str
    reduction: str = "scalar"
    analysis_kind: str | None = None
    selectors: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = METRIC_SPEC_SCHEMA
    schema_version: int = METRIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != METRIC_SPEC_SCHEMA or self.schema_version != METRIC_SCHEMA_VERSION:
            raise MetricContractError("unsupported MetricSpec schema")
        if not all(item.strip() for item in (self.metric_id, self.provider_id, self.source, self.quantity)):
            raise MetricContractError("metric identity, provider, source, and quantity are required")
        if not self.reduction.strip():
            raise MetricContractError("metric reduction must not be empty")
        parse_unit(self.unit)
        object.__setattr__(self, "unit", normalize_unit(self.unit))
        object.__setattr__(self, "selectors", _freeze_mapping(self.selectors))
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    @property
    def name(self) -> str:
        """Compatibility alias used by the pre-v1 SimulationTask API."""
        return str(self.attributes.get("legacy_name", self.metric_id))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "metric_id": self.metric_id,
            "provider_id": self.provider_id,
            "source": self.source,
            "quantity": self.quantity,
            "unit": self.unit,
            "reduction": self.reduction,
            "analysis_kind": self.analysis_kind,
            "selectors": _thaw(self.selectors),
            "attributes": _thaw(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricSpec":
        return cls(
            metric_id=str(data.get("metric_id", data.get("name", ""))),
            provider_id=str(data.get("provider_id", "")),
            source=str(data.get("source", "")),
            quantity=str(data.get("quantity", data.get("source", ""))),
            unit=str(data.get("unit", "")),
            reduction=str(data.get("reduction", "scalar")),
            analysis_kind=(str(data["analysis_kind"]) if data.get("analysis_kind") is not None else None),
            selectors=dict(data.get("selectors", {})),
            attributes=dict(data.get("attributes", {})),
            schema=str(data.get("schema", METRIC_SPEC_SCHEMA)),
            schema_version=int(data.get("schema_version", METRIC_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class EvidenceReference:
    source_kind: str
    source_id: str
    evidence_hash: str = ""
    backend_id: str = ""
    backend_version: str = ""
    fidelity: str = "unspecified"
    level: str = "unverified"
    level_rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "evidence_hash": self.evidence_hash,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "fidelity": self.fidelity,
            "level": self.level,
            "level_rank": self.level_rank,
        }


@dataclass(frozen=True)
class MetricValue:
    metric_id: str
    status: MetricStatus
    value: float | None
    unit: str
    provider_id: str
    source: str
    evidence: tuple[EvidenceReference, ...] = ()
    diagnostics: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = METRIC_VALUE_SCHEMA
    schema_version: int = METRIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != METRIC_VALUE_SCHEMA or self.schema_version != METRIC_SCHEMA_VERSION:
            raise MetricContractError("unsupported MetricValue schema")
        if not all(item.strip() for item in (self.metric_id, self.provider_id, self.source)):
            raise MetricContractError("metric value identity, provider, and source are required")
        status = self.status if isinstance(self.status, MetricStatus) else MetricStatus(self.status)
        if status is MetricStatus.AVAILABLE:
            if self.value is None or not math.isfinite(float(self.value)):
                raise MetricContractError("available metric value must contain a finite scalar")
        elif self.value is not None:
            raise MetricContractError("unavailable metric value must not contain a scalar")
        parse_unit(self.unit)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unit", normalize_unit(self.unit))
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    @property
    def available(self) -> bool:
        return self.status is MetricStatus.AVAILABLE

    @property
    def evidence_level(self) -> str:
        if not self.evidence:
            return "unverified"
        return max(self.evidence, key=lambda item: item.level_rank).level

    @property
    def evidence_rank(self) -> int:
        return max((item.level_rank for item in self.evidence), default=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "metric_id": self.metric_id,
            "status": self.status.value,
            "value": self.value,
            "unit": self.unit,
            "provider_id": self.provider_id,
            "source": self.source,
            "evidence": [item.as_dict() for item in self.evidence],
            "evidence_level": self.evidence_level,
            "diagnostics": list(self.diagnostics),
            "attributes": _thaw(self.attributes),
        }


@dataclass(frozen=True)
class MetricSet:
    values: Mapping[str, MetricValue]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ordered = {key: self.values[key] for key in sorted(self.values)}
        if any(key != value.metric_id for key, value in ordered.items()):
            raise MetricContractError("MetricSet keys must match metric ids")
        object.__setattr__(self, "values", MappingProxyType(ordered))

    def get(self, metric_id: str) -> MetricValue | None:
        return self.values.get(metric_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": {key: value.as_dict() for key, value in self.values.items()},
            "diagnostics": list(self.diagnostics),
        }


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_thaw(item) for item in value]
    return value
