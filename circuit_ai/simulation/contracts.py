"""Versioned, backend-neutral simulation request and evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from ..graph import ModelRef
from ..graph.model import decode_graph_value, encode_graph_value, is_finite_graph_number


SIMULATION_REQUEST_SCHEMA = "circuit_ai.simulation_request"
SIMULATION_RESULT_SCHEMA = "circuit_ai.simulation_result"
SIMULATION_SCHEMA_VERSION = 1


class SimulationContractError(ValueError):
    """Raised when a request or result violates its serialized contract."""


class SimulationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class SweepSpec:
    variable: str
    unit: str
    values: tuple[float, ...] = ()
    start: float | None = None
    stop: float | None = None
    points: int | None = None
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not self.variable.strip():
            raise SimulationContractError("sweep variable must not be empty")
        explicit = bool(self.values)
        ranged = any(value is not None for value in (self.start, self.stop, self.points))
        if explicit == ranged:
            raise SimulationContractError(
                "sweep must define either explicit values or start/stop/points"
            )
        if explicit:
            if not all(is_finite_graph_number(value) for value in self.values):
                raise SimulationContractError("sweep values must be finite")
        else:
            if self.start is None or self.stop is None or self.points is None:
                raise SimulationContractError("range sweep requires start, stop, and points")
            if not math.isfinite(self.start) or not math.isfinite(self.stop):
                raise SimulationContractError("sweep bounds must be finite")
            if self.points < 1:
                raise SimulationContractError("sweep points must be positive")
            if self.scale not in {"linear", "log"}:
                raise SimulationContractError("sweep scale must be 'linear' or 'log'")
            if self.scale == "log" and (self.start <= 0.0 or self.stop <= 0.0):
                raise SimulationContractError("log sweep bounds must be positive")

    def materialize(self) -> tuple[float, ...]:
        if self.values:
            return self.values
        assert self.start is not None and self.stop is not None and self.points is not None
        if self.points == 1:
            return (float(self.start),)
        if self.scale == "linear":
            step = (self.stop - self.start) / (self.points - 1)
            return tuple(float(self.start + step * index) for index in range(self.points))
        start_log = math.log10(self.start)
        step_log = (math.log10(self.stop) - start_log) / (self.points - 1)
        return tuple(float(10.0 ** (start_log + step_log * index)) for index in range(self.points))

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "unit": self.unit,
            "values": list(self.values),
            "start": self.start,
            "stop": self.stop,
            "points": self.points,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepSpec":
        return cls(
            variable=str(data.get("variable", "")),
            unit=str(data.get("unit", "")),
            values=tuple(float(item) for item in data.get("values", [])),
            start=float(data["start"]) if data.get("start") is not None else None,
            stop=float(data["stop"]) if data.get("stop") is not None else None,
            points=int(data["points"]) if data.get("points") is not None else None,
            scale=str(data.get("scale", "linear")),
        )


@dataclass(frozen=True)
class AnalysisSpec:
    kind: str
    sweeps: tuple[SweepSpec, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise SimulationContractError("analysis kind must not be empty")
        if len({item.variable for item in self.sweeps}) != len(self.sweeps):
            raise SimulationContractError("analysis sweep variables must be unique")
        object.__setattr__(self, "options", _freeze_mapping(self.options))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sweeps": [item.as_dict() for item in self.sweeps],
            "options": encode_graph_value(self.options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisSpec":
        return cls(
            kind=str(data.get("kind", "")),
            sweeps=tuple(SweepSpec.from_dict(item) for item in data.get("sweeps", [])),
            options=decode_graph_value(data.get("options", {})),
        )


@dataclass(frozen=True)
class ObservableSpec:
    observable_id: str
    quantity: str
    unit: str
    target_kind: str
    target_id: str
    reference_id: str | None = None
    source_id: str | None = None
    representation: str = "scalar"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observable_id.strip():
            raise SimulationContractError("observable id must not be empty")
        if not self.quantity.strip():
            raise SimulationContractError("observable quantity must not be empty")
        if not self.target_kind.strip() or not self.target_id.strip():
            raise SimulationContractError("observable target kind and id must not be empty")
        if self.representation not in {"scalar", "real", "complex", "waveform", "tensor"}:
            raise SimulationContractError(
                f"unsupported observable representation {self.representation!r}"
            )
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "quantity": self.quantity,
            "unit": self.unit,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "reference_id": self.reference_id,
            "source_id": self.source_id,
            "representation": self.representation,
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservableSpec":
        return cls(
            observable_id=str(data.get("observable_id", "")),
            quantity=str(data.get("quantity", "")),
            unit=str(data.get("unit", "")),
            target_kind=str(data.get("target_kind", "")),
            target_id=str(data.get("target_id", "")),
            reference_id=(str(data["reference_id"]) if data.get("reference_id") is not None else None),
            source_id=str(data["source_id"]) if data.get("source_id") is not None else None,
            representation=str(data.get("representation", "scalar")),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class Excitation:
    excitation_id: str
    kind: str
    target_kind: str
    target_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.excitation_id, self.kind, self.target_kind, self.target_id)
        ):
            raise SimulationContractError("excitation identity, kind, and target must not be empty")
        if not _all_finite(self.parameters):
            raise SimulationContractError("excitation parameters contain a non-finite number")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))

    def as_dict(self) -> dict[str, Any]:
        return {
            "excitation_id": self.excitation_id,
            "kind": self.kind,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "parameters": encode_graph_value(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Excitation":
        return cls(
            excitation_id=str(data.get("excitation_id", "")),
            kind=str(data.get("kind", "")),
            target_kind=str(data.get("target_kind", "")),
            target_id=str(data.get("target_id", "")),
            parameters=decode_graph_value(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class OperatingConditions:
    temperature_c: float = 25.0
    corner: str = "typical"
    variables: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature_c):
            raise SimulationContractError("operating temperature must be finite")
        if not self.corner.strip():
            raise SimulationContractError("operating corner must not be empty")
        if not _all_finite(self.variables):
            raise SimulationContractError("operating conditions contain a non-finite number")
        object.__setattr__(self, "variables", _freeze_mapping(self.variables))

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "corner": self.corner,
            "variables": encode_graph_value(self.variables),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatingConditions":
        return cls(
            temperature_c=float(data.get("temperature_c", 25.0)),
            corner=str(data.get("corner", "typical")),
            variables=decode_graph_value(data.get("variables", {})),
        )


@dataclass(frozen=True)
class SimulationRequest:
    request_id: str
    analysis: AnalysisSpec
    graph_id: str
    parameter_values: Mapping[str, Any] = field(default_factory=dict)
    excitations: tuple[Excitation, ...] = ()
    conditions: OperatingConditions = field(default_factory=OperatingConditions)
    requested_observables: tuple[ObservableSpec, ...] = ()
    fidelity: str = "unspecified"
    timeout_s: float | None = None
    random_seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SIMULATION_REQUEST_SCHEMA
    schema_version: int = SIMULATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SIMULATION_REQUEST_SCHEMA or self.schema_version != SIMULATION_SCHEMA_VERSION:
            raise SimulationContractError("unsupported SimulationRequest schema")
        if not self.request_id.strip() or not self.graph_id.strip():
            raise SimulationContractError("request id and graph id must not be empty")
        if not self.fidelity.strip():
            raise SimulationContractError("request fidelity must not be empty")
        if self.timeout_s is not None and (not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0):
            raise SimulationContractError("request timeout must be finite and positive")
        if len({item.excitation_id for item in self.excitations}) != len(self.excitations):
            raise SimulationContractError("excitation ids must be unique")
        if len({item.observable_id for item in self.requested_observables}) != len(
            self.requested_observables
        ):
            raise SimulationContractError("observable ids must be unique")
        if not _all_finite(self.parameter_values):
            raise SimulationContractError("parameter values contain a non-finite number")
        object.__setattr__(self, "parameter_values", _freeze_mapping(self.parameter_values))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def request_hash(self) -> str:
        return _hash_payload(self._payload())

    @property
    def semantic_hash(self) -> str:
        """Hash of physical simulation inputs, excluding run metadata.

        ``request_hash`` intentionally protects the full wire contract.  A
        result cache must use a different key so a new request id, timeout, or
        ordinary reporting metadata does not duplicate the same simulation.
        """

        return _hash_payload(self._semantic_payload())

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "analysis": self.analysis.as_dict(),
            "parameter_values": encode_graph_value(self.parameter_values),
            "excitations": [item.as_dict() for item in self.excitations],
            "conditions": self.conditions.as_dict(),
            "requested_observables": [item.as_dict() for item in self.requested_observables],
            "fidelity": self.fidelity,
            "random_seed": self.random_seed,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "analysis": self.analysis.as_dict(),
            "graph_id": self.graph_id,
            "parameter_values": encode_graph_value(self.parameter_values),
            "excitations": [item.as_dict() for item in self.excitations],
            "conditions": self.conditions.as_dict(),
            "requested_observables": [item.as_dict() for item in self.requested_observables],
            "fidelity": self.fidelity,
            "timeout_s": self.timeout_s,
            "random_seed": self.random_seed,
            "metadata": encode_graph_value(self.metadata),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "request_hash": self.request_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationRequest":
        request = cls(
            request_id=str(data.get("request_id", "")),
            analysis=AnalysisSpec.from_dict(data.get("analysis", {})),
            graph_id=str(data.get("graph_id", "")),
            parameter_values=decode_graph_value(data.get("parameter_values", {})),
            excitations=tuple(Excitation.from_dict(item) for item in data.get("excitations", [])),
            conditions=OperatingConditions.from_dict(data.get("conditions", {})),
            requested_observables=tuple(
                ObservableSpec.from_dict(item) for item in data.get("requested_observables", [])
            ),
            fidelity=str(data.get("fidelity", "unspecified")),
            timeout_s=float(data["timeout_s"]) if data.get("timeout_s") is not None else None,
            random_seed=int(data["random_seed"]) if data.get("random_seed") is not None else None,
            metadata=decode_graph_value(data.get("metadata", {})),
            schema=str(data.get("schema", "")),
            schema_version=int(data.get("schema_version", 0)),
        )
        supplied_hash = data.get("request_hash")
        if supplied_hash is not None and str(supplied_hash) != request.request_hash:
            raise SimulationContractError("SimulationRequest request_hash mismatch")
        return request


@dataclass(frozen=True)
class Quantity:
    value: Any
    unit: str
    uncertainty: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_finite_graph_number(self.value):
            raise SimulationContractError("quantity value must be finite")
        if self.uncertainty is not None and (
            not math.isfinite(self.uncertainty) or self.uncertainty < 0.0
        ):
            raise SimulationContractError("quantity uncertainty must be finite and non-negative")
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": encode_graph_value(self.value),
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Quantity":
        return cls(
            value=decode_graph_value(data.get("value")),
            unit=str(data.get("unit", "")),
            uncertainty=(float(data["uncertainty"]) if data.get("uncertainty") is not None else None),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class ResultAxis:
    name: str
    unit: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.values:
            raise SimulationContractError("result axis requires a name and values")
        if not _all_finite(self.values):
            raise SimulationContractError("result axis contains a non-finite number")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "values": encode_graph_value(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResultAxis":
        return cls(
            name=str(data.get("name", "")),
            unit=str(data.get("unit", "")),
            values=tuple(decode_graph_value(data.get("values", []))),
        )


@dataclass(frozen=True)
class Waveform:
    observable_id: str
    unit: str
    axes: tuple[ResultAxis, ...]
    values: tuple[Any, ...]
    shape: tuple[int, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observable_id.strip() or not self.axes:
            raise SimulationContractError("waveform requires an observable id and at least one axis")
        shape = self.shape or tuple(len(axis.values) for axis in self.axes)
        if len(shape) != len(self.axes) or any(size < 1 for size in shape):
            raise SimulationContractError("waveform shape must match its axes")
        if any(len(axis.values) != size for axis, size in zip(self.axes, shape)):
            raise SimulationContractError("waveform axis lengths must match shape")
        if math.prod(shape) != len(self.values):
            raise SimulationContractError("waveform value count must equal the shape product")
        if not _all_finite(self.values):
            raise SimulationContractError("waveform contains a non-finite number")
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "unit": self.unit,
            "axes": [item.as_dict() for item in self.axes],
            "values": encode_graph_value(self.values),
            "shape": list(self.shape),
            "attributes": encode_graph_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Waveform":
        return cls(
            observable_id=str(data.get("observable_id", "")),
            unit=str(data.get("unit", "")),
            axes=tuple(ResultAxis.from_dict(item) for item in data.get("axes", [])),
            values=tuple(decode_graph_value(data.get("values", []))),
            shape=tuple(int(item) for item in data.get("shape", [])),
            attributes=decode_graph_value(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    path: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise SimulationContractError("diagnostic code and message must not be empty")
        if self.severity not in {"info", "warning", "error"}:
            raise SimulationContractError(f"unsupported diagnostic severity {self.severity!r}")
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "details": encode_graph_value(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Diagnostic":
        return cls(
            code=str(data.get("code", "")),
            severity=str(data.get("severity", "error")),
            message=str(data.get("message", "")),
            path=str(data.get("path", "")),
            details=decode_graph_value(data.get("details", {})),
        )


@dataclass(frozen=True)
class SimulationResult:
    request_id: str
    request_hash: str
    graph_hash: str
    backend_id: str
    backend_version: str
    model_manifest: tuple[ModelRef, ...]
    status: SimulationStatus
    scalars: Mapping[str, Quantity] = field(default_factory=dict)
    waveforms: Mapping[str, Waveform] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    runtime_s: float = 0.0
    fidelity: str = "unspecified"
    statistics: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SIMULATION_RESULT_SCHEMA
    schema_version: int = SIMULATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SIMULATION_RESULT_SCHEMA or self.schema_version != SIMULATION_SCHEMA_VERSION:
            raise SimulationContractError("unsupported SimulationResult schema")
        if not all(
            value.strip()
            for value in (self.request_id, self.request_hash, self.graph_hash, self.backend_id, self.backend_version)
        ):
            raise SimulationContractError("result identity and backend fields must not be empty")
        try:
            status = self.status if isinstance(self.status, SimulationStatus) else SimulationStatus(self.status)
        except ValueError as exc:
            raise SimulationContractError(f"unsupported simulation status {self.status!r}") from exc
        if not math.isfinite(self.runtime_s) or self.runtime_s < 0.0:
            raise SimulationContractError("simulation runtime must be finite and non-negative")
        if not self.fidelity.strip():
            raise SimulationContractError("result fidelity must not be empty")
        if status is SimulationStatus.PASSED and any(
            item.severity == "error" for item in self.diagnostics
        ):
            raise SimulationContractError("passed simulation result cannot contain error diagnostics")
        if any(key != value.observable_id for key, value in self.waveforms.items()):
            raise SimulationContractError("waveform mapping keys must match observable ids")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "waveforms", MappingProxyType(dict(self.waveforms)))
        object.__setattr__(self, "statistics", _freeze_mapping(self.statistics))

    @property
    def succeeded(self) -> bool:
        return self.status is SimulationStatus.PASSED

    @property
    def evidence_hash(self) -> str:
        return _hash_payload(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "graph_hash": self.graph_hash,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "model_manifest": [item.as_dict() for item in self.model_manifest],
            "status": self.status.value,
            "scalars": {
                key: self.scalars[key].as_dict() for key in sorted(self.scalars)
            },
            "waveforms": {
                key: self.waveforms[key].as_dict() for key in sorted(self.waveforms)
            },
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "runtime_s": self.runtime_s,
            "fidelity": self.fidelity,
            "statistics": encode_graph_value(self.statistics),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "evidence_hash": self.evidence_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationResult":
        result = cls(
            request_id=str(data.get("request_id", "")),
            request_hash=str(data.get("request_hash", "")),
            graph_hash=str(data.get("graph_hash", "")),
            backend_id=str(data.get("backend_id", "")),
            backend_version=str(data.get("backend_version", "")),
            model_manifest=tuple(ModelRef.from_dict(item) for item in data.get("model_manifest", [])),
            status=SimulationStatus(str(data.get("status", "failed"))),
            scalars={
                str(key): Quantity.from_dict(value)
                for key, value in data.get("scalars", {}).items()
            },
            waveforms={
                str(key): Waveform.from_dict(value)
                for key, value in data.get("waveforms", {}).items()
            },
            diagnostics=tuple(Diagnostic.from_dict(item) for item in data.get("diagnostics", [])),
            runtime_s=float(data.get("runtime_s", 0.0)),
            fidelity=str(data.get("fidelity", "unspecified")),
            statistics=decode_graph_value(data.get("statistics", {})),
            schema=str(data.get("schema", "")),
            schema_version=int(data.get("schema_version", 0)),
        )
        supplied_hash = data.get("evidence_hash")
        if supplied_hash is not None and str(supplied_hash) != result.evidence_hash:
            raise SimulationContractError("SimulationResult evidence_hash mismatch")
        return result


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


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_all_finite(item) for item in value)
    return is_finite_graph_number(value)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    wire = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()
