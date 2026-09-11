"""Stable capability contracts for synthesis services and backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


class CapabilityRole(str, Enum):
    """A service role that can participate in a synthesis execution plan."""

    PARAMETER_OPTIMIZER = "parameter_optimizer"
    SIMULATION_BACKEND = "simulation_backend"
    VERIFICATION_BACKEND = "verification_backend"
    CONSTRAINT_VALIDATOR = "constraint_validator"
    SCHEMATIC_RENDERER = "schematic_renderer"
    PROJECT_EXPORTER = "project_exporter"
    METRIC_PROVIDER = "metric_provider"


_ROLE_METHODS = {
    CapabilityRole.PARAMETER_OPTIMIZER: "optimize",
    CapabilityRole.SIMULATION_BACKEND: "simulate",
    CapabilityRole.VERIFICATION_BACKEND: "verify",
    CapabilityRole.CONSTRAINT_VALIDATOR: "validate",
    CapabilityRole.SCHEMATIC_RENDERER: "render",
    CapabilityRole.PROJECT_EXPORTER: "export",
    CapabilityRole.METRIC_PROVIDER: "extract",
}


@runtime_checkable
class Optimizer(Protocol):
    def optimize(self, ir: Any, candidate: Any) -> Any: ...


@runtime_checkable
class Simulator(Protocol):
    def compile(self, graph: Any) -> Any: ...

    def simulate(self, compiled: Any, request: Any) -> Any: ...


@runtime_checkable
class VerificationBackend(Protocol):
    def verify(self, result: Any, specification: Any, output_dir: Any = None) -> Any: ...


@runtime_checkable
class ConstraintValidator(Protocol):
    def validate(self, ir: Any, candidate: Any, simulation_result: Any) -> Any: ...


@runtime_checkable
class MetricProvider(Protocol):
    def extract(self, metric_spec: Any, context: Any) -> Any: ...


@runtime_checkable
class SchematicRenderer(Protocol):
    def render(self, result: Any) -> str: ...


@runtime_checkable
class Exporter(Protocol):
    def export(self, result: Any, output_dir: Any, **options: Any) -> Any: ...


@dataclass(frozen=True)
class CapabilityTarget:
    """Immutable, inspectable description of what one implementation supports.

    Empty constraint sets are wildcards. A target describes capability only;
    runtime configuration belongs to the implementation object.
    """

    capability_id: str
    version: str
    role: CapabilityRole
    description: str
    families: frozenset[str] = field(default_factory=frozenset)
    solver_ids: frozenset[str] = field(default_factory=frozenset)
    analysis_kinds: frozenset[str] = field(default_factory=frozenset)
    model_kinds: frozenset[str] = field(default_factory=frozenset)
    export_formats: frozenset[str] = field(default_factory=frozenset)
    fidelity: str = "unspecified"
    priority: int = 0
    source: str = "builtin"

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be empty")
        if not self.version.strip():
            raise ValueError("capability version must not be empty")
        if not self.description.strip():
            raise ValueError("capability description must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "version": self.version,
            "role": self.role.value,
            "description": self.description,
            "families": sorted(self.families),
            "solver_ids": sorted(self.solver_ids),
            "analysis_kinds": sorted(self.analysis_kinds),
            "model_kinds": sorted(self.model_kinds),
            "export_formats": sorted(self.export_formats),
            "fidelity": self.fidelity,
            "priority": self.priority,
            "source": self.source,
        }


@dataclass(frozen=True)
class CapabilityRegistration:
    """Bind a descriptive target to an executable implementation."""

    target: CapabilityTarget
    implementation: Any
    availability: Callable[[], bool | tuple[bool, str]] | None = None

    def __post_init__(self) -> None:
        method_name = _ROLE_METHODS[self.target.role]
        if not callable(getattr(self.implementation, method_name, None)):
            raise TypeError(
                f"capability {self.target.capability_id!r} with role "
                f"{self.target.role.value!r} requires a callable {method_name}()"
            )

    def availability_status(self) -> tuple[bool, str]:
        if self.availability is None:
            return True, "available"
        try:
            status = self.availability()
        except Exception as exc:  # pragma: no cover - defensive plugin boundary
            return False, f"availability check failed: {type(exc).__name__}: {exc}"
        if isinstance(status, tuple):
            available, reason = status
            return bool(available), str(reason)
        return bool(status), "available" if status else "runtime dependency is unavailable"


@dataclass(frozen=True)
class CapabilityRequest:
    """Requirements for one role in an execution plan."""

    role: CapabilityRole
    family: str | None = None
    solver_id: str | None = None
    analysis_kinds: frozenset[str] = field(default_factory=frozenset)
    model_kinds: frozenset[str] = field(default_factory=frozenset)
    export_format: str | None = None
    require_available: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "family": self.family,
            "solver_id": self.solver_id,
            "analysis_kinds": sorted(self.analysis_kinds),
            "model_kinds": sorted(self.model_kinds),
            "export_format": self.export_format,
            "require_available": self.require_available,
        }


@dataclass(frozen=True)
class CapabilityRejection:
    capability_id: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class CapabilityGap:
    request: CapabilityRequest
    rejections: tuple[CapabilityRejection, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "rejections": [item.as_dict() for item in self.rejections],
        }

    def summary(self) -> str:
        context = [f"role={self.request.role.value}"]
        if self.request.family:
            context.append(f"family={self.request.family}")
        if self.request.solver_id:
            context.append(f"solver={self.request.solver_id}")
        closest = sorted(self.rejections, key=lambda item: (len(item.reasons), item.capability_id))[:3]
        details = "; ".join(
            f"{item.capability_id}: {', '.join(item.reasons)}" for item in closest
        )
        return "no capability matched " + ", ".join(context) + (f"; closest: {details}" if details else "")


@dataclass(frozen=True)
class CapabilityResolution:
    request: CapabilityRequest
    selected: CapabilityRegistration
    alternatives: tuple[CapabilityRegistration, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "selected": self.selected.target.as_dict(),
            "alternatives": [item.target.as_dict() for item in self.alternatives],
        }


class CapabilityResolutionError(LookupError):
    def __init__(self, gap: CapabilityGap):
        self.gap = gap
        super().__init__(gap.summary())

    def as_dict(self) -> dict[str, Any]:
        return {"error_type": "capability_resolution", "gap": self.gap.as_dict()}


class CapabilityPlanError(LookupError):
    """Raised when no complete execution plan can be assembled."""

    def __init__(self, gaps: Iterable[CapabilityGap]):
        self.gaps = tuple(gaps)
        summaries = "; ".join(gap.summary() for gap in self.gaps)
        super().__init__(
            "topology candidates exist, but the execution plan has capability gaps"
            + (f": {summaries}" if summaries else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_type": "capability_plan",
            "gaps": [gap.as_dict() for gap in self.gaps],
        }


class CapabilityRegistrationError(ValueError):
    pass


class CapabilityPluginError(RuntimeError):
    pass


def normalize_plugin_registrations(value: Any) -> tuple[CapabilityRegistration, ...]:
    """Normalize the public entry-point contract into registrations."""

    if isinstance(value, CapabilityRegistration):
        return (value,)
    if callable(value):
        value = value()
        if isinstance(value, CapabilityRegistration):
            return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise CapabilityPluginError(
            "capability entry point must expose a CapabilityRegistration, a zero-argument "
            "factory, or an iterable of registrations"
        )
    registrations = tuple(value)
    if not all(isinstance(item, CapabilityRegistration) for item in registrations):
        raise CapabilityPluginError("capability entry point returned a non-registration item")
    return registrations
