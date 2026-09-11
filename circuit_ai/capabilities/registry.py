"""Capability registration, matching, diagnostics, and explicit plugin loading."""

from __future__ import annotations

from importlib import metadata
from typing import Iterable

from .contracts import (
    CapabilityGap,
    CapabilityPluginError,
    CapabilityRegistration,
    CapabilityRegistrationError,
    CapabilityRejection,
    CapabilityRequest,
    CapabilityResolution,
    CapabilityResolutionError,
    normalize_plugin_registrations,
)


CAPABILITY_ENTRY_POINT_GROUP = "circuit_ai.capabilities"


class CapabilityRegistry:
    """A single registry for composable synthesis capabilities."""

    def __init__(self, registrations: Iterable[CapabilityRegistration] = ()) -> None:
        self._registrations: dict[str, CapabilityRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: CapabilityRegistration) -> CapabilityRegistration:
        capability_id = registration.target.capability_id
        if capability_id in self._registrations:
            existing = self._registrations[capability_id]
            raise CapabilityRegistrationError(
                f"duplicate capability id {capability_id!r}: "
                f"{existing.target.version!r} is already registered"
            )
        self._registrations[capability_id] = registration
        return registration

    def get(self, capability_id: str) -> CapabilityRegistration:
        try:
            return self._registrations[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability id {capability_id!r}") from exc

    def registrations(self) -> tuple[CapabilityRegistration, ...]:
        return tuple(
            self._registrations[key] for key in sorted(self._registrations)
        )

    def manifest(self) -> dict[str, object]:
        return {
            "entry_point_group": CAPABILITY_ENTRY_POINT_GROUP,
            "capabilities": [item.target.as_dict() for item in self.registrations()],
        }

    def discover_entry_points(
        self,
        group: str = CAPABILITY_ENTRY_POINT_GROUP,
    ) -> tuple[CapabilityRegistration, ...]:
        """Explicitly load installed plugins from Python package metadata."""

        loaded: list[CapabilityRegistration] = []
        entry_points = sorted(
            metadata.entry_points(group=group),
            key=lambda item: (item.name, item.value),
        )
        for entry_point in entry_points:
            try:
                registrations = normalize_plugin_registrations(entry_point.load())
                for registration in registrations:
                    self.register(registration)
                    loaded.append(registration)
            except Exception as exc:
                if isinstance(exc, CapabilityPluginError):
                    raise
                raise CapabilityPluginError(
                    f"failed to load capability entry point {entry_point.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return tuple(loaded)


class CapabilityResolver:
    """Resolve a role request and preserve evidence for every rejection."""

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def resolve(self, request: CapabilityRequest) -> CapabilityResolution:
        matches: list[CapabilityRegistration] = []
        rejections: list[CapabilityRejection] = []
        for registration in self.registry.registrations():
            reasons = _mismatch_reasons(registration, request)
            if reasons:
                rejections.append(
                    CapabilityRejection(registration.target.capability_id, tuple(reasons))
                )
            else:
                matches.append(registration)
        if not matches:
            gap = CapabilityGap(request, tuple(rejections))
            raise CapabilityResolutionError(gap)
        matches.sort(
            key=lambda item: (
                -item.target.priority,
                item.target.capability_id,
                item.target.version,
            )
        )
        return CapabilityResolution(request, matches[0], tuple(matches[1:]))


def _mismatch_reasons(
    registration: CapabilityRegistration,
    request: CapabilityRequest,
) -> list[str]:
    target = registration.target
    reasons: list[str] = []
    if target.role != request.role:
        reasons.append(f"role {target.role.value!r} does not provide {request.role.value!r}")
    if request.family and target.families and request.family not in target.families:
        reasons.append(f"family {request.family!r} is not supported")
    if request.solver_id and target.solver_ids and request.solver_id not in target.solver_ids:
        reasons.append(f"solver {request.solver_id!r} is not supported")
    missing_analyses = request.analysis_kinds - target.analysis_kinds if target.analysis_kinds else set()
    if missing_analyses:
        reasons.append(f"analysis kinds are missing: {sorted(missing_analyses)}")
    missing_models = request.model_kinds - target.model_kinds if target.model_kinds else set()
    if missing_models:
        reasons.append(f"model kinds are missing: {sorted(missing_models)}")
    if request.export_format and target.export_formats and request.export_format not in target.export_formats:
        reasons.append(f"export format {request.export_format!r} is not supported")
    # Runtime checks may touch executables, devices, or credentials. Only
    # probe a provider after all cheap descriptive constraints have matched.
    if request.require_available and not reasons:
        available, reason = registration.availability_status()
        if not available:
            reasons.append(reason)
    return reasons
