"""Explicit, loss-aware migrations for serialized CircuitGraph documents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from .model import CIRCUIT_GRAPH_SCHEMA, CIRCUIT_GRAPH_VERSION


class CircuitGraphMigrationError(ValueError):
    """Raised when a graph document cannot be upgraded without guessing."""


Migration = Callable[[dict[str, Any]], dict[str, Any]]

_GRAPH_V1_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "graph_id",
        "name",
        "description",
        "family",
        "preferred_solver_ids",
        "domains",
        "ports",
        "nets",
        "components",
        "variables",
        "provenance",
        "extensions",
    }
)


def _v0_to_v1(payload: dict[str, Any]) -> dict[str, Any]:
    solver_id = payload.pop("preferred_solver_id", None)
    if "preferred_solver_ids" not in payload:
        payload["preferred_solver_ids"] = [str(solver_id)] if solver_id else []
    payload.setdefault("ports", [])
    payload.setdefault("variables", [])
    payload.setdefault("provenance", [])
    payload.setdefault("extensions", {})
    payload["schema"] = CIRCUIT_GRAPH_SCHEMA
    payload["schema_version"] = 1
    return payload


_MIGRATIONS: dict[int, Migration] = {0: _v0_to_v1}


def migrate_circuit_graph_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a current-schema copy of a serialized graph document."""

    payload = deepcopy(dict(data))
    raw_version = payload.get("schema_version", 0)
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise CircuitGraphMigrationError(
            f"invalid circuit graph schema_version={raw_version!r}"
        ) from exc

    schema = payload.get("schema")
    if version > 0 and schema != CIRCUIT_GRAPH_SCHEMA:
        raise CircuitGraphMigrationError(f"unsupported circuit graph schema {schema!r}")
    if version > CIRCUIT_GRAPH_VERSION:
        raise CircuitGraphMigrationError(
            f"unsupported future circuit graph schema_version={version}"
        )

    while version < CIRCUIT_GRAPH_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise CircuitGraphMigrationError(
                f"no circuit graph migration registered for schema_version={version}"
            )
        payload = migration(payload)
        migrated_version = int(payload.get("schema_version", version))
        if migrated_version != version + 1:
            raise CircuitGraphMigrationError(
                f"migration {version} must produce schema_version={version + 1}"
            )
        version = migrated_version

    if payload.get("schema") != CIRCUIT_GRAPH_SCHEMA:
        raise CircuitGraphMigrationError(
            f"unsupported circuit graph schema {payload.get('schema')!r}"
        )

    _preserve_unknown_top_level_fields(payload)
    return payload


def _preserve_unknown_top_level_fields(payload: dict[str, Any]) -> None:
    unknown = {
        key: payload.pop(key)
        for key in tuple(payload)
        if key not in _GRAPH_V1_FIELDS
    }
    if not unknown:
        return
    extensions = payload.setdefault("extensions", {})
    if not isinstance(extensions, dict):
        raise CircuitGraphMigrationError("CircuitGraph extensions must be an object")
    preserved = extensions.setdefault("unknown_top_level_fields", {})
    if not isinstance(preserved, dict):
        raise CircuitGraphMigrationError(
            "extensions.unknown_top_level_fields must be an object"
        )
    for key, value in unknown.items():
        preserved.setdefault(key, value)
