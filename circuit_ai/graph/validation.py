"""Electrical graph invariants and structured diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .model import (
    CIRCUIT_GRAPH_SCHEMA,
    CIRCUIT_GRAPH_VERSION,
    CircuitGraph,
    is_finite_graph_number,
)
from .primitives import ISOLATION_BARRIER_KIND


@dataclass(frozen=True)
class GraphIssue:
    code: str
    path: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class GraphValidationReport:
    issues: tuple[GraphIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    @property
    def errors(self) -> tuple[GraphIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "error")

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [item.as_dict() for item in self.issues],
        }


class GraphValidationError(ValueError):
    def __init__(self, report: GraphValidationReport):
        self.report = report
        details = "; ".join(f"{item.path}: {item.message}" for item in report.errors[:5])
        super().__init__("invalid CircuitGraph" + (f": {details}" if details else ""))

    def as_dict(self) -> dict[str, Any]:
        return {"error_type": "graph_validation", "validation": self.report.as_dict()}


def validate_circuit_graph(graph: CircuitGraph) -> GraphValidationReport:
    issues: list[GraphIssue] = []

    def error(code: str, path: str, message: str) -> None:
        issues.append(GraphIssue(code, path, message))

    def warning(code: str, path: str, message: str) -> None:
        issues.append(GraphIssue(code, path, message, "warning"))

    if graph.schema != CIRCUIT_GRAPH_SCHEMA:
        error("schema", "schema", f"expected {CIRCUIT_GRAPH_SCHEMA!r}")
    if graph.schema_version != CIRCUIT_GRAPH_VERSION:
        error("schema_version", "schema_version", f"expected {CIRCUIT_GRAPH_VERSION}")
    if not graph.graph_id.strip():
        error("empty_id", "graph_id", "graph id must not be empty")
    if not graph.name.strip():
        error("empty_name", "name", "graph name must not be empty")

    _check_unique((item.domain_id for item in graph.domains), "domains", "domain_id", error)
    _check_unique((item.net_id for item in graph.nets), "nets", "net_id", error)
    _check_unique((item.port_id for item in graph.ports), "ports", "port_id", error)
    _check_unique((item.instance_id for item in graph.components), "components", "instance_id", error)
    _check_unique((item.reference for item in graph.components), "components", "reference", error)
    _check_unique((item.variable_id for item in graph.variables), "variables", "variable_id", error)

    domains = {item.domain_id: item for item in graph.domains}
    nets = {item.net_id: item for item in graph.nets}
    variables = {item.variable_id: item for item in graph.variables}

    for index, domain in enumerate(graph.domains):
        path = f"domains[{index}]"
        if not domain.domain_id.strip():
            error("empty_id", f"{path}.domain_id", "domain id must not be empty")
        if domain.reference_net_id is not None:
            reference = nets.get(domain.reference_net_id)
            if reference is None:
                error("missing_reference_net", f"{path}.reference_net_id", "reference net does not exist")
            elif reference.domain_id != domain.domain_id:
                error("cross_domain_reference", f"{path}.reference_net_id", "reference net belongs to another domain")
            elif not reference.is_reference:
                error("reference_flag", f"{path}.reference_net_id", "reference net is not marked is_reference")
        for other in domain.isolated_from:
            if other not in domains:
                error("missing_isolation_domain", f"{path}.isolated_from", f"domain {other!r} does not exist")
            elif domain.domain_id not in domains[other].isolated_from:
                error("asymmetric_isolation", f"{path}.isolated_from", f"isolation with {other!r} is not symmetric")

    connected_nets: set[str] = set()
    for index, net in enumerate(graph.nets):
        path = f"nets[{index}]"
        if not net.net_id.strip():
            error("empty_id", f"{path}.net_id", "net id must not be empty")
        if net.domain_id not in domains:
            error("missing_domain", f"{path}.domain_id", f"domain {net.domain_id!r} does not exist")

    for index, port in enumerate(graph.ports):
        path = f"ports[{index}]"
        if port.domain_id not in domains:
            error("missing_domain", f"{path}.domain_id", f"domain {port.domain_id!r} does not exist")
        _check_unique((item.name for item in port.terminals), path, "terminal name", error)
        if not port.terminals:
            error("empty_port", f"{path}.terminals", "port must have at least one terminal")
        for terminal_index, terminal in enumerate(port.terminals):
            terminal_path = f"{path}.terminals[{terminal_index}]"
            net = nets.get(terminal.net_id)
            if net is None:
                error("missing_net", f"{terminal_path}.net_id", f"net {terminal.net_id!r} does not exist")
            else:
                connected_nets.add(terminal.net_id)
                if net.domain_id != port.domain_id:
                    error("cross_domain_port", f"{terminal_path}.net_id", "port terminal crosses domains")

    used_variables: set[str] = set()
    for index, component in enumerate(graph.components):
        path = f"components[{index}]"
        if not component.instance_id.strip():
            error("empty_id", f"{path}.instance_id", "component id must not be empty")
        if not component.model.model_id.strip():
            error("empty_model", f"{path}.model.model_id", "model id must not be empty")
        if component.model.kind != ISOLATION_BARRIER_KIND:
            touched = {
                nets[connection.net_id].domain_id
                for connection in component.connections
                if connection.net_id in nets
            }
            if any(
                other in touched
                for domain_id in touched
                for other in domains[domain_id].isolated_from
                if domain_id in domains
            ):
                error(
                    "isolation_short",
                    f"{path}.connections",
                    f"{component.reference} bridges isolated graph domains {sorted(touched)}",
                )
        model_terminals = [item.name for item in component.model.terminals]
        connection_terminals = [item.terminal for item in component.connections]
        _check_unique(model_terminals, f"{path}.model", "terminal name", error)
        _check_unique(connection_terminals, path, "connection terminal", error)
        if set(model_terminals) != set(connection_terminals):
            error(
                "terminal_contract",
                f"{path}.connections",
                f"model terminals {sorted(model_terminals)} do not match connections {sorted(connection_terminals)}",
            )
        for connection_index, connection in enumerate(component.connections):
            if connection.net_id not in nets:
                error(
                    "missing_net",
                    f"{path}.connections[{connection_index}].net_id",
                    f"net {connection.net_id!r} does not exist",
                )
            else:
                connected_nets.add(connection.net_id)
        _check_unique((item.name for item in component.parameters), path, "parameter name", error)
        for parameter_index, parameter in enumerate(component.parameters):
            parameter_path = f"{path}.parameters[{parameter_index}]"
            sources = sum(
                (
                    parameter.value is not None,
                    parameter.variable_id is not None,
                    parameter.expression is not None,
                )
            )
            if sources != 1:
                error("parameter_source", parameter_path, "parameter needs exactly one value, variable, or expression")
            if not is_finite_graph_number(parameter.value):
                error("non_finite", f"{parameter_path}.value", "parameter value must be finite")
            if parameter.variable_id is not None:
                variable = variables.get(parameter.variable_id)
                if variable is None:
                    error("missing_variable", f"{parameter_path}.variable_id", "parameter variable does not exist")
                else:
                    used_variables.add(variable.variable_id)
                    if parameter.unit and variable.unit and parameter.unit != variable.unit:
                        error("unit_mismatch", parameter_path, f"parameter unit {parameter.unit!r} differs from variable unit {variable.unit!r}")
        for rating_index, rating in enumerate(component.ratings):
            rating_path = f"{path}.ratings[{rating_index}]"
            if rating.minimum is not None and rating.maximum is not None and rating.minimum > rating.maximum:
                error("invalid_range", rating_path, "rating minimum exceeds maximum")

    for index, variable in enumerate(graph.variables):
        path = f"variables[{index}]"
        if variable.lower is not None and variable.upper is not None and variable.lower > variable.upper:
            error("invalid_range", path, "variable lower bound exceeds upper bound")
        if variable.scale not in {"linear", "log", "log10"}:
            error("invalid_scale", f"{path}.scale", f"unsupported scale {variable.scale!r}")
        if variable.value_type == "categorical" and not variable.choices:
            error("missing_choices", f"{path}.choices", "categorical variable needs choices")
        if variable.variable_id not in used_variables:
            warning("unused_variable", path, "variable is not bound to a component parameter")

    for index, net in enumerate(graph.nets):
        if net.net_id not in connected_nets:
            warning("unused_net", f"nets[{index}]", "net has no port or component connection")

    return GraphValidationReport(tuple(issues))


def require_valid_graph(graph: CircuitGraph) -> CircuitGraph:
    report = validate_circuit_graph(graph)
    if not report.passed:
        raise GraphValidationError(report)
    return graph


def _check_unique(
    values: Iterable[str],
    path: str,
    label: str,
    error,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        error("duplicate_id", path, f"duplicate {label}: {sorted(duplicates)}")
