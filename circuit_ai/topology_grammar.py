"""Typed, bounded topology grammar for PBDL power-stage synthesis.

The grammar builds candidates from small electrical construction rules.  A
candidate is executable only when a matching physics model is registered; this
keeps topology discovery separate from claims about circuit correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .experts import TopologyCandidate, UnsupportedTopology
from .ir import UnifiedIR
from .knowledge import TopologyKnowledgeBase, load_topology_knowledge


GRAMMAR_VERSION = "power-grammar-v2"


@dataclass(frozen=True)
class TopologyRejection:
    name: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason}


@dataclass(frozen=True)
class TopologySearchCertificate:
    grammar_version: str
    max_components: int
    allowed_elements: tuple[str, ...]
    required_elements: tuple[str, ...]
    registered_productions: tuple[str, ...]
    constructed_candidates: int
    accepted_candidates: int
    rejected: tuple[TopologyRejection, ...]
    exhaustive_within_registered_productions: bool = True
    native_search: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "grammar_version": self.grammar_version,
            "bounds": {
                "max_components": self.max_components,
                "allowed_elements": list(self.allowed_elements),
                "required_elements": list(self.required_elements),
                "registered_productions": list(self.registered_productions),
            },
            "constructed_candidates": self.constructed_candidates,
            "accepted_candidates": self.accepted_candidates,
            "rejected": [item.as_dict() for item in self.rejected],
            "exhaustive_within_registered_productions": self.exhaustive_within_registered_productions,
            "native_search": dict(self.native_search) if self.native_search is not None else None,
        }


@dataclass(frozen=True)
class TopologySearchResult:
    candidates: tuple[TopologyCandidate, ...]
    certificate: TopologySearchCertificate


class PowerTopologyGrammar:
    """Generate all registered ideal power families within explicit bounds."""

    def __init__(self, knowledge_path: str | Path | None = None) -> None:
        self.knowledge: TopologyKnowledgeBase = load_topology_knowledge(knowledge_path)

    def search(
        self,
        ir: UnifiedIR,
        *,
        additional_candidates: tuple[TopologyCandidate, ...] = (),
    ) -> TopologySearchResult:
        context = _power_context(ir)
        isolated = context["isolated"]
        vin = context["vin"]
        vout = context["vout"]
        constructed: list[TopologyCandidate] = []
        rejected: list[TopologyRejection] = []
        native_certificate: dict[str, Any] | None = None

        native_candidates, native_certificate = self._native_candidates(ir, context)
        constructed.extend(native_candidates)

        for production in self.knowledge.productions:
            if not production.is_applicable(isolated=isolated, vin=vin, vout=vout):
                continue
            constructed.append(self.knowledge.instantiate(production, context))

        constructed.extend(additional_candidates)
        accepted: list[TopologyCandidate] = []
        seen: set[str] = set()
        for candidate in constructed:
            candidate = _ensure_candidate_graph(candidate, ir)
            reason = _rejection_reason(ir, candidate, isolated=isolated)
            if reason is not None:
                rejected.append(TopologyRejection(candidate.name, reason))
                continue
            signature = _candidate_signature(candidate)
            if signature in seen:
                rejected.append(TopologyRejection(candidate.name, "duplicate electrical graph"))
                continue
            seen.add(signature)
            accepted.append(candidate)

        constraints = ir.constraints
        allowed = tuple(sorted(str(item) for item in constraints.get("element_types", [])))
        required = tuple(sorted(str(item) for item in constraints.get("required_elements", [])))
        certificate = TopologySearchCertificate(
            grammar_version=f"{GRAMMAR_VERSION}:{self.knowledge.knowledge_id}",
            max_components=int(constraints.get("max_component_count", ir.optimization.get("max_components", 8))),
            allowed_elements=allowed,
            required_elements=required,
            registered_productions=tuple(item.name for item in self.knowledge.productions),
            constructed_candidates=len(constructed),
            accepted_candidates=len(accepted),
            rejected=tuple(rejected),
            exhaustive_within_registered_productions=native_certificate is None,
            native_search=native_certificate,
        )
        return TopologySearchResult(tuple(accepted), certificate)

    def _native_candidates(
        self,
        ir: UnifiedIR,
        context: dict[str, Any],
    ) -> tuple[tuple[TopologyCandidate, ...], dict[str, Any] | None]:
        options = dict(ir.optimization.get("graph_search", {}) or {})
        if not bool(options.get("native_enabled", False)):
            return (), None
        if context["isolated"]:
            return (), {"enabled": True, "skipped": "isolated_power_not_supported_by_native_v1"}
        if context["vout"] <= context["vin"]:
            return (), {"enabled": True, "skipped": "native_v1_only_supports_step_up"}

        from .graph import (
            GraphSearchPolicy,
            ParetoBeamSearch,
            PowerGraphActionGenerator,
            graph_to_topology_candidate,
            initial_power_graph_state,
        )

        # The order is a construction hint, not a registered production:
        # energy storage creates an internal switch node before the switching
        # and rectification branches are attached.
        default_kinds = ("L", "ideal_switch", "ideal_diode", "C", "R")
        allowed_config = options.get("allowed_model_kinds")
        if allowed_config is None:
            allowed_config = ir.constraints.get("element_types") or default_kinds
        allowed_tokens = {str(item).casefold() for item in allowed_config}
        allowed = tuple(
            kind for kind in default_kinds
            if kind.casefold() in allowed_tokens
        )
        required_config = options.get("required_model_kinds")
        canonical_names = {kind.casefold(): kind for kind in default_kinds}
        required = tuple(
            canonical_names[str(item).casefold()]
            for item in (required_config if required_config is not None else default_kinds)
            if str(item).casefold() in canonical_names and str(item).casefold() in allowed_tokens
        )
        if not required or not set(required).issubset(set(allowed)):
            return (), {
                "enabled": True,
                "skipped": "native_required_models_not_available",
                "allowed_model_kinds": list(allowed),
                "required_model_kinds": list(required),
            }
        max_components = int(
            options.get(
                "max_components",
                ir.constraints.get("max_component_count", ir.optimization.get("max_components", 8)),
            )
        )
        if max_components < len(required):
            return (), {
                "enabled": True,
                "skipped": "native_component_budget_below_required_models",
                "max_components": max_components,
                "required_model_kinds": list(required),
            }
        max_nodes = int(options.get("max_nodes", max(4, max_components + 1)))
        policy = GraphSearchPolicy(
            allowed_model_kinds=allowed,
            required_model_kinds=required,
            min_components=max(len(required), int(options.get("min_components", len(required)))),
            max_components=max_components,
            max_nodes=max_nodes,
            max_depth=int(options.get("max_depth", max(5, len(required) + 1))),
            beam_width=int(options.get("beam_width", 24)),
            target_count=int(options.get("max_candidates", 4)),
            max_expansions=int(options.get("max_expansions", 2_000)),
            max_simulations=0,
            allow_parallel=bool(options.get("allow_parallel", True)),
            allow_input_ground=bool(options.get("allow_input_ground", True)),
            variable_parameters=True,
            parameter_ranges=ir.constraints.get("parameter_ranges", {}),
            grammar_version="power_graph_actions.v1",
        )
        initial = initial_power_graph_state(ir, policy)
        search = ParetoBeamSearch(
            policy=policy,
            seed_states=(),
            action_generator=PowerGraphActionGenerator(policy),
        ).search(initial, target_count=policy.target_count)
        candidates: list[TopologyCandidate] = []
        rejected_native: list[dict[str, str]] = []
        for state in search.complete_states:
            graph = state.graph
            if not _is_native_boost_shape(graph, ir):
                rejected_native.append(
                    {"name": graph.name, "reason": "native_v1_shape_not_supported_by_ideal_boost_solver"}
                )
                continue
            candidate = graph_to_topology_candidate(graph)
            candidate = _native_candidate_metadata(candidate, ir)
            candidate = replace(
                candidate,
                metadata={
                    **dict(candidate.metadata),
                    "derivation": [record.as_dict() for record in state.derivation],
                },
            )
            candidates.append(candidate)
        return tuple(candidates), {
            "enabled": True,
            "policy": {
                "allowed_model_kinds": list(allowed),
                "required_model_kinds": list(required),
                "max_components": max_components,
                "max_nodes": max_nodes,
                "beam_width": policy.beam_width,
                "max_expansions": policy.max_expansions,
            },
            "search": search.certificate.as_dict(),
            "constructed_complete_graphs": len(search.complete_states),
            "accepted_candidates": len(candidates),
            "rejected": rejected_native,
        }


def _power_context(ir: UnifiedIR) -> dict[str, Any]:
    if ir.intent_kind != "dc":
        raise UnsupportedTopology(f"power grammar does not support intent {ir.intent_kind!r}")
    if len(ir.ports) != 2:
        raise UnsupportedTopology("power grammar currently requires exactly two ports")
    target = ir.primary_target
    vin = _positive_float(target, "input_voltage_v")
    vout = _positive_float(target, "output_voltage_v")
    input_port = _port_by_role(ir, "input", ir.ports[0])
    output_port = _port_by_role(ir, "output", ir.ports[1])
    isolated = any(
        str(item.get("kind", "")).casefold() in {"galvanic_isolation", "isolated"}
        for item in ir.relations
    )
    if isolated and input_port.negative == output_port.negative:
        raise UnsupportedTopology("galvanic isolation requires distinct reference nodes")
    if not isolated and input_port.negative != output_port.negative:
        raise UnsupportedTopology("non-isolated power conversion requires a shared reference node")
    return {
        "ir": ir,
        "vin": vin,
        "vout": vout,
        "input_port": input_port,
        "output_port": output_port,
        "isolated": isolated,
    }


def _is_native_boost_shape(graph, ir: UnifiedIR) -> bool:
    """Recognize the topology currently executable by the ideal boost solver."""

    input_port = _port_by_role(ir, "input", ir.ports[0])
    output_port = _port_by_role(ir, "output", ir.ports[-1])
    reference = input_port.negative
    components = tuple(graph.components)
    if len(components) != 5:
        return False
    by_kind: dict[str, list[Any]] = {}
    for component in components:
        by_kind.setdefault(component.model.kind, []).append(component)
    if set(by_kind) != {"R", "C", "L", "ideal_switch", "ideal_diode"}:
        return False
    if any(len(items) != 1 for items in by_kind.values()):
        return False

    def connections(kind: str) -> dict[str, str]:
        component = by_kind[kind][0]
        return {item.terminal: item.net_id for item in component.connections}

    inductor = connections("L")
    switch = connections("ideal_switch")
    diode = connections("ideal_diode")
    capacitor = connections("C")
    load = by_kind["R"][0]
    load_connections = connections("R")
    inductor_nodes = set(inductor.values())
    internal_nodes = inductor_nodes - {input_port.positive}
    if len(internal_nodes) != 1:
        return False
    switch_node = next(iter(internal_nodes))
    return (
        set(inductor.values()) == {input_port.positive, switch_node}
        and set(switch.values()) == {switch_node, reference}
        and diode.get("anode") == switch_node
        and diode.get("cathode") == output_port.positive
        and set(capacitor.values()) == {output_port.positive, reference}
        and set(load_connections.values()) == {output_port.positive, reference}
        and str(load.attributes.get("role", "")) == "external_load"
    )


def _native_candidate_metadata(candidate: TopologyCandidate, ir: UnifiedIR) -> TopologyCandidate:
    input_port = _port_by_role(ir, "input", ir.ports[0])
    output_port = _port_by_role(ir, "output", ir.ports[-1])
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "origin": "native_power_graph_search",
            "solver": "ideal_boost_averaged",
            "input_port": input_port.name,
            "output_port": output_port.name,
            "input_node": input_port.positive,
            "output_node": output_port.positive,
            "input_reference_node": input_port.negative,
            "output_reference_node": output_port.negative,
            "reference_node": input_port.negative,
            "conduction_mode": "ideal_ccm",
        }
    )
    # The detailed action certificate is stored by the grammar result.  Keep
    # the candidate metadata compact and deterministic for replay rows.
    canonical_names = {
        "L": "L1",
        "ideal_switch": "Q1",
        "ideal_diode": "D1",
        "C": "C1",
        "R": "Rload",
    }
    metadata.pop("graph_hash", None)
    metadata.pop("topology_hash", None)
    components = tuple(
        replace(component, name=canonical_names.get(component.kind, component.name))
        for component in candidate.components
    )
    normalized = replace(
        candidate,
        family="dc_boost",
        components=components,
        metadata=metadata,
        solver_id="ideal_boost_averaged",
        graph=None,
    )
    from .graph import topology_candidate_to_graph

    graph = topology_candidate_to_graph(normalized, ir)
    metadata.update({"graph_hash": graph.graph_hash, "topology_hash": graph.topology_hash})
    return replace(normalized, graph=graph, metadata=metadata)


def _rejection_reason(ir: UnifiedIR, candidate: TopologyCandidate, *, isolated: bool) -> str | None:
    design = tuple(c for c in candidate.components if c.attributes.get("role") != "external_load")
    constraints = ir.constraints
    allowed = {str(item).casefold() for item in constraints.get("element_types", [])}
    kinds = {component.kind for component in design}
    if allowed and any(kind.casefold() not in allowed for kind in kinds):
        return f"uses elements outside library: {sorted(kind for kind in kinds if kind.casefold() not in allowed)}"
    required = {str(item).casefold() for item in constraints.get("required_elements", [])}
    present = {kind.casefold() for kind in kinds}
    if not required.issubset(present):
        return f"missing required elements: {sorted(required - present)}"
    max_components = int(constraints.get("max_component_count", ir.optimization.get("max_components", 8)))
    if len(design) > max_components:
        return f"component bound exceeded: {len(design)} > {max_components}"
    if len({component.name for component in candidate.components}) != len(candidate.components):
        return "component names are not unique"
    for component in candidate.components:
        if len(component.nodes) < 2 or len(set(component.nodes)) < 2:
            return f"{component.name} has an invalid connection"
    if not _ports_connected(candidate, ir):
        return "component graph does not connect both port domains"
    if isolated != bool(candidate.metadata.get("isolated", False)):
        return "isolation contract does not match candidate"
    if isolated and _has_cross_domain_short(candidate, ir):
        return "a non-transformer path crosses the galvanic isolation boundary"
    if not candidate.metadata.get("solver"):
        return "no physics solver is bound to this production"
    unreachable = _unreachable_ratio_reason(candidate, ir)
    if unreachable is not None:
        return unreachable
    return None


def _unreachable_ratio_reason(candidate: TopologyCandidate, ir: UnifiedIR) -> str | None:
    """Reject a candidate whose family cannot produce the requested ratio.

    Applicability in the knowledge base is deliberately coarse -- ``step_up`` or
    ``step_down`` -- which says nothing about whether the *target* ratio is
    reachable.  A boost cannot serve unity gain, and a 10x step-up needs a duty
    cycle above the usable band.  Those candidates used to be constructed and
    then fail at optimisation time, which wastes the search and reports a
    confusing failure; the family's own ratio function answers this directly.

    A family that declares no ratio is never rejected on a claim it did not
    make, so undeclared models keep their previous behaviour.
    """

    from .power import stage_model_by_solver_id

    stage = stage_model_by_solver_id(str(candidate.metadata.get("solver")))
    if stage is None:
        return None
    target = ir.primary_target
    vin = float(target.get("input_voltage_v") or ir.operating_point.get("supply_voltage_v") or 0.0)
    vout = float(target.get("output_voltage_v") or 0.0)
    if vin <= 0.0 or vout <= 0.0:
        return None
    required_ratio = vout / vin
    if stage.can_reach_ratio(required_ratio):
        return None
    bounds = stage.reachable_ratio_range()
    span = f"{bounds[0]:.4g}..{bounds[1]:.4g}" if bounds is not None else "unknown"
    return (
        f"{stage.family} cannot produce a conversion ratio of {required_ratio:.4g} "
        f"inside its usable duty band (reachable {span})"
    )


def _ports_connected(candidate: TopologyCandidate, ir: UnifiedIR) -> bool:
    adjacency: dict[str, set[str]] = {}
    for component in candidate.components:
        nodes = component.nodes
        for index, first in enumerate(nodes):
            for second in nodes[index + 1 :]:
                adjacency.setdefault(first, set()).add(second)
                adjacency.setdefault(second, set()).add(first)
    start = ir.ports[0].positive
    pending = [start]
    seen = {start}
    while pending:
        node = pending.pop()
        for nxt in adjacency.get(node, set()):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return ir.ports[1].positive in seen


def _has_cross_domain_short(candidate: TopologyCandidate, ir: UnifiedIR) -> bool:
    input_nodes = {ir.ports[0].positive, ir.ports[0].negative}
    output_nodes = {ir.ports[1].positive, ir.ports[1].negative}
    adjacency: dict[str, set[str]] = {}
    for component in candidate.components:
        if component.kind == "ideal_transformer":
            continue
        if len(component.nodes) != 2:
            continue
        a, b = component.nodes
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    pending = list(input_nodes)
    seen = set(input_nodes)
    while pending:
        node = pending.pop()
        for nxt in adjacency.get(node, set()):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return bool(seen & output_nodes)


def _candidate_signature(candidate: TopologyCandidate) -> str:
    if candidate.graph is None:
        raise ValueError(f"candidate {candidate.name!r} has no CircuitGraph")
    return candidate.graph.topology_hash


def _ensure_candidate_graph(candidate: TopologyCandidate, ir: UnifiedIR) -> TopologyCandidate:
    if candidate.graph is not None:
        return candidate
    from dataclasses import replace
    from .graph import topology_candidate_to_graph

    return replace(candidate, graph=topology_candidate_to_graph(candidate, ir))


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping.get(key) or 0.0)
    if value <= 0:
        raise UnsupportedTopology(f"DC target {key} must be positive")
    return value


def _port_by_role(ir: UnifiedIR, role: str, fallback):
    for port in ir.ports:
        if port.role == role:
            return port
    return fallback
