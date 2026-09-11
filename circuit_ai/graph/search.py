"""Open-ended CircuitGraph beam search with a replayable certificate."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement
from typing import Any, Iterable, Mapping

from ..experts import TopologyCandidate
from ..ir import IRPort, UnifiedIR
from .actions import (
    AddComponent,
    AddFeedback,
    ConnectTerminal,
    GraphAction,
    GraphActionError,
    OpenTerminal,
    action_payload,
    apply_graph_action,
)
from .canonical import exact_isomorphism_key
from .model import (
    CircuitGraph,
    ComponentInstance,
    GraphDomain,
    GraphPort,
    Net,
    ParameterBinding,
    PortTerminal,
)
from .primitives import primitive_model_ref


@dataclass(frozen=True)
class GraphSearchPolicy:
    """Hard limits and grammar knobs for one structural search run."""

    allowed_model_kinds: tuple[str, ...] = ("R", "C", "L")
    required_model_kinds: tuple[str, ...] = ()
    min_components: int = 2
    max_components: int = 4
    max_nodes: int = 6
    max_depth: int = 8
    beam_width: int = 24
    target_count: int = 8
    max_expansions: int = 1_000
    max_simulations: int = 0
    allow_parallel: bool = True
    allow_input_ground: bool = True
    variable_parameters: bool = True
    parameter_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    grammar_version: str = "circuit_graph_actions.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_model_kinds", tuple(str(item) for item in self.allowed_model_kinds))
        object.__setattr__(self, "required_model_kinds", tuple(str(item) for item in self.required_model_kinds))
        if self.min_components < 0 or self.max_components < self.min_components:
            raise ValueError("invalid component bounds")
        if self.max_nodes < 1 or self.max_depth < 0 or self.beam_width < 1:
            raise ValueError("graph search budgets must be positive")


@dataclass(frozen=True)
class SearchActionRecord:
    action: dict[str, Any]
    accepted: bool
    reason: str = ""
    graph_hash: str | None = None
    parent_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": dict(self.action),
            "accepted": self.accepted,
            "reason": self.reason,
            "graph_hash": self.graph_hash,
            "parent_hash": self.parent_hash,
        }


@dataclass(frozen=True)
class GraphSearchState:
    graph: CircuitGraph
    open_terminals: tuple[OpenTerminal, ...] = ()
    unmet_requirements: tuple[str, ...] = ()
    derivation: tuple[SearchActionRecord, ...] = ()
    depth: int = 0
    resource_usage: Mapping[str, int] = field(default_factory=dict)
    available_solver_ids: tuple[str, ...] = ()
    source_family: str = "open_skeleton"

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_terminals", tuple(sorted(set(self.open_terminals), key=_open_key)))
        object.__setattr__(self, "resource_usage", dict(self.resource_usage))

    @classmethod
    def from_graph(
        cls,
        graph: CircuitGraph,
        *,
        open_terminals: Iterable[OpenTerminal] = (),
        source_family: str = "expert",
    ) -> "GraphSearchState":
        graph.require_valid()
        return cls(
            graph=graph,
            open_terminals=tuple(open_terminals),
            source_family=source_family,
            available_solver_ids=graph.preferred_solver_ids,
        )

    @property
    def topology_hash(self) -> str:
        return self.graph.topology_hash

    @property
    def exact_key(self) -> str:
        components = {item.instance_id: item for item in self.graph.components}
        nets = {item.net_id: item for item in self.graph.nets}
        open_key = tuple(
            sorted(
                (
                    components[item.component_id].model.model_id,
                    tuple(sorted(terminal.name for terminal in components[item.component_id].model.terminals)),
                    item.terminal_name,
                    nets[item.net_id].kind,
                    nets[item.net_id].is_reference,
                )
                for item in self.open_terminals
            )
        )
        return exact_isomorphism_key(self.graph) + ":open=" + repr(open_key)

    @property
    def design_components(self) -> tuple[ComponentInstance, ...]:
        return tuple(
            component
            for component in self.graph.components
            if component.model.kind not in {"voltage_source", "current_source"}
        )

    @property
    def is_complete(self) -> bool:
        return self._is_complete(GraphSearchPolicy())

    def complete_for(self, policy: GraphSearchPolicy) -> bool:
        return self._is_complete(policy)

    def apply(self, action: GraphAction, policy: GraphSearchPolicy | None = None) -> "GraphSearchState":
        updated_graph, updated_open = apply_graph_action(
            self.graph,
            self.open_terminals,
            action,
            policy=policy,
        )
        usage = dict(self.resource_usage)
        usage["actions"] = usage.get("actions", 0) + 1
        if isinstance(action, AddComponent):
            priority = action.attributes.get("search_priority", 0)
            if isinstance(priority, (int, float)):
                usage["search_priority"] = usage.get("search_priority", 0) + int(priority)
        record = SearchActionRecord(
            action=action_payload(action),
            accepted=True,
            graph_hash=updated_graph.topology_hash,
            parent_hash=self.graph.topology_hash,
        )
        return replace(
            self,
            graph=updated_graph,
            open_terminals=updated_open,
            derivation=self.derivation + (record,),
            depth=self.depth + 1,
            resource_usage=usage,
            available_solver_ids=updated_graph.preferred_solver_ids,
        )

    def validate(self, policy: GraphSearchPolicy | None = None):
        """Validate both the schema graph and state-level search budgets."""

        report = self.graph.validate()
        if not report.passed:
            self.graph.require_valid()
        if policy is not None:
            if len(self.design_components) > policy.max_components:
                raise GraphActionError("component_budget", "state exceeds max_components")
            if len(self.graph.nets) > policy.max_nodes:
                raise GraphActionError("node_budget", "state exceeds max_nodes")
        components = {item.instance_id for item in self.graph.components}
        if any(item.component_id not in components for item in self.open_terminals):
            raise GraphActionError("dangling_terminal", "open terminal refers to a missing component")
        return report

    def _is_complete(self, policy: GraphSearchPolicy) -> bool:
        if self.open_terminals:
            return False
        if len(self.design_components) < policy.min_components:
            return False
        present_models = {component.model.kind for component in self.design_components}
        if not set(policy.required_model_kinds).issubset(present_models):
            return False
        if len(self.graph.nets) > policy.max_nodes:
            return False
        for component in self.design_components:
            net_ids = [item.net_id for item in component.connections]
            if len(net_ids) != len(set(net_ids)):
                return False
        input_port = next((item for item in self.graph.ports if item.direction == "input"), None)
        output_port = next((item for item in self.graph.ports if item.direction == "output"), None)
        if input_port is None or output_port is None:
            return False
        if not _nets_connected(self.graph, input_port.terminals[0].net_id, output_port.terminals[0].net_id):
            return False
        if len(output_port.terminals) > 1 and not _nets_connected(
            self.graph,
            output_port.terminals[0].net_id,
            output_port.terminals[1].net_id,
        ):
            return False
        if len(input_port.terminals) > 1 and len(output_port.terminals) > 1:
            if not _nets_connected(self.graph, input_port.terminals[1].net_id, output_port.terminals[1].net_id):
                return False
        return True


@dataclass(frozen=True)
class GraphSearchCertificate:
    grammar_version: str
    expanded_states: int
    generated_actions: int
    accepted_states: int
    pruned_states: int
    duplicate_states: int
    failed_actions: int
    simulation_budget: int
    simulation_calls: int
    stop_reason: str
    records: tuple[SearchActionRecord, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "grammar_version": self.grammar_version,
            "expanded_states": self.expanded_states,
            "generated_actions": self.generated_actions,
            "accepted_states": self.accepted_states,
            "pruned_states": self.pruned_states,
            "duplicate_states": self.duplicate_states,
            "failed_actions": self.failed_actions,
            "simulation_budget": self.simulation_budget,
            "simulation_calls": self.simulation_calls,
            "stop_reason": self.stop_reason,
            "records": [item.as_dict() for item in self.records],
        }


@dataclass(frozen=True)
class GraphSearchResult:
    complete_states: tuple[GraphSearchState, ...]
    frontier: tuple[GraphSearchState, ...]
    certificate: GraphSearchCertificate

    @property
    def complete_graphs(self) -> tuple[CircuitGraph, ...]:
        return tuple(item.graph for item in self.complete_states)


class LinearGraphActionGenerator:
    """Generate a bounded but open R/C/L action vocabulary.

    Existing nets are paired with replacement, so parallel elements are
    legal.  A ``None`` terminal creates one new internal net, which also makes
    serial and feedback-like branches reachable without a template catalogue.
    """

    def __init__(self, policy: GraphSearchPolicy):
        self.policy = policy

    def actions(self, state: GraphSearchState) -> tuple[GraphAction, ...]:
        actions: list[GraphAction] = []
        nets = tuple(sorted(item.net_id for item in state.graph.nets))
        design_count = len(state.design_components)
        if design_count < self.policy.max_components:
            for kind in self.policy.allowed_model_kinds:
                model = primitive_model_ref(kind)
                instance_id = _next_component_id(state, kind)
                for first, second in combinations_with_replacement(nets, 2):
                    if first == second:
                        continue
                    if not self.policy.allow_input_ground and {first, second} == {"in", "0"}:
                        continue
                    actions.append(
                        AddComponent(
                            model=model,
                            instance_id=instance_id,
                            connections={"p": first, "n": second},
                        )
                    )
                if len(state.graph.nets) < self.policy.max_nodes and nets:
                    new_reference = nets[0]
                    actions.append(
                        AddComponent(
                            model=model,
                            instance_id=instance_id,
                            connections={"p": new_reference, "n": None},
                        )
                    )

        for terminal in state.open_terminals:
            for net in nets:
                actions.append(ConnectTerminal(terminal, net))

        # Feedback is represented as a semantic graph extension.  It is only
        # emitted when the requirement explicitly asks for it, avoiding a
        # combinatorial branch in ordinary passive synthesis.
        if _requires_feedback(state.graph):
            for component in state.design_components:
                if component.model.kind in {"vcvs", "opamp"}:
                    actions.append(AddFeedback(component.reference, "output"))
        return _dedupe_actions(actions)


class PowerGraphActionGenerator:
    """Generate typed actions for a bounded non-isolated power-stage search.

    The generator deliberately keeps the first power grammar small: every
    required model is introduced at most once, while its terminals may be
    connected to any existing net, including a reserved internal-net pool in
    the power skeleton. This is enough to discover the canonical boost wiring
    without registering the complete production as a seed. More permissive
    module fragments can be added later without changing the search state or
    certificate contract.
    """

    _parameter_defaults = {
        "ideal_switch": {"duty_cycle": 0.5, "frequency_hz": 100_000.0},
        "ideal_transformer": {"turns_ratio": 1.0},
    }

    def __init__(self, policy: GraphSearchPolicy):
        self.policy = policy

    def actions(self, state: GraphSearchState) -> tuple[GraphAction, ...]:
        if len(state.design_components) >= self.policy.max_components:
            return ()
        existing_counts: dict[str, int] = {}
        for component in state.design_components:
            existing_counts[component.model.kind] = existing_counts.get(component.model.kind, 0) + 1
        nets = tuple(sorted(item.net_id for item in state.graph.nets))
        actions: list[GraphAction] = []
        for terminal in state.open_terminals:
            # A newly allocated net is an open typed terminal until the
            # search explicitly terminates it. Keeping the self-net option
            # lets a component introduce an internal node without forcing a
            # second component to own that node.
            ordered_nets = (terminal.net_id,) + tuple(net for net in nets if net != terminal.net_id)
            actions.extend(ConnectTerminal(terminal, net) for net in ordered_nets)
        remaining_required = tuple(
            kind for kind in self.policy.required_model_kinds
            if existing_counts.get(kind, 0) == 0
        )
        kinds = remaining_required[:1] if remaining_required else self.policy.allowed_model_kinds
        for kind in kinds:
            # Required model kinds describe a single structural role in this
            # first power grammar. Optional library models remain repeatable.
            if kind in self.policy.required_model_kinds and existing_counts.get(kind, 0) >= 1:
                continue
            model = primitive_model_ref(kind)
            if len(model.terminals) != 2:
                continue
            instance_id = _next_component_id(state, kind)
            first_name, second_name = (item.name for item in model.terminals)
            for first, second in _power_connection_options(nets, self.policy):
                attributes = {"search_origin": "native_power_graph"}
                if kind == "R":
                    attributes["role"] = "external_load"
                priority = _power_action_priority(kind, {first_name: first, second_name: second}, state)
                if priority:
                    attributes["search_priority"] = priority
                actions.append(
                    AddComponent(
                        model=model,
                        instance_id=instance_id,
                        connections={first_name: first, second_name: second},
                        parameters=self._parameter_defaults.get(kind, {}),
                        attributes=attributes,
                    )
                )
        return _dedupe_actions(actions)


def initial_power_graph_state(
    ir: UnifiedIR,
    policy: GraphSearchPolicy | None = None,
    *,
    family: str = "dc_boost",
    solver_id: str = "ideal_boost_averaged",
) -> GraphSearchState:
    """Create a non-isolated DC two-port skeleton for native graph search."""

    policy = policy or GraphSearchPolicy()
    if ir.intent_kind != "dc":
        raise ValueError("native power graph search requires a DC intent")
    if len(ir.ports) != 2:
        raise ValueError("native power graph search needs exactly two ports")
    input_port = _port_by_role(ir.ports, "input", ir.ports[0])
    output_port = _port_by_role(ir.ports, "output", ir.ports[-1])
    if input_port.negative != output_port.negative:
        raise ValueError("native power graph search currently requires a shared reference net")
    reference = input_port.negative
    nodes = set(ir.node_names)
    nodes.update({input_port.positive, output_port.positive, reference, "native_internal_1"})
    domain = GraphDomain("electrical", reference_net_id=reference)
    nets = tuple(
        Net(
            net_id=node,
            name=node,
            domain_id="electrical",
            kind="reference" if node == reference else "signal",
            is_reference=node == reference,
        )
        for node in sorted(nodes)
    )
    ports = tuple(
        _ir_port_to_graph(
            port,
            reference,
            direction_override=(
                "input"
                if port.name == input_port.name
                else "output"
                if port.name == output_port.name
                else None
            ),
        )
        for port in ir.ports
    )
    graph = CircuitGraph(
        graph_id=f"{ir.name}.power.open",
        name=f"{ir.name} native power graph",
        description="Native CircuitGraph open power topology skeleton",
        family=family,
        preferred_solver_ids=(solver_id,),
        domains=(domain,),
        nets=nets,
        components=(),
        ports=ports,
        extensions={
            "search_root": True,
            "source_ir_name": ir.name,
            "native_power_search": True,
            "isolated": False,
        },
    ).require_valid()
    return GraphSearchState.from_graph(graph, source_family="native_power_skeleton")


class CircuitGraphBeamSearcher:
    """Pareto beam search over immutable native graph states."""

    def __init__(
        self,
        ir: UnifiedIR | None = None,
        *,
        policy: GraphSearchPolicy | None = None,
        seed_states: Iterable[GraphSearchState] = (),
        action_generator: LinearGraphActionGenerator | None = None,
    ):
        self.ir = ir
        self.policy = policy or GraphSearchPolicy()
        self.seed_states = tuple(seed_states)
        self.action_generator = action_generator or LinearGraphActionGenerator(self.policy)

    def search(
        self,
        initial_state: GraphSearchState | None = None,
        *,
        target_count: int | None = None,
    ) -> GraphSearchResult:
        root = initial_state or (initial_linear_graph_state(self.ir, self.policy) if self.ir else None)
        if root is None:
            raise ValueError("CircuitGraphBeamSearcher needs ir or initial_state")
        roots = (root,) + self.seed_states
        buckets: dict[str, dict[str, GraphSearchState]] = {}
        beam: list[GraphSearchState] = []
        for state in roots:
            state.graph.require_valid()
            if _register_state(state, buckets):
                beam.append(state)

        complete: list[GraphSearchState] = []
        records: list[SearchActionRecord] = []
        expanded = generated = accepted = pruned = duplicates = failed = 0
        target = target_count if target_count is not None else self.policy.target_count
        stop_reason = "frontier_empty"

        while beam:
            if len(complete) >= target:
                stop_reason = "target_count"
                break
            if expanded >= self.policy.max_expansions:
                stop_reason = "expansion_budget"
                break
            next_states: list[GraphSearchState] = []
            for state in beam:
                if expanded >= self.policy.max_expansions:
                    break
                expanded += 1
                if state.complete_for(self.policy):
                    complete.append(state)
                    continue
                if state.depth >= self.policy.max_depth:
                    records.append(
                        SearchActionRecord({}, False, "max_depth", state.topology_hash, state.topology_hash)
                    )
                    pruned += 1
                    continue
                actions = self.action_generator.actions(state)
                generated += len(actions)
                for action in actions:
                    if expanded + len(next_states) >= self.policy.max_expansions * self.policy.beam_width:
                        break
                    try:
                        candidate = state.apply(action, self.policy)
                    except GraphActionError as exc:
                        failed += 1
                        pruned += 1
                        records.append(
                            SearchActionRecord(
                                action_payload(action),
                                False,
                                exc.code,
                                state.topology_hash,
                                state.topology_hash,
                            )
                        )
                        continue
                    accepted += 1
                    if not _register_state(candidate, buckets):
                        duplicates += 1
                        records.append(
                            SearchActionRecord(
                                action_payload(action),
                                False,
                                "duplicate_graph",
                                candidate.topology_hash,
                                state.topology_hash,
                            )
                        )
                        continue
                    records.append(
                        SearchActionRecord(
                            action_payload(action),
                            True,
                            "accepted",
                            candidate.topology_hash,
                            state.topology_hash,
                        )
                    )
                    next_states.append(candidate)

            complete.extend(state for state in next_states if state.complete_for(self.policy))
            remaining = [state for state in next_states if not state.complete_for(self.policy)]
            beam = _pareto_beam(remaining, self.policy.beam_width)
            if not beam:
                stop_reason = "target_count" if len(complete) >= target else "frontier_empty"
            elif expanded >= self.policy.max_expansions:
                stop_reason = "expansion_budget"

        complete = _unique_complete(complete)
        certificate = GraphSearchCertificate(
            grammar_version=self.policy.grammar_version,
            expanded_states=expanded,
            generated_actions=generated,
            accepted_states=accepted,
            pruned_states=pruned,
            duplicate_states=duplicates,
            failed_actions=failed,
            simulation_budget=self.policy.max_simulations,
            simulation_calls=0,
            stop_reason=stop_reason,
            records=tuple(records),
        )
        return GraphSearchResult(tuple(complete[:target]), tuple(beam), certificate)


class ParetoBeamSearch(CircuitGraphBeamSearcher):
    """Short public name for the native graph search implementation."""


def initial_linear_graph_state(
    ir: UnifiedIR,
    policy: GraphSearchPolicy | None = None,
) -> GraphSearchState:
    """Create the open two-port skeleton and its fixed AC excitation."""

    policy = policy or GraphSearchPolicy()
    if ir.intent_kind == "dc":
        raise ValueError("native linear graph search currently supports AC analyses only")
    if not ir.ports:
        raise ValueError("native graph search needs at least one declared port")
    input_port = _port_by_role(ir.ports, "input", ir.ports[0])
    output_port = _port_by_role(ir.ports, "output", ir.ports[-1])
    reference = input_port.negative
    nodes = set(ir.node_names)
    nodes.update({input_port.positive, reference, output_port.positive, output_port.negative})
    domain = GraphDomain("electrical", reference_net_id=reference)
    nets = tuple(
        Net(
            net_id=node,
            name=node,
            domain_id="electrical",
            kind="reference" if node == reference else "signal",
            is_reference=node == reference,
        )
        for node in sorted(nodes)
    )
    ports = tuple(
        _ir_port_to_graph(
            port,
            reference,
            direction_override=(
                "input"
                if port.name == input_port.name
                else "output"
                if port.name == output_port.name
                else None
            ),
        )
        for port in ir.ports
    )
    graph = CircuitGraph(
        graph_id=f"{ir.name}.open",
        name=f"{ir.name} open graph",
        description="Native CircuitGraph open topology skeleton",
        family="open_linear_graph",
        preferred_solver_ids=("linear_mna",),
        domains=(domain,),
        nets=nets,
        components=(),
        ports=ports,
        extensions={"search_root": True, "source_ir_name": ir.name},
    ).require_valid()
    state = GraphSearchState.from_graph(graph, source_family="open_skeleton")
    source = AddComponent(
        model=primitive_model_ref("voltage_source"),
        instance_id="Vin",
        connections={"p": input_port.positive, "n": reference},
        parameters={"value": 1.0},
        attributes={"role": "analysis_excitation"},
    )
    state = state.apply(source, policy)
    return replace(state, derivation=state.derivation, depth=0)


def _ir_port_to_graph(
    port: IRPort,
    reference: str,
    *,
    direction_override: str | None = None,
) -> GraphPort:
    terminal_names = ("positive", "reference") if len(port.terminals) == 2 else tuple(
        f"terminal_{index + 1}" for index in range(len(port.terminals))
    )
    terminals = tuple(
        PortTerminal(
            name=name,
            net_id=node,
            quantity="voltage" if index == 0 else "ground" if node == reference else "voltage",
            role="potential" if index == 0 else "reference",
        )
        for index, (name, node) in enumerate(zip(terminal_names, port.terminals))
    )
    return GraphPort(
        port_id=port.port_id or port.name,
        name=port.name,
        direction=direction_override
        or (port.role if port.role in {"input", "output", "bidirectional"} else "bidirectional"),
        domain_id="electrical",
        terminals=terminals,
    )


def _port_by_role(ports: tuple[IRPort, ...], role: str, fallback: IRPort) -> IRPort:
    return next((port for port in ports if port.role == role), fallback)


def _nets_connected(graph: CircuitGraph, start: str, end: str) -> bool:
    if start == end:
        return True
    adjacency: dict[str, set[str]] = {}
    for component in graph.components:
        connections = [item.net_id for item in component.connections]
        for first in connections:
            adjacency.setdefault(first, set()).update(item for item in connections if item != first)
    pending = [start]
    visited = {start}
    while pending:
        current = pending.pop()
        if current == end:
            return True
        for neighbor in adjacency.get(current, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return False


def _register_state(state: GraphSearchState, buckets: dict[str, dict[str, GraphSearchState]]) -> bool:
    bucket = buckets.setdefault(state.topology_hash, {})
    if state.exact_key in bucket:
        return False
    bucket[state.exact_key] = state
    return True


def _pareto_beam(states: list[GraphSearchState], width: int) -> list[GraphSearchState]:
    if not states:
        return []
    remaining = list(states)
    selected: list[GraphSearchState] = []
    while remaining and len(selected) < width:
        front = [
            state
            for state in remaining
            if not any(_dominates(other, state) for other in remaining if other is not state)
        ]
        front.sort(key=_state_sort_key)
        chosen = front[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _dominates(first: GraphSearchState, second: GraphSearchState) -> bool:
    first_values = _objectives(first)
    second_values = _objectives(second)
    return all(a <= b for a, b in zip(first_values, second_values)) and any(
        a < b for a, b in zip(first_values, second_values)
    )


def _objectives(state: GraphSearchState) -> tuple[int, int, int, int, int]:
    priority = int(state.resource_usage.get("search_priority", 0))
    if state.source_family != "native_power_skeleton":
        return (
            len(state.open_terminals),
            _port_distance(state),
            len(state.design_components),
            len(state.graph.nets),
            0,
        )
    return (
        len(state.open_terminals),
        -priority,
        _port_distance(state),
        len(state.design_components),
        len(state.graph.nets),
    )


def _port_distance(state: GraphSearchState) -> int:
    input_port = next((item for item in state.graph.ports if item.direction == "input"), None)
    output_port = next((item for item in state.graph.ports if item.direction == "output"), None)
    if input_port is None or output_port is None:
        return 1
    return 0 if _nets_connected(state.graph, input_port.terminals[0].net_id, output_port.terminals[0].net_id) else 1


def _state_sort_key(state: GraphSearchState) -> tuple[Any, ...]:
    return _objectives(state) + (state.source_family, state.exact_key)


def _unique_complete(states: list[GraphSearchState]) -> list[GraphSearchState]:
    unique: dict[str, GraphSearchState] = {}
    for state in states:
        unique.setdefault(state.exact_key, state)
    return sorted(unique.values(), key=_state_sort_key)


def _next_component_id(state: GraphSearchState, kind: str) -> str:
    prefix = kind.upper()[:1]
    used = {item.instance_id for item in state.graph.components}
    index = 1
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _open_key(item: OpenTerminal) -> tuple[str, str, str]:
    return item.component_id, item.terminal_name, item.net_id


def _dedupe_actions(actions: list[GraphAction]) -> tuple[GraphAction, ...]:
    seen: set[str] = set()
    unique: list[GraphAction] = []
    for action in actions:
        key = repr(action_payload(action))
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return tuple(unique)


def _power_action_priority(
    kind: str,
    connections: Mapping[str, str | None],
    state: GraphSearchState,
) -> int:
    """Score role-compatible actions so useful native paths survive the beam."""

    input_port = next((item for item in state.graph.ports if item.direction == "input"), None)
    output_port = next((item for item in state.graph.ports if item.direction == "output"), None)
    if input_port is None or output_port is None:
        return 0
    input_net = input_port.terminals[0].net_id
    output_net = output_port.terminals[0].net_id
    reference = input_port.terminals[1].net_id if len(input_port.terminals) > 1 else None
    connected = {value for value in connections.values() if value is not None}
    if kind == "L" and input_net in connected:
        return 5 if any(
            value is not None and str(value).startswith("native_internal_")
            for value in connections.values()
        ) else 4
    inductor_nets = {
        connection.net_id
        for component in state.graph.components
        if component.model.kind == "L"
        for connection in component.connections
    }
    internal = inductor_nets - {input_net}
    if len(internal) == 1:
        switch_node = next(iter(internal))
        if kind == "ideal_switch" and switch_node in connected and reference in connected:
            return 5
        if kind == "ideal_diode" and connections.get("anode") == switch_node and connections.get("cathode") == output_net:
            return 5
    if kind in {"C", "R"} and output_net in connected and reference in connected:
        return 5
    return 0


def _power_connection_options(
    nets: tuple[str, ...],
    policy: GraphSearchPolicy,
) -> tuple[tuple[str | None, str | None], ...]:
    """Return ordered two-terminal connections over the current net pool."""

    options: list[tuple[str | None, str | None]] = []
    for first in nets:
        for second in nets:
            if first == second:
                continue
            if not policy.allow_input_ground and {first, second} == {"in", "0"}:
                continue
            options.append((first, second))
    return tuple(options)


def _requires_feedback(graph: CircuitGraph) -> bool:
    return any(
        str(item.get("kind", "")).casefold() in {"feedback", "closed_loop"}
        for item in graph.extensions.get("requirements", ())
        if isinstance(item, Mapping)
    )
