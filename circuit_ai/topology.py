from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from .graph_templates import ElementSlot, GraphCircuitTemplate
from .graph_templates import NativeCircuitGraphTemplate
from .ir import IRPort, UnifiedIR
from .graph import GraphSearchPolicy, ParetoBeamSearch
from .spec import LibrarySpec, SynthesisSpec
from .structure import analyze_slots
from .templates import CircuitTemplate


LINEAR_KINDS = ("R", "C", "L")


def generate_linear_graph_templates(
    library: LibrarySpec,
    max_components: int,
    internal_nodes: int = 1,
    limit: int = 64,
    behavior_kind: str = "*",
    min_components: int = 2,
) -> list[GraphCircuitTemplate]:
    """Generate small legal R/C/L two-port graph candidates.

    This is a deliberately conservative simple-graph enumerator.  It avoids
    parallel elements and voltage-source shorts in the first version; richer
    graph grammars can later replace it behind the same output type.
    """

    kinds = tuple(kind for kind in LINEAR_KINDS if kind in library.allowed)
    if not kinds:
        return []

    nodes = ("in", "out", "0") + tuple(f"n{i}" for i in range(1, internal_nodes + 1))
    edge_pool = [edge for edge in combinations(nodes, 2) if set(edge) != {"in", "0"}]
    templates: list[GraphCircuitTemplate] = []
    seen: set[tuple] = set()

    for count in range(min_components, max_components + 1):
        for edges in combinations(edge_pool, count):
            if not _is_structurally_legal(edges):
                continue
            for kind_assignment in product(kinds, repeat=count):
                slots = tuple(
                    ElementSlot(f"X{idx}", kind, n1, n2)
                    for idx, ((n1, n2), kind) in enumerate(zip(edges, kind_assignment), start=1)
                )
                report = analyze_slots(slots)
                if not report.passed:
                    continue
                key = report.canonical_key
                if key in seen:
                    continue
                seen.add(key)
                templates.append(GraphCircuitTemplate(slots))

    templates.sort(key=lambda template: _rank_template(template, behavior_kind))
    return templates[:limit]


def _is_structurally_legal(edges: tuple[tuple[str, str], ...]) -> bool:
    if not _has_passive_path(edges, "in", "out"):
        return False
    if not _has_passive_path(edges, "out", "0"):
        return False

    used_nodes = {node for edge in edges for node in edge}
    for node in used_nodes:
        if node.startswith("n") and not _has_passive_path(edges, node, "out"):
            return False
    return True


def _has_passive_path(edges: tuple[tuple[str, str], ...], start: str, end: str) -> bool:
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    pending = [start]
    seen = {start}
    while pending:
        node = pending.pop()
        if node == end:
            return True
        for nxt in adjacency.get(node, set()):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return False


def _rank_template(template: GraphCircuitTemplate, behavior_kind: str) -> tuple[int, int, str]:
    kind = behavior_kind.strip().lower()
    slots = template.slots
    score = 0

    def has(edge_kind: str, a: str, b: str) -> bool:
        edge = tuple(sorted((a, b)))
        return any(slot.kind == edge_kind and slot.normalized_edge() == edge for slot in slots)

    kinds = {slot.kind for slot in slots}
    if kind == "lowpass":
        score += 0 if has("R", "in", "out") else 8
        score += 0 if has("C", "out", "0") else 8
        score += 4 if "L" in kinds else 0
    elif kind == "highpass":
        score += 0 if has("C", "in", "out") else 8
        score += 0 if has("R", "out", "0") else 8
        score += 4 if "L" in kinds else 0
    elif kind == "bandpass":
        score += 0 if {"R", "L", "C"}.issubset(kinds) else 10
        score += 0 if has("R", "out", "0") else 3
    else:
        score += len(kinds)

    return (score, template.component_count, template.name)


@dataclass
class GraphSearchProposer:
    """Small-scale topology search proposer for generated linear graphs."""

    fallback: object | None = None
    max_candidates: int = 48
    internal_nodes: int = 1
    include_fallback: bool = True

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        behavior_kind = str(spec.behavior["kind"]).strip().lower()
        generated = generate_linear_graph_templates(
            spec.library,
            max_components=spec.optimization.max_components,
            internal_nodes=self.internal_nodes,
            limit=self.max_candidates,
            behavior_kind=behavior_kind,
        )

        if not self.include_fallback or self.fallback is None:
            return generated

        proposed: list[CircuitTemplate] = []
        seen: set[str] = set()
        for template in generated + self.fallback.propose(spec):
            if template.name in seen:
                continue
            proposed.append(template)
            seen.add(template.name)
        return proposed


@dataclass
class NativeGraphSearchProposer:
    """Expose native CircuitGraph search through ``CircuitSynthesizer``.

    This adapter is opt-in so existing template-search behavior and its
    compatibility tests remain stable while the native path becomes the
    canonical open-topology experiment.
    """

    fallback: object | None = None
    max_candidates: int = 8
    include_fallback: bool = True
    beam_width: int = 24

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        allowed = tuple(kind for kind in LINEAR_KINDS if kind in spec.library.allowed)
        required = {kind for kind in spec.library.required if kind in LINEAR_KINDS}
        if not allowed or not required.issubset(set(allowed)):
            return list(self.fallback.propose(spec)) if self.fallback is not None else []

        ir = UnifiedIR(
            name=spec.name,
            description="SynthesisSpec projected into native graph search",
            ports=(
                IRPort("input", ("in", "0"), "input", "electrical"),
                IRPort("output", ("out", "0"), "output", "electrical"),
            ),
            relations=(),
            analyses=(spec.analysis.as_dict(),),
            targets=(),
            constraints={"required_elements": sorted(required)},
            operating_point={},
            optimization={
                "graph_search": {
                    "native_enabled": True,
                    "max_candidates": self.max_candidates,
                }
            },
        )
        policy = GraphSearchPolicy(
            allowed_model_kinds=allowed,
            min_components=(
                1
                if "impedance" in spec.analysis.kind or "impedance" in spec.behavior.get("kind", "")
                else max(2, len(required))
            ),
            max_components=spec.optimization.max_components,
            max_nodes=max(3, spec.optimization.max_components + 2),
            max_depth=max(2, min(10, spec.optimization.max_components + 2)),
            beam_width=self.beam_width,
            target_count=self.max_candidates,
            max_expansions=max(100, self.max_candidates * self.beam_width * 4),
            parameter_ranges=spec.library.parameter_ranges,
        )
        search = ParetoBeamSearch(ir, policy=policy).search(target_count=self.max_candidates)
        generated = [NativeCircuitGraphTemplate(state.graph) for state in search.complete_states]
        proposed: list[CircuitTemplate] = []
        seen: set[str] = set()
        for template in generated + (
            list(self.fallback.propose(spec))
            if self.include_fallback and self.fallback is not None
            else []
        ):
            if template.name in seen:
                continue
            seen.add(template.name)
            proposed.append(template)
        return proposed
