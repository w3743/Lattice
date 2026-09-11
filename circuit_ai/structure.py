from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from .graph_templates import ElementSlot


EXTERNAL_NODES = ("0", "in", "out")


@dataclass(frozen=True)
class StructureIssue:
    severity: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class StructureReport:
    passed: bool
    canonical_key: tuple[tuple[str, str, str], ...]
    issues: tuple[StructureIssue, ...]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "canonical_key": ["|".join(item) for item in self.canonical_key],
            "issues": [issue.as_dict() for issue in self.issues],
        }

    @property
    def errors(self) -> tuple[StructureIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[StructureIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def analyze_slots(slots: Iterable[ElementSlot]) -> StructureReport:
    slots = tuple(slots)
    issues: list[StructureIssue] = []
    if not slots:
        issues.append(StructureIssue("error", "empty_graph", "graph has no elements"))

    nodes = _nodes(slots)
    edges = tuple((slot.n1, slot.n2) for slot in slots)
    adjacency = _adjacency(edges)

    for slot in slots:
        if slot.n1 == slot.n2:
            issues.append(
                StructureIssue("error", "self_loop", f"{slot.name} connects {slot.n1} to itself")
            )
        if set((slot.n1, slot.n2)) == {"in", "0"}:
            issues.append(
                StructureIssue(
                    "error",
                    "source_shunt",
                    f"{slot.name} directly shunts the ideal input voltage source",
                )
            )

    parallel_same_kind = _parallel_same_kind(slots)
    for kind, n1, n2 in parallel_same_kind:
        issues.append(
            StructureIssue(
                "warning",
                "parallel_same_kind",
                f"parallel {kind} elements exist between {n1} and {n2}; they can be merged",
            )
        )

    if not _has_path(adjacency, "in", "out"):
        issues.append(StructureIssue("error", "no_input_output_path", "no passive path from input to output"))
    if not _has_path(adjacency, "out", "0"):
        issues.append(StructureIssue("error", "no_output_ground_path", "no passive path from output to ground"))

    connected_to_main = _reachable(adjacency, "out")
    for node in nodes:
        if node not in connected_to_main:
            issues.append(
                StructureIssue("error", "disconnected_node", f"{node} is disconnected from the output network")
            )

    degrees = {node: len(adjacency.get(node, set())) for node in nodes}
    for node, degree in degrees.items():
        if _is_internal(node) and degree <= 1:
            issues.append(
                StructureIssue(
                    "error",
                    "dangling_internal_node",
                    f"{node} is a dangling internal node; its branch is electrically ineffective or singular",
                )
            )
        if _is_internal(node) and degree == 2:
            incident = [slot.kind for slot in slots if node in (slot.n1, slot.n2)]
            if len(incident) == 2 and incident[0] == incident[1]:
                issues.append(
                    StructureIssue(
                        "warning",
                        "series_same_kind",
                        f"{node} connects two series {incident[0]} elements that may be compressible",
                    )
                )

    canonical = canonical_key(slots)
    passed = not any(issue.severity == "error" for issue in issues)
    return StructureReport(passed=passed, canonical_key=canonical, issues=tuple(issues))


def canonical_key(slots: Iterable[ElementSlot]) -> tuple[tuple[str, str, str], ...]:
    slots = tuple(slots)
    internal_nodes = sorted({node for slot in slots for node in (slot.n1, slot.n2) if _is_internal(node)})
    if not internal_nodes:
        return _key_with_mapping(slots, {})

    canonical_internal = [f"n{idx}" for idx in range(1, len(internal_nodes) + 1)]
    best: tuple[tuple[str, str, str], ...] | None = None
    for permuted in permutations(canonical_internal):
        mapping = dict(zip(internal_nodes, permuted))
        key = _key_with_mapping(slots, mapping)
        if best is None or key < best:
            best = key
    return best or ()


def is_structurally_valid(slots: Iterable[ElementSlot]) -> bool:
    return analyze_slots(slots).passed


def _key_with_mapping(
    slots: tuple[ElementSlot, ...],
    mapping: dict[str, str],
) -> tuple[tuple[str, str, str], ...]:
    items = []
    for slot in slots:
        n1 = mapping.get(slot.n1, slot.n1)
        n2 = mapping.get(slot.n2, slot.n2)
        a, b = sorted((n1, n2))
        items.append((slot.kind, a, b))
    return tuple(sorted(items))


def _nodes(slots: tuple[ElementSlot, ...]) -> set[str]:
    nodes = set(EXTERNAL_NODES)
    for slot in slots:
        nodes.update([slot.n1, slot.n2])
    return nodes


def _adjacency(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def _has_path(adjacency: dict[str, set[str]], start: str, end: str) -> bool:
    return end in _reachable(adjacency, start)


def _reachable(adjacency: dict[str, set[str]], start: str) -> set[str]:
    pending = [start]
    seen = {start}
    while pending:
        node = pending.pop()
        for nxt in adjacency.get(node, set()):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return seen


def _is_internal(node: str) -> bool:
    return node not in EXTERNAL_NODES


def _parallel_same_kind(slots: tuple[ElementSlot, ...]) -> set[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    duplicates: set[tuple[str, str, str]] = set()
    for slot in slots:
        n1, n2 = sorted((slot.n1, slot.n2))
        key = (slot.kind, n1, n2)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates
