from __future__ import annotations

from circuit_ai.graph_templates import ElementSlot, GraphCircuitTemplate
from circuit_ai.structure import analyze_slots, canonical_key
from circuit_ai.topology import generate_linear_graph_templates
from circuit_ai.spec import LibrarySpec


def test_structure_accepts_basic_rc_lowpass() -> None:
    slots = (
        ElementSlot("X1", "R", "in", "out"),
        ElementSlot("X2", "C", "out", "0"),
    )
    report = analyze_slots(slots)

    assert report.passed
    assert report.errors == ()
    assert report.canonical_key == (("C", "0", "out"), ("R", "in", "out"))


def test_structure_rejects_dangling_internal_branch() -> None:
    slots = (
        ElementSlot("X1", "R", "in", "out"),
        ElementSlot("X2", "C", "out", "0"),
        ElementSlot("X3", "R", "out", "n1"),
    )
    report = analyze_slots(slots)

    assert not report.passed
    assert any(issue.code == "dangling_internal_node" for issue in report.errors)


def test_canonical_key_ignores_internal_node_renaming() -> None:
    a = (
        ElementSlot("X1", "R", "in", "n1"),
        ElementSlot("X2", "C", "n1", "out"),
        ElementSlot("X3", "R", "out", "0"),
    )
    b = (
        ElementSlot("X1", "R", "in", "n7"),
        ElementSlot("X2", "C", "n7", "out"),
        ElementSlot("X3", "R", "out", "0"),
    )

    assert canonical_key(a) == canonical_key(b)


def test_graph_generator_filters_dangling_internal_branches() -> None:
    templates = generate_linear_graph_templates(
        LibrarySpec.from_dict({"allowed": ["R", "C"]}),
        max_components=3,
        internal_nodes=1,
        behavior_kind="lowpass",
        limit=64,
    )

    assert templates
    assert all(template.structural_report().passed for template in templates)
    assert not any(
        isinstance(template, GraphCircuitTemplate)
        and any(slot.n1 == "n1" or slot.n2 == "n1" for slot in template.slots)
        and sum(1 for slot in template.slots if "n1" in (slot.n1, slot.n2)) == 1
        for template in templates
    )
