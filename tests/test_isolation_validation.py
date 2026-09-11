"""Isolation-domain integrity tests (plan §18.2 隔离域非法短接在仿真前拒绝测试).

Requirement: a conductive path that bridges two electrically isolated domains must be
rejected *before* simulation, with a stable, assertable issue code, while a legitimate
isolated graph (whose only cross-domain element is the isolation device itself) must be
accepted.  See ``计划:1853`` and ``circuit_ai/knowledge/power_topologies.yaml:156``
("functional blocks cross the declared isolation boundary only through a transformer").
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_ai.graph import (
    CircuitGraph,
    ComponentInstance,
    GraphDomain,
    GraphPort,
    GraphValidationError,
    Net,
    ParameterBinding,
    PortTerminal,
    TerminalConnection,
    primitive_model_ref,
)
from circuit_ai.simulation import LinearMNASimulatorBackend


ISOLATION_BRIDGE_CODE = "isolation_short"


def _resistor(reference: str, first: str, second: str) -> ComponentInstance:
    return ComponentInstance(
        instance_id=reference,
        reference=reference,
        model=primitive_model_ref("R", 2),
        connections=(TerminalConnection("p", first), TerminalConnection("n", second)),
        parameters=(ParameterBinding("value", "ohm", 1000.0),),
    )


def _isolated_graph(
    *,
    extra_components: tuple[ComponentInstance, ...] = (),
    include_barrier: bool = True,
) -> CircuitGraph:
    """Two galvanically isolated domains, each with its own reference net.

    The only legitimate cross-domain element of an isolated topology is the isolation
    barrier itself (``ideal_transformer``); ``extra_components`` lets a test inject a
    conductive path that violates that rule.
    """
    barrier = (
        ComponentInstance(
            instance_id="T1",
            reference="T1",
            model=primitive_model_ref("ideal_transformer", 4),
            connections=(
                TerminalConnection("primary_p", "pri_a"),
                TerminalConnection("primary_n", "in_0"),
                TerminalConnection("secondary_p", "sec_a"),
                TerminalConnection("secondary_n", "out_0"),
            ),
        ),
    ) if include_barrier else ()
    return CircuitGraph(
        graph_id="isolated_pair",
        name="isolated_pair",
        description="Two isolated domains joined only through the isolation barrier",
        family="isolated_flyback",
        domains=(
            GraphDomain("input_domain", reference_net_id="in_0", isolated_from=("output_domain",)),
            GraphDomain("output_domain", reference_net_id="out_0", isolated_from=("input_domain",)),
        ),
        nets=(
            Net("in_0", "in_0", "input_domain", kind="reference", is_reference=True),
            Net("in", "in", "input_domain"),
            Net("pri_a", "pri_a", "input_domain"),
            Net("out_0", "out_0", "output_domain", kind="reference", is_reference=True),
            Net("out", "out", "output_domain"),
            Net("sec_a", "sec_a", "output_domain"),
        ),
        components=(
            *barrier,
            _resistor("Rbias_in", "pri_a", "in"),
            _resistor("Rbias_out", "sec_a", "out"),
        )
        + extra_components,
        ports=(
            GraphPort(
                port_id="input",
                name="input",
                direction="input",
                domain_id="input_domain",
                terminals=(
                    PortTerminal("positive", "in", "voltage", "potential"),
                    PortTerminal("reference", "in_0", "ground", "reference"),
                ),
            ),
            GraphPort(
                port_id="output",
                name="output",
                direction="output",
                domain_id="output_domain",
                terminals=(
                    PortTerminal("positive", "out", "voltage", "potential"),
                    PortTerminal("reference", "out_0", "ground", "reference"),
                ),
            ),
        ),
    )


def _codes(graph: CircuitGraph) -> set[str]:
    return {issue.code for issue in graph.validate().issues}


def test_legitimate_isolated_graph_validates_through_the_barrier() -> None:
    graph = _isolated_graph()

    report = graph.validate()

    assert report.passed is True
    assert report.errors == ()
    assert graph.require_valid() is graph
    assert {domain.domain_id for domain in graph.domains} == {"input_domain", "output_domain"}


def test_reference_short_across_isolation_domains_is_rejected_before_simulation() -> None:
    graph = _isolated_graph(extra_components=(_resistor("Rbridge", "in_0", "out_0"),))

    report = graph.validate()

    assert report.passed is False
    assert ISOLATION_BRIDGE_CODE in {issue.code for issue in report.errors}
    with pytest.raises(GraphValidationError) as excinfo:
        graph.require_valid()
    assert ISOLATION_BRIDGE_CODE in {issue.code for issue in excinfo.value.report.errors}


def test_signal_path_across_isolation_domains_is_rejected_before_simulation() -> None:
    graph = _isolated_graph(extra_components=(_resistor("Rleak", "in", "out"),))

    assert _codes(graph) >= {ISOLATION_BRIDGE_CODE}
    assert graph.validate().passed is False


def test_linear_backend_refuses_isolation_bridge_instead_of_simulating() -> None:
    graph = _isolated_graph(
        extra_components=(_resistor("Rbridge", "in_0", "out_0"),),
        include_barrier=False,
    )

    with pytest.raises(GraphValidationError):
        LinearMNASimulatorBackend().compile(graph)


def test_linear_backend_compiles_multi_domain_graph_without_bridge() -> None:
    # No barrier here: every element stays inside one domain, so the gate must not
    # blanket-reject multi-domain graphs and the linear backend must accept them.
    graph = _isolated_graph(include_barrier=False)

    assert graph.validate().passed is True
    compiled = LinearMNASimulatorBackend().compile(graph)

    assert compiled.graph.graph_hash == graph.graph_hash
    assert {item.reference for item in compiled.graph.components} == {
        "Rbias_in",
        "Rbias_out",
    }


def test_barrier_is_the_only_legitimate_cross_domain_component() -> None:
    graph = _isolated_graph()

    spanning = [
        component.reference
        for component in graph.components
        if len(
            {
                net.domain_id
                for connection in component.connections
                for net in graph.nets
                if net.net_id == connection.net_id
            }
        )
        > 1
    ]

    assert spanning == ["T1"]
