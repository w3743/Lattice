from __future__ import annotations

from dataclasses import replace

from circuit_ai.graph import (
    AddComponent,
    AddFeedback,
    ConnectTerminal,
    GraphSearchPolicy,
    GraphSearchState,
    OpenTerminal,
    ParetoBeamSearch,
    RemoveComponent,
    ReplaceModel,
    SplitNet,
    TerminatePort,
    exact_isomorphism_key,
    initial_linear_graph_state,
    primitive_model_ref,
)
from circuit_ai.graph_templates import NativeCircuitGraphTemplate
from circuit_ai.ir import IRPort, UnifiedIR


def _ir() -> UnifiedIR:
    return UnifiedIR(
        name="native_search_test",
        description="two-port test problem",
        ports=(
            IRPort("input", ("in", "0"), "input", "electrical"),
            IRPort("output", ("out", "0"), "output", "electrical"),
        ),
        relations=(),
        analyses=({"kind": "voltage_transfer"},),
        targets=(),
        constraints={},
        operating_point={},
        optimization={},
    )


def test_typed_actions_support_parallel_input_ground_and_lifecycle() -> None:
    policy = GraphSearchPolicy(max_components=4, max_nodes=6)
    state = initial_linear_graph_state(_ir(), policy)

    state = state.apply(AddComponent(primitive_model_ref("R"), "R1"), policy)
    assert {item.terminal_name for item in state.open_terminals} == {"p", "n"}
    state = state.apply(ConnectTerminal(OpenTerminal("R1", "p", "R1_p"), "in"), policy)
    state = state.apply(ConnectTerminal(("R1", "n"), "out"), policy)
    state = state.apply(
        AddComponent(
            primitive_model_ref("R"),
            "R2",
            {"p": "in", "n": "0"},
        ),
        policy,
    )
    assert len(state.open_terminals) == 0
    assert len(state.design_components) == 2

    state = state.apply(ReplaceModel("R2", primitive_model_ref("C")), policy)
    state = state.apply(AddFeedback("R1", "output"), policy)
    state = state.apply(TerminatePort("output"), policy)
    assert "output" in state.graph.extensions["terminated_ports"]
    assert state.graph.extensions["feedback_edges"]

    state = state.apply(RemoveComponent("R2"), policy)
    assert [item.reference for item in state.design_components] == ["R1"]
    state.graph.require_valid()


def test_split_net_and_exact_key_ignore_internal_and_instance_labels() -> None:
    policy = GraphSearchPolicy(max_components=4, max_nodes=6)
    state = initial_linear_graph_state(_ir(), policy)
    state = state.apply(
        AddComponent(
            primitive_model_ref("R"),
            "R1",
            {"p": "in", "n": None},
        ),
        policy,
    )
    internal = next(item.net_id for item in state.graph.nets if item.net_id.startswith("R1_n"))
    state = state.apply(
        AddComponent(
            primitive_model_ref("R"),
            "R2",
            {"p": internal, "n": "out"},
        ),
        policy,
    )
    state = state.apply(ConnectTerminal(("R1", "n"), internal), policy)
    state = state.apply(AddComponent(primitive_model_ref("C"), "C1", {"p": "out", "n": "0"}), policy)
    state = state.apply(SplitNet("out", "out_branch", (("R2", "n"),)), policy)
    assert any(item.net_id == "out_branch" for item in state.graph.nets)
    state.graph.require_valid()

    graph = state.graph
    net_map = {item.net_id: item.net_id for item in graph.nets}
    net_map[internal] = "renamed_internal"
    components = tuple(
        replace(
            component,
            instance_id=f"Y{index}",
            reference=f"Y{index}",
            connections=tuple(
                replace(connection, net_id=net_map[connection.net_id])
                for connection in component.connections
            ),
        )
        for index, component in enumerate(reversed(graph.components))
    )
    renamed = replace(
        graph,
        graph_id="renamed",
        name="renamed",
        nets=tuple(
            replace(net, net_id=net_map[net.net_id], name=net_map[net.net_id])
            for net in graph.nets
        ),
        components=components,
    )
    assert exact_isomorphism_key(graph) == exact_isomorphism_key(renamed)


def test_pareto_beam_search_certificate_is_deterministic_and_finds_open_graphs() -> None:
    policy = GraphSearchPolicy(
        min_components=2,
        max_components=2,
        max_nodes=4,
        max_depth=3,
        beam_width=16,
        target_count=3,
        max_expansions=100,
    )
    first = ParetoBeamSearch(_ir(), policy=policy).search(target_count=3)
    second = ParetoBeamSearch(_ir(), policy=policy).search(target_count=3)
    assert first.complete_states
    assert first.certificate.stop_reason == "target_count"
    assert first.certificate.duplicate_states > 0
    assert first.certificate.as_dict() == second.certificate.as_dict()
    assert [item.exact_key for item in first.complete_states] == [
        item.exact_key for item in second.complete_states
    ]


def test_native_graph_template_materializes_variable_bindings() -> None:
    policy = GraphSearchPolicy(min_components=2, max_components=2, max_nodes=4, target_count=1)
    result = ParetoBeamSearch(_ir(), policy=policy).search(target_count=1)
    template = NativeCircuitGraphTemplate(result.complete_states[0].graph)
    values = {item.name: 1.0 for item in template.params}
    circuit = template.to_circuit(values)
    assert circuit.voltage_sources[0].name == "Vin"
    assert template.to_graph(values).validate().passed
