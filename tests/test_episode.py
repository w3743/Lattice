from __future__ import annotations

import json
from dataclasses import replace
from typing import get_args

import pytest

from circuit_ai.evidence import (
    EPISODE_SCHEMA,
    EPISODE_SCHEMA_VERSION,
    STRUCTURAL_COMPLETE,
    STRUCTURAL_NOT_APPLIED,
    STRUCTURAL_REVERTED,
    STRUCTURAL_VALID,
    DesignEpisode,
    DesignStep,
    EpisodeChainError,
    EpisodeContractError,
    decode_action,
    encode_action,
    validate_state_chain,
)
from circuit_ai.graph import (
    AddComponent,
    AddFeedback,
    AddFragment,
    ConnectTerminal,
    GraphSearchPolicy,
    LinearGraphActionGenerator,
    OpenTerminal,
    ParetoBeamSearch,
    RemoveComponent,
    ReplaceModel,
    SplitNet,
    TerminatePort,
    initial_linear_graph_state,
    initial_power_graph_state,
)
from circuit_ai.graph.actions import GraphAction
from circuit_ai.graph.primitives import primitive_model_ref
from circuit_ai.ir import pbdl_to_ir


def _dc_ir():
    return pbdl_to_ir(
        {
            "name": "episode_dc",
            "ports": [
                {
                    "name": "input",
                    "role": "input",
                    "terminals": [
                        {"name": "in", "quantity": "voltage"},
                        {"name": "0", "quantity": "ground"},
                    ],
                },
                {
                    "name": "output",
                    "role": "output",
                    "terminals": [
                        {"name": "out", "quantity": "voltage"},
                        {"name": "0", "quantity": "ground"},
                    ],
                },
            ],
            "analyses": [
                {"kind": "dc_transfer", "source_port": "input", "output_port": "output"}
            ],
            "targets": [
                {
                    "target_kind": "dc",
                    "input_voltage_v": 5.0,
                    "output_voltage_v": 10.0,
                    "output_current_a": 1.0,
                }
            ],
            "constraints": {
                "element_types": ["R", "C", "L", "ideal_switch", "ideal_diode"],
                "max_component_count": 5,
            },
        }
    )


def _ac_ir():
    return pbdl_to_ir(
        {
            "name": "episode_ac",
            "ports": [
                {
                    "name": "input",
                    "terminals": [
                        {"name": "in", "quantity": "voltage"},
                        {"name": "0", "quantity": "ground"},
                    ],
                },
                {
                    "name": "output",
                    "terminals": [
                        {"name": "out", "quantity": "voltage"},
                        {"name": "0", "quantity": "ground"},
                    ],
                },
            ],
            "analyses": [
                {"kind": "ac_sweep", "source_port": "input", "output_port": "output"}
            ],
            "targets": [
                {
                    "target_kind": "filter",
                    "filter_kind": "lowpass",
                    "cutoff_hz": 1000.0,
                    "gain": 1.0,
                    "order": 1,
                    "frequency_range_hz": [10, 100000],
                }
            ],
        }
    )


def _action_cases() -> tuple[tuple[str, GraphAction], ...]:
    fragment = initial_power_graph_state(_dc_ir(), GraphSearchPolicy()).graph
    return (
        (
            "add_component",
            AddComponent(
                model=primitive_model_ref("R"),
                instance_id="R1",
                connections={"p": "in", "n": None},
                parameters={"value": 1000.0},
                attributes={"role": "load"},
            ),
        ),
        ("add_fragment", AddFragment(fragment=fragment, prefix="frag_")),
        ("connect_terminal", ConnectTerminal(OpenTerminal("R1", "p", "R1_p"), "in")),
        ("split_net", SplitNet("out", "out_branch", (("R2", "n"),))),
        ("add_feedback", AddFeedback("X1", "output")),
        ("terminate_port", TerminatePort("in")),
        ("replace_model", ReplaceModel("R1", primitive_model_ref("C"))),
        ("remove_component", RemoveComponent("R1")),
    )


ACTION_CASES = _action_cases()


def _rich_episode() -> DesignEpisode:
    actions = dict(ACTION_CASES)
    steps = (
        DesignStep(
            step_index=0,
            state_before_hash="hash:root",
            action=actions["add_component"],
            state_after_hash="hash:r1",
            structural_status=STRUCTURAL_VALID,
            simulation_request_ids=("req:op",),
            simulation_result_ids=("res:op",),
            constraint_evaluation_ids=("cons:fit",),
            cost={"objective": 0.25, "selection_score": 0.75},
        ),
        DesignStep(
            step_index=1,
            state_before_hash="hash:r1",
            action=actions["connect_terminal"],
            state_after_hash="hash:r1_closed",
            structural_status=STRUCTURAL_COMPLETE,
            cost={"objective": 0.1},
        ),
        DesignStep(
            step_index=2,
            state_before_hash="hash:r1_closed",
            action=actions["terminate_port"],
            state_after_hash=None,
            structural_status=STRUCTURAL_REVERTED,
            failure_category="unsupported",
            cost={},
        ),
    )
    return DesignEpisode(
        episode_id="episode:test",
        spec_hash="spec:test",
        environment_manifest_hash="env:test",
        initial_state_hash="hash:root",
        steps=steps,
        final_graph_ids=("episode_dc.native",),
        termination_reason="budget_exhausted",
        budget={"max_expansions": 200.0, "beam_width": 8.0},
        random_seed=17,
    )


def _searched_certificate(*, max_nodes: int = 4, beam_width: int = 12, max_expansions: int = 200):
    policy = GraphSearchPolicy(
        allowed_model_kinds=("R", "C"),
        required_model_kinds=("R", "C"),
        min_components=2,
        max_components=2,
        max_nodes=max_nodes,
        max_depth=3,
        beam_width=beam_width,
        target_count=2,
        max_expansions=max_expansions,
        grammar_version="episode_test.v1",
    )
    ir = _ac_ir()
    result = ParetoBeamSearch(
        ir=ir,
        policy=policy,
        action_generator=LinearGraphActionGenerator(policy),
    ).search(initial_linear_graph_state(ir, policy))
    return result.certificate


def test_episode_survives_json_round_trip_with_stable_hash() -> None:
    episode = _rich_episode()

    wire = json.loads(json.dumps(episode.as_dict(), allow_nan=False))
    restored = DesignEpisode.from_dict(wire)

    assert restored == episode
    assert restored.episode_hash == episode.episode_hash
    assert restored.schema == EPISODE_SCHEMA
    assert restored.schema_version == EPISODE_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("label", "action"),
    ACTION_CASES,
    ids=[case[0] for case in ACTION_CASES],
)
def test_every_graph_action_round_trips(label: str, action: GraphAction) -> None:
    wire = json.loads(json.dumps(encode_action(action), allow_nan=False))

    assert decode_action(wire) == action, label


def test_action_codec_covers_the_whole_action_union() -> None:
    assert {type(action).__name__ for _, action in ACTION_CASES} == {
        item.__name__ for item in get_args(GraphAction)
    }


def test_unknown_action_type_is_rejected_explicitly() -> None:
    payload = encode_action(dict(ACTION_CASES)["remove_component"])
    payload["type"] = "FinishGraph"

    with pytest.raises(EpisodeContractError, match="unknown graph action type"):
        decode_action(payload)


def test_state_chain_validation_accepts_rejected_steps_and_names_a_break() -> None:
    episode = _rich_episode()
    validate_state_chain(episode)
    assert episode.steps[2].applied is False
    assert episode.steps[2].failure_category == "unsupported"

    broken = replace(
        episode,
        steps=(episode.steps[0], replace(episode.steps[1], state_before_hash="hash:other")),
    )
    with pytest.raises(EpisodeChainError) as error:
        validate_state_chain(broken)

    assert error.value.step_index == 1
    assert error.value.expected == "hash:r1"
    assert error.value.actual == "hash:other"


def test_failure_paths_survive_round_trip() -> None:
    episode = replace(
        _rich_episode(),
        steps=(
            replace(
                _rich_episode().steps[2],
                structural_status=STRUCTURAL_NOT_APPLIED,
                simulation_request_ids=("req:unsupported",),
            ),
        ),
    )

    restored = DesignEpisode.from_dict(json.loads(json.dumps(episode.as_dict())))

    step = restored.steps[0]
    assert step.state_after_hash is None
    assert step.failure_category == "unsupported"
    assert step.structural_status == STRUCTURAL_NOT_APPLIED
    assert step.simulation_request_ids == ("req:unsupported",)
    assert restored.termination_reason == "budget_exhausted"
    assert restored.budget == {"max_expansions": 200.0, "beam_width": 8.0}


def test_certificate_rejected_actions_become_failure_steps_with_their_reason() -> None:
    certificate = _searched_certificate()
    rejected = [record for record in certificate.records if not record.accepted and record.action]
    assert rejected, "search log must contain at least one rejected action"
    assert all(record.reason for record in rejected)
    assert all(record.parent_hash for record in rejected)

    episode = DesignEpisode.from_search_certificate(
        certificate,
        spec_hash="spec:ac",
        environment_manifest_hash="env:test",
        initial_state_hash=certificate.records[0].parent_hash,
    )

    failure_steps = [step for step in episode.steps if step.failure_category]
    assert failure_steps
    assert {step.failure_category for step in failure_steps} == {record.reason for record in rejected}
    for step in failure_steps:
        assert step.state_after_hash is None
        assert step.structural_status == STRUCTURAL_NOT_APPLIED

    accepted = [record for record in certificate.records if record.accepted and record.action]
    assert [
        repr(encode_action(step.action))
        for step in episode.steps
        if not step.failure_category
    ] == [repr(record_action(record.action)) for record in accepted]
    assert episode.steps[0].state_before_hash == certificate.records[0].parent_hash


def record_action(payload: dict) -> dict:
    """The certificate shape of an action, rebuilt from its encoded payload."""

    return encode_action(decode_action(payload))


def test_unknown_fields_are_preserved_instead_of_dropped() -> None:
    payload = json.loads(json.dumps(_rich_episode().as_dict()))
    payload["producer_annotation"] = {"tool": "episode_test", "revision": 3}
    payload["steps"][0]["producer_step_note"] = "kept"

    episode = DesignEpisode.from_dict(payload)

    assert episode.extensions["unknown_top_level_fields"]["producer_annotation"] == {
        "tool": "episode_test",
        "revision": 3,
    }
    assert episode.steps[0].extensions["unknown_step_fields"]["producer_step_note"] == "kept"
    assert episode.steps[0].state_after_hash == "hash:r1"
    assert episode.steps[0].action == _rich_episode().steps[0].action
