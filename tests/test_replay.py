from __future__ import annotations

import numpy as np

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.graph_learning import SklearnGraphProposer, graph_to_record
from circuit_ai.graph_templates import ElementSlot, GraphCircuitTemplate
from circuit_ai.io import write_jsonl
from circuit_ai.ranking import rank_replay_rows_experiment, replay_top_k_recall
from circuit_ai.replay import (
    export_graph_replay_rows,
    load_training_rows_from_replay,
    resample_rows_to_common_grid,
    result_to_graph_replay_row,
    replay_error_features_from_arrays,
    summarize_replay_labels,
    train_from_replay,
)
from circuit_ai.topology import GraphSearchProposer


def _graph_spec() -> SynthesisSpec:
    return SynthesisSpec.from_dict(
        {
            "name": "replay_lowpass",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {"points": 48, "max_components": 2, "max_iterations": 12, "top_k": 2, "seed": 44},
        }
    )


def test_graph_result_exports_to_replay_and_retrains(tmp_path) -> None:
    spec = _graph_spec()
    results = CircuitSynthesizer(
        proposer=GraphSearchProposer(fallback=None, max_candidates=8, include_fallback=False)
    ).synthesize(spec)
    assert isinstance(results[0].template, GraphCircuitTemplate)

    row = result_to_graph_replay_row(results[0], spec)
    assert row is not None
    assert row["graph"]["slots"]
    assert row["target"]["magnitude_db"]
    assert row["error_features"]["rmse_db"] >= 0
    assert row["label"]["accepted"]

    replay_path = tmp_path / "replay.jsonl"
    summary = export_graph_replay_rows(results, spec, replay_path, append=False)
    assert summary.rows_written >= 1
    rows = load_training_rows_from_replay(replay_path)
    assert rows
    shorter = dict(rows[0])
    shorter["frequency_hz"] = shorter["frequency_hz"][:24]
    shorter["magnitude_db"] = shorter["magnitude_db"][:24]
    shorter["phase_deg"] = shorter["phase_deg"][:24]
    resampled = resample_rows_to_common_grid([shorter, rows[0]], points=16)
    assert all(len(row["frequency_hz"]) == 16 for row in resampled)
    assert all(len(row.get("target", {}).get("frequency_hz", [])) == 16 for row in resampled)
    assert all("error_features" in row for row in resampled)

    model_path = tmp_path / "replay_model.joblib"
    bundle = train_from_replay(
        replay_path,
        model_path,
        hidden_layer_sizes=(12,),
        max_iter=200,
        seed=22,
    )
    assert model_path.exists()
    assert bundle["training_source"]["kind"] == "replay"
    assert bundle["training_source"]["label_summary"]["total"] >= bundle["training_source"]["rows"]
    assert bundle["validation"]["validation_rows"] >= 0
    assert "ranking_model" in bundle

    proposer = SklearnGraphProposer(model_path, fallback=None, top_n=4)
    replay_result = CircuitSynthesizer(proposer=proposer).synthesize(spec)[0]
    assert isinstance(replay_result.template, GraphCircuitTemplate)
    assert replay_result.metrics.rmse_db < 0.2


def test_replay_label_summary_counts_accepted_rejected_and_unlabeled() -> None:
    summary = summarize_replay_labels(
        [
            {"label": {"accepted": True}},
            {"label": {"accepted": False}},
            {"label": {}},
            {},
        ]
    )

    assert summary == {"total": 4, "accepted": 1, "rejected": 1, "unlabeled": 2}


def test_replay_ranker_experiment_keeps_feasibility_tiers_primary() -> None:
    rows = [
        {"candidate_id": "unknown", "label": {"feasibility_rank": 2, "score": 0.0}},
        {"candidate_id": "verified", "label": {"feasibility_rank": 0, "score": 10.0}},
        {"candidate_id": "failed", "label": {"feasibility_rank": 3, "score": 0.0}},
    ]

    ranked = rank_replay_rows_experiment(rows, None)

    assert [row["candidate_id"] for row in ranked] == ["verified", "unknown", "failed"]
    assert [row["experimental_rank"] for row in ranked] == [0, 1, 2]


def test_replay_top_k_recall_is_reported_for_verified_candidates() -> None:
    rows = [
        {
            "candidate_id": "good",
            "problem_id": "recall_problem",
            "label": {"spec_name": "recall_problem", "accepted": True, "score": 0.1},
        },
        {
            "candidate_id": "bad",
            "problem_id": "recall_problem",
            "label": {"spec_name": "recall_problem", "accepted": False, "score": 5.0},
        },
    ]

    report = replay_top_k_recall(rows, None, top_k=1)

    assert report == {"top_k": 1, "groups": 1, "hit_groups": 1, "recall": 1.0}


def test_unified_replay_trains_quality_and_pairwise_models_on_mixed_domains(tmp_path) -> None:
    graph = {
        "components": [
            {
                "model": {"kind": "R"},
                "connections": [
                    {"net_id": "in"},
                    {"net_id": "out"},
                ],
            }
        ]
    }
    rows = []
    for candidate_id, domain, accepted, score, feasibility_rank, evidence_status in (
        ("ac-good", "ac", True, 0.1, 0, "verified"),
        ("dc-known-bad", "dc", False, 2.0, 1, "known_infeasible"),
        ("ac-spice-failed", "ac", False, 4.0, 3, "execution_failed"),
    ):
        rows.append(
            {
                "schema": "circuit_ai.design_replay",
                "schema_version": 1,
                "problem_id": "mixed_problem",
                "domain": domain,
                "candidate_id": candidate_id,
                "name": candidate_id,
                "graph": graph,
                "parameters": {},
                "label": {
                    "accepted": accepted,
                    "score": score,
                    "feasibility_rank": feasibility_rank,
                    "evidence_status": evidence_status,
                    "spec_name": "mixed_problem",
                },
            }
        )

    replay_path = tmp_path / "mixed_design_replay.jsonl"
    write_jsonl(rows, replay_path)
    bundle = train_from_replay(
        replay_path,
        tmp_path / "mixed_ranker.joblib",
        accepted_only=False,
        resample_points=None,
        max_iter=80,
        seed=17,
    )

    assert bundle["quality_model"]["enabled"]
    assert bundle["ranking_model"]["enabled"]
    assert bundle["ranking_model"]["groups"] == 1
    assert "1" in bundle["ranking_model"]["group_top_k_recall"]
    assert bundle["training_source"]["ranking_model_enabled"]
    assert bundle["model"] is None


def test_replay_training_uses_rejected_rows_for_quality_model_not_classifier(tmp_path) -> None:
    good_lowpass = GraphCircuitTemplate(
        (
            ElementSlot("X1", "R", "in", "out"),
            ElementSlot("X2", "C", "out", "0"),
        )
    )
    good_highpass = GraphCircuitTemplate(
        (
            ElementSlot("X1", "C", "in", "out"),
            ElementSlot("X2", "R", "out", "0"),
        )
    )
    rejected_graph = GraphCircuitTemplate(
        (
            ElementSlot("X1", "R", "in", "out"),
            ElementSlot("X2", "R", "out", "0"),
        )
    )

    def row(template: GraphCircuitTemplate, accepted: bool, rmse_db: float) -> dict:
        freqs = [10.0, 100.0, 1000.0, 10000.0]
        magnitude = [-0.1, -0.2, -3.0, -20.0]
        phase = [0.0, -5.0, -45.0, -85.0]
        target_mag = [0.0, -0.1, -3.0, -20.0] if accepted else [0.0, 0.0, 0.0, 0.0]
        target_phase = [0.0, -5.0, -45.0, -85.0]
        return {
            "template": template.name,
            "graph": graph_to_record(template),
            "component_count": template.component_count,
            "parameters": {"X1": 1000.0, "X2": 1e-8},
            "frequency_hz": freqs,
            "magnitude_db": magnitude,
            "phase_deg": phase,
            "target": {
                "frequency_hz": freqs,
                "magnitude_db": target_mag,
                "phase_deg": target_phase,
                "has_phase": True,
                "analysis": {"kind": "voltage_transfer"},
            },
            "error_features": replay_error_features_from_arrays(
                np.asarray(magnitude),
                np.asarray(phase),
                np.asarray(target_mag),
                np.asarray(target_phase),
                has_phase=True,
            ),
            "label": {
                "accepted": accepted,
                "score": rmse_db,
                "rmse_db": rmse_db,
                "max_abs_db": rmse_db * 2.0,
                "phase_rmse_deg": None,
                "spec_name": "quality_replay",
                "behavior_kind": "lowpass",
            },
        }

    replay_path = tmp_path / "quality_replay.jsonl"
    write_jsonl(
        [
            row(good_lowpass, True, 0.05),
            row(good_highpass, True, 0.2),
            row(rejected_graph, False, 12.0),
        ],
        replay_path,
    )
    model_path = tmp_path / "quality_model.joblib"

    bundle = train_from_replay(
        replay_path,
        model_path,
        accepted_only=False,
        resample_points=None,
        hidden_layer_sizes=(8,),
        max_iter=80,
        seed=7,
    )

    assert bundle["quality_model"]["enabled"]
    assert bundle["quality_model"]["accepted_rows"] == 2
    assert bundle["quality_model"]["rejected_rows"] == 1
    assert bundle["ranking_model"]["enabled"]
    assert bundle["ranking_model"]["pairs"] > 0
    assert bundle["training_source"]["ranking_model_enabled"]
    assert bundle["training_source"]["classifier_accepted_only"]
    assert not bundle["training_source"]["rejected_rows_used_as_positive_classes"]

    proposed = SklearnGraphProposer(model_path, fallback=None, top_n=3).propose(_graph_spec())
    assert proposed
    assert all(isinstance(template, GraphCircuitTemplate) for template in proposed)
