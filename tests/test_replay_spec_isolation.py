"""Replay training/validation split must keep one spec on one side (plan §18.2 / §12).

Requirement (``计划:1857``, §12 数据准入): candidates produced for the same PBDL spec are
highly correlated (same requirement, same search trace), so the replay train/validation
split is grouped by ``label.spec_name`` -- a candidate's spec may never appear in both
splits.

The observable contract asserted here:

* ``circuit_ai.learned.group_holdout_indices`` keeps every group on one side, is
  deterministic for a given seed, and degrades to an explicit all-training split (with
  no validation metrics invented) when fewer than two groups exist.
* ``train_from_replay`` records that grouping in the bundle
  (``split_strategy``/``group_key``/``train_groups``/``validation_groups``), and the
  recorded train/validation group sets are disjoint for the classifier, quality and
  ranking models.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit_ai import learned, replay
from circuit_ai.learned import (
    ALL_TRAINING_STRATEGY,
    GROUP_KEY_SOURCE,
    SPEC_GROUPED_STRATEGY,
    group_holdout_indices,
)


SPEC_IDS = ("spec_alpha", "spec_beta", "spec_gamma")
ROWS_PER_SPEC = 3


def _replay_row(spec_name: str, candidate_id: str, accepted: bool, score: float) -> dict:
    return {
        "schema": "circuit_ai.design_replay",
        "schema_version": 1,
        "problem_id": spec_name,
        "domain": "ac",
        "candidate_id": candidate_id,
        "name": candidate_id,
        "template": "rc_lowpass",
        "graph": {
            "slots": [
                {"name": "X1", "kind": "R", "n1": "in", "n2": "out"},
                {"name": "X2", "kind": "C", "n1": "out", "n2": "0"},
            ],
            "output_node": "out",
            "source_name": "Vin",
        },
        "graph_record": {
            "slots": [
                {"name": "X1", "kind": "R", "n1": "in", "n2": "out"},
                {"name": "X2", "kind": "C", "n1": "out", "n2": "0"},
            ],
            "output_node": "out",
            "source_name": "Vin",
        },
        "parameters": {"X1": 1000.0, "X2": 1e-6},
        "frequency_hz": [10.0, 100.0, 1000.0, 10000.0, 100000.0],
        "magnitude_db": [-0.1, -0.2, -3.0, -20.0, -40.0],
        "target": {"frequencies_hz": [10.0, 100.0, 1000.0, 10000.0, 100000.0]},
        "label": {
            "accepted": accepted,
            "score": score,
            "rmse_db": score,
            "max_abs_db": score,
            "feasibility_rank": 0 if accepted else 3,
            "evidence_status": "verified" if accepted else "execution_failed",
            "spec_name": spec_name,
            "behavior_kind": "lowpass",
        },
    }


def _rows() -> list[dict]:
    rows: list[dict] = []
    for spec_index, spec_name in enumerate(SPEC_IDS):
        for candidate_index, (accepted, score) in enumerate(
            ((True, 0.1), (False, 1.0), (False, 4.0))
        ):
            rows.append(
                _replay_row(
                    spec_name,
                    f"{spec_name}-{candidate_index}",
                    accepted,
                    score + spec_index,
                )
            )
    return rows


def _specs_of(rows: list[dict]) -> np.ndarray:
    return np.asarray([str(row["label"]["spec_name"]) for row in rows])


def test_group_holdout_never_splits_a_spec_across_sides() -> None:
    rows = _rows()
    specs = _specs_of(rows)
    labels = np.asarray([row["label"]["accepted"] for row in rows], dtype=int)

    split = group_holdout_indices(labels, list(specs), validation_fraction=0.34, seed=17)

    assert split.strategy == SPEC_GROUPED_STRATEGY
    assert split.spec_grouped is True
    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert len(split.train) + len(split.validation) == len(rows)
    train_specs = set(specs[split.train].tolist())
    validation_specs = set(specs[split.validation].tolist())
    assert train_specs & validation_specs == set()
    assert train_specs | validation_specs == set(SPEC_IDS)
    assert set(split.train_groups) | set(split.validation_groups) == set(SPEC_IDS)


def test_group_holdout_is_deterministic_and_seed_dependent() -> None:
    rows = _rows()
    specs = list(_specs_of(rows))
    labels = np.asarray([row["label"]["accepted"] for row in rows], dtype=int)

    first = group_holdout_indices(labels, specs, validation_fraction=0.34, seed=17)
    same = group_holdout_indices(labels, specs, validation_fraction=0.34, seed=17)
    other = group_holdout_indices(labels, specs, validation_fraction=0.34, seed=18)

    assert sorted(first.validation_groups) == sorted(same.validation_groups)
    assert first.train.tolist() == same.train.tolist()
    # A different seed explores a different holdout, still spec-disjoint.
    other_specs = set(_specs_of(rows)[other.train].tolist())
    assert other_specs & set(_specs_of(rows)[other.validation].tolist()) == set()


def test_group_holdout_degrades_without_two_specs() -> None:
    rows = [row for row in _rows() if row["label"]["spec_name"] == "spec_alpha"]
    labels = np.asarray([row["label"]["accepted"] for row in rows], dtype=int)

    split = group_holdout_indices(
        labels, [row["label"]["spec_name"] for row in rows], validation_fraction=0.5, seed=17
    )

    assert split.strategy == ALL_TRAINING_STRATEGY
    assert split.spec_grouped is False
    assert len(split.validation) == 0
    assert len(split.train) == len(rows)
    assert split.validation_groups == ()


def test_sample_group_keys_fall_back_to_problem_id_then_row_identity() -> None:
    rows = _rows()
    keys = learned.sample_group_keys(rows)

    assert keys == [row["label"]["spec_name"] for row in rows]

    without_spec = [{"problem_id": "spec_x"}, {"problem_id": "spec_x"}, {}]
    assert learned.sample_group_keys(without_spec) == ["spec_x", "spec_x", "sample:2"]


def test_train_from_replay_records_spec_disjoint_holdout(tmp_path) -> None:
    rows = _rows()
    specs = _specs_of(rows)
    replay_path = tmp_path / "replay.jsonl"
    replay.write_jsonl(rows, replay_path)

    bundle = replay.train_from_replay(
        replay_path,
        tmp_path / "replay_ranker.joblib",
        accepted_only=False,
        resample_points=None,
        max_iter=80,
        seed=17,
    )

    holdout = bundle["training_source"]["holdout"]
    assert set(holdout) == {"classifier", "quality_model", "ranking_model"}
    for model_name, record in holdout.items():
        assert record["split_strategy"] == SPEC_GROUPED_STRATEGY, model_name
        assert record["group_key"] == GROUP_KEY_SOURCE, model_name
        train_groups = set(record["train_groups"])
        validation_groups = set(record["validation_groups"])
        assert validation_groups, f"{model_name} held out no spec"
        assert train_groups, f"{model_name} kept no spec for training"
        assert train_groups & validation_groups == set(), (
            f"{model_name}: spec on both sides of the holdout split: "
            f"{sorted(train_groups & validation_groups)}"
        )
        assert train_groups | validation_groups <= set(specs.tolist())

    # The bundle's own top level describes the classifier split as well.
    assert bundle["split_strategy"] == SPEC_GROUPED_STRATEGY
    assert bundle["validation_groups"] == holdout["classifier"]["validation_groups"]


def test_train_from_replay_reports_all_training_for_single_spec(tmp_path) -> None:
    rows = [row for row in _rows() if row["label"]["spec_name"] == "spec_alpha"]
    replay_path = tmp_path / "single_spec_replay.jsonl"
    replay.write_jsonl(rows, replay_path)

    bundle = replay.train_from_replay(
        replay_path,
        tmp_path / "replay_ranker.joblib",
        accepted_only=False,
        resample_points=None,
        max_iter=80,
        seed=17,
    )

    for model_name, record in bundle["training_source"]["holdout"].items():
        assert record["split_strategy"] == ALL_TRAINING_STRATEGY, model_name
        assert record["validation_groups"] == [], model_name
    # No validation metric may be invented when nothing was held out.
    assert bundle["validation"]["validation_accuracy"] is None
    assert bundle["quality_model"]["validation_accuracy"] is None
    assert bundle["ranking_model"]["validation_pairwise_accuracy"] is None


def test_holdout_draw_is_row_order_invariant_and_repeatable(tmp_path) -> None:
    """Determinism regression for the spec-grouped draw.

    The choice of spec groups must come from the sorted group names plus the seed only --
    never from dict/set iteration order, insertion order, or an unseeded RNG.  A row
    permutation therefore leaves the held-out specs unchanged, and re-running the whole
    replay trainer on the same file with the same seed records an identical split.
    """
    rows = _rows()
    specs = list(_specs_of(rows))
    labels = np.asarray([row["label"]["accepted"] for row in rows], dtype=int)

    baseline = group_holdout_indices(labels, specs, validation_fraction=0.34, seed=17)
    repeat = group_holdout_indices(labels, specs, validation_fraction=0.34, seed=17)
    assert repeat.train.tolist() == baseline.train.tolist()
    assert repeat.validation.tolist() == baseline.validation.tolist()
    assert repeat.validation_groups == baseline.validation_groups

    order = np.random.default_rng(5).permutation(len(rows))
    permuted = group_holdout_indices(
        labels[order],
        [specs[index] for index in order],
        validation_fraction=0.34,
        seed=17,
    )
    assert permuted.validation_groups == baseline.validation_groups
    assert permuted.train_groups == baseline.train_groups

    replay_path = tmp_path / "determinism_replay.jsonl"
    replay.write_jsonl(rows, replay_path)
    recorded: list[dict] = []
    for model_name in ("first.joblib", "second.joblib"):
        bundle = replay.train_from_replay(
            replay_path,
            tmp_path / model_name,
            accepted_only=False,
            resample_points=None,
            max_iter=80,
            seed=17,
        )
        recorded.append(bundle["training_source"]["holdout"])

    assert recorded[0] == recorded[1]
    assert recorded[0]["ranking_model"]["validation_groups"]
