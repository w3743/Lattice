from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .learned import HoldoutSplit, group_holdout_indices
from .quality import graph_quality_features


RANKING_ERROR_FEATURES = (
    "rmse_db",
    "max_abs_db",
    "mean_db_error",
    "std_db_error",
    "p95_abs_db",
    "phase_rmse_deg",
    "phase_max_abs_deg",
)
RANKING_FEATURE_NAMES = (
    *RANKING_ERROR_FEATURES,
    "label_component_count",
    *[f"graph_{name}" for name in (
        "component_count",
        "count_R",
        "count_C",
        "count_L",
        "internal_node_count",
        "degree_in",
        "degree_out",
        "degree_ground",
        "has_in_out",
        "has_out_ground",
        "has_in_ground",
        "has_R_in_out",
        "has_C_in_out",
        "has_R_out_ground",
        "has_C_out_ground",
        "behavior_lowpass",
        "behavior_highpass",
        "behavior_bandpass",
        "behavior_samples",
        "behavior_zpk",
        "behavior_constant_impedance",
        "behavior_impedance",
        "behavior_other",
    )],
)


def replay_ranking_features(row: dict[str, Any]) -> np.ndarray:
    label = row.get("label", {})
    behavior_kind = str(label.get("behavior_kind", ""))
    error = _row_error_features(row)
    graph = graph_quality_features(row.get("graph_record", row.get("graph", {})), behavior_kind)
    values = [
        *[float(error.get(name, 0.0)) for name in RANKING_ERROR_FEATURES],
        float(row.get("component_count", label.get("component_count", 0))),
        *graph.tolist(),
    ]
    return np.asarray(values, dtype=np.float64)


def train_replay_pairwise_ranker(
    rows: list[dict[str, Any]],
    validation_fraction: float = 0.2,
    seed: int = 83,
    min_score_delta: float = 1e-9,
) -> dict[str, Any]:
    usable = [row for row in rows if _score(row) is not None]
    groups = _group_by_spec(usable)
    pairs_x: list[np.ndarray] = []
    pairs_y: list[int] = []
    pair_groups: list[str] = []
    for group_id, group_rows in groups.items():
        for i in range(len(group_rows)):
            for j in range(i + 1, len(group_rows)):
                score_i = _score(group_rows[i])
                score_j = _score(group_rows[j])
                if score_i is None or score_j is None or abs(score_i - score_j) <= min_score_delta:
                    continue
                fi = replay_ranking_features(group_rows[i])
                fj = replay_ranking_features(group_rows[j])
                y = 1 if score_i < score_j else 0
                pairs_x.append(fi - fj)
                pairs_y.append(y)
                pair_groups.append(group_id)
                pairs_x.append(fj - fi)
                pairs_y.append(1 - y)
                pair_groups.append(group_id)

    if not pairs_x:
        return _disabled_ranker("no comparable replay pairs")
    x = np.vstack(pairs_x)
    y = np.asarray(pairs_y, dtype=int)
    if len(set(y.tolist())) < 2:
        return _disabled_ranker("pairwise replay labels contain only one class")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ranker",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x, y)
    split = group_holdout_indices(
        y,
        pair_groups,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    validation_accuracy = _validation_accuracy(x, y, split, seed=seed)
    metrics = {
        str(k): replay_top_k_recall(usable, {"enabled": True, "model": model}, k)
        for k in (1, 3, 5)
    }
    return {
        "enabled": True,
        "kind": "replay_pairwise_ranker",
        "model": model,
        "feature_names": list(RANKING_FEATURE_NAMES),
        "rows": len(usable),
        "groups": len(groups),
        "pairs": len(y),
        "training_pairwise_accuracy": float(model.score(x, y)),
        "validation_pairwise_accuracy": validation_accuracy,
        "group_top1_accuracy": group_top1_accuracy({"enabled": True, "model": model}, usable),
        "group_top_k_recall": metrics,
        "message": "trained pairwise ranker from replay candidates grouped by spec_name",
        **split.as_dict(),
    }


def score_replay_candidate_ranker(ranker: dict[str, Any] | None, row: dict[str, Any]) -> float:
    if not ranker or not ranker.get("enabled") or ranker.get("model") is None:
        return 0.0
    model = ranker["model"]
    feature = replay_ranking_features(row).reshape(1, -1)
    try:
        return float(model.decision_function(feature)[0])
    except AttributeError:
        probability = model.predict_proba(feature)[0]
        classes = list(model.classes_)
        return float(probability[classes.index(1)]) if 1 in classes else float(probability[-1])


def rank_replay_rows_experiment(
    rows: list[dict[str, Any]],
    ranker: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply an ML score as a deterministic within-tier ranking experiment.

    Feasibility evidence remains the primary key.  This helper is deliberately
    offline and does not change the orchestrator's hard-constraint decision.
    """
    scored = []
    for index, row in enumerate(rows):
        label = row.get("label", {})
        try:
            feasibility_rank = int(label.get("feasibility_rank", _row_feasibility_rank(row)))
        except (TypeError, ValueError):
            feasibility_rank = _row_feasibility_rank(row)
        score = score_replay_candidate_ranker(ranker, row)
        scored.append((
            feasibility_rank,
            -score,
            _label_float(label, "score"),
            str(row.get("candidate_id", row.get("template", ""))),
            index,
            row,
        ))
    ordered = sorted(scored, key=lambda item: item[:-1])
    return [
        {
            **row,
            "experimental_rank": position,
            "experimental_rank_score": float(-sort_item[1]),
        }
        for position, sort_item in enumerate(ordered)
        for row in [sort_item[-1]]
    ]


def group_top1_accuracy(ranker: dict[str, Any], rows: list[dict[str, Any]]) -> float | None:
    groups = _group_by_spec([row for row in rows if _score(row) is not None])
    comparable = [group for group in groups.values() if len(group) >= 2]
    if not comparable:
        return None
    hits = 0
    for group in comparable:
        predicted = max(group, key=lambda row: score_replay_candidate_ranker(ranker, row))
        actual = min(group, key=lambda row: float(_score(row)))
        hits += int(predicted is actual)
    return float(hits / len(comparable))


def replay_top_k_recall(
    rows: list[dict[str, Any]],
    ranker: dict[str, Any] | None,
    top_k: int = 3,
) -> dict[str, float | int | None]:
    """Measure fixed-budget recall of groups containing a verified candidate."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    groups = _group_by_spec([row for row in rows if _score(row) is not None])
    eligible = [
        group for group in groups.values()
        if any(bool(row.get("label", {}).get("accepted", False)) for row in group)
    ]
    if not eligible:
        return {
            "top_k": top_k,
            "groups": 0,
            "hit_groups": 0,
            "recall": None,
        }
    hits = 0
    for group in eligible:
        ordered = sorted(
            group,
            key=lambda row: (
                -score_replay_candidate_ranker(ranker, row),
                float(_score(row)),
                str(row.get("candidate_id", row.get("template", ""))),
            ),
        )
        hits += int(any(bool(row.get("label", {}).get("accepted", False)) for row in ordered[:top_k]))
    return {
        "top_k": top_k,
        "groups": len(eligible),
        "hit_groups": hits,
        "recall": float(hits / len(eligible)),
    }


def _validation_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    split: HoldoutSplit,
    *,
    seed: int,
) -> float | None:
    if len(y) < 4 or not split.spec_grouped:
        return None
    train = split.train
    validation = split.validation
    if len(train) < 2 or len(validation) == 0:
        return None
    if len(set(y[train].tolist())) < 2:
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ranker",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x[train], y[train])
    return float(model.score(x[validation], y[validation]))


def _row_error_features(row: dict[str, Any]) -> dict[str, float]:
    error = row.get("error_features")
    if isinstance(error, dict):
        return {name: float(error.get(name, 0.0)) for name in RANKING_ERROR_FEATURES}
    label = row.get("label", {})
    return {
        "rmse_db": _label_float(label, "rmse_db"),
        "max_abs_db": _label_float(label, "max_abs_db"),
        "mean_db_error": 0.0,
        "std_db_error": 0.0,
        "p95_abs_db": _label_float(label, "max_abs_db"),
        "phase_rmse_deg": _label_float(label, "phase_rmse_deg"),
        "phase_max_abs_deg": _label_float(label, "phase_rmse_deg"),
    }


def _group_by_spec(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row.get("label", {})
        group_id = label.get("spec_name") or row.get("problem_id") or row.get("domain")
        groups[str(group_id or "")].append(row)
    return {key: value for key, value in groups.items() if key and len(value) >= 2}


def _score(row: dict[str, Any]) -> float | None:
    label = row.get("label", {})
    try:
        value = float(label.get("score", label.get("rmse_db")))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _label_float(label: dict[str, Any], key: str) -> float:
    try:
        value = float(label.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _disabled_ranker(message: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "kind": "replay_pairwise_ranker",
        "model": None,
        "feature_names": list(RANKING_FEATURE_NAMES),
        "rows": 0,
        "groups": 0,
        "pairs": 0,
        "training_pairwise_accuracy": None,
        "validation_pairwise_accuracy": None,
        "group_top1_accuracy": None,
        "group_top_k_recall": {},
        "message": message,
        **HoldoutSplit.all_training().as_dict(),
    }


def _row_feasibility_rank(row: dict[str, Any]) -> int:
    label = row.get("label", {})
    evidence = str(label.get("evidence_status", label.get("feasibility", "indeterminate")))
    return {
        "verified": 0,
        "verified_feasible": 0,
        "known_infeasible": 1,
        "indeterminate": 2,
        "execution_failed": 3,
    }.get(evidence, 2)
