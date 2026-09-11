from __future__ import annotations

from typing import Any

import numpy as np

from .learned import HoldoutSplit, group_holdout_indices, sample_group_keys


QUALITY_BEHAVIORS = (
    "lowpass",
    "highpass",
    "bandpass",
    "samples",
    "zpk",
    "constant_impedance",
    "impedance",
    "other",
)
QUALITY_FEATURE_NAMES = (
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
    *[f"behavior_{kind}" for kind in QUALITY_BEHAVIORS],
)


def graph_quality_features(record: dict[str, Any], behavior_kind: str) -> np.ndarray:
    record = record.get("graph_record", record) if isinstance(record, dict) else {}
    slots = list(record.get("slots", []))
    if not slots:
        slots = _circuit_graph_slots(record)
    nodes: set[str] = {"in", "out", "0"}
    degrees: dict[str, int] = {}
    counts = {"R": 0, "C": 0, "L": 0}
    edge_kinds: dict[tuple[str, str], set[str]] = {}

    for slot in slots:
        kind = str(slot.get("kind", "")).upper()
        n1 = str(slot.get("n1", ""))
        n2 = str(slot.get("n2", ""))
        nodes.update([n1, n2])
        degrees[n1] = degrees.get(n1, 0) + 1
        degrees[n2] = degrees.get(n2, 0) + 1
        if kind in counts:
            counts[kind] += 1
        edge = tuple(sorted((n1, n2)))
        edge_kinds.setdefault(edge, set()).add(kind)

    def has_edge(a: str, b: str, kind: str | None = None) -> float:
        edge = tuple(sorted((a, b)))
        if kind is None:
            return float(edge in edge_kinds)
        return float(kind in edge_kinds.get(edge, set()))

    behavior = _normalized_behavior(behavior_kind)
    behavior_one_hot = [1.0 if behavior == kind else 0.0 for kind in QUALITY_BEHAVIORS]
    values = [
        float(len(slots)),
        float(counts["R"]),
        float(counts["C"]),
        float(counts["L"]),
        float(sum(1 for node in nodes if node not in {"in", "out", "0"})),
        float(degrees.get("in", 0)),
        float(degrees.get("out", 0)),
        float(degrees.get("0", 0)),
        has_edge("in", "out"),
        has_edge("out", "0"),
        has_edge("in", "0"),
        has_edge("in", "out", "R"),
        has_edge("in", "out", "C"),
        has_edge("out", "0", "R"),
        has_edge("out", "0", "C"),
        *behavior_one_hot,
    ]
    return np.asarray(values, dtype=np.float64)


def _circuit_graph_slots(record: dict[str, Any]) -> list[dict[str, str]]:
    """Project native CircuitGraph components onto the legacy feature shape."""
    projected: list[dict[str, str]] = []
    for component in record.get("components", []):
        model = component.get("model", {})
        kind = str(model.get("kind", "")).upper()
        connections = component.get("connections", [])
        nets = [str(item.get("net_id", "")) for item in connections]
        if len(nets) < 2:
            continue
        projected.append({"kind": kind, "n1": nets[0], "n2": nets[-1]})
    return projected


def train_graph_quality_model(
    rows: list[dict[str, Any]],
    validation_fraction: float = 0.2,
    seed: int = 71,
) -> dict[str, Any]:
    labeled = [
        row for row in rows if isinstance(row.get("label"), dict) and "accepted" in row["label"]
    ]
    stats = graph_quality_template_stats(labeled)
    if not labeled:
        return _disabled_quality_model("no labeled replay rows", stats)

    labels = np.asarray([bool(row["label"]["accepted"]) for row in labeled], dtype=bool)
    if len(set(labels.tolist())) < 2:
        return _disabled_quality_model("quality model needs both accepted and rejected rows", stats)

    x = np.vstack(
        [
            graph_quality_features(
                row["graph"],
                str(row.get("label", {}).get("behavior_kind", "")),
            )
            for row in labeled
        ]
    )
    y = labels.astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x, y)

    validation_accuracy = None
    split = group_holdout_indices(
        y,
        sample_group_keys(labeled),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_idx, validation_idx = split.train, split.validation
    if _has_two_classes(y[train_idx]) and len(validation_idx) > 0:
        eval_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        eval_model.fit(x[train_idx], y[train_idx])
        validation_accuracy = float(eval_model.score(x[validation_idx], y[validation_idx]))

    return {
        "enabled": True,
        "kind": "graph_quality_binary_classifier",
        "model": model,
        "feature_names": list(QUALITY_FEATURE_NAMES),
        "rows": len(labeled),
        "accepted_rows": int(np.sum(y == 1)),
        "rejected_rows": int(np.sum(y == 0)),
        "training_accuracy": float(model.score(x, y)),
        "validation_accuracy": validation_accuracy,
        "template_stats": stats,
        "message": "trained replay quality model from accepted/rejected graph candidates",
        **split.as_dict(),
    }


def graph_quality_template_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for row in rows:
        name = str(row.get("template", ""))
        if not name:
            continue
        label = row.get("label", {})
        bucket = stats.setdefault(
            name,
            {
                "rows": 0,
                "accepted_rows": 0,
                "rejected_rows": 0,
                "mean_score": 0.0,
                "mean_rmse_db": 0.0,
            },
        )
        rows_seen = int(bucket["rows"])
        bucket["rows"] = rows_seen + 1
        accepted = bool(label.get("accepted", False))
        if accepted:
            bucket["accepted_rows"] = int(bucket["accepted_rows"]) + 1
        else:
            bucket["rejected_rows"] = int(bucket["rejected_rows"]) + 1
        bucket["mean_score"] = _running_mean(float(bucket["mean_score"]), rows_seen, label.get("score"))
        bucket["mean_rmse_db"] = _running_mean(
            float(bucket["mean_rmse_db"]),
            rows_seen,
            label.get("rmse_db"),
        )

    for bucket in stats.values():
        total = max(1, int(bucket["rows"]))
        bucket["acceptance_rate"] = float(int(bucket["accepted_rows"]) / total)
    return stats


def score_graph_quality(quality_model: dict[str, Any] | None, record: dict[str, Any], behavior_kind: str) -> float:
    if not quality_model:
        return 1.0
    if quality_model.get("enabled") and quality_model.get("model") is not None:
        features = graph_quality_features(record, behavior_kind).reshape(1, -1)
        model = quality_model["model"]
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        if 1 in classes:
            return float(probabilities[classes.index(1)])
    stats = quality_model.get("template_stats", {})
    name = str(record.get("name", ""))
    if name in stats:
        return float(stats[name].get("acceptance_rate", 0.5))
    return 0.5


def _disabled_quality_model(message: str, stats: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    return {
        "enabled": False,
        "kind": "graph_quality_binary_classifier",
        "model": None,
        "feature_names": list(QUALITY_FEATURE_NAMES),
        "rows": 0,
        "accepted_rows": sum(int(item["accepted_rows"]) for item in stats.values()),
        "rejected_rows": sum(int(item["rejected_rows"]) for item in stats.values()),
        "training_accuracy": None,
        "validation_accuracy": None,
        "template_stats": stats,
        "message": message,
        **HoldoutSplit.all_training().as_dict(),
    }


def _normalized_behavior(behavior_kind: str) -> str:
    behavior = behavior_kind.strip().lower()
    return behavior if behavior in QUALITY_BEHAVIORS else "other"


def _running_mean(current: float, count: int, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return current
    if not np.isfinite(numeric):
        return current
    return (current * count + numeric) / (count + 1)


def _has_two_classes(values: np.ndarray) -> bool:
    return len(set(values.tolist())) >= 2
