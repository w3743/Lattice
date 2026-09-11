from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .formatting import db20
from .proposers import HeuristicProposer, TopologyProposer
from .spec import SynthesisSpec
from .targets import target_from_behavior
from .templates import CircuitTemplate, default_templates


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_feature(row: dict[str, Any], use_phase: bool = True) -> np.ndarray:
    mag = np.asarray(row["magnitude_db"], dtype=np.float64)
    mag = np.clip(mag, -160.0, 80.0)
    if not use_phase or "phase_deg" not in row:
        return mag

    phase = np.asarray(row["phase_deg"], dtype=np.float64)
    phase = np.unwrap(np.deg2rad(phase))
    phase = np.clip(phase / np.pi, -8.0, 8.0)
    return np.concatenate([mag, phase])


def target_feature(spec: SynthesisSpec, frequency_hz: np.ndarray, use_phase: bool = True) -> np.ndarray:
    target = target_from_behavior(spec.behavior, frequency_hz, spec.analysis)
    mag = np.clip(db20(target.values), -160.0, 80.0)
    if not use_phase:
        return mag
    phase = np.unwrap(np.angle(target.values))
    phase = np.clip(phase / np.pi, -8.0, 8.0)
    return np.concatenate([mag, phase])


def train_template_classifier(
    rows: list[dict[str, Any]],
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 1000,
    seed: int = 31,
    use_phase: bool = True,
    validation_fraction: float = 0.2,
    top_k: int = 3,
) -> dict[str, Any]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning

    if not rows:
        raise ValueError("cannot train learned proposer without rows")

    first_freq = np.asarray(rows[0]["frequency_hz"], dtype=float)
    for row in rows:
        if len(row["frequency_hz"]) != len(first_freq):
            raise ValueError("all training rows must use the same frequency grid")

    x = np.vstack([row_feature(row, use_phase=use_phase) for row in rows])
    y = np.asarray([row["template"] for row in rows])
    split = group_holdout_indices(
        y,
        sample_group_keys(rows),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_idx, validation_idx = split.train, split.validation
    validation_metrics: dict[str, Any] = {
        "validation_fraction": validation_fraction,
        "validation_rows": len(validation_idx),
        "validation_accuracy": None,
        "validation_top_k_accuracy": None,
    }
    evaluation_warnings = 0
    if len(validation_idx) > 0 and len(train_idx) > 0:
        eval_model, evaluation_warnings = _fit_mlp_classifier(
            x[train_idx],
            y[train_idx],
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            seed=seed,
            warning_category=ConvergenceWarning,
        )
        validation_metrics.update(
            {
                "validation_accuracy": float(eval_model.score(x[validation_idx], y[validation_idx])),
                "validation_top_k_accuracy": top_k_accuracy(
                    eval_model,
                    x[validation_idx],
                    y[validation_idx],
                    top_k,
                ),
            }
        )

    model, convergence_warnings = _fit_mlp_classifier(
        x,
        y,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        seed=seed,
        warning_category=ConvergenceWarning,
    )
    return {
        "model": model,
        "frequency_hz": first_freq,
        "use_phase": use_phase,
        "classes": sorted(set(y.tolist())),
        "training_accuracy": float(model.score(x, y)),
        "training_top_k_accuracy": top_k_accuracy(model, x, y, top_k),
        "training_rows": len(rows),
        "training_class_counts": _class_counts(y),
        "validation": validation_metrics,
        "validation_accuracy": validation_metrics["validation_accuracy"],
        "validation_top_k_accuracy": validation_metrics["validation_top_k_accuracy"],
        "top_k": min(int(top_k), len(set(y.tolist()))),
        "convergence_warnings": convergence_warnings,
        "evaluation_convergence_warnings": evaluation_warnings,
        **split.as_dict(),
    }


def _fit_mlp_classifier(
    x: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple[int, ...],
    max_iter: int,
    seed: int,
    warning_category,
):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=seed,
                    solver="lbfgs",
                    alpha=1e-4,
                    tol=1e-3,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", warning_category)
        model.fit(x, y)
    convergence_warnings = [
        warning for warning in caught if issubclass(warning.category, warning_category)
    ]
    return model, len(convergence_warnings)


GROUP_KEY_SOURCE = "label.spec_name | problem_id | row_index"
SPEC_GROUPED_STRATEGY = "spec_grouped"
ALL_TRAINING_STRATEGY = "all_training"


@dataclass(frozen=True)
class HoldoutSplit:
    """A deterministic holdout split that never splits a group across sides.

    For replay data the group is ``label.spec_name``: candidates produced for the same
    requirement are correlated (identical target, shared search trace), so a row-wise
    split leaks the spec from training into validation.  ``strategy`` is
    ``spec_grouped`` when a grouped validation set was actually held out and
    ``all_training`` when it was not (no validation requested, or fewer than two
    groups); the validation metrics of an ``all_training`` split are explicitly absent
    instead of being fabricated.
    """

    train: np.ndarray
    validation: np.ndarray
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    strategy: str

    @property
    def spec_grouped(self) -> bool:
        return self.strategy == SPEC_GROUPED_STRATEGY

    @classmethod
    def all_training(cls, groups: Sequence[str] = ()) -> "HoldoutSplit":
        return cls(
            train=np.asarray([], dtype=int),
            validation=np.asarray([], dtype=int),
            train_groups=tuple(sorted(groups)),
            validation_groups=(),
            strategy=ALL_TRAINING_STRATEGY,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_strategy": self.strategy,
            "group_key": GROUP_KEY_SOURCE,
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
        }


def sample_group_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the split-group key of every row.

    ``label.spec_name`` is the requirement identity and therefore the group; rows that
    only expose ``problem_id`` use that.  A row carrying neither has no spec to leak, so
    it becomes its own group (independent sample).
    """
    keys: list[str] = []
    for index, row in enumerate(rows):
        label = row.get("label")
        spec_name = label.get("spec_name") if isinstance(label, Mapping) else None
        key = str(spec_name or row.get("problem_id") or "").strip()
        keys.append(key or f"sample:{index}")
    return keys


def group_holdout_indices(
    labels: np.ndarray,
    groups: Sequence[str],
    validation_fraction: float = 0.2,
    seed: int = 31,
) -> HoldoutSplit:
    """Split ``labels`` into train/validation while keeping every group on one side.

    Groups are ordered by name and then shuffled with ``seed``, so the result depends
    only on the inputs and the seed -- never on dict/insertion order.  At least one
    group always stays in training; with fewer than two groups nothing is held out.
    """
    label_array = np.asarray(labels)
    group_array = np.asarray([str(item) for item in groups], dtype=object)
    if len(label_array) != len(group_array):
        raise ValueError("labels and groups must have the same length")
    if validation_fraction >= 1:
        raise ValueError("validation_fraction must be < 1")

    ordered = sorted(set(group_array.tolist()))
    count = len(label_array)
    if validation_fraction <= 0 or len(ordered) < 2:
        return HoldoutSplit(
            train=np.arange(count, dtype=int),
            validation=np.asarray([], dtype=int),
            train_groups=tuple(ordered),
            validation_groups=(),
            strategy=ALL_TRAINING_STRATEGY,
        )

    members: dict[str, list[int]] = {group: [] for group in ordered}
    for index, group in enumerate(group_array.tolist()):
        members[group].append(index)

    shuffled = [ordered[position] for position in np.random.default_rng(seed).permutation(len(ordered))]
    target = min(max(1, int(round(count * float(validation_fraction)))), count - 1)
    validation_groups: list[str] = []
    validation_count = 0
    for group in shuffled[:-1]:  # the last group in the draw order always trains
        validation_groups.append(group)
        validation_count += len(members[group])
        if validation_count >= target:
            break

    validation_set = set(validation_groups)
    validation = np.asarray(
        sorted(index for group in validation_groups for index in members[group]), dtype=int
    )
    train = np.asarray(
        sorted(index for group in ordered if group not in validation_set for index in members[group]),
        dtype=int,
    )
    return HoldoutSplit(
        train=train,
        validation=validation,
        train_groups=tuple(group for group in ordered if group not in validation_set),
        validation_groups=tuple(validation_groups),
        strategy=SPEC_GROUPED_STRATEGY,
    )


def top_k_accuracy(model, x: np.ndarray, y: np.ndarray, top_k: int = 3) -> float | None:
    if len(y) == 0:
        return None
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    k = max(1, min(int(top_k), len(classes)))
    top = np.argsort(probabilities, axis=1)[:, -k:]
    hits = [label in classes[top[index]] for index, label in enumerate(y)]
    return float(np.mean(hits))


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))}


def save_model(bundle: dict[str, Any], path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str | Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


@dataclass
class SklearnTemplateProposer:
    """Learned frequency-response-to-template proposer.

    The model ranks topology classes from target port behavior.  The physics
    loop still optimizes and verifies each candidate, so bad guesses are merely
    expensive, not trusted.
    """

    model_path: str | Path
    fallback: TopologyProposer | None = None
    top_n: int = 8

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        bundle = load_model(self.model_path)
        model = bundle["model"]
        freqs = np.asarray(bundle["frequency_hz"], dtype=float)
        feature = target_feature(spec, freqs, use_phase=bool(bundle.get("use_phase", True)))
        template_by_name = {template.name: template for template in default_templates()}

        probabilities = model.predict_proba(feature.reshape(1, -1))[0]
        classes = list(model.classes_)
        ranked_names = [
            name for _, name in sorted(zip(probabilities, classes), key=lambda item: item[0], reverse=True)
        ]

        behavior_kind = str(spec.behavior["kind"]).strip().lower()
        proposed: list[CircuitTemplate] = []
        seen: set[str] = set()
        for name in ranked_names[: self.top_n]:
            template = template_by_name.get(name)
            if template is None or name in seen:
                continue
            if template.is_compatible(spec.library, spec.optimization.max_components, behavior_kind):
                proposed.append(template)
                seen.add(name)

        fallback = self.fallback or HeuristicProposer()
        for template in fallback.propose(spec):
            if template.name not in seen:
                proposed.append(template)
                seen.add(template.name)

        return proposed
