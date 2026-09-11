"""Validated-data surrogate models for fast circuit response screening.

The surrogate is deliberately an advisory oracle.  It reports an ensemble
uncertainty and refuses to claim trust outside the parameter domain represented
by validated training rows.  High-fidelity simulation remains the authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import write_jsonl


VALIDATED_STATUSES = frozenset({"passed", "verified", "validated"})


@dataclass(frozen=True)
class SurrogatePrediction:
    template: str
    frequency_hz: np.ndarray
    response: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    uncertainty_db: float
    uncertainty_phase_deg: float
    in_domain: bool
    trusted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "frequency_hz": self.frequency_hz.tolist(),
            "magnitude_db": self.magnitude_db.tolist(),
            "phase_deg": self.phase_deg.tolist(),
            "uncertainty_db": self.uncertainty_db,
            "uncertainty_phase_deg": self.uncertainty_phase_deg,
            "in_domain": self.in_domain,
            "trusted": self.trusted,
            "reason": self.reason,
        }


def row_is_validated(row: dict[str, Any]) -> bool:
    """Return whether a row carries explicit high-fidelity validation evidence."""
    for key in ("validation", "spice_verification"):
        evidence = row.get(key)
        if isinstance(evidence, dict):
            if bool(evidence.get("validated")):
                return True
            if str(evidence.get("status", "")).casefold() in VALIDATED_STATUSES:
                return True
    provenance = row.get("provenance")
    return isinstance(provenance, dict) and bool(provenance.get("validated"))


def result_to_surrogate_row(result, validation: Any | None = None, *, source: str = "synthesis") -> dict[str, Any]:
    """Convert a synthesized result into a response row with validation provenance."""
    validation_data = _as_dict(validation)
    trace = _trace_from_validation(validation)
    if trace is None:
        frequency_hz = np.asarray(result.target.frequencies_hz, dtype=float)
        magnitude_db = _db20(result.response)
        phase_deg = np.rad2deg(np.unwrap(np.angle(result.response)))
    else:
        frequency_hz, magnitude_db, phase_deg = trace
    row: dict[str, Any] = {
        "template": result.template.name,
        "parameters": {str(key): float(value) for key, value in result.parameters.items()},
        "frequency_hz": frequency_hz.tolist(),
        "magnitude_db": magnitude_db.tolist(),
        "phase_deg": phase_deg.tolist(),
        "provenance": {
            "source": source,
            "validated": row_is_validated({"validation": validation_data}) if validation_data else False,
        },
    }
    if validation_data:
        row["validation"] = validation_data
    return row


def export_verified_surrogate_rows(
    results: Iterable[Any],
    validations: Iterable[Any],
    path: str | Path,
    *,
    append: bool = True,
) -> dict[str, int | str]:
    """Export only successful validations that contain a simulator response trace."""
    result_list = list(results)
    validation_list = list(validations)
    if len(result_list) != len(validation_list):
        raise ValueError("results and validations must have the same length")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for result, validation in zip(result_list, validation_list):
        evidence = _as_dict(validation)
        if str(evidence.get("status", "")).casefold() not in VALIDATED_STATUSES:
            skipped += 1
            continue
        if _trace_from_validation(validation) is None:
            raise ValueError("a passed validation must expose a response trace")
        rows.append(result_to_surrogate_row(result, validation, source="external_spice"))
    target = Path(path)
    if append and target.exists():
        with target.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    else:
        write_jsonl(rows, target)
    return {
        "rows_written": len(rows),
        "validated_rows": len(rows),
        "skipped_rows": skipped,
        "path": str(target),
    }


def export_surrogate_rows(
    results: Iterable[Any],
    path: str | Path,
    validations: Iterable[Any] | None = None,
    *,
    append: bool = True,
) -> dict[str, int | str]:
    """Persist response rows and preserve which rows have truth validation."""
    result_list = list(results)
    validation_list = list(validations) if validations is not None else []
    rows = [
        result_to_surrogate_row(
            result,
            validation_list[index] if index < len(validation_list) else None,
        )
        for index, result in enumerate(result_list)
    ]
    target = Path(path)
    if append and target.exists():
        with target.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    else:
        write_jsonl(rows, target)
    return {
        "rows_written": len(rows),
        "validated_rows": sum(row_is_validated(row) for row in rows),
        "path": str(target),
    }


def load_surrogate_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_response_surrogate(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    n_estimators: int = 64,
    min_samples_leaf: int = 2,
    seed: int = 41,
    require_validated: bool = True,
    trust_uncertainty_db: float = 1.0,
) -> dict[str, Any]:
    """Train one uncertainty-aware ensemble per topology from response rows."""
    from sklearn.ensemble import ExtraTreesRegressor

    if not rows:
        raise ValueError("cannot train a surrogate without rows")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if require_validated:
        rows = [row for row in rows if row_is_validated(row)]
        if not rows:
            raise ValueError("no explicitly validated rows available for surrogate training")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        template = str(row.get("template", "")).strip()
        if not template:
            raise ValueError("surrogate row is missing template")
        grouped.setdefault(template, []).append(row)

    models: dict[str, dict[str, Any]] = {}
    for template, template_rows in sorted(grouped.items()):
        first_frequency = _frequency_grid(template_rows[0])
        parameter_names = tuple(sorted(str(key) for key in template_rows[0].get("parameters", {})))
        if not parameter_names:
            raise ValueError(f"surrogate rows for {template!r} have no parameters")
        features: list[list[float]] = []
        targets: list[np.ndarray] = []
        for row in template_rows:
            frequency = _frequency_grid(row)
            if not np.array_equal(frequency, first_frequency):
                raise ValueError(f"surrogate rows for {template!r} must share one frequency grid")
            parameters = row.get("parameters", {})
            if set(str(key) for key in parameters) != set(parameter_names):
                raise ValueError(f"surrogate rows for {template!r} must share parameter names")
            feature = _log_parameter_vector(parameters, parameter_names)
            magnitude = np.asarray(row.get("magnitude_db", []), dtype=float)
            phase = np.asarray(row.get("phase_deg", []), dtype=float)
            if len(magnitude) != len(first_frequency) or len(phase) != len(first_frequency):
                raise ValueError(f"surrogate row for {template!r} has a response length mismatch")
            target = np.concatenate([magnitude, phase])
            if not np.all(np.isfinite(feature)) or not np.all(np.isfinite(target)):
                continue
            features.append(feature)
            targets.append(target)
        if len(features) < 4:
            raise ValueError(f"topology {template!r} needs at least four finite validated rows")

        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        train_indices, validation_indices = _holdout_indices(len(x), validation_fraction, seed)
        estimator = _fit_ensemble(
            ExtraTreesRegressor,
            x[train_indices],
            y[train_indices],
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
        )
        validation_metrics: dict[str, float | int | None] = {
            "rows": int(len(validation_indices)),
            "magnitude_rmse_db": None,
            "phase_rmse_deg": None,
        }
        if len(validation_indices):
            predicted = estimator.predict(x[validation_indices])
            split = len(first_frequency)
            validation_metrics["magnitude_rmse_db"] = float(
                np.sqrt(np.mean((predicted[:, :split] - y[validation_indices, :split]) ** 2))
            )
            validation_metrics["phase_rmse_deg"] = float(
                np.rad2deg(np.sqrt(np.mean(np.deg2rad(predicted[:, split:] - y[validation_indices, split:]) ** 2)))
            )

        models[template] = {
            "model": estimator,
            "frequency_hz": first_frequency,
            "parameter_names": parameter_names,
            "input_min": np.min(x, axis=0),
            "input_max": np.max(x, axis=0),
            "training_rows": int(len(x)),
            "validation": validation_metrics,
        }

    return {
        "kind": "validated_response_surrogate",
        "version": 1,
        "models": models,
        "require_validated": require_validated,
        "trust_uncertainty_db": float(trust_uncertainty_db),
        "training_rows": int(sum(item["training_rows"] for item in models.values())),
    }


def predict_response(
    bundle: dict[str, Any],
    template: str,
    parameters: dict[str, float],
    frequency_hz: Iterable[float] | None = None,
    *,
    trust_uncertainty_db: float | None = None,
) -> SurrogatePrediction:
    """Predict a response and return an explicit trust decision."""
    if template not in bundle.get("models", {}):
        raise KeyError(f"surrogate has no topology {template!r}")
    record = bundle["models"][template]
    parameter_names = tuple(record["parameter_names"])
    feature = _log_parameter_vector(parameters, parameter_names).reshape(1, -1)
    estimator = record["model"]
    tree_predictions = np.asarray([tree.predict(feature)[0] for tree in estimator.estimators_])
    predicted = np.mean(tree_predictions, axis=0)
    uncertainty = np.std(tree_predictions, axis=0)
    native_frequency = np.asarray(record["frequency_hz"], dtype=float)
    split = len(native_frequency)
    magnitude = predicted[:split]
    phase = predicted[split:]
    magnitude_uncertainty = uncertainty[:split]
    phase_uncertainty = uncertainty[split:]
    requested_frequency = native_frequency if frequency_hz is None else np.asarray(list(frequency_hz), dtype=float)
    if not np.all(np.isfinite(requested_frequency)) or np.any(requested_frequency <= 0):
        raise ValueError("frequency_hz must contain finite positive values")
    if not np.array_equal(requested_frequency, native_frequency):
        source_log = np.log10(native_frequency)
        target_log = np.log10(requested_frequency)
        magnitude = np.interp(target_log, source_log, magnitude)
        phase = np.interp(target_log, source_log, phase)
        magnitude_uncertainty = np.interp(target_log, source_log, magnitude_uncertainty)
        phase_uncertainty = np.interp(target_log, source_log, phase_uncertainty)

    lower = np.asarray(record["input_min"], dtype=float)
    upper = np.asarray(record["input_max"], dtype=float)
    in_domain = bool(np.all(feature[0] >= lower) and np.all(feature[0] <= upper))
    uncertainty_db = float(np.max(magnitude_uncertainty))
    uncertainty_phase_deg = float(np.max(phase_uncertainty))
    threshold = float(bundle.get("trust_uncertainty_db", 1.0) if trust_uncertainty_db is None else trust_uncertainty_db)
    trusted = in_domain and uncertainty_db <= threshold
    if not in_domain:
        reason = "parameter point is outside validated training domain"
    elif uncertainty_db > threshold:
        reason = "ensemble disagreement exceeds trust threshold"
    else:
        reason = "within validated domain and uncertainty threshold"
    response = 10.0 ** (magnitude / 20.0) * np.exp(1j * np.deg2rad(phase))
    return SurrogatePrediction(
        template=template,
        frequency_hz=requested_frequency,
        response=response,
        magnitude_db=magnitude,
        phase_deg=phase,
        uncertainty_db=uncertainty_db,
        uncertainty_phase_deg=uncertainty_phase_deg,
        in_domain=in_domain,
        trusted=trusted,
        reason=reason,
    )


def save_response_surrogate(bundle: dict[str, Any], path: str | Path) -> None:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)


def load_response_surrogate(path: str | Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def _fit_ensemble(estimator_type, x: np.ndarray, y: np.ndarray, *, n_estimators: int, min_samples_leaf: int, seed: int):
    if n_estimators < 4:
        raise ValueError("n_estimators must be at least 4 for uncertainty estimation")
    model = estimator_type(
        n_estimators=int(n_estimators),
        min_samples_leaf=max(1, int(min_samples_leaf)),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def _holdout_indices(count: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if fraction == 0.0 or count < 5:
        return np.arange(count), np.asarray([], dtype=int)
    validation_count = min(max(1, int(round(count * fraction))), count - 1)
    rng = np.random.default_rng(seed)
    shuffled = np.arange(count)
    rng.shuffle(shuffled)
    return np.sort(shuffled[validation_count:]), np.sort(shuffled[:validation_count])


def _frequency_grid(row: dict[str, Any]) -> np.ndarray:
    frequency = np.asarray(row.get("frequency_hz", []), dtype=float)
    if len(frequency) < 2 or not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("surrogate rows need a finite positive frequency grid")
    return frequency


def _log_parameter_vector(parameters: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([float(parameters[name]) for name in names], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("surrogate parameters must be finite and positive")
    return np.log10(values)


def _db20(response: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(response, dtype=complex)), 1e-18))


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    raise TypeError("validation evidence must be a dict or expose as_dict()")


def _trace_from_validation(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        trace = value.get("trace")
        if not isinstance(trace, dict):
            return None
        frequency = trace.get("frequency_hz")
        magnitude = trace.get("magnitude_db")
        phase = trace.get("phase_deg")
    else:
        frequency = getattr(value, "trace_frequency_hz", ())
        magnitude = getattr(value, "trace_magnitude_db", ())
        phase = getattr(value, "trace_phase_deg", ())
    if not frequency or not magnitude or not phase:
        return None
    arrays = tuple(np.asarray(item, dtype=float) for item in (frequency, magnitude, phase))
    if any(len(item) == 0 for item in arrays) or len({len(item) for item in arrays}) != 1:
        raise ValueError("validation trace arrays must have the same non-zero length")
    if any(not np.all(np.isfinite(item)) for item in arrays):
        raise ValueError("validation trace arrays must be finite")
    return arrays
