from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import numpy as np

from .formatting import db20
from .targets import TargetResponse
from .templates import CircuitTemplate


@dataclass(frozen=True)
class RobustnessMetrics:
    enabled: bool
    tolerance_fraction: float
    samples: int
    yield_fraction: float
    finite_fraction: float
    rmse_db_mean: float
    rmse_db_p95: float
    rmse_db_worst: float
    max_abs_db_p95: float
    max_abs_db_worst: float
    acceptance_rmse_db: float
    acceptance_max_abs_db: float
    seed: int | None = None
    distribution: str = "uniform"
    yield_confidence_interval: tuple[float, float] | None = None
    corner_results: tuple[dict[str, Any], ...] = ()
    corner_yield_fraction: float | None = None
    total_evaluations: int | None = None

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.enabled,
            "tolerance_fraction": self.tolerance_fraction,
            "samples": self.samples,
            "yield_fraction": self.yield_fraction,
            "finite_fraction": self.finite_fraction,
            "rmse_db_mean": self.rmse_db_mean,
            "rmse_db_p95": self.rmse_db_p95,
            "rmse_db_worst": self.rmse_db_worst,
            "max_abs_db_p95": self.max_abs_db_p95,
            "max_abs_db_worst": self.max_abs_db_worst,
            "acceptance_rmse_db": self.acceptance_rmse_db,
            "acceptance_max_abs_db": self.acceptance_max_abs_db,
            "seed": self.seed,
            "distribution": self.distribution,
            "yield_confidence_interval": self.yield_confidence_interval,
            "corner_results": [dict(item) for item in self.corner_results],
            "corner_yield_fraction": self.corner_yield_fraction,
            "total_evaluations": self.total_evaluations or self.samples,
        }


def analyze_robustness(
    template: CircuitTemplate,
    parameters: dict[str, float],
    target: TargetResponse,
    options: dict[str, Any],
    seed: int,
) -> RobustnessMetrics | None:
    if not options.get("enabled", False):
        return None

    tolerance = float(options.get("tolerance_fraction", options.get("tolerance", 0.05)))
    samples = int(options.get("samples", 64))
    acceptance_rmse_db = float(options.get("acceptance_rmse_db", 1.0))
    acceptance_max_abs_db = float(options.get("acceptance_max_abs_db", 6.0))
    distribution = str(options.get("distribution", "uniform")).lower()
    workers = int(options.get("workers", 1))
    batch_size = int(options.get("batch_size", max(1, min(samples, 32))))
    confidence_level = float(options.get("confidence_level", 0.95))
    target_half_width = options.get("target_yield_half_width")
    min_samples = int(options.get("min_samples", min(16, samples)))

    if tolerance < 0:
        raise ValueError("robustness tolerance must be non-negative")
    if samples <= 0:
        raise ValueError("robustness samples must be positive")
    if workers < 1:
        raise ValueError("robustness workers must be positive")
    if batch_size < 1:
        raise ValueError("robustness batch_size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("robustness confidence_level must be in (0, 1)")
    if target_half_width is not None and float(target_half_width) <= 0.0:
        raise ValueError("robustness target_yield_half_width must be positive")
    min_samples = max(1, min(min_samples, samples))

    perturbations = _parameter_samples(
        parameters,
        samples,
        tolerance,
        distribution,
        seed,
    )
    corner_specs = _normalize_corners(
        options.get("pvt_corners", options.get("corners", ())),
        parameters,
    )
    corner_results = _evaluate_corners(
        template,
        corner_specs,
        target,
        acceptance_rmse_db,
        acceptance_max_abs_db,
        workers=workers,
    )
    rmse_values: list[float] = []
    max_values: list[float] = []
    finite_count = 0
    pass_count = 0

    for start in range(0, samples, batch_size):
        batch = perturbations[start : start + batch_size]
        responses = _analyze_batch(
            template,
            batch,
            target,
            workers=workers,
        )
        for response in responses:
            if not np.all(np.isfinite(response)):
                rmse_values.append(1e9)
                max_values.append(1e9)
                continue

            finite_count += 1
            rmse_db, max_abs_db = _magnitude_error_metrics(target, response)
            rmse_values.append(rmse_db)
            max_values.append(max_abs_db)
            if rmse_db <= acceptance_rmse_db and max_abs_db <= acceptance_max_abs_db:
                pass_count += 1

        if target_half_width is not None and len(rmse_values) >= min_samples:
            _, _, half_width = _wilson_interval(
                pass_count,
                len(rmse_values),
                confidence_level,
            )
            if half_width <= float(target_half_width):
                break

    rmse = np.asarray(rmse_values, dtype=float)
    max_abs = np.asarray(max_values, dtype=float)
    corner_rmse = np.asarray(
        [float(item["rmse_db"]) for item in corner_results if item.get("finite")],
        dtype=float,
    )
    corner_max_abs = np.asarray(
        [float(item["max_abs_db"]) for item in corner_results if item.get("finite")],
        dtype=float,
    )
    combined_rmse = np.concatenate((rmse, corner_rmse)) if corner_rmse.size else rmse
    combined_max_abs = np.concatenate((max_abs, corner_max_abs)) if corner_max_abs.size else max_abs
    corner_passes = sum(bool(item.get("passed")) for item in corner_results)
    corner_yield = (
        float(corner_passes / len(corner_results)) if corner_results else None
    )
    total_evaluations = len(rmse_values) + len(corner_results)
    confidence_interval = _wilson_interval(
        pass_count,
        len(rmse_values),
        confidence_level,
    )[:2]
    return RobustnessMetrics(
        enabled=True,
        tolerance_fraction=tolerance,
        samples=len(rmse_values),
        yield_fraction=float(pass_count / len(rmse_values)),
        finite_fraction=float(finite_count / len(rmse_values)),
        rmse_db_mean=float(np.mean(combined_rmse)),
        rmse_db_p95=float(np.percentile(combined_rmse, 95)),
        rmse_db_worst=float(np.max(combined_rmse)),
        max_abs_db_p95=float(np.percentile(combined_max_abs, 95)),
        max_abs_db_worst=float(np.max(combined_max_abs)),
        acceptance_rmse_db=acceptance_rmse_db,
        acceptance_max_abs_db=acceptance_max_abs_db,
        seed=seed,
        distribution=distribution,
        yield_confidence_interval=confidence_interval,
        corner_results=tuple(corner_results),
        corner_yield_fraction=corner_yield,
        total_evaluations=total_evaluations,
    )


def robustness_penalty(metrics: RobustnessMetrics | None, options: dict[str, Any]) -> float:
    if metrics is None:
        return 0.0

    weight = float(options.get("score_weight", 0.35))
    yield_weight = float(options.get("yield_weight", 4.0))
    p95_weight = float(options.get("p95_weight", 1.0))
    worst_weight = float(options.get("worst_weight", 0.05))

    yield_penalty = yield_weight * (1.0 - metrics.yield_fraction)
    p95_penalty = p95_weight * metrics.rmse_db_p95
    worst_penalty = worst_weight * metrics.max_abs_db_worst
    return weight * (yield_penalty + p95_penalty + worst_penalty)


def _normalize_corners(raw: Any, parameters: dict[str, float]) -> tuple[tuple[str, dict[str, float]], ...]:
    if not raw:
        return ()
    if isinstance(raw, Mapping):
        raw = [
            {"name": str(name), **(dict(value) if isinstance(value, Mapping) else {"multipliers": value})}
            for name, value in raw.items()
        ]
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise ValueError("robustness corners must be a list or mapping")
    normalized: list[tuple[str, dict[str, float]]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("each robustness corner must be a mapping")
        label = str(item.get("name", item.get("id", f"corner_{index}")))
        values = dict(parameters)
        multipliers = item.get("multipliers", item.get("scales", {}))
        if multipliers is not None:
            if not isinstance(multipliers, Mapping):
                raise ValueError(f"corner {label!r} multipliers must be a mapping")
            for key, factor in multipliers.items():
                if str(key) in values:
                    values[str(key)] = max(
                        float(values[str(key)]) * float(factor),
                        np.finfo(float).tiny,
                    )
        overrides = item.get("parameters", item.get("values", {}))
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise ValueError(f"corner {label!r} parameters must be a mapping")
            for key, value in overrides.items():
                if str(key) in values:
                    values[str(key)] = max(float(value), np.finfo(float).tiny)
        normalized.append((label, values))
    if len({label for label, _ in normalized}) != len(normalized):
        raise ValueError("robustness corner names must be unique")
    return tuple(normalized)


def _evaluate_corners(
    template: CircuitTemplate,
    corners: tuple[tuple[str, dict[str, float]], ...],
    target: TargetResponse,
    acceptance_rmse_db: float,
    acceptance_max_abs_db: float,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if not corners:
        return []
    responses = _analyze_batch(
        template,
        [values for _, values in corners],
        target,
        workers=workers,
    )
    results: list[dict[str, Any]] = []
    for (label, values), response in zip(corners, responses):
        finite = bool(np.all(np.isfinite(response)))
        if finite:
            rmse_db, max_abs_db = _magnitude_error_metrics(target, response)
        else:
            rmse_db, max_abs_db = 1e9, 1e9
        results.append(
            {
                "name": label,
                "parameters": dict(values),
                "finite": finite,
                "passed": finite
                and rmse_db <= acceptance_rmse_db
                and max_abs_db <= acceptance_max_abs_db,
                "rmse_db": float(rmse_db),
                "max_abs_db": float(max_abs_db),
            }
        )
    return results


def _perturb_parameters(
    parameters: dict[str, float],
    rng: np.random.Generator,
    tolerance: float,
    distribution: str,
) -> dict[str, float]:
    if tolerance == 0:
        return dict(parameters)

    values: dict[str, float] = {}
    for key, value in parameters.items():
        if distribution == "normal":
            factor = 1.0 + rng.normal(0.0, tolerance / 3.0)
            factor = float(np.clip(factor, 1.0 - tolerance, 1.0 + tolerance))
        elif distribution == "lognormal":
            sigma = np.log1p(tolerance) / 3.0
            factor = float(np.exp(rng.normal(0.0, sigma)))
            factor = float(np.clip(factor, 1.0 - tolerance, 1.0 + tolerance))
        else:
            factor = float(rng.uniform(1.0 - tolerance, 1.0 + tolerance))
        values[key] = max(value * factor, np.finfo(float).tiny)
    return values


def _parameter_samples(
    parameters: dict[str, float],
    samples: int,
    tolerance: float,
    distribution: str,
    seed: int,
) -> list[dict[str, float]]:
    keys = tuple(parameters)
    if not keys:
        return [{} for _ in range(samples)]
    if tolerance == 0.0:
        return [dict(parameters) for _ in range(samples)]

    unit_samples = _unit_samples(samples, len(keys), distribution, seed)
    if distribution in {"normal", "gaussian"}:
        from scipy.special import ndtri

        factors = 1.0 + (tolerance / 3.0) * ndtri(unit_samples)
        factors = np.clip(factors, 1.0 - tolerance, 1.0 + tolerance)
    elif distribution == "lognormal":
        from scipy.special import ndtri

        sigma = np.log1p(tolerance) / 3.0
        factors = np.exp(sigma * ndtri(unit_samples))
        factors = np.clip(factors, 1.0 - tolerance, 1.0 + tolerance)
    else:
        factors = 1.0 - tolerance + 2.0 * tolerance * unit_samples

    return [
        {
            key: max(float(parameters[key]) * float(factors[row, column]), np.finfo(float).tiny)
            for column, key in enumerate(keys)
        }
        for row in range(samples)
    ]


def _unit_samples(
    samples: int,
    dimensions: int,
    distribution: str,
    seed: int,
) -> np.ndarray:
    if distribution in {"sobol", "sobol_sequence"}:
        try:
            from scipy.stats import qmc

            return qmc.Sobol(dimensions, scramble=True, seed=seed).random(samples)
        except ImportError:
            pass
    if distribution in {"lhs", "latin_hypercube", "latin-hypercube"}:
        try:
            from scipy.stats import qmc

            return qmc.LatinHypercube(dimensions, seed=seed).random(samples)
        except ImportError:
            pass
    return np.random.default_rng(seed).random((samples, dimensions))


def _analyze_batch(
    template: CircuitTemplate,
    parameters_batch: list[dict[str, float]],
    target: TargetResponse,
    *,
    workers: int,
) -> tuple[np.ndarray, ...]:
    if workers == 1:
        return template.analyze_many(
            parameters_batch,
            target.frequencies_hz,
            target.analysis,
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(
            executor.map(
                lambda values: template.analyze(
                    values,
                    target.frequencies_hz,
                    target.analysis,
                ),
                parameters_batch,
            )
        )


def _wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    from scipy.stats import norm

    if trials <= 0:
        return 0.0, 1.0, 1.0
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    denominator = 1.0 + z * z / trials
    center = (successes / trials + z * z / (2.0 * trials)) / denominator
    margin = z * np.sqrt(
        successes * (trials - successes) / trials**3 + z * z / (4.0 * trials**2)
    ) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper), float((upper - lower) / 2.0)


def _magnitude_error_metrics(target: TargetResponse, response: np.ndarray) -> tuple[float, float]:
    error_db = db20(response) - db20(target.values)
    return float(np.sqrt(np.mean(error_db**2))), float(np.max(np.abs(error_db)))
