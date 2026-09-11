from __future__ import annotations

import numpy as np

from .formatting import db20
from .io import write_jsonl
from .spec import LibrarySpec
from .targets import frequency_grid
from .templates import CircuitTemplate, default_templates


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))


def sample_parameters(
    template: CircuitTemplate,
    library: LibrarySpec,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = {}
    for param, bounds in zip(template.params, template.parameter_bounds(library)):
        values[param.name] = log_uniform(rng, bounds[0], bounds[1])
    return values


def generate_rows(
    library: LibrarySpec,
    samples_per_template: int = 64,
    f_min_hz: float = 10.0,
    f_max_hz: float = 1_000_000.0,
    points: int = 96,
    seed: int = 23,
    templates: list[CircuitTemplate] | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    freqs = frequency_grid(f_min_hz, f_max_hz, points)
    rows: list[dict] = []

    for template in templates or default_templates():
        if not template.required_elements.issubset(library.allowed):
            continue
        for _ in range(samples_per_template):
            params = sample_parameters(template, library, rng)
            response = template.response(params, freqs)
            if not np.all(np.isfinite(response)):
                continue
            rows.append(
                {
                    "template": template.name,
                    "component_count": template.component_count,
                    "parameters": params,
                    "frequency_hz": freqs.tolist(),
                    "magnitude_db": db20(response).tolist(),
                    "phase_deg": np.rad2deg(np.unwrap(np.angle(response))).tolist(),
                }
            )

    return rows
