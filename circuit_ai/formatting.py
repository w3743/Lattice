from __future__ import annotations

import math


_PREFIXES = [
    (-12, "p"),
    (-9, "n"),
    (-6, "u"),
    (-3, "m"),
    (0, ""),
    (3, "k"),
    (6, "Meg"),
    (9, "G"),
]


def eng(value: float, unit: str = "") -> str:
    """Return a SPICE-friendly engineering string."""

    if value == 0 or not math.isfinite(value):
        return f"{value:g}{unit}"

    exponent = int(math.floor(math.log10(abs(value)) / 3.0) * 3)
    exponent = min(max(exponent, -12), 9)
    prefix = dict(_PREFIXES).get(exponent, "")
    scaled = value / (10**exponent)
    return f"{scaled:.6g}{prefix}{unit}"


def db20(values):
    import numpy as np

    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-18))
