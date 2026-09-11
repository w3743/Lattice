"""Project-wide physical-unit parsing and scalar conversion.

Serialized contracts keep units as strings.  Pint is deliberately isolated in
this module so schemas do not expose library-specific objects and every layer
uses the same aliases and failure semantics.
"""

from __future__ import annotations

import math

import pint


class UnitContractError(ValueError):
    """Raised when a serialized unit is unknown or dimensionally incompatible."""


UNIT_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=False)
UNIT_REGISTRY.define("CNY = [currency]")

_ALIASES = {
    "": "1",
    "dimensionless": "1",
    "unitless": "1",
    "ohms": "ohm",
    "Ω": "ohm",
    "mm2": "mm**2",
    "cm2": "cm**2",
    "m2": "m**2",
    "degC_delta": "delta_degC",
}
_LOGARITHMIC_UNITS = frozenset({"dB", "dBm", "dBW"})


def normalize_unit(unit: str | None) -> str:
    text = str(unit or "").strip()
    return _ALIASES.get(text, text)


def parse_unit(unit: str | None):
    normalized = normalize_unit(unit)
    try:
        return UNIT_REGISTRY.parse_units(normalized)
    except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError) as exc:
        raise UnitContractError(f"unknown unit {unit!r}") from exc


def units_compatible(left: str | None, right: str | None) -> bool:
    try:
        left_unit = parse_unit(left)
        right_unit = parse_unit(right)
        if _is_logarithmic(left) or _is_logarithmic(right):
            return normalize_unit(left) == normalize_unit(right)
        return left_unit.dimensionality == right_unit.dimensionality
    except UnitContractError:
        return False


def convert_scalar(value: float, from_unit: str | None, to_unit: str | None) -> float:
    if not math.isfinite(float(value)):
        raise UnitContractError("unit conversion requires a finite scalar")
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if source == target:
        return float(value)
    if _is_logarithmic(source) or _is_logarithmic(target):
        raise UnitContractError(
            f"logarithmic units must match exactly, cannot convert {source!r} to {target!r}"
        )
    try:
        return float(UNIT_REGISTRY.Quantity(float(value), parse_unit(source)).to(parse_unit(target)).magnitude)
    except pint.errors.DimensionalityError as exc:
        raise UnitContractError(
            f"incompatible units {source!r} and {target!r}"
        ) from exc
    except (pint.errors.OffsetUnitCalculusError, ValueError) as exc:
        raise UnitContractError(
            f"cannot convert {value!r} from {source!r} to {target!r}"
        ) from exc


def dimensionality(unit: str | None) -> str:
    return str(parse_unit(unit).dimensionality)


def _is_logarithmic(unit: str | None) -> bool:
    return normalize_unit(unit) in _LOGARITHMIC_UNITS
