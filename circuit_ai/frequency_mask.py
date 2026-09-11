"""Frequency-band acceptance masks for AC designs.

The DC power path got an operating envelope and a worst-case reduction; the AC
path had nothing equivalent. An AC design was scored by ``rmse_db`` and
``max_abs_db`` against an analytic target, which answers "how close is the shape
on average" but never "does the passband hold its ripple, and does the stopband
reach its attenuation" -- the two numbers a filter requirement is actually
written in.

This module adds the missing acceptance layer:

* :class:`BandSpec` states one frequency band's gain limits.
* :class:`FilterMask` is an ordered set of bands plus the transition regions
  between them, derived rather than restated.
* :func:`evaluate_filter_mask` reports per-band margin, the worst frequency in
  each band, and the overall verdict.

The band verdict is deliberately separate from the shape score: a design can
track a target closely on average and still break its stopband floor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MASK_SCHEMA",
    "MASK_VERSION",
    "BandMargin",
    "BandSpec",
    "FilterMask",
    "FilterMaskReport",
    "TransitionRegion",
    "build_mask",
    "evaluate_filter_mask",
    "mask_from_mapping",
]


MASK_SCHEMA = "circuit_ai.filter_mask"
MASK_VERSION = 1


@dataclass(frozen=True)
class BandSpec:
    """One frequency band of a requirement, in absolute gain limits.

    Limits are in dB relative to the same reference the response is expressed
    in, so a passband of ``min_db=-0.5, max_db=+0.5`` around 0 dB gain means the
    usual +-0.5 dB ripple.
    """

    band_id: str
    kind: str
    upper_hz: float
    lower_hz: float = 0.0
    min_db: float | None = None
    max_db: float | None = None
    #: A band may legitimately contain no sample; the stopband above the last
    #: analysed frequency often does.
    allow_empty: bool = True

    def __post_init__(self) -> None:
        if not self.band_id.strip():
            raise ValueError("band id must not be empty")
        if self.kind not in {"passband", "stopband", "transition"}:
            raise ValueError(f"unsupported band kind {self.kind!r}")
        if self.lower_hz < 0.0 or self.upper_hz <= self.lower_hz:
            raise ValueError(
                f"band {self.band_id!r} needs 0 <= lower_hz < upper_hz "
                f"(got {self.lower_hz}, {self.upper_hz})"
            )
        if self.min_db is None and self.max_db is None:
            raise ValueError(f"band {self.band_id!r} must state min_db or max_db")
        if self.min_db is not None and self.max_db is not None and self.min_db > self.max_db:
            raise ValueError(
                f"band {self.band_id!r} has min_db above max_db "
                f"({self.min_db}, {self.max_db})"
            )

    def contains(self, frequency_hz: float) -> bool:
        return self.lower_hz <= frequency_hz <= self.upper_hz

    def as_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "kind": self.kind,
            "lower_hz": self.lower_hz,
            "upper_hz": self.upper_hz,
            "min_db": self.min_db,
            "max_db": self.max_db,
            "allow_empty": self.allow_empty,
        }


@dataclass(frozen=True)
class TransitionRegion:
    """The gap between two consecutive bands, where no requirement is stated."""

    lower_hz: float
    upper_hz: float
    from_band: str
    to_band: str

    @property
    def width_ratio(self) -> float:
        if self.lower_hz <= 0.0:
            return math.inf
        return self.upper_hz / self.lower_hz

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_band": self.from_band,
            "to_band": self.to_band,
            "lower_hz": self.lower_hz,
            "upper_hz": self.upper_hz,
            "width_ratio": self.width_ratio,
        }


@dataclass(frozen=True)
class BandMargin:
    """One band's outcome, with the frequency that decided it."""

    band_id: str
    kind: str
    sample_count: int
    passed: bool
    worst_frequency_hz: float | None
    worst_gain_db: float | None
    margin_db: float | None
    violated_limit: str | None
    min_observed_db: float | None = None
    max_observed_db: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "kind": self.kind,
            "sample_count": self.sample_count,
            "passed": self.passed,
            "worst_frequency_hz": self.worst_frequency_hz,
            "worst_gain_db": self.worst_gain_db,
            "margin_db": self.margin_db,
            "violated_limit": self.violated_limit,
            "min_observed_db": self.min_observed_db,
            "max_observed_db": self.max_observed_db,
            "note": self.note,
        }


@dataclass(frozen=True)
class FilterMaskReport:
    """The whole mask verdict across bands."""

    bands: tuple[BandMargin, ...]
    transitions: tuple[TransitionRegion, ...]
    worst_band: str | None
    worst_margin_db: float | None
    schema: str = MASK_SCHEMA
    schema_version: int = MASK_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every band held its limits.

        A band with no samples does not pass by default: absence of evidence is
        not evidence of compliance, so an empty band is reported as not passed
        unless it was explicitly allowed to be empty.
        """

        return all(band.passed for band in self.bands)

    @property
    def failed_bands(self) -> tuple[str, ...]:
        return tuple(band.band_id for band in self.bands if not band.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "passed": self.passed,
            "failed_bands": list(self.failed_bands),
            "worst_band": self.worst_band,
            "worst_margin_db": self.worst_margin_db,
            "bands": [band.as_dict() for band in self.bands],
            "transitions": [item.as_dict() for item in self.transitions],
            "extensions": dict(self.extensions),
        }


@dataclass(frozen=True)
class FilterMask:
    """An ordered set of bands; transitions are derived, never restated."""

    mask_id: str
    bands: tuple[BandSpec, ...]
    schema: str = MASK_SCHEMA
    schema_version: int = MASK_VERSION

    def __post_init__(self) -> None:
        if self.schema != MASK_SCHEMA or self.schema_version != MASK_VERSION:
            raise ValueError("unsupported filter mask schema")
        if not self.mask_id.strip() or not self.bands:
            raise ValueError("filter mask needs an id and at least one band")
        if len({band.band_id for band in self.bands}) != len(self.bands):
            raise ValueError("filter mask band ids must be unique")
        ordered = tuple(sorted(self.bands, key=lambda band: band.lower_hz))
        previous: BandSpec | None = None
        for band in ordered:
            if previous is not None and band.lower_hz < previous.upper_hz:
                raise ValueError(
                    f"filter mask bands overlap: {previous.band_id!r} and {band.band_id!r}"
                )
            previous = band
        object.__setattr__(self, "bands", ordered)

    @property
    def transitions(self) -> tuple[TransitionRegion, ...]:
        regions: list[TransitionRegion] = []
        for first, second in zip(self.bands, self.bands[1:]):
            if second.lower_hz > first.upper_hz:
                regions.append(
                    TransitionRegion(
                        lower_hz=first.upper_hz,
                        upper_hz=second.lower_hz,
                        from_band=first.band_id,
                        to_band=second.band_id,
                    )
                )
        return tuple(regions)

    def band_for(self, frequency_hz: float) -> BandSpec | None:
        for band in self.bands:
            if band.contains(frequency_hz):
                return band
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "mask_id": self.mask_id,
            "bands": [band.as_dict() for band in self.bands],
            "transitions": [item.as_dict() for item in self.transitions],
        }


def evaluate_filter_mask(
    mask: FilterMask,
    frequencies_hz: Sequence[float],
    magnitude_db: Sequence[float],
) -> FilterMaskReport:
    """Check an achieved magnitude response against every band.

    ``magnitude_db`` must already be in the same reference as the band limits.
    Frequencies outside every band are ignored: the mask states what is
    required, and a transition region deliberately requires nothing.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float)
    gains = np.asarray(magnitude_db, dtype=float)
    if frequencies.shape != gains.shape:
        raise ValueError(
            f"frequency and gain arrays must match: {frequencies.shape} vs {gains.shape}"
        )
    if frequencies.size == 0:
        raise ValueError("filter mask needs at least one sample")

    outcomes: list[BandMargin] = []
    for band in mask.bands:
        selected = (frequencies >= band.lower_hz) & (frequencies <= band.upper_hz)
        band_gains = gains[selected]
        band_frequencies = frequencies[selected]
        if band_gains.size == 0:
            outcomes.append(
                BandMargin(
                    band_id=band.band_id,
                    kind=band.kind,
                    sample_count=0,
                    passed=bool(band.allow_empty),
                    worst_frequency_hz=None,
                    worst_gain_db=None,
                    margin_db=None,
                    violated_limit=None,
                    note=(
                        "no analysed sample falls in this band"
                        if not band.allow_empty
                        else "no analysed sample falls in this band; allowed empty"
                    ),
                )
            )
            continue

        low_margin = (
            float(np.min(band_gains - band.min_db)) if band.min_db is not None else math.inf
        )
        high_margin = (
            float(np.min(band.max_db - band_gains)) if band.max_db is not None else math.inf
        )
        if low_margin <= high_margin:
            margin = low_margin
            violated = "min_db" if band.min_db is not None else None
            index = int(np.argmin(band_gains)) if band.min_db is not None else 0
        else:
            margin = high_margin
            violated = "max_db" if band.max_db is not None else None
            index = int(np.argmax(band_gains)) if band.max_db is not None else 0

        outcomes.append(
            BandMargin(
                band_id=band.band_id,
                kind=band.kind,
                sample_count=int(band_gains.size),
                passed=bool(margin >= 0.0),
                worst_frequency_hz=float(band_frequencies[index]),
                worst_gain_db=float(band_gains[index]),
                margin_db=float(margin),
                violated_limit=None if margin >= 0.0 else violated,
                min_observed_db=float(np.min(band_gains)),
                max_observed_db=float(np.max(band_gains)),
            )
        )

    failing = [item for item in outcomes if not item.passed and item.margin_db is not None]
    if failing:
        worst = min(failing, key=lambda item: float(item.margin_db))
        worst_band: str | None = worst.band_id
        worst_margin: float | None = float(worst.margin_db)
    else:
        scored = [item for item in outcomes if item.margin_db is not None]
        if scored:
            best = min(scored, key=lambda item: float(item.margin_db))
            worst_band = best.band_id
            worst_margin = float(best.margin_db)
        else:
            worst_band, worst_margin = None, None

    return FilterMaskReport(
        bands=tuple(outcomes),
        transitions=mask.transitions,
        worst_band=worst_band,
        worst_margin_db=worst_margin,
    )


def build_mask(
    mask_id: str,
    bands: Sequence[Mapping[str, Any]],
) -> FilterMask:
    """Build a mask from plain mappings, for spec parsing."""

    specs = [
        BandSpec(
            band_id=str(item["band_id"]),
            kind=str(item.get("kind", "passband")),
            lower_hz=float(item.get("lower_hz", 0.0)),
            upper_hz=float(item["upper_hz"]),
            min_db=_optional_float(item.get("min_db")),
            max_db=_optional_float(item.get("max_db")),
            allow_empty=bool(item.get("allow_empty", True)),
        )
        for item in bands
    ]
    return FilterMask(mask_id=mask_id, bands=tuple(specs))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def mask_from_mapping(spec: Mapping[str, Any]) -> FilterMask | None:
    """Read an optional ``filter_mask`` block from a PBDL spec.

    Returns ``None`` when absent, so a spec that states only an analytic target
    keeps its previous behaviour and reports no band verdict.
    """

    block = spec.get("filter_mask")
    if not isinstance(block, Mapping):
        return None
    bands = block.get("bands")
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValueError("filter_mask must declare a non-empty bands list")
    return build_mask(str(block.get("mask_id", "mask")), bands)
