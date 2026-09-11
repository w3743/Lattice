"""Topology-agnostic selection records for staged simulation fidelity.

This module also owns the *comparison* half of the multi-fidelity policy
(plan §7.3 rules 3-4).  Selection decides which candidates are promoted to a
higher fidelity; comparison decides whether the low-fidelity conclusion
survives contact with the high-fidelity result.  Keeping both here means the
threshold and its semantics are stated exactly once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

FIDELITY_SCHEDULE_SCHEMA = "circuit_ai.fidelity_schedule"
FIDELITY_SCHEDULE_VERSION = 1

#: Stable diagnostic code required by plan §7.3 rule 4.  Grep for this exact
#: string to find every producer and consumer of the disagreement signal.
MODEL_DISAGREEMENT_CODE = "model_disagreement"

FIDELITY_DISAGREEMENT_SCHEMA = "circuit_ai.fidelity_disagreement"
FIDELITY_DISAGREEMENT_VERSION = 1

#: Default agreement threshold, in the unit of the quantity being compared
#: (dB for complex-valued observables).  This is the single source of truth
#: for the previously duplicated ``accept_rmse_db = 1.0`` literals.
DEFAULT_DISAGREEMENT_THRESHOLD = 1.0

#: Below this magnitude a complex observable is treated as numerical noise and
#: clamped before the decibel conversion, matching the metric providers.
_DB_FLOOR = 1e-300


class FidelityRole(str, Enum):
    SCREEN = "screen"
    TRUTH = "truth"


@dataclass(frozen=True)
class FidelityStage:
    """One ordered selection gate in a multi-fidelity execution plan."""

    stage_id: str
    fidelity: str
    role: FidelityRole
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.stage_id.strip() or not self.fidelity.strip():
            raise ValueError("fidelity stage id and fidelity must not be empty")
        role = self.role if isinstance(self.role, FidelityRole) else FidelityRole(self.role)
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("fidelity stage max_candidates must be positive")
        object.__setattr__(self, "role", role)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "fidelity": self.fidelity,
            "role": self.role.value,
            "max_candidates": self.max_candidates,
        }


@dataclass(frozen=True)
class FidelitySchedule:
    """A versioned policy that promotes ranked candidates across fidelities."""

    schedule_id: str
    stages: tuple[FidelityStage, ...]
    schema: str = FIDELITY_SCHEDULE_SCHEMA
    schema_version: int = FIDELITY_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.schema != FIDELITY_SCHEDULE_SCHEMA or self.schema_version != FIDELITY_SCHEDULE_VERSION:
            raise ValueError("unsupported fidelity schedule schema")
        if not self.schedule_id.strip() or not self.stages:
            raise ValueError("fidelity schedule id and at least one stage are required")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("fidelity stage ids must be unique")
        if self.stages[-1].role is not FidelityRole.TRUTH:
            raise ValueError("the final fidelity stage must be a truth stage")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "stages": [stage.as_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class FidelityStageSelection:
    stage_id: str
    fidelity: str
    role: FidelityRole
    candidate_ids: tuple[str, ...]
    input_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "fidelity": self.fidelity,
            "role": self.role.value,
            "candidate_ids": list(self.candidate_ids),
            "input_count": self.input_count,
            "selected_count": len(self.candidate_ids),
        }


@dataclass(frozen=True)
class FidelityScheduleRun:
    schedule: FidelitySchedule
    selections: tuple[FidelityStageSelection, ...]

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return self.selections[-1].candidate_ids

    def as_dict(self) -> dict[str, object]:
        return {
            **self.schedule.as_dict(),
            "selections": [selection.as_dict() for selection in self.selections],
            "selected_candidate_ids": list(self.selected_candidate_ids),
        }


class FidelityScheduler:
    """Apply a declared schedule to candidates already sorted by low-cost rank."""

    def schedule(
        self,
        policy: FidelitySchedule,
        candidate_ids: tuple[str, ...],
        *,
        stage_candidate_ids: Mapping[str, tuple[str, ...]] | None = None,
    ) -> FidelityScheduleRun:
        if len(set(candidate_ids)) != len(candidate_ids) or any(not item.strip() for item in candidate_ids):
            raise ValueError("fidelity scheduler requires unique, non-empty candidate ids")
        selected = candidate_ids
        selections: list[FidelityStageSelection] = []
        for stage in policy.stages:
            input_count = len(selected)
            requested = (stage_candidate_ids or {}).get(stage.stage_id)
            if requested is not None:
                if len(set(requested)) != len(requested) or any(
                    candidate not in selected for candidate in requested
                ):
                    raise ValueError(
                        f"fidelity stage {stage.stage_id!r} contains an invalid promotion set"
                    )
                selected = tuple(requested)
            elif stage.max_candidates is not None:
                selected = selected[: stage.max_candidates]
            selections.append(
                FidelityStageSelection(
                    stage_id=stage.stage_id,
                    fidelity=stage.fidelity,
                    role=stage.role,
                    candidate_ids=selected,
                    input_count=input_count,
                )
            )
        return FidelityScheduleRun(policy, tuple(selections))


def default_ac_fidelity_schedule(top_k: int) -> FidelitySchedule:
    """Record the existing AC flow: optimize all, then truth-check the Top-K."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    return FidelitySchedule(
        schedule_id="ac.default.v1",
        stages=(
            FidelityStage("parameter_optimization", "optimization_model", FidelityRole.SCREEN),
            FidelityStage("linear_mna_truth", "linear_frequency_domain", FidelityRole.TRUTH, top_k),
        ),
    )


# ---------------------------------------------------------------------------
# Multi-fidelity disagreement (plan §7.3 rules 3-4)
# ---------------------------------------------------------------------------


class FidelityDisagreementError(ValueError):
    """Raised when two results cannot be meaningfully compared at all."""


@runtime_checkable
class _SimulationResultLike(Protocol):
    """Structural view of ``SimulationResult`` without importing it.

    ``circuit_ai.simulation`` imports this package, so importing it back here
    would create an import cycle.  Comparison only needs these four attributes.
    """

    status: Any
    scalars: Mapping[str, Any]
    waveforms: Mapping[str, Any]
    diagnostics: tuple[Any, ...]


@dataclass(frozen=True)
class ObservableDeviation:
    """One compared quantity, in the unit its deviation was measured in."""

    name: str
    kind: str
    unit: str
    deviation: float
    sample_count: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise FidelityDisagreementError("deviation name must not be empty")
        if self.kind not in {"scalar", "waveform"}:
            raise FidelityDisagreementError(f"unsupported deviation kind {self.kind!r}")
        if not math.isfinite(self.deviation) or self.deviation < 0.0:
            raise FidelityDisagreementError(
                f"deviation for {self.name!r} must be finite and non-negative"
            )
        if self.sample_count < 1:
            raise FidelityDisagreementError(
                f"deviation for {self.name!r} must cover at least one sample"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "unit": self.unit,
            "deviation": self.deviation,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class FidelityDisagreement:
    """The full, re-derivable record of one low-vs-high fidelity comparison.

    Plan §7.3 rule 3 requires the final report to carry the low/high
    difference, and rule 4 requires a threshold breach to enter the
    diagnostics queue.  This record serves the first; :meth:`as_diagnostic`
    serves the second.  Every field needed to recompute the verdict is stored,
    so a reader never has to trust the boolean alone.
    """

    threshold: float
    deviations: tuple[ObservableDeviation, ...]
    skipped_observables: tuple[str, ...]
    low_fidelity: str
    high_fidelity: str
    low_backend_id: str
    high_backend_id: str
    low_evidence_hash: str = ""
    high_evidence_hash: str = ""
    schema: str = FIDELITY_DISAGREEMENT_SCHEMA
    schema_version: int = FIDELITY_DISAGREEMENT_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != FIDELITY_DISAGREEMENT_SCHEMA:
            raise FidelityDisagreementError("unsupported fidelity disagreement schema")
        if self.schema_version != FIDELITY_DISAGREEMENT_VERSION:
            raise FidelityDisagreementError("unsupported fidelity disagreement schema version")
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise FidelityDisagreementError("disagreement threshold must be finite and non-negative")

    @property
    def max_deviation(self) -> float:
        """Worst deviation across compared observables (0.0 when none)."""

        return max((item.deviation for item in self.deviations), default=0.0)

    @property
    def worst_observable(self) -> str | None:
        if not self.deviations:
            return None
        return max(self.deviations, key=lambda item: item.deviation).name

    @property
    def disagreed(self) -> bool:
        """Whether any compared observable breached the threshold.

        An empty comparison is *not* agreement: with nothing compared we cannot
        claim the fidelities agree, so a caller that needs a verdict must treat
        ``compared_anything is False`` as unresolved rather than as passing.
        """

        return self.max_deviation > self.threshold

    @property
    def compared_anything(self) -> bool:
        return bool(self.deviations)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "threshold": self.threshold,
            "max_deviation": self.max_deviation,
            "worst_observable": self.worst_observable,
            "disagreed": self.disagreed,
            "compared_anything": self.compared_anything,
            "low_fidelity": self.low_fidelity,
            "high_fidelity": self.high_fidelity,
            "low_backend_id": self.low_backend_id,
            "high_backend_id": self.high_backend_id,
            "low_evidence_hash": self.low_evidence_hash,
            "high_evidence_hash": self.high_evidence_hash,
            "deviations": [item.as_dict() for item in self.deviations],
            "skipped_observables": list(self.skipped_observables),
            "extensions": dict(self.extensions),
        }

    def as_diagnostic(self) -> Any | None:
        """Return the ``model_disagreement`` diagnostic, or ``None`` if agreed.

        Severity is ``warning``, never ``error``: disagreement is a signal that
        the low-fidelity conclusion is not trustworthy enough to stand alone,
        not proof that the candidate is wrong.  Verified results are left
        success-typed so the uncertainty is visible instead of hidden.
        """

        if not self.disagreed:
            return None
        from ..simulation.contracts import Diagnostic

        worst = self.worst_observable or "unspecified"
        return Diagnostic(
            code=MODEL_DISAGREEMENT_CODE,
            severity="warning",
            message=(
                f"backends {self.low_backend_id!r} ({self.low_fidelity}) and "
                f"{self.high_backend_id!r} ({self.high_fidelity}) disagree by "
                f"{self.max_deviation:.6g} > threshold {self.threshold:.6g} "
                f"on observable {worst!r}"
            ),
            path=worst,
            details=self.as_dict(),
        )


def _as_float_array(values: Any) -> np.ndarray:
    """Flatten to a numeric array *without* forcing complex.

    Casting to ``complex`` here would make :func:`_is_complex_like` true for
    every input and silently route real-valued observables through the decibel
    branch, so the dtype is preserved unless the values genuinely are complex
    (or arrive as Python objects holding complex numbers).
    """

    array = np.asarray(values)
    if array.dtype.kind in {"c", "O"}:
        return array.astype(complex).reshape(-1)
    return array.astype(float).reshape(-1)


def _is_complex_like(array: np.ndarray) -> bool:
    return bool(np.iscomplexobj(array))


def compare_observable(
    name: str,
    low: Any,
    high: Any,
    unit: str,
    *,
    kind: str = "waveform",
) -> tuple[ObservableDeviation | None, str | None]:
    """Measure the deviation between two samples of the same observable.

    Returns ``(deviation, skip_reason)``; exactly one is ``None``.

    Deviation is measured in the observable's own unit, so the threshold stays
    physically interpretable:

    * complex-valued samples (AC phasors) are compared in dB, because a linear
      scale would make the verdict depend on where the operating point sits;
    * real-valued samples (DC levels, ripple) are compared as a fraction of the
      larger magnitude, so an arbitrary unit does not set the threshold.
    """

    low_array = _as_float_array(low)
    high_array = _as_float_array(high)
    if low_array.shape != high_array.shape:
        return None, f"shape mismatch {low_array.shape} vs {high_array.shape}"
    if low_array.size == 0:
        return None, "empty sample"
    if _is_complex_like(low_array) or _is_complex_like(high_array):
        low_db = 20.0 * np.log10(np.maximum(np.abs(low_array), _DB_FLOOR))
        high_db = 20.0 * np.log10(np.maximum(np.abs(high_array), _DB_FLOOR))
        deviation = float(np.max(np.abs(low_db - high_db)))
        return ObservableDeviation(name, kind, "dB", deviation, low_array.size), None
    low_real = np.abs(np.real(low_array))
    high_real = np.abs(np.real(high_array))
    scale = float(max(np.max(low_real), np.max(high_real), 1e-300))
    deviation = float(np.max(np.abs(low_real - high_real))) / scale
    return ObservableDeviation(name, kind, "ratio", deviation, low_array.size), None


def _observable_item(name: str, source: Mapping[str, Any]) -> tuple[Any, str] | None:
    """Extract comparable values and a unit from a scalar or waveform entry."""

    entry = source.get(name)
    if entry is None:
        return None
    if hasattr(entry, "values"):
        return entry.values, str(getattr(entry, "unit", ""))
    return getattr(entry, "value", entry), str(getattr(entry, "unit", ""))


def compare_fidelity_results(
    low_result: _SimulationResultLike,
    high_result: _SimulationResultLike,
    *,
    threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> FidelityDisagreement:
    """Compare one candidate's low- and high-fidelity results (plan §7.3).

    Both results must describe *the same candidate*; the point of the
    comparison is to isolate the fidelity difference, so comparing different
    graphs would measure the candidates, not the models.  Use
    :func:`graphs_are_comparable` when the caller cannot guarantee that.

    Only observables present in both results with matching shape are compared;
    the rest are reported in ``skipped_observables`` instead of being silently
    folded into "agreement".
    """

    if not math.isfinite(threshold) or threshold < 0.0:
        raise FidelityDisagreementError("disagreement threshold must be finite and non-negative")

    deviations: list[ObservableDeviation] = []
    skipped: list[str] = []
    scalar_names = sorted(set(low_result.scalars) & set(high_result.scalars))
    waveform_names = sorted(set(low_result.waveforms) & set(high_result.waveforms))
    skipped.extend(f"scalar:{name}" for name in sorted(set(low_result.scalars) - set(high_result.scalars)))
    skipped.extend(f"scalar:{name}" for name in sorted(set(high_result.scalars) - set(low_result.scalars)))
    skipped.extend(
        f"waveform:{name}" for name in sorted(set(low_result.waveforms) - set(high_result.waveforms))
    )
    skipped.extend(
        f"waveform:{name}" for name in sorted(set(high_result.waveforms) - set(low_result.waveforms))
    )
    for name in scalar_names:
        low_item = _observable_item(name, low_result.scalars)
        high_item = _observable_item(name, high_result.scalars)
        if low_item is None or high_item is None:
            skipped.append(f"scalar:{name}")
            continue
        deviation, reason = compare_observable(
            name,
            np.asarray([low_item[0]]),
            np.asarray([high_item[0]]),
            low_item[1] or high_item[1],
            kind="scalar",
        )
        if deviation is None:
            skipped.append(f"scalar:{name}: {reason}")
            continue
        deviations.append(deviation)
    for name in waveform_names:
        low_waveform = low_result.waveforms[name]
        high_waveform = high_result.waveforms[name]
        low_unit = str(getattr(low_waveform, "unit", ""))
        high_unit = str(getattr(high_waveform, "unit", ""))
        if low_unit != high_unit:
            skipped.append(f"waveform:{name}: unit mismatch {low_unit!r} vs {high_unit!r}")
            continue
        deviation, reason = compare_observable(name, low_waveform.values, high_waveform.values, low_unit)
        if deviation is None:
            skipped.append(f"waveform:{name}: {reason}")
            continue
        deviations.append(deviation)

    return FidelityDisagreement(
        threshold=threshold,
        deviations=tuple(deviations),
        skipped_observables=tuple(skipped),
        low_fidelity=str(getattr(low_result, "fidelity", "unspecified")),
        high_fidelity=str(getattr(high_result, "fidelity", "unspecified")),
        low_backend_id=str(getattr(low_result, "backend_id", "")),
        high_backend_id=str(getattr(high_result, "backend_id", "")),
        low_evidence_hash=str(getattr(low_result, "evidence_hash", "") or ""),
        high_evidence_hash=str(getattr(high_result, "evidence_hash", "") or ""),
    )


def graphs_are_comparable(
    low_result: _SimulationResultLike,
    high_result: _SimulationResultLike,
) -> bool:
    """Whether two results describe the same candidate graph.

    A disagreement verdict is only meaningful between results of one
    candidate.  Results that omit ``graph_hash`` are treated as comparable
    (the caller has already asserted sameness); results that carry different
    hashes are not.
    """

    low_hash = str(getattr(low_result, "graph_hash", "") or "")
    high_hash = str(getattr(high_result, "graph_hash", "") or "")
    if not low_hash or not high_hash:
        return True
    return low_hash == high_hash


def attach_disagreement_diagnostics(
    results: Sequence[Any],
    comparisons: Sequence[FidelityDisagreement],
) -> tuple[Any, ...]:
    """Append ``model_disagreement`` diagnostics to the high-fidelity results.

    The diagnostic is attached to the *high-fidelity* result because that is
    the result whose conclusion is being qualified: the low-fidelity value is
    the one that proved untrustworthy.  Existing diagnostics are preserved and
    duplicates of the same code are not added twice.
    """

    pending: dict[int, list[Any]] = {}
    for comparison in comparisons:
        diagnostic = comparison.as_diagnostic()
        if diagnostic is None:
            continue
        for index, result in enumerate(results):
            if str(getattr(result, "fidelity", "")) != comparison.high_fidelity:
                continue
            if str(getattr(result, "backend_id", "")) != comparison.high_backend_id:
                continue
            if any(
                getattr(existing, "code", None) == MODEL_DISAGREEMENT_CODE
                for existing in getattr(result, "diagnostics", ())
            ):
                continue
            pending.setdefault(index, []).append(diagnostic)
    if not pending:
        return tuple(results)
    return tuple(
        replace(result, diagnostics=tuple(getattr(result, "diagnostics", ())) + tuple(pending[index]))
        if index in pending
        else result
        for index, result in enumerate(results)
    )


def disagreement_from_options(
    options: Mapping[str, Any] | None,
    *,
    default_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> tuple[bool, float]:
    """Read the ``fidelity_disagreement`` policy from a spec mapping.

    Absent or ``enabled: false`` means the caller asked for no comparison, so
    the verdict is not evaluated at all rather than defaulted to agreement.
    """

    if not options:
        return False, default_threshold
    enabled = bool(options.get("enabled", False))
    raw = options.get("threshold", default_threshold)
    try:
        threshold = float(raw)
    except (TypeError, ValueError) as exc:
        raise FidelityDisagreementError(
            f"fidelity_disagreement.threshold must be a number, got {raw!r}"
        ) from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise FidelityDisagreementError(
            "fidelity_disagreement.threshold must be finite and non-negative"
        )
    return enabled, threshold
