from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spec import SynthesisSpec


ACTIVE_ELEMENTS = {"opamp", "E"}


@dataclass(frozen=True)
class FeasibilityIssue:
    severity: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class FeasibilityReport:
    enabled: bool
    passed: bool
    enforce: bool
    issues: tuple[FeasibilityIssue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "passed": self.passed,
            "enforce": self.enforce,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    @property
    def errors(self) -> tuple[FeasibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[FeasibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


class FeasibilityError(ValueError):
    def __init__(self, report: FeasibilityReport):
        self.report = report
        details = "; ".join(issue.message for issue in report.errors)
        super().__init__(f"spec failed feasibility checks: {details}")


def analyze_feasibility(spec: SynthesisSpec) -> FeasibilityReport:
    options = spec.optimization.feasibility
    enabled = bool(options.get("enabled", True))
    enforce = bool(options.get("enforce", True))
    if not enabled:
        return FeasibilityReport(enabled=False, passed=True, enforce=enforce, issues=())

    issues: list[FeasibilityIssue] = []
    behavior = spec.behavior
    kind = str(behavior.get("kind", "")).strip().lower()

    _check_library(spec, issues)
    _check_optimization(spec, issues)

    if kind == "lowpass":
        _check_filter_common(behavior, issues, cutoff_key="cutoff_hz")
        _check_order_q(behavior, issues)
    elif kind == "highpass":
        _check_filter_common(behavior, issues, cutoff_key="cutoff_hz")
        _check_order_q(behavior, issues)
    elif kind == "bandpass":
        _check_filter_common(behavior, issues, cutoff_key="center_hz")
        _check_positive(behavior, "q", issues, required=False)
    elif kind == "zpk":
        _check_zpk(behavior, issues, options)
    elif kind == "samples":
        _check_samples(behavior, issues)
    elif kind in {"constant_impedance", "impedance"}:
        _check_impedance_target(spec, issues)
    elif kind in {"constant_output_impedance", "output_impedance"}:
        _check_output_impedance_target(spec, issues)
    elif kind in {"transimpedance", "tia"}:
        _check_transimpedance_target(spec, issues)
    else:
        issues.append(
            FeasibilityIssue("error", "unsupported_behavior", f"unsupported behavior kind: {kind!r}")
        )

    _check_gain_and_passive_hint(spec, issues, options)
    passed = not any(issue.severity == "error" for issue in issues)
    return FeasibilityReport(enabled=True, passed=passed, enforce=enforce, issues=tuple(issues))


def enforce_feasibility(spec: SynthesisSpec) -> FeasibilityReport:
    report = analyze_feasibility(spec)
    if report.enabled and report.enforce and not report.passed:
        raise FeasibilityError(report)
    return report


def _check_library(spec: SynthesisSpec, issues: list[FeasibilityIssue]) -> None:
    if not spec.library.allowed:
        issues.append(FeasibilityIssue("error", "empty_library", "element library is empty"))
    for kind, (lo, hi) in spec.library.parameter_ranges.items():
        if lo <= 0 or hi <= lo:
            issues.append(
                FeasibilityIssue(
                    "error",
                    "invalid_parameter_range",
                    f"invalid range for {kind}: [{lo}, {hi}]",
                )
            )


def _check_optimization(spec: SynthesisSpec, issues: list[FeasibilityIssue]) -> None:
    opt = spec.optimization
    if opt.points < 2:
        issues.append(FeasibilityIssue("error", "too_few_points", "frequency grid needs at least 2 points"))
    if opt.max_components < 1:
        issues.append(FeasibilityIssue("error", "invalid_max_components", "max_components must be positive"))
    if opt.top_k < 1:
        issues.append(FeasibilityIssue("error", "invalid_top_k", "top_k must be positive"))


def _check_filter_common(
    behavior: dict[str, Any],
    issues: list[FeasibilityIssue],
    cutoff_key: str,
) -> None:
    _check_positive(behavior, cutoff_key, issues, required=True)
    _check_finite_number(behavior, "gain", issues, required=False)


def _check_order_q(behavior: dict[str, Any], issues: list[FeasibilityIssue]) -> None:
    try:
        order = int(behavior.get("order", 1))
    except (TypeError, ValueError):
        issues.append(FeasibilityIssue("error", "invalid_order", "filter order must be an integer"))
        return
    if order < 1:
        issues.append(FeasibilityIssue("error", "invalid_order", "filter order must be at least 1"))
    if order == 2:
        _check_positive(behavior, "q", issues, required=False)
    if order > 2:
        issues.append(
            FeasibilityIssue(
                "warning",
                "approximate_high_order_target",
                "orders above 2 are represented by repeated first-order factors in this prototype",
            )
        )


def _check_zpk(behavior: dict[str, Any], issues: list[FeasibilityIssue], options: dict[str, Any]) -> None:
    zeros = _complex_list(behavior.get("zeros_rad_s", []), "zeros_rad_s", issues)
    poles = _complex_list(behavior.get("poles_rad_s", []), "poles_rad_s", issues)
    require_stable = bool(options.get("require_stable", True))
    require_causal = bool(options.get("require_causal", True))
    stable_tol = float(options.get("stable_real_tolerance", 1e-12))

    if require_causal and len(zeros) > len(poles):
        issues.append(
            FeasibilityIssue(
                "error",
                "noncausal_improper_transfer",
                "ZPK transfer has more zeros than poles, so it is not a proper causal rational transfer",
            )
        )

    if require_stable:
        unstable = [pole for pole in poles if pole.real >= -stable_tol]
        if unstable:
            issues.append(
                FeasibilityIssue(
                    "error",
                    "unstable_poles",
                    f"ZPK transfer has non-left-half-plane poles: {unstable}",
                )
            )

    if not poles:
        issues.append(
            FeasibilityIssue(
                "warning",
                "no_zpk_poles",
                "ZPK target has no poles; this may require a direct/ideal path rather than a realizable filter",
            )
        )


def _check_samples(behavior: dict[str, Any], issues: list[FeasibilityIssue]) -> None:
    required = ["frequency_hz", "magnitude_db"]
    for key in required:
        if key not in behavior:
            issues.append(FeasibilityIssue("error", "missing_samples", f"samples behavior missing {key}"))
            return

    freq = np.asarray(behavior["frequency_hz"], dtype=float)
    mag = np.asarray(behavior["magnitude_db"], dtype=float)
    if len(freq) != len(mag):
        issues.append(
            FeasibilityIssue("error", "sample_length_mismatch", "frequency_hz and magnitude_db lengths differ")
        )
    if len(freq) < 2:
        issues.append(FeasibilityIssue("error", "too_few_samples", "sample behavior needs at least 2 samples"))
    if np.any(~np.isfinite(freq)) or np.any(freq <= 0):
        issues.append(FeasibilityIssue("error", "invalid_sample_frequency", "sample frequencies must be finite and positive"))
    if np.any(np.diff(freq) <= 0):
        issues.append(FeasibilityIssue("error", "nonmonotonic_samples", "sample frequencies must be strictly increasing"))
    if np.any(~np.isfinite(mag)):
        issues.append(FeasibilityIssue("error", "invalid_sample_magnitude", "sample magnitudes must be finite"))
    if "phase_deg" in behavior:
        phase = np.asarray(behavior["phase_deg"], dtype=float)
        if len(phase) != len(freq):
            issues.append(
                FeasibilityIssue("error", "sample_length_mismatch", "phase_deg length differs from frequency_hz")
            )
        if np.any(~np.isfinite(phase)):
            issues.append(FeasibilityIssue("error", "invalid_sample_phase", "sample phases must be finite"))
    issues.append(
        FeasibilityIssue(
            "warning",
            "sample_feasibility_limited",
            "sampled behavior can only be checked for numeric consistency; causality/stability are not proven",
        )
    )


def _check_impedance_target(spec: SynthesisSpec, issues: list[FeasibilityIssue]) -> None:
    if spec.analysis.kind != "input_impedance":
        issues.append(
            FeasibilityIssue(
                "error",
                "analysis_behavior_mismatch",
                "impedance behavior requires analysis.kind='input_impedance'",
            )
        )
    behavior = spec.behavior
    if "ohms" in behavior:
        _check_positive(behavior, "ohms", issues, required=True)
    else:
        _check_positive(behavior, "resistance_ohm", issues, required=True)


def _check_output_impedance_target(spec: SynthesisSpec, issues: list[FeasibilityIssue]) -> None:
    if spec.analysis.kind != "output_impedance":
        issues.append(
            FeasibilityIssue(
                "error",
                "analysis_behavior_mismatch",
                "output impedance behavior requires analysis.kind='output_impedance'",
            )
        )
    behavior = spec.behavior
    if "ohms" in behavior:
        _check_positive(behavior, "ohms", issues, required=True)
    else:
        _check_positive(behavior, "resistance_ohm", issues, required=True)


def _check_transimpedance_target(spec: SynthesisSpec, issues: list[FeasibilityIssue]) -> None:
    if spec.analysis.kind != "transimpedance":
        issues.append(
            FeasibilityIssue(
                "error",
                "analysis_behavior_mismatch",
                "transimpedance behavior requires analysis.kind='transimpedance'",
            )
        )
    behavior = spec.behavior
    if "transimpedance_ohm" in behavior:
        _check_positive(behavior, "transimpedance_ohm", issues, required=True)
    elif "ohms" in behavior:
        _check_positive(behavior, "ohms", issues, required=True)
    else:
        _check_positive(behavior, "resistance_ohm", issues, required=True)
    _check_positive(behavior, "cutoff_hz", issues, required=False)
    _check_positive(behavior, "bandwidth_hz", issues, required=False)


def _check_gain_and_passive_hint(
    spec: SynthesisSpec,
    issues: list[FeasibilityIssue],
    options: dict[str, Any],
) -> None:
    gain = spec.behavior.get("gain", 1.0)
    try:
        gain_value = abs(float(gain))
    except (TypeError, ValueError):
        return

    has_active = bool(spec.library.allowed.intersection(ACTIVE_ELEMENTS))
    if (
        bool(options.get("warn_passive_gain", True))
        and not has_active
        and gain_value > 1.0 + float(options.get("passive_gain_tolerance", 1e-9))
    ):
        issues.append(
            FeasibilityIssue(
                "warning",
                "passive_library_gain_gt_one",
                "target gain exceeds unity while the library has no active elements; passive two-port realization may be impossible under normal loading assumptions",
            )
        )


def _check_positive(
    behavior: dict[str, Any],
    key: str,
    issues: list[FeasibilityIssue],
    required: bool,
) -> None:
    value = behavior.get(key)
    if value is None:
        if required:
            issues.append(FeasibilityIssue("error", "missing_positive_parameter", f"missing required {key}"))
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        issues.append(FeasibilityIssue("error", "invalid_numeric_parameter", f"{key} must be numeric"))
        return
    if not np.isfinite(numeric) or numeric <= 0:
        issues.append(FeasibilityIssue("error", "invalid_positive_parameter", f"{key} must be finite and positive"))


def _check_finite_number(
    behavior: dict[str, Any],
    key: str,
    issues: list[FeasibilityIssue],
    required: bool,
) -> None:
    value = behavior.get(key)
    if value is None:
        if required:
            issues.append(FeasibilityIssue("error", "missing_numeric_parameter", f"missing required {key}"))
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        issues.append(FeasibilityIssue("error", "invalid_numeric_parameter", f"{key} must be numeric"))
        return
    if not np.isfinite(numeric):
        issues.append(FeasibilityIssue("error", "invalid_numeric_parameter", f"{key} must be finite"))


def _complex_list(raw: Any, key: str, issues: list[FeasibilityIssue]) -> list[complex]:
    values: list[complex] = []
    try:
        for item in raw:
            values.append(complex(item))
    except (TypeError, ValueError):
        issues.append(FeasibilityIssue("error", "invalid_complex_list", f"{key} must contain complex-compatible values"))
    return values
