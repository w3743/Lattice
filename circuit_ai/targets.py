from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analysis import AnalysisRequest


@dataclass(frozen=True)
class TargetResponse:
    frequencies_hz: np.ndarray
    values: np.ndarray
    has_phase: bool = True
    analysis: AnalysisRequest = field(default_factory=AnalysisRequest)


def frequency_grid(f_min_hz: float, f_max_hz: float, points: int) -> np.ndarray:
    return np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), points)


def target_from_behavior(
    behavior: dict[str, Any],
    frequencies_hz: np.ndarray,
    analysis: AnalysisRequest | None = None,
) -> TargetResponse:
    analysis = analysis or AnalysisRequest.from_dict(behavior.get("analysis"))
    kind = behavior["kind"].strip().lower()
    s = 2j * np.pi * frequencies_hz
    gain = float(behavior.get("gain", 1.0))

    if kind == "lowpass":
        wc = 2.0 * np.pi * float(behavior["cutoff_hz"])
        order = int(behavior.get("order", 1))
        if order == 1:
            values = gain / (1.0 + s / wc)
        elif order == 2:
            q = float(behavior.get("q", 1.0 / np.sqrt(2.0)))
            values = gain * wc**2 / (s**2 + (wc / q) * s + wc**2)
        else:
            values = gain / (1.0 + s / wc) ** order
        return TargetResponse(frequencies_hz, values, analysis=analysis)

    if kind == "highpass":
        wc = 2.0 * np.pi * float(behavior["cutoff_hz"])
        order = int(behavior.get("order", 1))
        if order == 1:
            values = gain * (s / wc) / (1.0 + s / wc)
        elif order == 2:
            q = float(behavior.get("q", 1.0 / np.sqrt(2.0)))
            values = gain * s**2 / (s**2 + (wc / q) * s + wc**2)
        else:
            values = gain * (s / wc) ** order / (1.0 + s / wc) ** order
        return TargetResponse(frequencies_hz, values, analysis=analysis)

    if kind == "bandpass":
        wc = 2.0 * np.pi * float(behavior["center_hz"])
        q = float(behavior.get("q", 3.0))
        values = gain * ((wc / q) * s) / (s**2 + (wc / q) * s + wc**2)
        return TargetResponse(frequencies_hz, values, analysis=analysis)

    if kind == "zpk":
        zeros = [complex(z) for z in behavior.get("zeros_rad_s", [])]
        poles = [complex(p) for p in behavior.get("poles_rad_s", [])]
        values = np.full_like(s, gain, dtype=np.complex128)
        for zero in zeros:
            values *= s - zero
        for pole in poles:
            values /= s - pole
        return TargetResponse(frequencies_hz, values, analysis=analysis)

    if kind == "samples":
        src_f = np.asarray(behavior["frequency_hz"], dtype=float)
        if np.any(src_f <= 0):
            raise ValueError("sample frequencies must be positive")
        mag_db = np.asarray(behavior["magnitude_db"], dtype=float)
        interp_mag_db = np.interp(np.log10(frequencies_hz), np.log10(src_f), mag_db)
        if "phase_deg" in behavior:
            phase_deg = np.asarray(behavior["phase_deg"], dtype=float)
            interp_phase = np.interp(np.log10(frequencies_hz), np.log10(src_f), phase_deg)
            values = 10 ** (interp_mag_db / 20.0) * np.exp(1j * np.deg2rad(interp_phase))
            return TargetResponse(frequencies_hz, values, has_phase=True, analysis=analysis)
        values = 10 ** (interp_mag_db / 20.0)
        return TargetResponse(
            frequencies_hz,
            values.astype(np.complex128),
            has_phase=False,
            analysis=analysis,
        )

    if kind in {"constant_impedance", "impedance"}:
        if analysis.kind != "input_impedance":
            raise ValueError("impedance behavior requires analysis.kind='input_impedance'")
        if "ohms" in behavior:
            impedance = float(behavior["ohms"])
        else:
            impedance = float(behavior["resistance_ohm"])
        values = np.full_like(frequencies_hz, impedance, dtype=np.complex128)
        return TargetResponse(frequencies_hz, values, has_phase=False, analysis=analysis)

    if kind in {"constant_output_impedance", "output_impedance"}:
        if analysis.kind != "output_impedance":
            raise ValueError("output impedance behavior requires analysis.kind='output_impedance'")
        if "ohms" in behavior:
            impedance = float(behavior["ohms"])
        else:
            impedance = float(behavior["resistance_ohm"])
        values = np.full_like(frequencies_hz, impedance, dtype=np.complex128)
        return TargetResponse(frequencies_hz, values, has_phase=False, analysis=analysis)

    if kind in {"transimpedance", "tia"}:
        if analysis.kind != "transimpedance":
            raise ValueError("transimpedance behavior requires analysis.kind='transimpedance'")
        if "transimpedance_ohm" in behavior:
            zt = float(behavior["transimpedance_ohm"])
        elif "ohms" in behavior:
            zt = float(behavior["ohms"])
        else:
            zt = float(behavior["resistance_ohm"])
        cutoff = behavior.get("cutoff_hz", behavior.get("bandwidth_hz"))
        if cutoff is None:
            values = np.full_like(frequencies_hz, zt, dtype=np.complex128)
        else:
            wc = 2.0 * np.pi * float(cutoff)
            values = zt / (1.0 + s / wc)
        return TargetResponse(frequencies_hz, values, has_phase=False, analysis=analysis)

    raise ValueError(f"unsupported behavior kind: {behavior['kind']!r}")
