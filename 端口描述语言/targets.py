"""目标规格 —— 你期望电路达到什么数值。

目标与分析方法分离：
  Analysis      →  测量什么（电压传输、阻抗...）
  Target        →  测量结果应该是什么

所有目标共享一个统一的数值描述层：
  - 解析表达式 (filter 类)
  - 采样点插值 (samples)
  - 上下界掩码 (mask)
  - 容忍度区间 (tolerance)

每种目标都包含：
  - value:  理想值（可以是标量、向量、函数）
  - tolerance: 允许的误差区间
  - weight:  多目标组合时的权重
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import math

import numpy as np


# ── 数据类型 ────────────────────────────────────────────


@dataclass(frozen=True)
class Tolerance:
    """容忍度区间。

    relative: 相对误差（0.01 = 1% 误差）
    absolute: 绝对误差（在目标空间中）
    """

    relative: float | None = None
    absolute: float | None = None

    def is_satisfied(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        bound = np.zeros_like(actual)
        if self.relative is not None:
            bound += np.abs(target) * self.relative
        if self.absolute is not None:
            bound += self.absolute
        return np.abs(actual - target) <= bound

    def as_dict(self) -> dict:
        return {"relative": self.relative, "absolute": self.absolute}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Tolerance":
        if data is None:
            return cls()
        return cls(
            relative=float(data["relative"]) if data.get("relative") is not None else None,
            absolute=float(data["absolute"]) if data.get("absolute") is not None else None,
        )


@dataclass(frozen=True)
class Target:
    """基础目标类型。"""
    weight: float = 1.0
    description: str = ""
    tolerance: Tolerance = field(default_factory=Tolerance)

    def as_dict(self) -> dict:
        return {
            "weight": self.weight,
            "description": self.description,
            "tolerance": self.tolerance.as_dict(),
        }


# ── 频域目标的基础类型 ────────────────────────────────


@dataclass(frozen=True)
class FrequencyDomainTarget(Target):
    """频域目标的公共基类。"""

    def evaluate(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """返回复数响应值（幅度+相位）或实数值（仅幅度）。"""
        raise NotImplementedError

    def evaluate_magnitude_db(self, frequencies_hz: np.ndarray) -> np.ndarray:
        resp = self.evaluate(frequencies_hz)
        return 20.0 * np.log10(np.maximum(np.abs(resp), 1e-18))

    def evaluate_phase_deg(self, frequencies_hz: np.ndarray) -> np.ndarray:
        resp = self.evaluate(frequencies_hz)
        return np.rad2deg(np.angle(resp))


# ── 滤波器目标 ────────────────────────────────────────


FilterKind = Literal["lowpass", "highpass", "bandpass", "bandstop", "allpass"]
FilterResponse = Literal["butterworth", "chebyshev", "bessel", "elliptic"]


@dataclass(frozen=True)
class FilterTarget(FrequencyDomainTarget):
    """标准滤波器目标——由解析传递函数定义。

    参数：
      kind:      lowpass / highpass / bandpass / bandstop / allpass
      order:     滤波器阶数（≥1）
      cutoff_hz: 截止频率（低通/高通）或中心频率（带通/带阻）
      gain_db:   通带增益
      q:         二阶节的品质因子（order=2 时有效）
      response:  逼近类型（Butterworth / Chebyshev / Bessel / Elliptic）
      bandwidth_hz: 带通/带阻的 -3dB 带宽
      ripple_db: 通带纹波（Chebyshev/Elliptic 时有效）
    """

    kind: FilterKind = "lowpass"
    order: int = 1
    cutoff_hz: float = 1000.0
    gain_db: float = 0.0
    q: float | None = None
    response: FilterResponse = "butterworth"
    bandwidth_hz: float | None = None
    ripple_db: float = 0.0

    def evaluate(self, frequencies_hz: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * frequencies_hz
        gain = 10.0 ** (self.gain_db / 20.0)

        if self.kind == "lowpass":
            return _lowpass_complex(s, gain, self.cutoff_hz, self.order, self.q, self.response)
        if self.kind == "highpass":
            return _highpass_complex(s, gain, self.cutoff_hz, self.order, self.q, self.response)
        if self.kind == "bandpass":
            bw = self.bandwidth_hz or self.cutoff_hz / (self.q or 3.0)
            return _bandpass_complex(s, gain, self.cutoff_hz, bw, self.order)
        if self.kind == "bandstop":
            bw = self.bandwidth_hz or self.cutoff_hz / (self.q or 3.0)
            return _bandstop_complex(s, gain, self.cutoff_hz, bw, self.order)
        if self.kind == "allpass":
            return _allpass_complex(s, self.cutoff_hz, self.order)
        raise ValueError(f"unsupported filter kind: {self.kind!r}")

    def as_dict(self) -> dict:
        d = {
            **super().as_dict(),
            "target_kind": "filter",
            "filter_kind": self.kind,
            "order": self.order,
            "cutoff_hz": self.cutoff_hz,
            "gain_db": self.gain_db,
            "response": self.response,
        }
        if self.q is not None:
            d["q"] = self.q
        if self.bandwidth_hz is not None:
            d["bandwidth_hz"] = self.bandwidth_hz
        if self.ripple_db != 0.0:
            d["ripple_db"] = self.ripple_db
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FilterTarget":
        return cls(
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            tolerance=Tolerance.from_dict(data.get("tolerance")),
            kind=data.get("filter_kind", data.get("kind", "lowpass")),
            order=int(data.get("order", 1)),
            cutoff_hz=float(data.get("cutoff_hz", data.get("cutoff", 1000.0))),
            gain_db=float(data.get("gain_db", data.get("gain", 0.0))),
            q=float(data["q"]) if data.get("q") is not None else None,
            response=data.get("response", "butterworth"),
            bandwidth_hz=float(data["bandwidth_hz"]) if data.get("bandwidth_hz") is not None else None,
            ripple_db=float(data.get("ripple_db", 0.0)),
        )


# ── 放大器目标 ────────────────────────────────────────


@dataclass(frozen=True)
class AmplifierTarget(FrequencyDomainTarget):
    """放大器目标——指定增益、带宽、输入输出阻抗。

    在频域里它是一个平直增益 + 低/高频滚降的组合。
    """

    gain_db: float = 20.0
    bandwidth_hz: tuple[float, float] = (10.0, 100_000.0)  # (f_low, f_high)
    input_impedance_ohm: float | None = None
    output_impedance_ohm: float | None = None
    slew_rate_v_us: float | None = None

    def evaluate(self, frequencies_hz: np.ndarray) -> np.ndarray:
        gain = 10.0 ** (self.gain_db / 20.0)
        f_lo, f_hi = self.bandwidth_hz

        def bandpass_envelope(f: np.ndarray) -> np.ndarray:
            h = np.ones_like(f, dtype=np.complex128)
            if f_lo > 0:
                h *= 1.0 / (1.0 + f_lo / (1j * f + 1e-18))
            if f_hi < float("inf"):
                h *= 1.0 / (1.0 + 1j * f / f_hi)
            return h

        return gain * bandpass_envelope(frequencies_hz)

    def as_dict(self) -> dict:
        return {
            **super().as_dict(),
            "target_kind": "amplifier",
            "gain_db": self.gain_db,
            "bandwidth_hz": list(self.bandwidth_hz),
            "input_impedance_ohm": self.input_impedance_ohm,
            "output_impedance_ohm": self.output_impedance_ohm,
            "slew_rate_v_us": self.slew_rate_v_us,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AmplifierTarget":
        bw = data.get("bandwidth_hz", [10.0, 100_000.0])
        return cls(
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            tolerance=Tolerance.from_dict(data.get("tolerance")),
            gain_db=float(data.get("gain_db", 20.0)),
            bandwidth_hz=(float(bw[0]), float(bw[1])),
            input_impedance_ohm=float(data["input_impedance_ohm"])
            if data.get("input_impedance_ohm") is not None
            else None,
            output_impedance_ohm=float(data["output_impedance_ohm"])
            if data.get("output_impedance_ohm") is not None
            else None,
            slew_rate_v_us=float(data["slew_rate_v_us"]) if data.get("slew_rate_v_us") is not None else None,
        )


# ── 阻抗目标 ────────────────────────────────────────


@dataclass(frozen=True)
class ImpedanceTarget(FrequencyDomainTarget):
    """阻抗目标——描述端口上的 V/I。

    可以是：
      ohms:    恒定阻抗（如 50Ω）
      samples: 频率相关的复阻抗采样点
    """

    ohms: float | None = None
    samples_frequency_hz: list[float] | None = None
    samples_impedance_ohm: list[float] | None = None

    def evaluate(self, frequencies_hz: np.ndarray) -> np.ndarray:
        if self.ohms is not None:
            return np.full_like(frequencies_hz, self.ohms, dtype=np.complex128)
        if self.samples_frequency_hz is not None and self.samples_impedance_ohm is not None:
            return np.interp(
                np.log10(frequencies_hz),
                np.log10(np.asarray(self.samples_frequency_hz, dtype=float)),
                np.asarray(self.samples_impedance_ohm, dtype=float),
            ).astype(np.complex128)
        raise ValueError("ImpedanceTarget needs either ohms or samples")

    def as_dict(self) -> dict:
        d = {
            **super().as_dict(),
            "target_kind": "impedance",
        }
        if self.ohms is not None:
            d["ohms"] = self.ohms
        if self.samples_frequency_hz is not None:
            d["samples_frequency_hz"] = list(self.samples_frequency_hz)
            d["samples_impedance_ohm"] = list(self.samples_impedance_ohm)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ImpedanceTarget":
        return cls(
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            tolerance=Tolerance.from_dict(data.get("tolerance")),
            ohms=float(data["ohms"]) if data.get("ohms") is not None else None,
            samples_frequency_hz=list(data["samples_frequency_hz"])
            if data.get("samples_frequency_hz") is not None
            else None,
            samples_impedance_ohm=list(data["samples_impedance_ohm"])
            if data.get("samples_impedance_ohm") is not None
            else None,
        )


# ── 采样点目标 ────────────────────────────────────────


@dataclass(frozen=True)
class SampledTarget(FrequencyDomainTarget):
    """采样点目标——用户在离散频率点上指定幅度和相位。

    适用于：任意自定义频响曲线。
    """

    frequency_hz: tuple[float, ...] = ()
    magnitude_db: tuple[float, ...] = ()
    phase_deg: tuple[float, ...] | None = None

    def evaluate(self, frequencies_hz: np.ndarray) -> np.ndarray:
        src_f = np.asarray(self.frequency_hz, dtype=float)
        src_mag_db = np.asarray(self.magnitude_db, dtype=float)

        interp_mag_db = np.interp(
            np.log10(frequencies_hz),
            np.log10(src_f),
            src_mag_db,
        )

        if self.phase_deg is not None:
            src_phase = np.asarray(self.phase_deg, dtype=float)
            interp_phase = np.interp(np.log10(frequencies_hz), np.log10(src_f), src_phase)
            return 10 ** (interp_mag_db / 20.0) * np.exp(1j * np.deg2rad(interp_phase))

        return 10 ** (interp_mag_db / 20.0).astype(np.complex128)

    def as_dict(self) -> dict:
        d = {
            **super().as_dict(),
            "target_kind": "sampled",
            "frequency_hz": list(self.frequency_hz),
            "magnitude_db": list(self.magnitude_db),
        }
        if self.phase_deg is not None:
            d["phase_deg"] = list(self.phase_deg)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SampledTarget":
        return cls(
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            tolerance=Tolerance.from_dict(data.get("tolerance")),
            frequency_hz=tuple(float(f) for f in data["frequency_hz"]),
            magnitude_db=tuple(float(m) for m in data["magnitude_db"]),
            phase_deg=tuple(float(p) for p in data["phase_deg"])
            if data.get("phase_deg") is not None
            else None,
        )


# ── 掩码目标 ────────────────────────────────────────


@dataclass(frozen=True)
class MaskTarget(FrequencyDomainTarget):
    """掩码目标——上下界定义允许区域。

    upper_bound_db:  频率采样点上的上限（必须低于此值）
    lower_bound_db:  频率采样点上的下限（必须高于此值）
    phase_deg:       期望相位（可选）

    适用于：规定的滤波器模板、EMC 兼容性要求。
    """

    frequency_hz: tuple[float, ...] = ()
    lower_bound_db: tuple[float, ...] = ()
    upper_bound_db: tuple[float, ...] = ()
    phase_deg: tuple[float, ...] | None = None

    def evaluate(self, frequencies_hz: np.ndarray) -> np.ndarray:
        # 掩码目标不定义单一理想值，返回中值作为参考
        mid_db = tuple((lo + hi) / 2.0 for lo, hi in zip(self.lower_bound_db, self.upper_bound_db, strict=False))
        src_f = np.asarray(self.frequency_hz, dtype=float)
        interp_mid = np.interp(np.log10(frequencies_hz), np.log10(src_f), np.asarray(mid_db, dtype=float))
        return 10 ** (interp_mid / 20.0).astype(np.complex128)

    def satisfied(self, frequencies_hz: np.ndarray, response_db: np.ndarray) -> np.ndarray:
        """返回每个频率点是否在掩码内。"""
        src_f = np.asarray(self.frequency_hz, dtype=float)
        lower = np.interp(np.log10(frequencies_hz), np.log10(src_f), np.asarray(self.lower_bound_db, dtype=float))
        upper = np.interp(np.log10(frequencies_hz), np.log10(src_f), np.asarray(self.upper_bound_db, dtype=float))
        return (response_db >= lower) & (response_db <= upper)

    def as_dict(self) -> dict:
        d = {
            **super().as_dict(),
            "target_kind": "mask",
            "frequency_hz": list(self.frequency_hz),
            "lower_bound_db": list(self.lower_bound_db),
            "upper_bound_db": list(self.upper_bound_db),
        }
        if self.phase_deg is not None:
            d["phase_deg"] = list(self.phase_deg)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MaskTarget":
        return cls(
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            tolerance=Tolerance.from_dict(data.get("tolerance")),
            frequency_hz=tuple(float(f) for f in data["frequency_hz"]),
            lower_bound_db=tuple(float(v) for v in data["lower_bound_db"]),
            upper_bound_db=tuple(float(v) for v in data["upper_bound_db"]),
            phase_deg=tuple(float(p) for p in data["phase_deg"])
            if data.get("phase_deg") is not None
            else None,
        )


# ── 目标类型注册表 ──────────────────────────────────────

from .dc_targets import DcTarget  # noqa: E402

TARGET_REGISTRY: dict[str, type] = {
    "filter": FilterTarget,
    "amplifier": AmplifierTarget,
    "impedance": ImpedanceTarget,
    "sampled": SampledTarget,
    "mask": MaskTarget,
    "dc": DcTarget,
}


def target_from_dict(data: dict):
    kind = data.get("target_kind", data.get("kind", "filter"))
    cls = TARGET_REGISTRY.get(kind)
    if cls is not None and hasattr(cls, "from_dict"):
        return cls.from_dict(data)
    raise ValueError(f"unsupported target kind: {kind!r}")


# ── 频域传递函数实现 ─────────────────────────────────────


def _butterworth_poles(order: int) -> list[complex]:
    poles = []
    for k in range(1, order + 1):
        theta = math.pi * (2 * k + order - 1) / (2 * order)
        poles.append(complex(math.cos(theta), math.sin(theta)))
    return poles


def _cascade(s: np.ndarray, gain: float, poles: list[complex], zeros: list[complex] | None = None) -> np.ndarray:
    zeros = zeros or []
    h = np.ones_like(s, dtype=np.complex128) * gain
    for p in poles:
        h /= (s - p)
    for z in zeros:
        h *= (s - z)
    return h


def _lowpass_complex(
    s: np.ndarray,
    gain: float,
    fc_hz: float,
    order: int,
    q: float | None,
    response: FilterResponse,
) -> np.ndarray:
    w0 = 2.0 * math.pi * fc_hz
    if response == "butterworth":
        poles = _butterworth_poles(order)
        denorm_poles = [complex(p.real * w0, p.imag * w0) for p in poles]
        dc_gain = gain * (w0 ** order)
        return _cascade(s, dc_gain, denorm_poles)
    if order == 1:
        return gain / (1.0 + s / w0)
    if order == 2:
        q_val = q or 1.0 / math.sqrt(2.0)
        return gain * w0**2 / (s**2 + (w0 / q_val) * s + w0**2)
    return gain / (1.0 + s / w0) ** order


def _highpass_complex(
    s: np.ndarray,
    gain: float,
    fc_hz: float,
    order: int,
    q: float | None,
    response: FilterResponse,
) -> np.ndarray:
    w0 = 2.0 * math.pi * fc_hz
    if order == 1:
        return gain * (s / w0) / (1.0 + s / w0)
    if order == 2:
        q_val = q or 1.0 / math.sqrt(2.0)
        return gain * s**2 / (s**2 + (w0 / q_val) * s + w0**2)
    return gain * (s / w0) ** order / (1.0 + s / w0) ** order


def _bandpass_complex(
    s: np.ndarray,
    gain: float,
    fc_hz: float,
    bw_hz: float,
    order: int,
) -> np.ndarray:
    w0 = 2.0 * math.pi * fc_hz
    bw = 2.0 * math.pi * bw_hz
    q_val = w0 / bw
    if order <= 2:
        return gain * ((w0 / q_val) * s) / (s**2 + (w0 / q_val) * s + w0**2)
    return gain * ((w0 / q_val) * s / (s**2 + (w0 / q_val) * s + w0**2)) ** (order // 2)


def _bandstop_complex(
    s: np.ndarray,
    gain: float,
    fc_hz: float,
    bw_hz: float,
    order: int,
) -> np.ndarray:
    w0 = 2.0 * math.pi * fc_hz
    bw = 2.0 * math.pi * bw_hz
    q_val = w0 / bw
    if order <= 2:
        return gain * (s**2 + w0**2) / (s**2 + (w0 / q_val) * s + w0**2)
    return gain * ((s**2 + w0**2) / (s**2 + (w0 / q_val) * s + w0**2)) ** (order // 2)


def _allpass_complex(s: np.ndarray, fc_hz: float, order: int) -> np.ndarray:
    w0 = 2.0 * math.pi * fc_hz
    if order == 1:
        return (1.0 - s / w0) / (1.0 + s / w0)
    if order == 2:
        q_val = 1.0 / math.sqrt(2.0)
        return (s**2 - (w0 / q_val) * s + w0**2) / (s**2 + (w0 / q_val) * s + w0**2)
    return ((1.0 - s / w0) / (1.0 + s / w0)) ** order
