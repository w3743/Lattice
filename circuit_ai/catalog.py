from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .spec import SynthesisSpec
from .templates import (
    BufferedCascadeRCLowPass,
    CircuitTemplate,
    GainRCHighPass,
    GainRCLowPass,
    OutputResistorImpedance,
    RCHighPass,
    RCLowPass,
    RLCBandPass,
    SallenKeyGainLowPass,
    SallenKeyLowPass,
    ShuntResistorImpedance,
    TransimpedanceAmplifier,
)


@dataclass(frozen=True)
class TopologyRecord:
    """Metadata for a known-good standard topology family."""

    name: str
    template_factory: Callable[[], CircuitTemplate]
    family: str
    behaviors: frozenset[str]
    analysis_kinds: frozenset[str]
    priority: int
    topology_risk: float
    tags: tuple[str, ...] = ()
    notes: str = ""

    def template(self) -> CircuitTemplate:
        return self.template_factory()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "behaviors": sorted(self.behaviors),
            "analysis_kinds": sorted(self.analysis_kinds),
            "priority": self.priority,
            "topology_risk": self.topology_risk,
            "tags": list(self.tags),
            "notes": self.notes,
        }


def standard_topology_catalog() -> list[TopologyRecord]:
    """Return the curated first-generation standard topology catalog."""

    return [
        TopologyRecord(
            name="rc_lowpass",
            template_factory=RCLowPass,
            family="passive_rc",
            behaviors=frozenset({"lowpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=10,
            topology_risk=0.05,
            tags=("one_pole", "passive", "minimum_parts"),
            notes="Canonical one-pole low-pass, useful as the simplest baseline.",
        ),
        TopologyRecord(
            name="rc_highpass",
            template_factory=RCHighPass,
            family="passive_rc",
            behaviors=frozenset({"highpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=10,
            topology_risk=0.05,
            tags=("one_pole", "passive", "minimum_parts"),
            notes="Canonical one-pole high-pass, output across the resistor.",
        ),
        TopologyRecord(
            name="rlc_bandpass",
            template_factory=RLCBandPass,
            family="passive_rlc",
            behaviors=frozenset({"bandpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=20,
            topology_risk=0.18,
            tags=("second_order", "passive", "inductor"),
            notes="Series RLC band-pass; practical inductor cost/area can dominate.",
        ),
        TopologyRecord(
            name="shunt_resistor_impedance",
            template_factory=ShuntResistorImpedance,
            family="termination",
            behaviors=frozenset({"impedance", "constant_impedance"}),
            analysis_kinds=frozenset({"input_impedance"}),
            priority=5,
            topology_risk=0.02,
            tags=("termination", "minimum_parts", "passive"),
            notes="One-resistor input impedance target.",
        ),
        TopologyRecord(
            name="output_resistor_impedance",
            template_factory=OutputResistorImpedance,
            family="termination",
            behaviors=frozenset({"output_impedance", "constant_output_impedance"}),
            analysis_kinds=frozenset({"output_impedance"}),
            priority=6,
            topology_risk=0.04,
            tags=("termination", "output_matching", "minimum_parts", "passive"),
            notes="One-resistor output impedance target for matching stages.",
        ),
        TopologyRecord(
            name="buffered_cascade_rc_lowpass",
            template_factory=BufferedCascadeRCLowPass,
            family="active_rc",
            behaviors=frozenset({"lowpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=35,
            topology_risk=0.22,
            tags=("two_stage", "buffered", "active_rc"),
            notes="Two RC stages isolated by an ideal unity-gain buffer.",
        ),
        TopologyRecord(
            name="sallen_key_lowpass",
            template_factory=SallenKeyLowPass,
            family="active_rc",
            behaviors=frozenset({"lowpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=22,
            topology_risk=0.16,
            tags=("second_order", "sallen_key", "unity_gain", "active_rc"),
            notes="Classic unity-gain Sallen-Key low-pass section.",
        ),
        TopologyRecord(
            name="sallen_key_gain_lowpass",
            template_factory=SallenKeyGainLowPass,
            family="active_rc",
            behaviors=frozenset({"lowpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=24,
            topology_risk=0.19,
            tags=("second_order", "sallen_key", "gain", "active_rc"),
            notes="Sallen-Key low-pass section with ideal non-inverting closed-loop gain.",
        ),
        TopologyRecord(
            name="gain_rc_lowpass",
            template_factory=GainRCLowPass,
            family="active_rc",
            behaviors=frozenset({"lowpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=25,
            topology_risk=0.2,
            tags=("gain", "active_rc", "non_inverting"),
            notes="One-pole low-pass followed by an ideal non-inverting gain block.",
        ),
        TopologyRecord(
            name="gain_rc_highpass",
            template_factory=GainRCHighPass,
            family="active_rc",
            behaviors=frozenset({"highpass", "samples", "zpk"}),
            analysis_kinds=frozenset({"voltage_transfer"}),
            priority=25,
            topology_risk=0.2,
            tags=("gain", "active_rc", "non_inverting"),
            notes="One-pole high-pass followed by an ideal non-inverting gain block.",
        ),
        TopologyRecord(
            name="transimpedance_amplifier",
            template_factory=TransimpedanceAmplifier,
            family="active_transimpedance",
            behaviors=frozenset({"transimpedance", "tia", "samples"}),
            analysis_kinds=frozenset({"transimpedance"}),
            priority=12,
            topology_risk=0.24,
            tags=("tia", "current_input", "photodiode", "active"),
            notes="Ideal op-amp TIA with feedback R and compensation C.",
        ),
    ]


def topology_record_by_name(name: str) -> TopologyRecord | None:
    for record in standard_topology_catalog():
        if record.name == name:
            return record
    return None


def compatible_topology_records(
    spec: SynthesisSpec,
    records: Iterable[TopologyRecord] | None = None,
) -> list[TopologyRecord]:
    behavior_kind = str(spec.behavior["kind"]).strip().lower()
    analysis_kind = spec.analysis.kind
    compatible: list[TopologyRecord] = []
    for record in records or standard_topology_catalog():
        if behavior_kind not in record.behaviors and "*" not in record.behaviors:
            continue
        if analysis_kind not in record.analysis_kinds and "*" not in record.analysis_kinds:
            continue
        template = record.template()
        if template.is_compatible(spec.library, spec.optimization.max_components, behavior_kind):
            compatible.append(record)
    return compatible


def ranked_topology_records(
    spec: SynthesisSpec,
    records: Iterable[TopologyRecord] | None = None,
) -> list[TopologyRecord]:
    compatible = compatible_topology_records(spec, records)
    return sorted(compatible, key=lambda record: _record_rank(record, spec))


def _record_rank(record: TopologyRecord, spec: SynthesisSpec) -> tuple[float, int, int, str]:
    behavior_kind = str(spec.behavior["kind"]).strip().lower()
    template = record.template()
    gain = float(spec.behavior.get("gain", 1.0))
    active_bonus = -6 if gain > 1.05 and "gain" in record.tags else 0
    passive_penalty = 8 if gain > 1.05 and "passive" in record.tags else 0
    exact_behavior = 0 if behavior_kind in record.behaviors else 1
    return (
        exact_behavior + record.topology_risk,
        record.priority + active_bonus + passive_penalty,
        template.component_count,
        record.name,
    )


@dataclass
class CatalogProposer:
    """Topology proposer backed by the curated standard topology catalog."""

    records: list[TopologyRecord] | None = None
    max_candidates: int | None = None

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        records = ranked_topology_records(spec, self.records)
        if self.max_candidates is not None:
            records = records[: self.max_candidates]
        return [record.template() for record in records]
