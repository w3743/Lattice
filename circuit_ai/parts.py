"""Concrete, supplier-backed component catalogues.

The synthesis engine historically optimized abstract ``R``/``C``/``L``
values and attached a real component late.  This module makes the opposite
contract possible: a candidate may opt into a catalogue and every
parameterized element must be selected from an imported, traceable part.
Numeric ratings are deliberately optional.  A symbol/footprint library is
not a SPICE model database; the catalog records that fact instead of filling
in guessed values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class PartSelectionPolicy:
    """Rules applied when a design is bound to concrete parts."""

    require_concrete: bool = False
    require_numeric_values: bool = True
    allowed_statuses: tuple[str, ...] = ("active",)
    preferred_part_numbers: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PartSelectionPolicy":
        data = dict(data or {})
        statuses = data.get("allowed_statuses", data.get("status_allowlist", ("active",)))
        if isinstance(statuses, str):
            statuses = (statuses,)
        preferred = data.get("preferred_part_numbers", data.get("preferred_parts", ()))
        if isinstance(preferred, str):
            preferred = (preferred,)
        return cls(
            require_concrete=bool(data.get("require_concrete", data.get("required", False))),
            require_numeric_values=bool(data.get("require_numeric_values", True)),
            allowed_statuses=tuple(str(item).strip().casefold() for item in statuses if str(item).strip()),
            preferred_part_numbers=tuple(str(item).strip() for item in preferred if str(item).strip()),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartRecord:
    part_id: str
    name: str
    library: str
    source_path: str
    symbol: str = ""
    footprint: str = ""
    reference: str = ""
    description: str = ""
    keywords: str = ""
    datasheet: str = ""
    detail_page: str = ""
    digi_key_part_number: str = ""
    manufacturer_part_number: str = ""
    manufacturer: str = ""
    category: str = ""
    family: str = ""
    status: str = ""
    pin_count: int = 0
    nominal_value: float | None = None
    nominal_unit: str = ""
    model_name: str = ""
    model_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_status(self) -> str:
        return self.status.strip().casefold()

    @property
    def identity(self) -> str:
        return self.digi_key_part_number or self.manufacturer_part_number or self.part_id

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class PartCatalog:
    records: tuple[PartRecord, ...]
    source: str = ""
    snapshot: str = ""

    @classmethod
    def from_index(cls, index_path: str | Path, *, source: str | None = None) -> "PartCatalog":
        path = Path(index_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = cls._records_from_parts(payload.get("parts", []))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(tuple(records), source=source or str(path), snapshot=digest)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str | Path) -> "PartCatalog":
        if isinstance(data, (str, Path)):
            return cls.from_index(data)
        records = data.get("records", data.get("parts", []))
        if records and isinstance(records[0], PartRecord):
            parsed = tuple(records)
        else:
            parsed = tuple(cls._records_from_parts(records))
        payload = json.dumps([item.as_dict() for item in parsed], sort_keys=True, ensure_ascii=False).encode("utf-8")
        return cls(
            parsed,
            source=str(data.get("source", "inline")),
            snapshot=str(data.get("snapshot", hashlib.sha256(payload).hexdigest())),
        )

    @staticmethod
    def _records_from_parts(parts: Iterable[Mapping[str, Any]]) -> list[PartRecord]:
        result: list[PartRecord] = []
        seen: set[tuple[str, str]] = set()
        for raw in parts:
            if str(raw.get("kind", "symbol")) != "symbol":
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            part_number = str(raw.get("part_number", raw.get("digi_key_part_number", ""))).strip()
            mpn = str(raw.get("manufacturer_part_number", raw.get("mpn", ""))).strip()
            part_id = part_number or mpn or name
            source_path = str(raw.get("source_path", ""))
            key = (part_id.casefold(), source_path.casefold())
            if key in seen:
                continue
            seen.add(key)
            nominal_value, nominal_unit = _parse_nominal_value(raw.get("nominal_value"), raw.get("value", ""))
            result.append(
                PartRecord(
                    part_id=part_id,
                    name=name,
                    library=str(raw.get("library", "")),
                    source_path=source_path,
                    symbol=name,
                    footprint=str(raw.get("footprint", "")),
                    reference=str(raw.get("reference", "")),
                    description=str(raw.get("description", "")),
                    keywords=str(raw.get("keywords", "")),
                    datasheet=str(raw.get("datasheet", "")),
                    detail_page=str(raw.get("detail_page", "")),
                    digi_key_part_number=part_number,
                    manufacturer_part_number=mpn,
                    manufacturer=str(raw.get("manufacturer", "")),
                    category=str(raw.get("category", "")),
                    family=str(raw.get("family", "")),
                    status=str(raw.get("status", "")),
                    pin_count=int(raw.get("pin_count", 0) or 0),
                    nominal_value=nominal_value,
                    nominal_unit=nominal_unit,
                    model_name=str(raw.get("simulation_name", raw.get("model_name", ""))),
                    model_path=str(raw.get("simulation_library", raw.get("model_path", ""))),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "snapshot": self.snapshot,
            "records": [record.as_dict() for record in self.records],
        }

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({family for record in self.records for family in _record_families(record) if family}))

    def records_for(
        self,
        element_type: str,
        *,
        policy: PartSelectionPolicy | None = None,
        real_components: Iterable[str] = (),
    ) -> tuple[PartRecord, ...]:
        policy = policy or PartSelectionPolicy()
        family = _normalize_family(element_type)
        requested = tuple(str(item).strip().casefold() for item in real_components if str(item).strip())
        statuses = set(policy.allowed_statuses)
        records = []
        for record in self.records:
            if statuses and record.normalized_status and record.normalized_status not in statuses:
                continue
            if not _record_matches_family(record, family):
                continue
            if requested and not _matches_requested(record, requested, family):
                continue
            records.append(record)
        preferred = {item.casefold(): index for index, item in enumerate(policy.preferred_part_numbers)}
        records.sort(key=lambda item: (preferred.get(item.identity.casefold(), len(preferred)), item.identity.casefold()))
        return tuple(records)

    def numeric_values(
        self,
        element_type: str,
        lower: float | None = None,
        upper: float | None = None,
        *,
        policy: PartSelectionPolicy | None = None,
        real_components: Iterable[str] = (),
    ) -> tuple[float, ...]:
        values = {
            float(record.nominal_value)
            for record in self.records_for(element_type, policy=policy, real_components=real_components)
            if record.nominal_value is not None and math.isfinite(record.nominal_value)
        }
        if lower is not None:
            values = {value for value in values if value >= float(lower)}
        if upper is not None:
            values = {value for value in values if value <= float(upper)}
        return tuple(sorted(values))

    def has_family(
        self,
        element_type: str,
        *,
        policy: PartSelectionPolicy | None = None,
        real_components: Iterable[str] = (),
        require_numeric: bool = False,
    ) -> bool:
        records = self.records_for(element_type, policy=policy, real_components=real_components)
        if not records:
            return False
        if require_numeric:
            return any(record.nominal_value is not None for record in records)
        return True

    def select(
        self,
        element_type: str,
        *,
        policy: PartSelectionPolicy | None = None,
        real_components: Iterable[str] = (),
        nominal_value: float | None = None,
    ) -> PartRecord | None:
        records = self.records_for(element_type, policy=policy, real_components=real_components)
        if nominal_value is not None:
            with_value = [record for record in records if record.nominal_value is not None]
            if with_value:
                return min(with_value, key=lambda record: abs(math.log10(record.nominal_value) - math.log10(nominal_value)))
            if policy and policy.require_numeric_values:
                return None
        return records[0] if records else None


def _normalize_family(identifier: str) -> str:
    text = str(identifier).strip().casefold()
    if text in {"r", "resistor", "resistors"} or "device:r" in text or "resistor" in text:
        return "R"
    if text in {"c", "capacitor", "capacitors"} or "device:c" in text or "capacitor" in text:
        return "C"
    if text in {"l", "inductor", "inductors"} or "device:l" in text or "inductor" in text:
        return "L"
    if any(token in text for token in ("opamp", "operational", "amplifier")):
        return "opamp"
    if "diode" in text:
        return "diode"
    if "mosfet" in text or "transistor" in text or text in {"q", "mos"}:
        return "transistor"
    return str(identifier).strip()


def _record_families(record: PartRecord) -> set[str]:
    text = " ".join((record.name, record.category, record.family, record.description, record.keywords)).casefold()
    families: set[str] = set()
    for family, tokens in {
        "R": ("resistor",),
        "C": ("capacitor",),
        "L": ("inductor",),
        "diode": ("diode",),
        "transistor": ("mosfet", "transistor"),
        "opamp": ("op amp", "opamp", "operational amplifier"),
    }.items():
        if any(token in text for token in tokens):
            families.add(family)
    return families


def _record_matches_family(record: PartRecord, family: str) -> bool:
    if family in _record_families(record):
        return True
    return family.casefold() in " ".join((record.category, record.family, record.name)).casefold()


def _matches_requested(record: PartRecord, requested: tuple[str, ...], family: str) -> bool:
    searchable = " ".join(
        (record.part_id, record.name, record.digi_key_part_number, record.manufacturer_part_number,
         record.category, record.family, record.library)
    ).casefold()
    for item in requested:
        if item in searchable or item == family.casefold():
            return True
    return False


def _parse_nominal_value(value: Any, fallback: Any) -> tuple[float | None, str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return (parsed, "") if math.isfinite(parsed) and parsed > 0 else (None, "")
    text = str(value or fallback or "").strip()
    match = re.fullmatch(
        r"(?i)([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)([pnumkmg]?)([fhΩohm]*)",
        text,
    )
    if not match:
        return None, ""
    scale = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "g": 1e9}
    parsed = float(match.group(1)) * scale[match.group(2).casefold()]
    return (parsed, match.group(3).casefold()) if parsed > 0 else (None, "")
