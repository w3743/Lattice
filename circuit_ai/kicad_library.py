"""Discovery and indexing for locally installed KiCad libraries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class LibraryPart:
    kind: str
    name: str
    library: str
    source_path: str
    description: str = ""
    keywords: str = ""
    footprint: str = ""
    datasheet: str = ""
    reference: str = ""
    value: str = ""
    pin_count: int = 0
    simulation_device: str = ""
    simulation_model: str = ""
    simulation_library: str = ""
    simulation_name: str = ""
    simulation_type: str = ""
    # Optional manufacturer/supplier metadata.  Legacy KiCad libraries often
    # store these in F4+ fields (or in a matching .dcm document library).
    part_number: str = ""
    manufacturer_part_number: str = ""
    manufacturer: str = ""
    category: str = ""
    family: str = ""
    status: str = ""
    detail_page: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SpiceModel:
    name: str
    kind: str
    source_path: str
    pins: tuple[str, ...] = ()
    model_type: str = ""

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "pins": list(self.pins)}


def discover_library_roots(extra_roots: Iterable[Path] = ()) -> list[Path]:
    candidates = [Path(path) for path in extra_roots]
    configured = os.environ.get("CIRCUIT_AI_KICAD_LIBRARY_PATH", "")
    candidates.extend(Path(path) for path in configured.split(os.pathsep) if path)
    for base in (Path(os.environ.get("ProgramFiles", "C:/Program Files")), Path(os.environ.get("LOCALAPPDATA", ""))):
        candidates.extend(base / "KiCad" / version / "share" / "kicad" for version in ("10.0", "9.0", "8.0"))
    result: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists() and candidate.resolve() not in result:
            result.append(candidate.resolve())
    return result


def build_index(index_path: Path, roots: Iterable[Path] = ()) -> dict[str, object]:
    discovered = discover_library_roots(roots)
    parts: list[LibraryPart] = []
    models: list[SpiceModel] = []
    for root in discovered:
        doc_metadata: dict[str, dict[str, str]] = {}
        for path in root.rglob("*.dcm"):
            doc_metadata.update(parse_doc_file(path))
        symbol_files = list(root.rglob("*.kicad_sym"))
        footprint_files = list(root.rglob("*.kicad_mod"))
        for path in symbol_files:
            parts.extend(parse_symbol_file(path))
        for path in footprint_files:
            text = path.read_text(encoding="utf-8", errors="replace")
            parts.append(LibraryPart("footprint", path.stem, path.parent.stem.removesuffix(".pretty"), str(path), pin_count=len(re.findall(r"\(pad\s+", text))))
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in {".lib", ".sub", ".cir", ".sp", ".spice"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                if path.suffix.casefold() == ".lib" and _looks_like_legacy_kicad_library(text):
                    parts.extend(parse_legacy_symbol_file(path, doc_metadata))
                else:
                    models.extend(parse_spice_file(path))
    payload = {"version": 2, "roots": [str(path) for path in discovered], "parts": [part.as_dict() for part in parts], "models": [model.as_dict() for model in models]}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"roots": payload["roots"], "symbol_count": sum(part.kind == "symbol" for part in parts), "footprint_count": sum(part.kind == "footprint" for part in parts), "model_count": len(models), "index_path": str(index_path)}


def parse_symbol_file(path: Path) -> list[LibraryPart]:
    text = path.read_text(encoding="utf-8", errors="replace")
    library = path.stem
    parts: list[LibraryPart] = []
    for match in re.finditer(r'^\s*\(symbol\s+"([^"]+)"', text, re.MULTILINE):
        name = match.group(1)
        if name.startswith(f"{library}:"):
            name = name[len(library) + 1 :]
        if re.search(r"_\d+_\d+$", name):
            continue
        block = text[match.start() : _balanced_end(text, match.start())]
        property_value = lambda key: _property(block, key)
        parts.append(LibraryPart(
            "symbol", name, library, str(path), property_value("Description"), property_value("ki_keywords"),
            property_value("Footprint"), property_value("Datasheet"), property_value("Reference"),
            property_value("Value"), len(re.findall(r"\(pin\s+", block)), property_value("Sim.Device"),
            property_value("Sim.Params"), property_value("Sim.Library"), property_value("Sim.Name"),
            property_value("Sim.Type"),
        ))
    return parts


def search_index(index_path: Path, query: str = "", kind: str | None = None, limit: int = 50) -> list[dict[str, object]]:
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    words = query.casefold().split()
    matches = []
    candidates = payload.get("parts", []) + [{**model, "kind": "model"} for model in payload.get("models", [])]
    for part in candidates:
        if kind and part.get("kind") != kind:
            continue
        haystack = " ".join(
            str(part.get(key, ""))
            for key in (
                "name", "library", "description", "keywords", "footprint",
                "part_number", "manufacturer_part_number", "manufacturer",
                "category", "family", "status",
            )
        ).casefold()
        if all(word in haystack for word in words):
            matches.append(part)
            if len(matches) >= max(1, min(limit, 200)):
                break
    return matches


def index_status(index_path: Path) -> dict[str, object]:
    if not index_path.exists():
        return {"ready": False, "roots": [], "symbol_count": 0, "footprint_count": 0, "model_count": 0, "index_path": str(index_path)}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    parts = payload.get("parts", [])
    return {"ready": True, "roots": payload.get("roots", []), "symbol_count": sum(part.get("kind") == "symbol" for part in parts), "footprint_count": sum(part.get("kind") == "footprint" for part in parts), "model_count": len(payload.get("models", [])), "index_path": str(index_path)}


def parse_spice_file(path: Path) -> list[SpiceModel]:
    models: list[SpiceModel] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        subckt = re.match(r"(?i)^\.subckt\s+(\S+)(?:\s+(.*))?$", line)
        if subckt:
            tokens = tuple((subckt.group(2) or "").split())
            pins = tuple(token for token in tokens if "=" not in token and not token.startswith("+") )
            models.append(SpiceModel(subckt.group(1), "subcircuit", str(path), pins))
            continue
        model = re.match(r"(?i)^\.model\s+(\S+)\s*\(?\s*([A-Za-z][A-Za-z0-9_]*)?", line)
        if model:
            models.append(SpiceModel(model.group(1), "model", str(path), (), model.group(2) or ""))
    return models


def parse_legacy_symbol_file(
    path: Path,
    doc_metadata: dict[str, dict[str, str]] | None = None,
) -> list[LibraryPart]:
    """Parse KiCad 5 ``.lib`` symbols, including supplier F4+ fields.

    Digi-Key's partner library intentionally ships a consolidated legacy
    ``.lib`` for KiCad 5 compatibility.  Treating every ``.lib`` as SPICE
    silently loses the actual component identities, so this parser keeps the
    symbol, footprint, and supplier metadata in the same index as modern
    ``.kicad_sym`` parts.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = doc_metadata or {}
    parts: list[LibraryPart] = []
    for match in re.finditer(r"(?ms)^DEF\s+(\S+)\s+(\S+).*?^ENDDEF\s*$", text):
        name, reference = match.group(1), match.group(2)
        block = match.group(0)
        fields: dict[int, str] = {}
        field_names: dict[str, str] = {}
        for raw_line in block.splitlines():
            field = re.match(r"^F(\d+)\s+\"((?:\\.|[^\"])*)\".*$", raw_line.strip())
            if not field:
                continue
            number = int(field.group(1))
            quoted = re.findall(r'\"((?:\\.|[^\"])*)\"', raw_line)
            if not quoted:
                continue
            value = quoted[0].replace('\\"', '"')
            fields[number] = value
            if len(quoted) > 1:
                field_names[quoted[-1]] = value
        doc = metadata.get(name, {})
        def field_value(label: str, number: int = -1) -> str:
            return field_names.get(label, fields.get(number, "")) or doc.get(label, "")

        parts.append(
            LibraryPart(
                kind="symbol",
                name=name,
                library=path.stem,
                source_path=str(path),
                description=field_value("Description") or doc.get("description", ""),
                keywords=doc.get("keywords", ""),
                footprint=fields.get(2, ""),
                datasheet=fields.get(3, "") or doc.get("datasheet", ""),
                reference=fields.get(0, reference),
                value=fields.get(1, name),
                pin_count=len(re.findall(r"(?m)^X\s+", block)),
                part_number=field_value("Digi-Key_PN") or doc.get("part_number", ""),
                manufacturer_part_number=field_value("MPN"),
                manufacturer=field_value("Manufacturer"),
                category=field_value("Category"),
                family=field_value("Family"),
                status=field_value("Status"),
                detail_page=field_value("DK_Detail_Page"),
            )
        )
    return parts


def parse_doc_file(path: Path) -> dict[str, dict[str, str]]:
    """Parse a legacy KiCad ``.dcm`` document library."""
    text = path.read_text(encoding="utf-8", errors="replace")
    records: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"(?ms)^\$CMP\s+(\S+)\s*$.*?^\$ENDCMP\s*$", text):
        name, block = match.group(1), match.group(0)
        description = re.search(r"(?m)^D\s*(.*)$", block)
        keywords = re.search(r"(?m)^K\s*(.*)$", block)
        datasheet = re.search(r"(?m)^F\s*(.*)$", block)
        part_number = (keywords.group(1).strip().split()[0] if keywords and keywords.group(1).strip() else "")
        records[name] = {
            "description": description.group(1).strip() if description else "",
            "keywords": keywords.group(1).strip() if keywords else "",
            "datasheet": datasheet.group(1).strip() if datasheet else "",
            "part_number": part_number,
        }
    return records


def _looks_like_legacy_kicad_library(text: str) -> bool:
    return bool(re.search(r"(?m)^EESchema-LIBRARY\s+Version\s+", text) and re.search(r"(?m)^DEF\s+", text))


def validate_model_binding(symbol: LibraryPart, models: Iterable[SpiceModel]) -> dict[str, object]:
    """Validate the model binding that a KiCad symbol would use."""
    models = tuple(models)
    reference = symbol.reference[:1].upper()
    if not symbol.simulation_library and not symbol.simulation_name:
        if symbol.pin_count == 2 and reference in {"R", "L", "C"}:
            return {"status": "inferred_ideal", "message": "KiCad can infer an ideal passive model"}
        return {"status": "missing", "message": "active or multi-pin symbol has no SPICE model binding"}
    candidates = [model for model in models if model.name == symbol.simulation_name]
    if not candidates:
        return {"status": "missing", "message": f"model {symbol.simulation_name!r} was not found"}
    model = candidates[0]
    if model.kind == "subcircuit" and model.pins and len(model.pins) != symbol.pin_count:
        return {"status": "pin_mismatch", "message": f"symbol has {symbol.pin_count} pins but model has {len(model.pins)}", "model": model.as_dict()}
    return {"status": "valid", "message": "model exists; explicit pin mapping is still required", "model": model.as_dict()}


def _property(block: str, name: str) -> str:
    match = re.search(rf'\(property\s+"{re.escape(name)}"\s+"([^"]*)"', block)
    return match.group(1) if match else ""


def _balanced_end(text: str, start: int) -> int:
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)
