"""Explicit CircuitGraph/device bindings -> native JLCEDA Pro project.

Uses the installed desktop client. Device selection and model equivalence are
caller responsibilities; the exporter verifies identity and every pin/net.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from pint.errors import UndefinedUnitError

from .graph import CircuitGraph
from .lceda import (LcedaNetlist, parse_protel2_netlist, verify_lceda_connectivity,
                    verify_epro2_library_references)
from .units import UNIT_REGISTRY


class LcedaClient:
    def __init__(self, executable: str | Path):
        self.executable = str(executable)
        self.closed_sessions: list[str] = []

    def command(self, *args: str, timeout: float = 45):
        result = subprocess.run([self.executable, *args], capture_output=True,
                                encoding="utf-8", timeout=timeout, check=False)
        try:
            response = json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("JLCEDA returned no valid JSON response") from exc
        if result.returncode or response.get("ok") is not True:
            raise RuntimeError(f"JLCEDA command failed: {response.get('error', result.stderr)}")
        return response.get("value")

    def invoke(self, session: str, code: str, args: dict | None = None):
        return self.command("invoke", "--session", session, "--ext-uuid", "eda",
                            "--code", code, "--args", json.dumps(args or {}, ensure_ascii=False),
                            "--timeout", "35000")

    @contextmanager
    def open(self, path: Path):
        session = self.command("open", "--path", path.resolve().as_posix(),
                               "--headless", "true")["sessionId"]
        try:
            deadline = time.monotonic() + 20
            while not self.invoke(session, "return Boolean(await eda.dmt_Project.getCurrentProjectInfo());"):
                if time.monotonic() >= deadline:
                    raise TimeoutError("JLCEDA project did not become ready")
                time.sleep(0.5)
            yield session
        finally:
            closed = self.command("session", "close", "--session", session, "--destroy")
            if not closed or closed.get("status") != "closed":
                raise RuntimeError(f"JLCEDA session not confirmed closed: {session}")
            self.closed_sessions.append(session)


def validate_export_plan(data: dict):
    graph = CircuitGraph.from_dict(data["graph"]).require_valid()
    if data.get("graph_hash") != graph.graph_hash:
        raise ValueError("device bindings belong to a different graph hash")
    devices = data["devices"]
    pins = {ref: item["terminal_pins"] for ref, item in devices.items()}
    names = {n.net_id: n.name for n in graph.nets}
    names.update(data.get("net_names", {}))
    expected = {}
    for component in graph.components:
        if any(parameter.variable_id is not None for parameter in component.parameters):
            raise ValueError("resolve graph parameter variables before native export")
        binding = devices.get(component.reference)
        if not binding:
            raise ValueError(f"missing device binding: {component.reference}")
        if not binding.get("library_uuid") or not binding.get("device_uuid"):
            raise ValueError("explicit library and device UUIDs are required")
        props = binding.get("expected_properties", {})
        if not props.get("PARTTYPE") or not props.get("FOOTPRINT"):
            raise ValueError("expected manufacturer part and footprint are required")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in props.items()):
            raise ValueError("expected editor properties must be strings")
        if component.model.kind in {"R", "C", "L"}:
            parameter = next(p for p in component.parameters if p.name == "value")
            try:
                nominal = float(UNIT_REGISTRY.Quantity(props["Value"]).to(parameter.unit).magnitude)
            except (KeyError, ValueError, TypeError, LookupError, UndefinedUnitError) as exc:
                raise ValueError(f"unreadable library nominal value for {component.reference}") from exc
            if not math.isclose(nominal, float(parameter.value), rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"library nominal value differs from graph for {component.reference}")
        for key in ("x", "y", "rotation"):
            if not isinstance(binding.get(key), (int, float)) or not math.isfinite(binding[key]):
                raise ValueError(f"finite placement {key} is required")
        if binding["rotation"] % 90:
            raise ValueError("only orthogonal device placement is supported")
        for connection in component.connections:
            for number in pins[component.reference].get(connection.terminal, ()):
                key = component.reference, number
                net = names[connection.net_id]
                if key in expected and expected[key] != net:
                    raise ValueError(f"physical pin assigned to different nets: {key}")
                expected[key] = net
    verify_lceda_connectivity(graph, LcedaNetlist({ref: {} for ref in devices}, expected),
                              pins, net_names=data.get("net_names"))
    return graph, pins, expected


_PLACE = """
const b=__CLI__.args.binding;
const c=await eda.sch_PrimitiveComponent.create(
    {libraryUuid:b.library_uuid,uuid:b.device_uuid},b.x,b.y,undefined,b.rotation);
if(!c)throw new Error('device creation returned no component');
const id=c.getState_PrimitiveId();
await eda.sch_PrimitiveComponent.modify(id,{designator:__CLI__.args.reference,
    otherProperty:{...c.getState_OtherProperty(),Designator:__CLI__.args.reference}});
return id;
"""

_READ = """
const result=[];
for(const c of await eda.sch_PrimitiveComponent.getAll()){
    if(c.componentType==='part')result.push({reference:c.designator,
        pins:await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.primitiveId)});
}return result;
"""

_EXPORT = """
async function pack(f){
    if(!f||!f.size)throw new Error('editor returned an empty export');
    return {name:f.name,b64:await new Promise(resolve=>{
        const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);
        r.readAsDataURL(f);})};
}
await eda.sch_Document.save();
const drc=await eda.sch_Drc.check(true,true,true);
return {drc,
    netlist:await pack(await eda.sch_ManufactureData.getNetlistFile('design','Protel2')),
    svg:await pack(await eda.sch_ManufactureData.getSvgFile('design'))};
"""


def _check_readback(data, graph, pins, exported):
    netlist = parse_protel2_netlist(base64.b64decode(exported["netlist"]["b64"]).decode("utf-8"))
    verify_lceda_connectivity(graph, netlist, pins, net_names=data.get("net_names"))
    for ref, binding in data["devices"].items():
        for key, expected in binding["expected_properties"].items():
            actual = netlist.components[ref].get(key)
            if actual != expected:
                raise ValueError(f"{ref} {key}: editor returned {actual!r}, expected {expected!r}")
    return netlist


def export_lceda_project(data: dict, *, client: LcedaClient, bootstrap: Path,
                         project_dir: Path, output_dir: Path, name: str) -> dict:
    """Create a new local project, export/import it, reopen and verify it.

    Refuses existing output/project names. Never edits the bootstrap original,
    changes global permissions, disables DRC rules, or generates a PDF.
    """
    graph, pins, expected = validate_export_plan(data)
    closed_start = len(client.closed_sessions)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("use a simple ASCII project name, at most 64 characters")
    project_dir = project_dir.resolve(strict=True)
    if not project_dir.is_dir() or not bootstrap.is_file() or bootstrap.suffix.lower() != ".eprj2":
        raise ValueError("bootstrap and local project directory must exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory must be empty")
    restored_path = project_dir / f"{name}-readback.eprj2"
    if any((project_dir / (base + ext)).exists() for base in (name, name + "-readback")
           for ext in (".eprj2", ".eprj3", ".eprj", "")):
        raise FileExistsError("project name already exists")
    doctor = client.command("doctor")
    if not doctor or not doctor.get("connected") or not doctor.get("versionMatch"):
        raise RuntimeError("JLCEDA bridge is unavailable or has a version mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = output_dir / "_bootstrap.eprj2"
    shutil.copyfile(bootstrap, seed)
    with client.open(seed) as session:
        if client.invoke(session, "return eda.sys_Environment.isHalfOfflineMode();") is not True:
            raise RuntimeError("this exporter requires the verified local half-offline mode")
        client.invoke(session, """
const a=__CLI__.args;
const id=await eda.dmt_Project.createProject(a.name,a.name,a.folder,undefined,'Lattice native export');
if(!id)throw new Error('project creation returned no UUID');
await eda.dmt_Project.openProject(id);return id;
""", {"name": name, "folder": project_dir.as_posix()})
        time.sleep(2)
        client.invoke(session, """
const p=await eda.dmt_Project.getCurrentProjectInfo();
if(p.friendlyName!==__CLI__.args.name)throw new Error('unexpected active project');
return await eda.dmt_EditorControl.openDocument(p.data.find(b=>b.schematic).schematic.page[0].uuid);
""", {"name": name})
        for ref, binding in data["devices"].items():
            client.invoke(session, _PLACE, {"reference": ref, "binding": binding})
        time.sleep(2)
        readback = client.invoke(session, _READ)
        actual = {(part["reference"], pin["pinNumber"]): pin
                  for part in readback for pin in part["pins"]}
        if set(actual) != set(expected):
            raise ValueError("library pin set differs from explicit device bindings")
        # Labelled stubs preserve named networks without guessing symbol geometry.
        # Positions and outward direction are read from the placed library pins.
        occupied = {}
        for key, net in expected.items():
            pin = actual[key]
            if pin["rotation"] % 90:
                raise ValueError("nonorthogonal library pin requires an explicit routing backend")
            point = (pin["x"], pin["y"])
            if point in occupied:
                if occupied[point] != net:
                    raise ValueError("device placements short different nets")
                continue
            occupied[point] = net
            angle = math.radians(pin["rotation"])
            line = [*point, round(point[0] + 30 * math.cos(angle), 6),
                    round(point[1] + 30 * math.sin(angle), 6)]
            client.invoke(session, """
const w=await eda.sch_PrimitiveWire.create(__CLI__.args.line,__CLI__.args.net);
if(!w)throw new Error('wire creation failed');return w.primitiveId;
""", {"line": line, "net": net})
        time.sleep(5)
        before = client.invoke(session, _EXPORT)
        before_netlist = _check_readback(data, graph, pins, before)
        # Import the editor-produced archive using the installed, working API.
        imported = client.invoke(session, """
const f=await eda.sys_FileManager.getProjectFile('design',undefined,'epro2');
if(!f?.size)throw new Error('empty archive');
const b64=await new Promise(resolve=>{const r=new FileReader();
 r.onload=()=>resolve(String(r.result).split(',')[1]);r.readAsDataURL(f);});
const result=await eda.sys_FileManager.importProjectByProjectFile(f,'JLCEDA Pro',undefined,
 {operation:'New Project',newProjectOwnerTeamUuid:__CLI__.args.folder,
  newProjectFriendlyName:__CLI__.args.name});
if(!result?.uuid)throw new Error('archive import returned no project');
return {uuid:result.uuid,archive:{b64}};
""", {"folder": project_dir.as_posix(), "name": name + "-readback"})
    if not imported or not restored_path.is_file():
        raise RuntimeError("archive import did not create the expected local project")
    archive_bytes = base64.b64decode(imported["archive"]["b64"])
    library_counts = verify_epro2_library_references(archive_bytes)
    native = output_dir / "design.eprj2"
    shutil.copyfile(restored_path, native)
    with client.open(native) as session:
        client.invoke(session, """
const p=await eda.dmt_Project.getCurrentProjectInfo();
return await eda.dmt_EditorControl.openDocument(p.data.find(b=>b.schematic).schematic.page[0].uuid);
""")
        time.sleep(2)
        after = client.invoke(session, _EXPORT)
        after_netlist = _check_readback(data, graph, pins, after)
        if before_netlist != after_netlist:
            raise ValueError("component properties or connectivity changed on archive import")
        # DOM is used only to read the active sample's DRC log: the API returns
        # counts, not individual messages, on the verified 4.1.60 desktop build.
        details = client.invoke(session, """
return Array.from(parent.document.querySelectorAll('[class*=log-container]'))
 .map(e=>e.innerText).filter(t=>t.includes('[警告]')||t.includes('[错误]'));
""")
    (output_dir / "design.epro2").write_bytes(archive_bytes)
    for kind, suffix in (("svg", ".svg"), ("netlist", ".net")):
        (output_dir / ("design" + suffix)).write_bytes(base64.b64decode(after[kind]["b64"]))
    report = {"status": "native_roundtrip_verified", "graph_hash": graph.graph_hash,
              "client_version": doctor.get("bridgeVersion"), "drc": after["drc"],
              "strict_drc_pass": after["drc"] == [], "drc_details": details,
              "model_device_equivalence_verified": False,
              "embedded_library_counts": library_counts,
              "owned_cli_sessions_closed": client.closed_sessions[closed_start:]}
    report["artifacts"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in output_dir.glob("design.*")}
    try:
        seed.unlink()
    except OSError:
        report["cleanup_note"] = "Client still holds the temporary bootstrap; it was not stopped."
    (output_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--client", required=True, type=Path)
    parser.add_argument("--bootstrap", required=True, type=Path)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    report = export_lceda_project(json.loads(args.plan.read_text(encoding="utf-8")),
        client=LcedaClient(args.client), bootstrap=args.bootstrap, project_dir=args.project_dir,
        output_dir=args.out, name=args.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
