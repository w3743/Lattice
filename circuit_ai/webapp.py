"""Local browser workbench for Circuit AI synthesis."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .capabilities import build_default_registry
from .formatting import db20
from .experts import ExpertTopologySelector
from .ir import pbdl_to_ir
from .kicad_library import build_index, index_status, search_index, validate_model_binding
from .catalog import ranked_topology_records
from .error_reporting import error_report, exception_report
from .pbdl_boundary import canonicalize_pbdl_dict, load_pbdl_dict, translate_pbdl_dict
from .pbdl_runner import run_pbdl_file
from .spec import SynthesisSpec
from .synthesis import CircuitSynthesizer, SynthesisResult, write_results
from 端口描述语言 import compile_function_graph, to_circuit_ai_plan


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "web"
OUTPUT_ROOT = ROOT / "outputs" / "workbench"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
COMPONENT_INDEX = ROOT / "data" / "kicad_component_index.json"


class SynthesisRequest(BaseModel):
    spec: dict[str, Any]


class PBDLRequest(BaseModel):
    spec: dict[str, Any]
    execute: bool = True
    top_k: int | None = None


class SpecCompileRequest(BaseModel):
    spec: dict[str, Any]


class TopologySuggestionRequest(BaseModel):
    spec: dict[str, Any]


class ComponentIndexRequest(BaseModel):
    roots: list[str] = []


class ComponentValidationRequest(BaseModel):
    identifier: str


def _raise_workbench_http(exc: BaseException, phase: str, status_code: int = 422) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=exception_report(exc, phase),
    ) from exc


app = FastAPI(title="Circuit AI Workbench")
app.mount("/artifacts", StaticFiles(directory=OUTPUT_ROOT), name="artifacts")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/store.js")
def store_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "store.js", media_type="application/javascript")


@app.get("/app.css")
def app_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.css", media_type="text/css")


@app.get("/power-results.css")
def power_results_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "power-results.css", media_type="text/css")


@app.get("/api/workbench/architecture")
def workbench_architecture() -> dict[str, Any]:
    """Expose the actual synthesis route the workbench is allowed to invoke."""

    manifest = build_default_registry().manifest()
    return {
        "pipeline": [
            {
                "id": "pbdl",
                "label": "PBDL specification",
                "detail": "ports, relations, analyses, targets, constraints",
                "status": "available",
            },
            {
                "id": "ir",
                "label": "Unified IR and CircuitGraph",
                "detail": "versioned ports, models, parameter bindings, provenance",
                "status": "available",
            },
            {
                "id": "candidates",
                "label": "Expert priors and bounded candidates",
                "detail": "catalog/grammar selection with explicit capability gaps",
                "status": "available",
            },
            {
                "id": "simulation",
                "label": "Simulation and optimization",
                "detail": "linear MNA for AC; ideal averaged models for supported DC power",
                "status": "available",
            },
            {
                "id": "verification",
                "label": "Metrics and constraints",
                "detail": "unit-safe ConstraintReport and feasibility-first selection",
                "status": "available",
            },
            {
                "id": "export",
                "label": "Artifacts",
                "detail": "SVG, KiCad and JSON evidence for supported synthesis routes",
                "status": "available",
            },
        ],
        "limits": [
            "AC final verification uses Linear MNA for currently supported linear models.",
            "DC power synthesis currently uses declared ideal averaged power models, not switch-level SPICE.",
            "Multi-port PBDL can be planned, but not every multi-port request has a joint solver yet.",
        ],
        "capabilities": manifest["capabilities"],
    }


@app.post("/api/spec/compile")
def compile_spec(request: SpecCompileRequest) -> dict[str, Any]:
    """Compile PBDL without executing synthesis.

    The browser uses this as the semantic gate before submission.  It exposes
    the canonical PBDL, lossless IR snapshot, stage plan and any capability
    gaps so a blocked request cannot look like a successful two-port design.
    """
    try:
        canonical_spec = canonicalize_pbdl_dict(request.spec)
        ir = pbdl_to_ir(canonical_spec)
        pbdl_spec = load_pbdl_dict(canonical_spec)
        plan = to_circuit_ai_plan(pbdl_spec)
    except Exception as exc:
        _raise_workbench_http(exc, "pbdl")

    stage_errors = []
    for stage in plan.stages:
        if stage.supported:
            continue
        stage_errors.append(
            error_report(
                "candidate_search" if "端口" in (stage.missing_capability or "") else "pbdl",
                "capability_gap",
                "当前阶段没有可执行能力",
                detail=stage.missing_capability or "stage is not supported",
                suggestion=stage.suggestion or "拆分需求或等待对应求解器能力。",
                severity="warning",
                context={
                    "stage_name": stage.name,
                    "stage_index": stage.analysis_index + 1,
                    "analysis_kind": stage.analysis_kind,
                    "target_kind": stage.target_kind,
                },
            )
        )

    function_compilation = None
    function_errors = []
    if pbdl_spec.functions:
        try:
            function_compilation = compile_function_graph(pbdl_spec.functions, pbdl_spec.ports)
            for behavior in function_compilation.behaviors:
                function = next(item for item in pbdl_spec.functions if item.function_id == behavior.function_id)
                if function.role not in {"target", "predicate"} or behavior.status == "executable":
                    continue
                function_errors.append(
                    error_report(
                        "pbdl",
                        "function_not_executable",
                        "函数已通过类型检查，但当前后端没有对应执行器",
                        detail="; ".join(item.get("message", "") for item in behavior.diagnostics) or behavior.function_id,
                        suggestion="改用 builtin/sampled 函数，或注册 expression_ast/rational 的求解器能力。",
                        severity="warning",
                        context={"function_id": behavior.function_id, "body_kind": function.body.kind},
                    )
                )
        except Exception as exc:
            _raise_workbench_http(exc, "pbdl")

    capability_gaps = [*stage_errors, *function_errors]

    unknown_top_level = sorted(set(request.spec) - set(canonical_spec))
    ready = plan.fully_supported and len(ir.ports) == 2 and not function_errors
    execution_mode = "direct_synthesis" if len(plan.stages) == 1 else "multi_stage_pbdl"
    return {
        "schema": "circuit_ai.spec_compile",
        "schema_version": 1,
        "name": ir.name,
        "canonical_pbdl": canonical_spec,
        "ir": ir.as_dict(),
        "plan": plan.as_dict(),
        "execution": {
            "status": "ready" if ready else "blocked",
            "mode": execution_mode,
            "adapter": "two_port_execution_adapter",
            "port_count": len(ir.ports),
            "required_port_count": 2,
        },
        "diagnostics": {
            "lossless_ir": True,
            "ignored_fields": unknown_top_level,
            "downgraded_fields": [],
            "capability_gaps": capability_gaps,
            "functions": function_compilation.as_dict() if function_compilation is not None else None,
        },
        "errors": capability_gaps,
    }


@app.post("/api/topology-suggestions")
def topology_suggestions(request: TopologySuggestionRequest) -> dict[str, Any]:
    try:
        pbdl_spec = load_pbdl_dict(request.spec)
        ir = pbdl_to_ir(pbdl_spec.as_dict())
        if ir.intent_kind == "dc":
            candidate = ExpertTopologySelector().select(ir)
            return {
                "catalog": [{
                    "name": candidate.name,
                    "family": candidate.family,
                    "tags": [candidate.rationale],
                    "priority": 0,
                }],
                "local_evidence": [],
            }
        spec = SynthesisSpec.from_dict(translate_pbdl_dict(pbdl_spec.as_dict()))
    except Exception as exc:
        _raise_workbench_http(exc, "pbdl")
    records = ranked_topology_records(spec)
    evidence = _local_template_evidence()
    return {
        "catalog": [record.as_dict() for record in records],
        "local_evidence": [
            {"template": record.name, "observations": evidence.get(record.name, 0)}
            for record in records
            if evidence.get(record.name, 0) > 0
        ],
    }


@app.get("/api/components/status")
def component_status() -> dict[str, object]:
    return index_status(COMPONENT_INDEX)


@app.post("/api/components/reindex")
def component_reindex(request: ComponentIndexRequest) -> dict[str, object]:
    try:
        return build_index(COMPONENT_INDEX, [Path(path) for path in request.roots])
    except Exception as exc:
        _raise_workbench_http(exc, "components")


@app.get("/api/components/search")
def component_search(q: str = "", kind: str | None = None, limit: int = 50) -> dict[str, object]:
    try:
        return {"items": search_index(COMPONENT_INDEX, q, kind, limit), **index_status(COMPONENT_INDEX)}
    except Exception as exc:
        _raise_workbench_http(exc, "components")


@app.post("/api/components/validate")
def component_validate(request: ComponentValidationRequest) -> dict[str, object]:
    if not COMPONENT_INDEX.exists():
        raise HTTPException(
            status_code=404,
            detail=error_report(
                "components",
                "component_index_unavailable",
                "本地元件索引尚未建立",
                suggestion="先重建本地 KiCad 元件索引，再验证模型绑定。",
                retryable=True,
            ),
        )
    payload = json.loads(COMPONENT_INDEX.read_text(encoding="utf-8"))
    library, _, name = request.identifier.partition(":")
    symbols = [item for item in payload.get("parts", []) if item.get("kind") == "symbol" and item.get("library") == library and item.get("name") == name]
    if not symbols:
        raise HTTPException(
            status_code=404,
            detail=error_report(
                "components",
                "component_not_found",
                "本地元件索引中没有这个符号",
                detail=request.identifier,
                suggestion="检查库名和符号名，或重新扫描正确的 KiCad 库根目录。",
            ),
        )
    from .kicad_library import LibraryPart, SpiceModel
    symbol = LibraryPart(**symbols[0])
    models = tuple(SpiceModel(**{**model, "pins": tuple(model.get("pins", []))}) for model in payload.get("models", []))
    return {"identifier": request.identifier, "validation": validate_model_binding(symbol, models)}


@app.post("/api/synthesize")
def synthesize(request: SynthesisRequest) -> dict[str, Any]:
    try:
        pbdl_spec = load_pbdl_dict(request.spec)
        canonical_spec = pbdl_spec.as_dict()
        spec = SynthesisSpec.from_dict(translate_pbdl_dict(canonical_spec))
        results = CircuitSynthesizer().synthesize(spec)
    except Exception as exc:
        _raise_workbench_http(exc, "pbdl")

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    output_dir = OUTPUT_ROOT / run_id
    write_results(results, output_dir)
    payload = {
        "run_id": run_id,
        "name": spec.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "spec": canonical_spec,
        "analysis": spec.analysis.as_dict(),
        "candidates": [_candidate_payload(result, run_id, index) for index, result in enumerate(results, start=1)],
    }
    (output_dir / "input_spec.json").write_text(json.dumps(canonical_spec, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "workbench.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


@app.post("/api/pbdl/synthesize")
def synthesize_pbdl(request: PBDLRequest) -> dict[str, Any]:
    run_id = f"pbdl-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "input_pbdl.json"
    try:
        canonical_spec = canonicalize_pbdl_dict(request.spec)
    except Exception as exc:
        _raise_workbench_http(exc, "pbdl")
    spec_path.write_text(json.dumps(canonical_spec, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        report = run_pbdl_file(spec_path, output_dir, execute=request.execute, top_k=request.top_k)
    except Exception as exc:
        _raise_workbench_http(exc, "execution")
    payload = report.as_dict()
    payload.update({"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"), "spec": request.spec})
    for stage in payload["stages"]:
        if stage.get("output_dir"):
            stage_path = Path(stage["output_dir"])
            try:
                relative = stage_path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()
            except ValueError:
                continue
            if stage.get("status") == "executed":
                stage["artifacts"] = {
                    "report": f"/artifacts/{relative}/report.json",
                    "schematic_svg": f"/artifacts/{relative}/best.svg",
                    "kicad_schematic": f"/artifacts/{relative}/best.kicad_sch",
                    "topology_search": f"/artifacts/{relative}/topology_search.json",
                    "simulation_tasks": f"/artifacts/{relative}/simulation_tasks.json",
                }
                stage["design"] = _power_stage_payload(stage_path)
    (output_dir / "pbdl_workbench.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _power_stage_payload(stage_path: Path) -> dict[str, Any] | None:
    report_path = stage_path / "report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("design") != "dc_power":
        return None

    candidates = []
    for item in report.get("candidates", []):
        candidate = item.get("candidate", {})
        task_evaluations = item.get("simulation_task_evaluations", [])
        candidates.append(
            {
                "name": candidate.get("name"),
                "family": candidate.get("family"),
                "objective": item.get("objective"),
                "selection_score": item.get("selection_score"),
                "passed": bool(item.get("validation", {}).get("passed"))
                and all(task.get("passed") for task in task_evaluations),
            }
        )

    topology = report.get("topology", {})
    metadata = topology.get("metadata", {})
    return {
        "topology": {
            "name": topology.get("name"),
            "family": topology.get("family"),
            "rationale": topology.get("rationale"),
            "origin": metadata.get("origin"),
            "knowledge_id": metadata.get("knowledge_id"),
            "solver": metadata.get("solver"),
        },
        "parameters": report.get("parameters", {}),
        "operating_point": report.get("operating_point", {}),
        "validation": report.get("validation", {}),
        "simulation_task_evaluations": report.get("simulation_task_evaluations", []),
        "topology_search": report.get("topology_search", {}),
        "candidates": candidates,
    }


def _candidate_payload(result: SynthesisResult, run_id: str, index: int) -> dict[str, Any]:
    metrics = result.metrics
    response = result.response
    target = result.target
    return {
        "rank": index,
        "template": result.template.name,
        "description": result.template.description,
        "parameters": result.parameters,
        "inventory": result.template.component_inventory(),
        "selected_real_components": list(result.selected_real_components),
        "model_bindings": list(result.model_bindings),
        "metrics": {
            "score": metrics.score,
            "rmse_db": metrics.rmse_db,
            "max_abs_db": metrics.max_abs_db,
            "component_count": metrics.component_count,
            "estimated_cost": metrics.estimated_cost,
            "estimated_area_mm2": metrics.estimated_area_mm2,
            "pareto_rank": metrics.pareto_rank,
            "optimizer_success": metrics.optimizer_success,
            "optimizer_message": metrics.optimizer_message,
            "surrogate": metrics.surrogate,
        },
        "verification": {
            "constraint_report": (
                result.constraint_report.as_dict()
                if result.constraint_report is not None
                else None
            ),
            "fidelity_schedule": (
                result.fidelity_schedule.as_dict()
                if result.fidelity_schedule is not None
                else None
            ),
            "optimization_run": (
                metrics.optimization_run.as_dict()
                if metrics.optimization_run is not None
                else None
            ),
            "simulation_statuses": [item.status.value for item in result.simulation_results],
        },
        "plot": {
            "frequency_hz": _numbers(target.frequencies_hz),
            "target_db": _numbers(db20(target.values)),
            "response_db": _numbers(db20(response)),
            "target_phase_deg": _numbers(np.rad2deg(np.unwrap(np.angle(target.values)))),
            "response_phase_deg": _numbers(np.rad2deg(np.unwrap(np.angle(response)))),
        },
        "artifacts": {
            "schematic_svg": f"/artifacts/{run_id}/candidate_{index}.svg",
            "kicad_schematic": f"/artifacts/{run_id}/candidate_{index}.kicad_sch",
            "spice_netlist": f"/artifacts/{run_id}/best.spice" if index == 1 else None,
            "report": f"/artifacts/{run_id}/report.json",
        },
    }


def _numbers(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _local_template_evidence() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in (ROOT / "data").glob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)
                    name = row.get("template")
                    if isinstance(name, str):
                        counts[name] = counts.get(name, 0) + 1
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    return counts
