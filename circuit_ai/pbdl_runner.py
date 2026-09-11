from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

from .feasibility import FeasibilityError
from .error_reporting import error_report, exception_report, stage_error
from .design_problem import compile_design_problem
from .ir import pbdl_to_ir
from .orchestrator import DesignOrchestrator
from .spec import SynthesisSpec


PBDL_MODULE = "\u7aef\u53e3\u63cf\u8ff0\u8bed\u8a00"


@dataclass(frozen=True)
class PBDLStageRun:
    stage_name: str
    status: str
    analysis_kind: str
    target_kind: str | None
    stage_index: int
    output_dir: str | None = None
    best_template: str | None = None
    best_rmse_db: float | None = None
    message: str = ""
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "analysis_kind": self.analysis_kind,
            "target_kind": self.target_kind,
            "stage_index": self.stage_index,
            "output_dir": self.output_dir,
            "best_template": self.best_template,
            "best_rmse_db": self.best_rmse_db,
            "message": self.message,
            "error": self.error,
        }


@dataclass(frozen=True)
class PBDLRunReport:
    spec_name: str
    plan_path: str
    stages: tuple[PBDLStageRun, ...]

    @property
    def executed_count(self) -> int:
        return sum(1 for stage in self.stages if stage.status == "executed")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for stage in self.stages if stage.status == "unsupported")

    @property
    def error_count(self) -> int:
        return sum(1 for stage in self.stages if stage.status == "error")

    @property
    def succeeded(self) -> bool:
        return self.error_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "plan_path": self.plan_path,
            "executed_count": self.executed_count,
            "unsupported_count": self.unsupported_count,
            "error_count": self.error_count,
            "succeeded": self.succeeded,
            "stages": [stage.as_dict() for stage in self.stages],
        }


def run_pbdl_file(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    execute: bool = True,
    top_k: int | None = None,
    replay_ranker_path: str | Path | None = None,
) -> PBDLRunReport:
    pbdl = importlib.import_module(PBDL_MODULE)
    spec = pbdl.load_spec(spec_path)
    plan = pbdl.to_circuit_ai_plan(spec)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Every PBDL run gets the same execution-facing IR snapshot.  The AC
    # adapter and the DC power expert may use different solvers, but they now
    # share one inspectable input contract.
    design_problem = compile_design_problem(spec.as_dict())
    (output_dir / "ir.json").write_text(
        json.dumps(design_problem.ir.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "design_problem.json").write_text(
        json.dumps(design_problem.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plan_path = output_dir / "plan.json"
    plan_path.write_text(
        json.dumps(plan.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    replay_ranker = None
    if replay_ranker_path is not None:
        from .learned import load_model

        bundle = load_model(replay_ranker_path)
        replay_ranker = bundle.get("ranking_model")
        if not replay_ranker or not replay_ranker.get("enabled"):
            raise ValueError("replay ranker bundle does not contain an enabled ranking_model")
        (output_dir / "replay_ranker_manifest.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "path": str(Path(replay_ranker_path)),
                    "kind": replay_ranker.get("kind", "replay_pairwise_ranker"),
                    "feature_names": list(replay_ranker.get("feature_names", [])),
                    "training_rows": replay_ranker.get("rows"),
                    "training_pairs": replay_ranker.get("pairs"),
                    "message": "experimental within-feasibility-tier candidate ranking",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    stage_runs: list[PBDLStageRun] = []
    orchestrator = DesignOrchestrator()
    for index, stage in enumerate(plan.stages, start=1):
        if not stage.supported or stage.synthesis_spec is None:
            unsupported = stage_error(
                error_report(
                    "pbdl",
                    "capability_gap",
                    "当前阶段没有可执行能力",
                    detail=stage.missing_capability or "stage was not translated into an executable specification",
                    suggestion=stage.suggestion or "拆分需求，或注册对应的分析/求解器能力。",
                    severity="warning",
                ),
                stage_name=stage.name,
                stage_index=index,
            )
            stage_runs.append(
                PBDLStageRun(
                    stage_name=stage.name,
                    status="unsupported",
                    analysis_kind=stage.analysis_kind,
                    target_kind=stage.target_kind,
                    stage_index=index,
                    message=stage.missing_capability or "unsupported stage",
                    error=unsupported,
                )
            )
            continue
        if not execute:
            stage_runs.append(
                PBDLStageRun(
                    stage_name=stage.name,
                    status="planned",
                    analysis_kind=stage.analysis_kind,
                    target_kind=stage.target_kind,
                    stage_index=index,
                    message="stage translated but execution was disabled",
                )
            )
            continue

        stage_dir = output_dir / f"stage_{index}_{_safe_suffix(stage.analysis_kind)}"
        try:
            orchestration = orchestrator.run(
                design_problem,
                stage_index=index - 1,
                translated_spec=stage.synthesis_spec,
                output_dir=stage_dir,
                top_k=top_k,
                replay_ranker=replay_ranker,
            )
            if not orchestration.succeeded:
                raise ValueError(
                    "stage validation failed: "
                    + "; ".join(orchestration.selected.validation.issues)
                )
            best = orchestration.selected
            native = orchestration.selected_native
            metrics = getattr(native, "metrics", None)
            stage_runs.append(
                PBDLStageRun(
                    stage_name=stage.name,
                    status="executed",
                    analysis_kind=stage.analysis_kind,
                    target_kind=stage.target_kind,
                    stage_index=index,
                    output_dir=str(stage_dir),
                    best_template=best.name,
                    best_rmse_db=(
                        float(metrics.rmse_db)
                        if metrics is not None and getattr(metrics, "rmse_db", None) is not None
                        else None
                    ),
                    message=(
                        str(metrics.optimizer_message)
                        if metrics is not None
                        else str(getattr(native, "optimizer_message", ""))
                    ),
                )
            )
        except FeasibilityError as exc:
            stage_dir.mkdir(parents=True, exist_ok=True)
            failure = stage_error(
                exception_report(exc, "verification", context={"output_dir": str(stage_dir)}),
                stage_name=stage.name,
                stage_index=index,
            )
            (stage_dir / "feasibility.json").write_text(
                json.dumps(exc.report.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stage_runs.append(
                PBDLStageRun(
                    stage_name=stage.name,
                    status="error",
                    analysis_kind=stage.analysis_kind,
                    target_kind=stage.target_kind,
                    stage_index=index,
                    output_dir=str(stage_dir),
                    message=str(exc),
                    error=failure,
                )
            )
        except Exception as exc:
            stage_dir.mkdir(parents=True, exist_ok=True)
            fallback_phase = "verification" if stage.target_kind == "dc" else "optimization"
            failure = stage_error(
                exception_report(exc, fallback_phase, context={"output_dir": str(stage_dir)}),
                stage_name=stage.name,
                stage_index=index,
            )
            error_payload = {"error_type": type(exc).__name__, "message": str(exc)}
            if callable(getattr(exc, "as_dict", None)):
                error_payload["details"] = exc.as_dict()
            (stage_dir / "error.json").write_text(
                json.dumps(error_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stage_runs.append(
                PBDLStageRun(
                    stage_name=stage.name,
                    status="error",
                    analysis_kind=stage.analysis_kind,
                    target_kind=stage.target_kind,
                    stage_index=index,
                    output_dir=str(stage_dir),
                    message=str(exc),
                    error=failure,
                )
            )

    report = PBDLRunReport(
        spec_name=spec.name,
        plan_path=str(plan_path),
        stages=tuple(stage_runs),
    )
    (output_dir / "pbdl_report.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _with_top_k(spec: SynthesisSpec, top_k: int) -> SynthesisSpec:
    return SynthesisSpec(
        name=spec.name,
        ports=spec.ports,
        analysis=spec.analysis,
        behavior=spec.behavior,
        library=spec.library,
        optimization=type(spec.optimization)(
            **{**spec.optimization.__dict__, "top_k": top_k}
        ),
    )


def _safe_suffix(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text) or "stage"
