"""Structured, user-facing error reports for the synthesis workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_SCHEMA = "circuit_ai.error_report"
ERROR_SCHEMA_VERSION = 1

PHASE_LABELS = {
    "pbdl": "PBDL 规范",
    "ir": "统一 IR",
    "candidate_search": "候选拓扑",
    "optimization": "参数优化",
    "simulation": "仿真",
    "verification": "约束验证",
    "export": "工件输出",
    "components": "元件库",
    "request": "网页请求",
    "execution": "执行",
}


@dataclass(frozen=True)
class ErrorReport:
    phase: str
    code: str
    message: str
    detail: str = ""
    suggestion: str = ""
    retryable: bool = False
    severity: str = "error"
    cause_type: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ERROR_SCHEMA,
            "schema_version": ERROR_SCHEMA_VERSION,
            "phase": self.phase,
            "phase_label": PHASE_LABELS.get(self.phase, self.phase),
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "retryable": self.retryable,
            "cause_type": self.cause_type,
            "context": dict(self.context),
        }


def error_report(
    phase: str,
    code: str,
    message: str,
    *,
    detail: str = "",
    suggestion: str = "",
    retryable: bool = False,
    severity: str = "error",
    cause_type: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ErrorReport(
        phase=phase,
        code=code,
        message=message,
        detail=detail,
        suggestion=suggestion,
        retryable=retryable,
        severity=severity,
        cause_type=cause_type,
        context=context or {},
    ).as_dict()


def exception_report(
    exc: BaseException,
    fallback_phase: str,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map internal exceptions to stable codes without exposing tracebacks."""

    name = type(exc).__name__
    detail = str(exc) or name
    phase = fallback_phase
    code = "execution_failed"
    suggestion = "检查当前阶段输入和能力边界，然后重试。"
    retryable = False

    if name == "TranslationError":
        phase = "pbdl"
        code = "capability_gap"
        suggestion = str(getattr(exc, "suggestion", "该需求需要尚未注册的分析或模型能力。"))
    elif name in {"JSONDecodeError", "ValidationError", "CircuitSpecError"}:
        phase = "pbdl"
        code = "invalid_pbdl"
        suggestion = "检查 JSON 结构、端口名称、分析类型、单位和目标字段。"
    elif name in {"UnsupportedTopology", "CapabilityPlanError", "CapabilityResolutionError"}:
        phase = "candidate_search"
        code = "capability_gap"
        suggestion = "扩大允许元件或搜索边界，或者补充对应的拓扑专家和求解器。"
    elif name in {"FeasibilityError", "ConstraintContractError", "ConstraintCompilationError"}:
        phase = "verification"
        code = "constraint_failed"
        suggestion = "查看具体约束的实际值、单位和证据等级；必要时调整规格或元件范围。"
    elif name in {"TimeoutExpired", "TimeoutError"}:
        phase = "simulation"
        code = "simulation_timeout"
        suggestion = "减少候选数或仿真点数，检查电路是否存在难收敛状态。"
        retryable = True
    elif name in {"FileNotFoundError", "PermissionError", "OSError"} and fallback_phase == "export":
        phase = "export"
        code = "artifact_write_failed"
        suggestion = "检查输出目录权限和 KiCad 工件路径。"
    elif fallback_phase == "pbdl":
        code = "invalid_pbdl"
        suggestion = "检查 PBDL 的端口、分析、目标和约束定义。"

    return error_report(
        phase,
        code,
        f"{PHASE_LABELS.get(phase, phase)}阶段失败",
        detail=detail,
        suggestion=suggestion,
        retryable=retryable,
        cause_type=name,
        context=context,
    )


def stage_error(report: dict[str, Any], *, stage_name: str, stage_index: int) -> dict[str, Any]:
    """Attach execution-local context to a reusable error report."""

    enriched = dict(report)
    context = dict(enriched.get("context") or {})
    context.update({"stage_name": stage_name, "stage_index": stage_index})
    enriched["context"] = context
    return enriched
