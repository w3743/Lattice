"""端口行为描述语言 (Port Behavior Description Language, PBDL).

统一的电路端口行为建模：
  Port(命名端子) → Analysis(测量什么) → Target(期望什么) → Constraints(限制条件)
"""

from .ports import Port, PortBundle, PortVariable, Terminal, VariableConstraint
from .analysis import Analysis, AcTransfer, Impedance, Transimpedance, DcTransfer, frequency_grid
from .targets import (
    Target,
    FilterTarget,
    AmplifierTarget,
    ImpedanceTarget,
    SampledTarget,
    MaskTarget,
    Tolerance,
)
from .dc_targets import DcTarget
from .constraints import Constraints, OperatingPoint
from .functions import (
    AxisSpec,
    BuiltinFunction,
    CompiledBehavior,
    FunctionBody,
    FunctionCompilation,
    FunctionCompileError,
    FunctionGraph,
    FunctionSignature,
    FunctionSpec,
    VariableRef,
    builtin_function_registry,
    compile_function_graph,
)
from .spec import CircuitSpec, load_spec, save_spec
from .translator import TranslationPlan, TranslationStage, to_circuit_ai_plan, to_circuit_ai_spec

__all__ = [
    # 端口
    "Port",
    "PortBundle",
    "Terminal",
    "PortVariable",
    "VariableConstraint",
    # 函数
    "VariableRef",
    "AxisSpec",
    "FunctionSignature",
    "FunctionBody",
    "FunctionSpec",
    "FunctionGraph",
    "FunctionCompilation",
    "CompiledBehavior",
    "BuiltinFunction",
    "FunctionCompileError",
    "builtin_function_registry",
    "compile_function_graph",
    # 分析
    "Analysis",
    "AcTransfer",
    "Impedance",
    "Transimpedance",
    "DcTransfer",
    "frequency_grid",
    # 目标
    "Target",
    "FilterTarget",
    "AmplifierTarget",
    "ImpedanceTarget",
    "SampledTarget",
    "MaskTarget",
    "DcTarget",
    "Tolerance",
    # 约束
    "Constraints",
    "OperatingPoint",
    # 规格
    "CircuitSpec",
    "load_spec",
    "save_spec",
    "TranslationPlan",
    "TranslationStage",
    "to_circuit_ai_plan",
    "to_circuit_ai_spec",
]
