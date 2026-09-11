"""Typed PBDL v2 function contracts and the safe function compiler.

Functions are the only objects allowed to describe behavior in PBDL v2.  A
function references typed port variables, carries an explicit domain, and
uses one of the closed function bodies below.  There is deliberately no
``eval``/``exec`` path in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable, Literal

import pint


FunctionBodyKind = Literal["builtin", "expression_ast", "sampled", "rational"]
FunctionRole = Literal["target", "excitation", "predicate", "derived", "report"]

_UNITS = pint.UnitRegistry(autoconvert_offset_to_baseunit=False)


class FunctionCompileError(ValueError):
    """A structured error in the typed function layer."""

    def __init__(self, message: str, *, code: str = "function_compile_error", path: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "path": self.path}


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_").lower()
    return text or "port"


@dataclass(frozen=True)
class VariableRef:
    """Stable reference to one variable exposed by one port."""

    port_id: str
    variable_id: str

    @property
    def path(self) -> str:
        return f"ports.{self.port_id}.variables.{self.variable_id}"

    def as_dict(self) -> dict[str, str]:
        return {"ref": self.path}

    @classmethod
    def from_value(cls, value: Any) -> "VariableRef":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            if "ref" in value:
                return cls.from_value(value["ref"])
            port_id = value.get("port_id", value.get("port", value.get("port_name")))
            variable_id = value.get("variable_id", value.get("variable", value.get("name")))
            if port_id and variable_id:
                return cls(str(port_id), str(variable_id))
        if isinstance(value, str):
            parts = value.split(".")
            if len(parts) == 4 and parts[0] == "ports" and parts[2] == "variables":
                return cls(parts[1], parts[3])
            if len(parts) == 2:
                return cls(parts[0], parts[1])
        raise FunctionCompileError(
            f"invalid variable reference: {value!r}",
            code="invalid_variable_ref",
        )


@dataclass(frozen=True)
class AxisSpec:
    """Independent axis used by a function or sampled target."""

    domain: str = "frequency"
    variable: str = "frequency"
    unit: str = "Hz"
    values: tuple[float, ...] = ()
    start: float | None = None
    stop: float | None = None
    points: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "domain": self.domain,
            "variable": self.variable,
            "unit": self.unit,
        }
        if self.values:
            result["values"] = list(self.values)
        if self.start is not None:
            result["start"] = self.start
        if self.stop is not None:
            result["stop"] = self.stop
        if self.points is not None:
            result["points"] = self.points
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "AxisSpec | None":
        if data is None:
            return None
        if isinstance(data, (list, tuple)):
            if len(data) != 3:
                raise FunctionCompileError("axis range must be [start, stop, points]", code="invalid_axis")
            return cls(start=float(data[0]), stop=float(data[1]), points=int(data[2]))
        if not isinstance(data, dict):
            raise FunctionCompileError("axis must be an object or [start, stop, points]", code="invalid_axis")
        values = tuple(float(item) for item in data.get("values", ()))
        return cls(
            domain=str(data.get("domain", "frequency")),
            variable=str(data.get("variable", "frequency")),
            unit=str(data.get("unit", "Hz")),
            values=values,
            start=float(data["start"]) if data.get("start") is not None else None,
            stop=float(data["stop"]) if data.get("stop") is not None else None,
            points=int(data["points"]) if data.get("points") is not None else None,
        )

    def frequency_range(self) -> tuple[float, float, int]:
        if self.values:
            if len(self.values) < 2:
                raise FunctionCompileError("frequency axis needs at least two values", code="invalid_axis")
            return float(self.values[0]), float(self.values[-1]), len(self.values)
        if self.start is None or self.stop is None:
            return 10.0, 100_000.0, int(self.points or 96)
        return float(self.start), float(self.stop), int(self.points or 96)


@dataclass(frozen=True)
class FunctionSignature:
    """Typed input/output signature of a function."""

    inputs: tuple[VariableRef, ...] = ()
    outputs: tuple[VariableRef, ...] = ()
    input_units: tuple[str, ...] = ()
    output_units: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "inputs": [item.path for item in self.inputs],
            "outputs": [item.path for item in self.outputs],
        }
        if self.input_units:
            result["input_units"] = list(self.input_units)
        if self.output_units:
            result["output_units"] = list(self.output_units)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionSignature":
        return cls(
            inputs=tuple(VariableRef.from_value(item) for item in data.get("inputs", ())),
            outputs=tuple(VariableRef.from_value(item) for item in data.get("outputs", ())),
            input_units=tuple(str(item) for item in data.get("input_units", ())),
            output_units=tuple(str(item) for item in data.get("output_units", ())),
        )


@dataclass(frozen=True)
class FunctionBody:
    """Closed union of safe function representations."""

    kind: FunctionBodyKind
    name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    expression: dict[str, Any] | None = None
    samples: dict[str, Any] | None = None
    numerator: dict[str, Any] | None = None
    denominator: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.name is not None:
            result["name"] = self.name
        if self.parameters:
            result["parameters"] = dict(self.parameters)
        if self.expression is not None:
            result["expression"] = dict(self.expression)
        if self.samples is not None:
            result["samples"] = dict(self.samples)
        if self.numerator is not None:
            result["numerator"] = dict(self.numerator)
        if self.denominator is not None:
            result["denominator"] = dict(self.denominator)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionBody":
        kind = str(data.get("kind", "builtin"))
        if kind not in {"builtin", "expression_ast", "sampled", "rational"}:
            raise FunctionCompileError(f"unsupported function body kind: {kind}", code="unsupported_function_body")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            name=str(data["name"]) if data.get("name") is not None else None,
            parameters=dict(data.get("parameters", {})),
            expression=dict(data["expression"]) if data.get("expression") is not None else None,
            samples=dict(data["samples"]) if data.get("samples") is not None else None,
            numerator=dict(data["numerator"]) if data.get("numerator") is not None else None,
            denominator=dict(data["denominator"]) if data.get("denominator") is not None else None,
        )


@dataclass(frozen=True)
class FunctionSpec:
    """One behavior, excitation, predicate or derived function."""

    function_id: str
    role: FunctionRole
    domain: str
    signature: FunctionSignature
    body: FunctionBody
    axis: AxisSpec | None = None
    description: str = ""
    constraints: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> tuple[VariableRef, ...]:
        return self.signature.inputs

    @property
    def outputs(self) -> tuple[VariableRef, ...]:
        return self.signature.outputs

    def as_dict(self) -> dict[str, Any]:
        result = {
            "function_id": self.function_id,
            "role": self.role,
            "domain": self.domain,
            "inputs": [item.path for item in self.inputs],
            "outputs": [item.path for item in self.outputs],
            "body": self.body.as_dict(),
        }
        if self.axis is not None:
            result["axis"] = self.axis.as_dict()
        if self.description:
            result["description"] = self.description
        if self.constraints:
            result["constraints"] = [dict(item) for item in self.constraints]
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionSpec":
        signature = FunctionSignature.from_dict(data.get("signature", data))
        role = str(data.get("role", "target"))
        if role not in {"target", "excitation", "predicate", "derived", "report"}:
            raise FunctionCompileError(f"unsupported function role: {role}", code="unsupported_function_role")
        return cls(
            function_id=str(data.get("function_id", data.get("id", "function"))),
            role=role,  # type: ignore[arg-type]
            domain=str(data.get("domain", "frequency")),
            signature=signature,
            body=FunctionBody.from_dict(dict(data.get("body", {}))),
            axis=AxisSpec.from_dict(data.get("axis")),
            description=str(data.get("description", "")),
            constraints=tuple(dict(item) for item in data.get("constraints", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class FunctionGraph:
    functions: tuple[FunctionSpec, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"functions": [item.as_dict() for item in self.functions]}

    def by_id(self) -> dict[str, FunctionSpec]:
        return {item.function_id: item for item in self.functions}

    def dependency_order(self) -> tuple[FunctionSpec, ...]:
        by_id = self.by_id()
        edges = {item.function_id: _function_dependencies(item) for item in self.functions}
        ordered: list[FunctionSpec] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(function_id: str) -> None:
            if function_id in visited:
                return
            if function_id in visiting:
                raise FunctionCompileError(
                    f"function dependency cycle includes {function_id!r}",
                    code="function_dependency_cycle",
                )
            visiting.add(function_id)
            for dependency in edges[function_id]:
                if dependency not in by_id:
                    raise FunctionCompileError(
                        f"function {function_id!r} references unknown function {dependency!r}",
                        code="unknown_function_ref",
                    )
                visit(dependency)
            visiting.remove(function_id)
            visited.add(function_id)
            ordered.append(by_id[function_id])

        for function in self.functions:
            visit(function.function_id)
        return tuple(ordered)


@dataclass(frozen=True)
class CompiledBehavior:
    function_id: str
    status: Literal["executable", "validated", "unsupported"]
    analysis: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    excitation: dict[str, Any] | None = None
    required_capabilities: tuple[str, ...] = ()
    diagnostics: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "status": self.status,
            "analysis": dict(self.analysis) if self.analysis else None,
            "target": dict(self.target) if self.target else None,
            "excitation": dict(self.excitation) if self.excitation else None,
            "required_capabilities": list(self.required_capabilities),
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


@dataclass(frozen=True)
class FunctionCompilation:
    graph: FunctionGraph
    behaviors: tuple[CompiledBehavior, ...]

    @property
    def analyses(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.analysis for item in self.behaviors if item.analysis is not None)

    @property
    def targets(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.target for item in self.behaviors if item.target is not None)

    @property
    def executable(self) -> bool:
        return bool(self.behaviors) and all(item.status == "executable" for item in self.behaviors if item.analysis or item.target)

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.as_dict(),
            "behaviors": [item.as_dict() for item in self.behaviors],
            "analyses": [dict(item) for item in self.analyses],
            "targets": [dict(item) for item in self.targets],
            "executable": self.executable,
        }


@dataclass(frozen=True)
class BuiltinFunction:
    name: str
    domains: tuple[str, ...]
    input_quantities: tuple[str, ...]
    output_quantities: tuple[str, ...]
    parameter_units: dict[str, str]
    required_capability: str


_BUILTINS: dict[str, BuiltinFunction] = {
    "lowpass": BuiltinFunction("lowpass", ("frequency", "frequency_domain"), ("voltage",), ("voltage",), {"cutoff": "Hz", "gain": "1", "order": "1", "q": "1"}, "linear_mna"),
    "highpass": BuiltinFunction("highpass", ("frequency", "frequency_domain"), ("voltage",), ("voltage",), {"cutoff": "Hz", "gain": "1", "order": "1", "q": "1"}, "linear_mna"),
    "bandpass": BuiltinFunction("bandpass", ("frequency", "frequency_domain"), ("voltage",), ("voltage",), {"center": "Hz", "cutoff": "Hz", "gain": "1", "bandwidth": "Hz", "q": "1"}, "linear_mna"),
    "gain": BuiltinFunction("gain", ("frequency", "frequency_domain"), ("voltage",), ("voltage",), {"gain": "1"}, "linear_mna"),
    "impedance": BuiltinFunction("impedance", ("frequency", "frequency_domain"), ("voltage",), ("voltage",), {"ohms": "ohm"}, "linear_mna"),
    "transimpedance": BuiltinFunction("transimpedance", ("frequency", "frequency_domain"), ("current",), ("voltage",), {"transimpedance": "ohm", "gain": "ohm", "cutoff": "Hz"}, "linear_mna"),
    "regulate": BuiltinFunction("regulate", ("dc", "dc_operating_point"), ("voltage",), ("voltage",), {"input_voltage": "V", "output_voltage": "V", "output_current": "A"}, "ideal_power_solver"),
    "sine": BuiltinFunction("sine", ("time", "transient", "frequency"), (), ("voltage", "current"), {"frequency": "Hz", "amplitude": "V", "offset": "V", "phase": "deg"}, "waveform_source"),
    "pulse": BuiltinFunction("pulse", ("time", "transient", "periodic"), (), ("voltage",), {"frequency": "Hz", "low": "V", "high": "V", "duty": "1", "rise": "s", "fall": "s"}, "waveform_source"),
    "sampled": BuiltinFunction("sampled", ("frequency", "time", "periodic"), (), ("voltage", "current"), {"period": "s"}, "sampled_source"),
}


def builtin_function_registry() -> dict[str, BuiltinFunction]:
    """Return a copy of the closed builtin registry."""

    return dict(_BUILTINS)


def compile_function_graph(functions: Iterable[FunctionSpec], ports: Iterable[Any]) -> FunctionCompilation:
    """Validate and lower typed functions without evaluating user code."""

    graph = FunctionGraph(tuple(functions))
    if not graph.functions:
        raise FunctionCompileError("at least one function is required", code="no_functions")
    if len(graph.by_id()) != len(graph.functions):
        raise FunctionCompileError("function_id values must be unique", code="duplicate_function_id")
    port_map = _port_variable_map(ports)
    graph.dependency_order()
    behaviors: list[CompiledBehavior] = []
    for function in graph.functions:
        _validate_function(function, port_map, graph.by_id())
        behaviors.append(_lower_function(function, port_map))
    return FunctionCompilation(graph, tuple(behaviors))


def _port_variable_map(ports: Iterable[Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for port in ports:
        port_id = str(getattr(port, "port_id", None) or _slug(port.name))
        variables = {str(item.name): {"quantity": item.quantity, "unit": item.unit} for item in getattr(port, "variables", ())}
        if not variables:
            variables = {"v": {"quantity": "voltage", "unit": "V"}, "i": {"quantity": "current", "unit": "A"}}
        result[port_id] = variables
        result.setdefault(str(port.name), variables)
    return result


def _resolve_ref(ref: VariableRef, port_map: dict[str, dict[str, dict[str, Any]]], *, path: str) -> dict[str, Any]:
    if ref.port_id not in port_map or ref.variable_id not in port_map[ref.port_id]:
        raise FunctionCompileError(
            f"unknown variable reference {ref.path}",
            code="unknown_variable_ref",
            path=path,
        )
    return port_map[ref.port_id][ref.variable_id]


def _validate_function(function: FunctionSpec, port_map: dict[str, dict[str, dict[str, Any]]], functions: dict[str, FunctionSpec]) -> None:
    if not function.function_id:
        raise FunctionCompileError("function_id cannot be empty", code="invalid_function_id")
    if not function.inputs and function.body.kind not in {"sampled", "builtin"}:
        raise FunctionCompileError("function needs at least one input", code="invalid_function_signature", path=function.function_id)
    for index, ref in enumerate(function.inputs):
        _resolve_ref(ref, port_map, path=f"functions.{function.function_id}.inputs[{index}]")
    for index, ref in enumerate(function.outputs):
        _resolve_ref(ref, port_map, path=f"functions.{function.function_id}.outputs[{index}]")
    if function.axis is not None:
        _validate_axis(function.axis, function.function_id)
    if function.body.kind == "builtin":
        name = function.body.name or ""
        definition = _BUILTINS.get(name)
        if definition is None:
            raise FunctionCompileError(f"unknown builtin function {name!r}", code="unknown_builtin", path=function.function_id)
        if function.domain not in definition.domains:
            raise FunctionCompileError(
                f"builtin {name!r} does not support domain {function.domain!r}",
                code="function_domain_mismatch",
                path=function.function_id,
            )
        _validate_quantities(function, definition, port_map)
        for key, unit in definition.parameter_units.items():
            if key in function.body.parameters:
                _numeric_parameter(function.body.parameters[key], unit, f"functions.{function.function_id}.body.parameters.{key}")
    elif function.body.kind == "expression_ast":
        if function.body.expression is None:
            raise FunctionCompileError("expression_ast requires expression", code="missing_expression", path=function.function_id)
        _infer_expression_unit(function.body.expression, port_map, functions, path=f"functions.{function.function_id}.body.expression")
    elif function.body.kind == "sampled":
        samples = function.body.samples or {}
        values = samples.get("magnitude_db", samples.get("values"))
        if not isinstance(values, list) or len(values) < 2:
            raise FunctionCompileError("sampled body requires at least two values", code="invalid_samples", path=function.function_id)
        axis = samples.get("frequency_hz", samples.get("axis"))
        if axis is None and function.axis is None:
            raise FunctionCompileError("sampled body requires an axis", code="missing_axis", path=function.function_id)
    else:
        if function.body.numerator is None or function.body.denominator is None:
            raise FunctionCompileError("rational body requires numerator and denominator", code="invalid_rational", path=function.function_id)
        _infer_expression_unit(function.body.numerator, port_map, functions, path=f"functions.{function.function_id}.body.numerator")
        _infer_expression_unit(function.body.denominator, port_map, functions, path=f"functions.{function.function_id}.body.denominator")


def _validate_axis(axis: AxisSpec, function_id: str) -> None:
    if axis.domain in {"frequency", "frequency_domain"} and axis.unit not in {"Hz", "kHz", "MHz", "1/s"}:
        raise FunctionCompileError("frequency axis must use Hz-compatible units", code="axis_unit_mismatch", path=function_id)
    if axis.values and any(not math.isfinite(item) or item <= 0 for item in axis.values):
        raise FunctionCompileError("axis values must be finite and positive", code="invalid_axis", path=function_id)
    if axis.start is not None and axis.stop is not None and (axis.start <= 0 or axis.stop <= axis.start):
        raise FunctionCompileError("axis range must be positive and increasing", code="invalid_axis", path=function_id)


def _validate_quantities(function: FunctionSpec, definition: BuiltinFunction, port_map: dict[str, dict[str, dict[str, Any]]]) -> None:
    if definition.input_quantities and len(function.inputs) != len(definition.input_quantities):
        raise FunctionCompileError(f"builtin {definition.name!r} expects {len(definition.input_quantities)} inputs", code="function_arity", path=function.function_id)
    if definition.output_quantities and len(function.outputs) != len(definition.output_quantities):
        raise FunctionCompileError(f"builtin {definition.name!r} expects {len(definition.output_quantities)} outputs", code="function_arity", path=function.function_id)
    for ref, expected in zip(function.inputs, definition.input_quantities, strict=False):
        actual = _resolve_ref(ref, port_map, path=function.function_id).get("quantity")
        if actual and actual != expected:
            raise FunctionCompileError(f"input {ref.path} is {actual}, expected {expected}", code="quantity_mismatch", path=function.function_id)
    for ref, expected in zip(function.outputs, definition.output_quantities, strict=False):
        actual = _resolve_ref(ref, port_map, path=function.function_id).get("quantity")
        if actual and actual != expected:
            raise FunctionCompileError(f"output {ref.path} is {actual}, expected {expected}", code="quantity_mismatch", path=function.function_id)


def _numeric_parameter(value: Any, expected_unit: str, path: str) -> float:
    raw_value = value.get("value") if isinstance(value, dict) else value
    unit = str(value.get("unit", expected_unit)) if isinstance(value, dict) else expected_unit
    try:
        numeric = float(raw_value)
        if not math.isfinite(numeric):
            raise ValueError
        source = _UNITS.parse_units(unit)
        target = _UNITS.parse_units(expected_unit)
        if source.dimensionality != target.dimensionality:
            raise ValueError
    except (TypeError, ValueError, pint.errors.PintError) as exc:
        raise FunctionCompileError(f"parameter at {path} must have unit compatible with {expected_unit}", code="unit_mismatch", path=path) from exc
    return numeric


def _infer_expression_unit(node: dict[str, Any], port_map: dict[str, dict[str, dict[str, Any]]], functions: dict[str, FunctionSpec], *, path: str) -> str:
    if not isinstance(node, dict):
        raise FunctionCompileError("expression node must be an object", code="invalid_expression_ast", path=path)
    if "ref" in node:
        ref = VariableRef.from_value(node["ref"])
        return str(_resolve_ref(ref, port_map, path=path).get("unit") or "1")
    if "constant" in node:
        value = node["constant"]
        unit = str(value.get("unit", "1")) if isinstance(value, dict) else "1"
        _numeric_parameter(value, unit, path)
        return unit
    if "function_ref" in node:
        function_id = str(node["function_ref"])
        if function_id not in functions:
            raise FunctionCompileError(f"unknown function reference {function_id!r}", code="unknown_function_ref", path=path)
        output_index = int(node.get("output_index", 0))
        outputs = functions[function_id].outputs
        if output_index >= len(outputs):
            raise FunctionCompileError("function output index is out of range", code="invalid_function_ref", path=path)
        return "1"
    op = node.get("op")
    args = node.get("args", [])
    if op in {"negate", "abs"}:
        if len(args) != 1:
            raise FunctionCompileError(f"{op} expects one argument", code="invalid_expression_ast", path=path)
        return _infer_expression_unit(args[0], port_map, functions, path=path + ".args[0]")
    if op in {"add", "subtract", "multiply", "divide", "power"}:
        if len(args) != 2:
            raise FunctionCompileError(f"{op} expects two arguments", code="invalid_expression_ast", path=path)
        left = _infer_expression_unit(args[0], port_map, functions, path=path + ".args[0]")
        right = _infer_expression_unit(args[1], port_map, functions, path=path + ".args[1]")
        if op in {"add", "subtract"}:
            _assert_compatible(left, right, path)
            return left
        if op == "multiply":
            return str((_UNITS.parse_units(left) * _UNITS.parse_units(right)).units)
        if op == "divide":
            return str((_UNITS.parse_units(left) / _UNITS.parse_units(right)).units)
        return "1"
    raise FunctionCompileError(f"unsupported expression operation {op!r}", code="unsupported_expression_op", path=path)


def _assert_compatible(left: str, right: str, path: str) -> None:
    try:
        if _UNITS.parse_units(left).dimensionality != _UNITS.parse_units(right).dimensionality:
            raise FunctionCompileError(f"units {left!r} and {right!r} are incompatible", code="unit_mismatch", path=path)
    except pint.errors.PintError as exc:
        raise FunctionCompileError("expression contains an unknown unit", code="unknown_unit", path=path) from exc


def _function_dependencies(function: FunctionSpec) -> set[str]:
    dependencies: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("function_ref"):
                dependencies.add(str(node["function_ref"]))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    body = function.body
    visit(body.expression)
    visit(body.numerator)
    visit(body.denominator)
    return dependencies


def _port_name(ref: VariableRef, port_map: dict[str, dict[str, dict[str, Any]]]) -> str:
    return ref.port_id


def _parameter(function: FunctionSpec, *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in function.body.parameters:
            return _numeric_parameter(function.body.parameters[name], _BUILTINS[function.body.name or ""].parameter_units.get(name, "1"), function.function_id)
    return default


def _frequency_axis(function: FunctionSpec) -> list[float]:
    if function.axis is not None:
        if function.axis.values:
            return list(function.axis.values)
        start, stop, points = function.axis.frequency_range()
        return [start, stop, points]
    return [10.0, 100_000.0, 96]


def _lower_function(function: FunctionSpec, port_map: dict[str, dict[str, dict[str, Any]]]) -> CompiledBehavior:
    body = function.body
    if body.kind in {"expression_ast", "rational"}:
        return CompiledBehavior(
            function.function_id,
            "validated",
            required_capabilities=("typed_function_evaluator",),
            diagnostics=({"code": "not_lowered_to_legacy_solver", "message": "function is type-checked but needs a native function evaluator"},),
        )
    if body.kind == "sampled":
        samples = body.samples or {}
        frequencies = samples.get("frequency_hz", samples.get("axis", {}).get("values", []) if isinstance(samples.get("axis"), dict) else [])
        magnitude = samples.get("magnitude_db", samples.get("values", []))
        if frequencies and magnitude:
            return CompiledBehavior(
                function.function_id,
                "executable",
                analysis={"kind": "voltage_transfer", "source_port": _port_name(function.inputs[0], port_map) if function.inputs else "input", "output_port": _port_name(function.outputs[0], port_map), "frequency_hz": list(frequencies)},
                target={"target_kind": "sampled", "frequency_hz": list(frequencies), "magnitude_db": list(magnitude), "phase_deg": samples.get("phase_deg")},
                required_capabilities=("sampled_target",),
            )
        return CompiledBehavior(function.function_id, "validated", required_capabilities=("sampled_target",))

    name = body.name or ""
    axis = _frequency_axis(function)
    source = _port_name(function.inputs[0], port_map) if function.inputs else "input"
    output = _port_name(function.outputs[0], port_map) if function.outputs else "output"
    params = body.parameters
    if name in {"lowpass", "highpass", "bandpass"}:
        cutoff = _parameter(function, "cutoff", "center", default=1000.0)
        gain = _parameter(function, "gain", default=1.0) or 1.0
        target: dict[str, Any] = {"target_kind": "filter", "kind": name, "cutoff_hz": cutoff, "gain_db": 20.0 * math.log10(max(gain, 1e-18))}
        if name == "bandpass":
            target["q"] = _parameter(function, "q", default=3.0)
        return CompiledBehavior(function.function_id, "executable", analysis={"kind": "voltage_transfer", "source_port": source, "output_port": output, "frequency_hz": axis}, target=target, required_capabilities=("linear_mna",))
    if name == "gain":
        gain = _parameter(function, "gain", default=1.0) or 1.0
        return CompiledBehavior(function.function_id, "executable", analysis={"kind": "voltage_transfer", "source_port": source, "output_port": output, "frequency_hz": axis}, target={"target_kind": "amplifier", "gain_db": 20.0 * math.log10(max(gain, 1e-18)), "bandwidth_hz": axis[:2]}, required_capabilities=("linear_mna",))
    if name == "impedance":
        return CompiledBehavior(function.function_id, "executable", analysis={"kind": "impedance", "port": output, "frequency_hz": axis}, target={"target_kind": "impedance", "ohms": _parameter(function, "ohms", default=50.0)}, required_capabilities=("linear_mna",))
    if name == "transimpedance":
        transimpedance = _parameter(function, "transimpedance", "gain", default=1000.0) or 1000.0
        return CompiledBehavior(function.function_id, "executable", analysis={"kind": "transimpedance", "source_port": source, "output_port": output, "frequency_hz": axis}, target={"target_kind": "amplifier", "gain_db": 20.0 * math.log10(max(transimpedance, 1e-18)), "bandwidth_hz": axis[:2]}, required_capabilities=("linear_mna",))
    if name == "regulate":
        return CompiledBehavior(function.function_id, "executable", analysis={"kind": "dc_transfer", "source_port": source, "output_port": output}, target={"target_kind": "dc", "input_voltage_v": _parameter(function, "input_voltage", default=5.0), "output_voltage_v": _parameter(function, "output_voltage", default=10.0), "output_current_a": _parameter(function, "output_current", default=0.0)}, required_capabilities=("ideal_power_solver",))
    if name in {"sine", "pulse", "sampled"}:
        return CompiledBehavior(function.function_id, "executable", excitation={"port": output, "waveform": dict(params) | {"kind": name}}, required_capabilities=("waveform_source",))
    return CompiledBehavior(function.function_id, "validated")


__all__ = [
    "AxisSpec",
    "BuiltinFunction",
    "CompiledBehavior",
    "FunctionBody",
    "FunctionBodyKind",
    "FunctionCompilation",
    "FunctionCompileError",
    "FunctionGraph",
    "FunctionRole",
    "FunctionSignature",
    "FunctionSpec",
    "VariableRef",
    "builtin_function_registry",
    "compile_function_graph",
]
