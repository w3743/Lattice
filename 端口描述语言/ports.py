"""端口定义 —— 电路的物理接口。

每个端口是一个或多个端子（terminal），端子承载某种物理量（电压、电流）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

TerminalQuantity = Literal["voltage", "current", "power", "ground", "unspecified"]
VariableRole = Literal["potential", "flow", "derived", "wave"]
ConstraintOperator = Literal["equal", "minimum", "maximum", "interval", "function", "samples"]
ConstraintSeverity = Literal["hard", "soft", "objective"]


@dataclass(frozen=True)
class PortVariable:
    """A physical variable exposed by a port.

    Electrical ports use ``voltage`` as the potential/across variable and
    ``current`` as the flow/through variable.  Positive current is always
    defined as entering the component through the first terminal.
    """

    name: str
    quantity: str
    role: VariableRole
    unit: str
    expression: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "quantity": self.quantity,
            "role": self.role,
            "unit": self.unit,
        }
        if self.expression is not None:
            data["expression"] = self.expression
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortVariable":
        return cls(
            name=str(data.get("name", data.get("symbol", "v"))),
            quantity=str(data.get("quantity", "unspecified")),
            role=str(data.get("role", "potential")),  # type: ignore[arg-type]
            unit=str(data.get("unit", "")),
            expression=data.get("expression", data.get("definition")),
        )


@dataclass(frozen=True)
class VariableConstraint:
    """One independently scoped constraint on one port variable."""

    variable: str
    analysis: dict[str, Any]
    operator: ConstraintOperator
    id: str | None = None
    value: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    severity: ConstraintSeverity = "hard"
    tolerance: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    reduction: str | None = None
    required_evidence_rank: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variable": self.variable,
            "analysis": dict(self.analysis),
            "operator": self.operator,
            "severity": self.severity,
            "weight": self.weight,
        }
        for key in ("id", "value", "minimum", "maximum", "unit", "reduction"):
            item = getattr(self, key)
            if item is not None:
                result[key] = item
        if self.tolerance:
            result["tolerance"] = dict(self.tolerance)
        if self.required_evidence_rank:
            result["required_evidence_rank"] = self.required_evidence_rank
        if self.data:
            result["data"] = dict(self.data)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariableConstraint":
        raw_analysis = data.get("analysis", {"kind": "dc_operating_point"})
        analysis = {"kind": raw_analysis} if isinstance(raw_analysis, str) else dict(raw_analysis)
        return cls(
            variable=str(data["variable"]),
            analysis=analysis,
            operator=str(data.get("operator", "equal")),  # type: ignore[arg-type]
            id=str(data["id"]) if data.get("id") is not None else None,
            value=float(data["value"]) if data.get("value") is not None else None,
            minimum=float(data["minimum"]) if data.get("minimum") is not None else None,
            maximum=float(data["maximum"]) if data.get("maximum") is not None else None,
            unit=data.get("unit"),
            severity=str(data.get("severity", data.get("kind", "hard"))),  # type: ignore[arg-type]
            tolerance=dict(data.get("tolerance", {})),
            weight=float(data.get("weight", 1.0)),
            reduction=str(data["reduction"]) if data.get("reduction") is not None else None,
            required_evidence_rank=int(data.get("required_evidence_rank", 0)),
            data=dict(data.get("data", {})),
        )


@dataclass(frozen=True)
class Terminal:
    """电路的一个物理端子。

    每个端子有一个名称和一个物理量类型。
    """

    name: str
    quantity: TerminalQuantity = "unspecified"

    def as_dict(self) -> dict:
        return {"name": self.name, "quantity": self.quantity}

    @classmethod
    def from_dict(cls, data: dict | str) -> "Terminal":
        if isinstance(data, str):
            return cls(name=data)
        return cls(name=data["name"], quantity=data.get("quantity", "unspecified"))


@dataclass(frozen=True)
class Port:
    """一个端口，由一或多个端子组成。

    两端口网络有两个端口（port_in, port_out）。
    每个端口可以是单端（一电压+一参考地）或差分（两电压）。
    """

    name: str
    terminals: tuple[Terminal, ...]
    description: str = ""
    variables: tuple[PortVariable, ...] = ()
    variable_constraints: tuple[VariableConstraint, ...] = ()
    role: str = "bidirectional"
    domain: str = "unspecified"
    excitation: dict[str, Any] | None = None
    port_id: str | None = None

    @property
    def stable_id(self) -> str:
        """Return the stable identifier used by PBDL function references."""
        raw = self.port_id or self.name
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw).strip()).strip("_").lower()
        return normalized or "port"

    @classmethod
    def single_ended(cls, name: str, signal: str = "in", ref: str = "0") -> "Port":
        return cls.electrical(
            name=name,
            positive=signal,
            negative=ref,
        )

    @classmethod
    def differential(cls, name: str, plus: str = "p", minus: str = "n") -> "Port":
        return cls.electrical(name=name, positive=plus, negative=minus)

    @classmethod
    def electrical(
        cls,
        name: str,
        positive: str,
        negative: str,
        *,
        constraints: tuple[VariableConstraint, ...] = (),
        description: str = "",
    ) -> "Port":
        return cls(
            name=name,
            terminals=(
                Terminal(positive, "voltage"),
                Terminal(negative, "ground" if negative in {"0", "gnd", "GND"} else "voltage"),
            ),
            description=description,
            variables=(
                PortVariable("v", "voltage", "potential", "V", f"V({positive}) - V({negative})"),
                PortVariable("i", "current", "flow", "A", f"current entering {positive}"),
                PortVariable("p", "power", "derived", "W", "v * i"),
            ),
            variable_constraints=constraints,
        )

    def voltage_terminal(self) -> Terminal:
        """返回承载信号电压的端子（单端端口）。"""
        for term in self.terminals:
            if term.quantity == "voltage":
                return term
        raise ValueError(f"port {self.name!r} has no voltage terminal")

    def ref_terminal(self) -> Terminal:
        """返回参考端子（通常是地）。"""
        for term in self.terminals:
            if term.quantity == "ground":
                return term
        raise ValueError(f"port {self.name!r} has no ground terminal")

    def as_dict(self) -> dict:
        result = {
            "port_id": self.stable_id,
            "name": self.name,
            "terminals": [t.as_dict() for t in self.terminals],
            "description": self.description,
            "role": self.role,
            "domain": self.domain,
        }
        if self.variables:
            result["variables"] = [variable.as_dict() for variable in self.variables]
        if self.variable_constraints:
            result["variable_constraints"] = [constraint.as_dict() for constraint in self.variable_constraints]
        if self.excitation is not None:
            result["excitation"] = dict(self.excitation)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Port":
        return cls(
            name=data["name"],
            terminals=tuple(Terminal.from_dict(t) for t in data.get("terminals", [])),
            description=data.get("description", ""),
            variables=tuple(PortVariable.from_dict(item) for item in data.get("variables", [])),
            variable_constraints=tuple(
                VariableConstraint.from_dict(item) for item in data.get("variable_constraints", [])
            ),
            role=str(data.get("role", "bidirectional")),
            domain=str(data.get("domain", "unspecified")),
            excitation=dict(data["excitation"]) if data.get("excitation") is not None else None,
            port_id=str(data.get("port_id", data.get("id"))) if data.get("port_id", data.get("id")) is not None else None,
        )


@dataclass(frozen=True)
class PortBundle:
    """一组端口，定义了一个电路的全部外部接口。

    例如一个放大器：
      input:  单端端口 (in_p, 0)
      output: 单端端口 (out_p, 0)
      vcc:    供电端口 (vcc, 0)
    """

    ports: tuple[Port, ...]

    def __getitem__(self, name: str) -> Port:
        for port in self.ports:
            if port.name == name:
                return port
        raise KeyError(f"port {name!r} not found")

    def __iter__(self):
        return iter(self.ports)

    def __len__(self) -> int:
        return len(self.ports)

    @classmethod
    def standard_two_port(
        cls,
        input_name: str = "input",
        output_name: str = "output",
    ) -> "PortBundle":
        return cls(
            ports=(
                Port.single_ended(input_name, signal="in", ref="0"),
                Port.single_ended(output_name, signal="out", ref="0"),
            )
        )

    def as_dict(self) -> list[dict]:
        return [port.as_dict() for port in self.ports]

    @classmethod
    def from_list(cls, data: list[dict]) -> "PortBundle":
        return cls(ports=tuple(Port.from_dict(item) for item in data))
