"""Explainable topology experts for the unified circuit IR.

The first expert is deliberately small: it recognizes a non-isolated DC
step-up requirement and emits the canonical asynchronous boost power stage.
Keeping this as a separate expert makes later learned routing and additional
topologies additive rather than entangled with the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

from .ir import IRComponent, UnifiedIR

if TYPE_CHECKING:
    from .graph import CircuitGraph


@dataclass(frozen=True)
class TopologyCandidate:
    name: str
    family: str
    rationale: str
    components: tuple[IRComponent, ...]
    metadata: dict[str, Any]
    solver_id: str | None = None
    graph: "CircuitGraph | None" = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "rationale": self.rationale,
            "components": [component.as_dict() for component in self.components],
            "metadata": dict(self.metadata),
            "solver_id": self.solver_id or self.metadata.get("solver"),
            "graph_hash": self.graph.graph_hash if self.graph is not None else None,
            "topology_hash": self.graph.topology_hash if self.graph is not None else None,
            "graph": self.graph.as_dict() if self.graph is not None else None,
        }


class UnsupportedTopology(ValueError):
    pass


class ExpertTopologySelector:
    """Select a standard topology from explicit PBDL intent."""

    def select(self, ir: UnifiedIR) -> TopologyCandidate:
        if ir.intent_kind != "dc":
            raise UnsupportedTopology(
                f"no power expert registered for intent {ir.intent_kind!r}"
            )
        if len(ir.ports) != 2:
            raise UnsupportedTopology("the boost expert currently requires exactly two ports")

        target = ir.primary_target
        vin = _required_float(target, "input_voltage_v")
        vout = _required_float(target, "output_voltage_v")
        if vout <= vin:
            raise UnsupportedTopology(
                "the boost expert requires output_voltage_v greater than input_voltage_v"
            )

        input_port = _port_by_role(ir, "input", fallback=ir.ports[0])
        output_port = _port_by_role(ir, "output", fallback=ir.ports[1])
        in_node = input_port.positive
        out_node = output_port.positive
        ground = input_port.negative
        isolated = any(
            str(relation.get("kind", "")).casefold() in {"galvanic_isolation", "isolated"}
            for relation in ir.relations
        )
        if isolated:
            return self._select_isolated_flyback(ir, input_port, output_port)
        if output_port.negative != ground:
            raise UnsupportedTopology("the non-isolated boost expert requires a shared reference node")

        components = (
            IRComponent("L1", "L", (in_node, "sw")),
            IRComponent("Q1", "ideal_switch", ("sw", ground), attributes={"control": "pwm"}),
            IRComponent("D1", "ideal_diode", ("sw", out_node)),
            IRComponent("C1", "C", (out_node, ground)),
            IRComponent(
                "Rload",
                "R",
                (out_node, ground),
                attributes={"role": "external_load", "derived_from": "output_current_a"},
            ),
        )
        candidate = TopologyCandidate(
            name="ideal_asynchronous_boost",
            family="dc_boost",
            rationale=(
                "vout is above vin, so the standard non-isolated boost stage "
                "stores energy in L1 and transfers it through D1 when Q1 is off"
            ),
            components=components,
            metadata={
                "origin": "standard_topology_expert",
                "solver": "ideal_boost_averaged",
                "input_port": input_port.name,
                "output_port": output_port.name,
                "input_node": input_port.positive,
                "output_node": output_port.positive,
                "input_reference_node": input_port.negative,
                "output_reference_node": output_port.negative,
                "reference_node": ground,
                "conduction_mode": "ideal_ccm",
            },
            solver_id="ideal_boost_averaged",
        )
        return _attach_graph(candidate, ir)

    def _select_isolated_flyback(self, ir: UnifiedIR, input_port, output_port) -> TopologyCandidate:
        if output_port.negative == input_port.negative:
            raise UnsupportedTopology("galvanic_isolation requires distinct input and output reference nodes")
        components = (
            IRComponent("T1", "ideal_transformer", (input_port.positive, "sw", "out_sw", output_port.negative)),
            IRComponent("Q1", "ideal_switch", ("sw", input_port.negative), attributes={"control": "pwm"}),
            IRComponent("D1", "ideal_diode", ("out_sw", output_port.positive)),
            IRComponent("C1", "C", (output_port.positive, output_port.negative)),
            IRComponent(
                "Rload",
                "R",
                (output_port.positive, output_port.negative),
                attributes={"role": "external_load", "derived_from": "output_current_a"},
            ),
        )
        candidate = TopologyCandidate(
            name="ideal_isolated_flyback",
            family="isolated_flyback",
            rationale=(
                "the PBDL relation requires galvanic isolation, so the expert "
                "uses a transformer-isolated flyback power stage"
            ),
            components=components,
            metadata={
                "origin": "standard_topology_expert",
                "solver": "ideal_flyback_averaged",
                "input_port": input_port.name,
                "output_port": output_port.name,
                "input_node": input_port.positive,
                "output_node": output_port.positive,
                "input_reference_node": input_port.negative,
                "output_reference_node": output_port.negative,
                "isolated": True,
                "conduction_mode": "ideal_ccm",
            },
            solver_id="ideal_flyback_averaged",
        )
        return _attach_graph(candidate, ir)


def _required_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        raise UnsupportedTopology(f"DC target is missing {key}")
    value = float(value)
    if value <= 0:
        raise UnsupportedTopology(f"DC target {key} must be positive")
    return value


def _port_by_role(ir: UnifiedIR, role: str, *, fallback):
    for port in ir.ports:
        if port.role == role:
            return port
    return fallback


def _attach_graph(candidate: TopologyCandidate, ir: UnifiedIR) -> TopologyCandidate:
    from .graph import topology_candidate_to_graph

    return replace(candidate, graph=topology_candidate_to_graph(candidate, ir))
