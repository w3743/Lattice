from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis import AnalysisRequest
from .mna import GROUND_NAMES
from .targets import TargetResponse


@dataclass(frozen=True)
class DifferentiableRefinementResult:
    enabled: bool
    backend: str
    status: str
    steps: int
    initial_loss: float | None
    final_loss: float | None
    improved: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "status": self.status,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improved": self.improved,
            "message": self.message,
        }


def refine_graph_parameters(
    template,
    parameters: dict[str, float],
    bounds: list[tuple[float, float]],
    target: TargetResponse,
    options: dict[str, Any],
) -> tuple[dict[str, float], DifferentiableRefinementResult | None]:
    if not options.get("enabled", False):
        return parameters, None

    backend = str(options.get("backend", "torch")).lower()
    if backend != "torch":
        return parameters, DifferentiableRefinementResult(
            enabled=True,
            backend=backend,
            status="skipped",
            steps=0,
            initial_loss=None,
            final_loss=None,
            improved=False,
            message=f"unsupported differentiable backend: {backend}",
        )

    if target.analysis.kind != "voltage_transfer":
        return parameters, DifferentiableRefinementResult(
            enabled=True,
            backend=backend,
            status="skipped",
            steps=0,
            initial_loss=None,
            final_loss=None,
            improved=False,
            message="differentiable refinement currently supports voltage_transfer analysis only",
        )

    try:
        return _torch_refine_parameters(template, parameters, bounds, target, options)
    except ImportError as exc:
        return parameters, DifferentiableRefinementResult(
            enabled=True,
            backend=backend,
            status="unavailable",
            steps=0,
            initial_loss=None,
            final_loss=None,
            improved=False,
            message=str(exc),
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return parameters, DifferentiableRefinementResult(
            enabled=True,
            backend=backend,
            status="error",
            steps=0,
            initial_loss=None,
            final_loss=None,
            improved=False,
            message=str(exc),
        )


def _torch_refine_parameters(
    template,
    parameters: dict[str, float],
    bounds: list[tuple[float, float]],
    target: TargetResponse,
    options: dict[str, Any],
) -> tuple[dict[str, float], DifferentiableRefinementResult]:
    import torch

    steps = int(options.get("steps", 120))
    learning_rate = float(options.get("learning_rate", 0.03))
    phase_weight = float(options.get("phase_weight", 0.05))
    patience = int(options.get("patience", max(20, steps // 4)))
    if steps <= 0:
        return parameters, DifferentiableRefinementResult(
            enabled=True,
            backend="torch",
            status="skipped",
            steps=0,
            initial_loss=None,
            final_loss=None,
            improved=False,
            message="differentiable steps must be positive",
        )

    model = TorchLinearCircuitMNA(template, target.frequencies_hz, target.values, target.analysis)
    names = [param.name for param in template.params]
    initial_log_values = np.asarray([np.log10(parameters[name]) for name in names], dtype=np.float64)
    log_bounds = np.asarray([(np.log10(lo), np.log10(hi)) for lo, hi in bounds], dtype=np.float64)
    log_params = torch.tensor(initial_log_values, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_params], lr=learning_rate)

    with torch.no_grad():
        initial_loss = float(model.loss(log_params, phase_weight=phase_weight).detach().cpu())
    best_loss = initial_loss
    best_log = initial_log_values.copy()
    stale = 0

    lower = torch.tensor(log_bounds[:, 0], dtype=torch.float64)
    upper = torch.tensor(log_bounds[:, 1], dtype=torch.float64)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = model.loss(log_params, phase_weight=phase_weight)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_params.clamp_(lower, upper)
            current_loss = float(model.loss(log_params, phase_weight=phase_weight).detach().cpu())
            if current_loss + 1e-12 < best_loss:
                best_loss = current_loss
                best_log = log_params.detach().cpu().numpy().copy()
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

    improved = best_loss + 1e-10 < initial_loss
    if improved:
        refined = {name: float(10 ** value) for name, value in zip(names, best_log)}
        message = "torch autograd linear-circuit MNA refinement improved local loss"
    else:
        refined = dict(parameters)
        message = "torch autograd linear-circuit MNA refinement did not improve local loss"

    return refined, DifferentiableRefinementResult(
        enabled=True,
        backend="torch",
        status="completed",
        steps=steps,
        initial_loss=initial_loss,
        final_loss=best_loss,
        improved=improved,
        message=message,
    )


class TorchLinearCircuitMNA:
    def __init__(
        self,
        template,
        frequencies_hz: np.ndarray,
        target_values: np.ndarray,
        analysis: AnalysisRequest | None = None,
    ):
        import torch

        self.template = template
        self.frequencies = torch.tensor(frequencies_hz, dtype=torch.float64)
        self.target = torch.tensor(target_values, dtype=torch.complex128)
        self.analysis = analysis or AnalysisRequest.voltage_transfer(
            output_node=template.output_node,
            source_name=template.source_name,
        )
        nominal_values = {param.name: 1.0 for param in template.params}
        self.nominal_circuit = template.to_circuit(nominal_values)
        self.nodes = self.nominal_circuit.nodes()
        self.node_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.source_rows = {
            source.name: len(self.nodes) + idx
            for idx, source in enumerate(self.nominal_circuit.voltage_sources)
        }
        self.controlled_source_rows = {
            source.name: len(self.nodes) + len(self.nominal_circuit.voltage_sources) + idx
            for idx, source in enumerate(self.nominal_circuit.controlled_voltage_sources)
        }
        self.size = (
            len(self.nodes)
            + len(self.nominal_circuit.voltage_sources)
            + len(self.nominal_circuit.controlled_voltage_sources)
        )

    def response(self, log_params):
        import torch

        values = torch.pow(torch.tensor(10.0, dtype=torch.float64), log_params)
        parameter_values = {
            param.name: value for param, value in zip(self.template.params, values)
        }
        circuit = self.template.to_circuit(parameter_values)
        if self.analysis.kind == "transimpedance":
            source = self._find_current_source(circuit, self.analysis.source_name)
        else:
            source = self._find_source(circuit, self.analysis.source_name)
        source_value = _as_complex_tensor(source.value, torch)
        s = torch.complex(
            torch.zeros_like(self.frequencies),
            2.0 * torch.pi * self.frequencies,
        )
        matrices = torch.zeros(
            (len(self.frequencies), self.size, self.size),
            dtype=torch.complex128,
        )
        rhs = torch.zeros((len(self.frequencies), self.size), dtype=torch.complex128)

        for element in circuit.elements:
            admittance = self._admittance(element, s).to(torch.complex128)
            self._stamp_admittance(matrices, element.n1, element.n2, admittance)

        for source in circuit.voltage_sources:
            row = self.source_rows[source.name]
            self._stamp_voltage_branch(matrices, row, source.n_plus, source.n_minus)
            rhs[:, row] = _as_complex_tensor(source.value, torch)

        for source in circuit.current_sources:
            value = _as_complex_tensor(source.value, torch)
            p = self._node(source.n_plus)
            m = self._node(source.n_minus)
            if p is not None:
                rhs[:, p] -= value
            if m is not None:
                rhs[:, m] += value

        for source in circuit.controlled_voltage_sources:
            row = self.controlled_source_rows[source.name]
            self._stamp_voltage_branch(matrices, row, source.n_plus, source.n_minus)
            gain = _as_complex_tensor(source.gain, torch)
            cp = self._node(source.control_plus)
            cm = self._node(source.control_minus)
            if cp is not None:
                matrices[:, row, cp] -= gain
            if cm is not None:
                matrices[:, row, cm] += gain

        solutions = torch.linalg.solve(matrices, rhs)
        output = self._voltage(solutions, self.analysis.output_node) - self._voltage(
            solutions,
            self.analysis.reference_node,
        )
        return output / source_value

    def loss(self, log_params, phase_weight: float = 0.05):
        import torch

        response = self.response(log_params)
        eps = torch.tensor(1e-18, dtype=torch.float64)
        response_mag = torch.clamp(torch.abs(response), min=eps)
        target_mag = torch.clamp(torch.abs(self.target), min=eps)
        db_error = 20.0 / torch.log(torch.tensor(10.0, dtype=torch.float64)) * (
            torch.log(response_mag) - torch.log(target_mag)
        )
        mag_loss = torch.mean(db_error * db_error)
        response_unit = response / torch.clamp(torch.abs(response), min=eps)
        target_unit = self.target / torch.clamp(torch.abs(self.target), min=eps)
        phase_loss = torch.mean(torch.abs(response_unit - target_unit) ** 2)
        return mag_loss + phase_weight * phase_loss

    def _admittance(self, element, s):
        import torch

        value = _as_real_tensor(element.value, torch)
        kind = element.kind.upper()
        if kind == "R":
            return torch.ones_like(s) / value
        if kind == "C":
            return s * value
        if kind == "L":
            return 1.0 / (s * value)
        raise ValueError(f"unsupported differentiable element kind: {element.kind}")

    def _stamp_admittance(self, matrix, n1: str, n2: str, admittance) -> None:
        a = self._node(n1)
        b = self._node(n2)
        if a is not None:
            matrix[:, a, a] += admittance
        if b is not None:
            matrix[:, b, b] += admittance
        if a is not None and b is not None:
            matrix[:, a, b] -= admittance
            matrix[:, b, a] -= admittance

    def _stamp_voltage_branch(self, matrix, row: int, n_plus: str, n_minus: str) -> None:
        p = self._node(n_plus)
        m = self._node(n_minus)
        if p is not None:
            matrix[:, p, row] += 1.0
            matrix[:, row, p] += 1.0
        if m is not None:
            matrix[:, m, row] -= 1.0
            matrix[:, row, m] -= 1.0

    def _voltage(self, solutions, node: str):
        import torch

        idx = self._node(node)
        if idx is None:
            return torch.zeros(len(self.frequencies), dtype=torch.complex128)
        return solutions[:, idx]

    def _node(self, name: str) -> int | None:
        if name in GROUND_NAMES:
            return None
        return self.node_index[name]

    @staticmethod
    def _find_source(circuit, source_name: str):
        for source in circuit.voltage_sources:
            if source.name == source_name:
                return source
        raise ValueError(f"voltage source {source_name!r} not found")

    @staticmethod
    def _find_current_source(circuit, source_name: str):
        for source in circuit.current_sources:
            if source.name == source_name:
                return source
        raise ValueError(f"current source {source_name!r} not found")


class TorchGraphMNA(TorchLinearCircuitMNA):
    """Backward-compatible name for generated graph differentiable MNA."""

    pass


def _as_real_tensor(value, torch):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64)
    return torch.tensor(float(value), dtype=torch.float64)


def _as_complex_tensor(value, torch):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.complex128)
    return torch.tensor(complex(value), dtype=torch.complex128)
