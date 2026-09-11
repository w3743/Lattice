from __future__ import annotations

import numpy as np
import pytest

from circuit_ai import CircuitSynthesizer, SynthesisSpec
from circuit_ai.differentiable import TorchGraphMNA, TorchLinearCircuitMNA, refine_graph_parameters
from circuit_ai.graph_templates import ElementSlot, GraphCircuitTemplate
from circuit_ai.targets import frequency_grid, target_from_behavior
from circuit_ai.templates import GainRCLowPass
from circuit_ai.topology import GraphSearchProposer


torch = pytest.importorskip("torch")


def test_torch_graph_mna_matches_template_response() -> None:
    template = GraphCircuitTemplate(
        (
            ElementSlot("X1", "R", "in", "out"),
            ElementSlot("X2", "C", "out", "0"),
        )
    )
    freqs = frequency_grid(10, 100000, 48)
    params = {"X1": 10_000.0, "X2": 15.915494309e-9}
    target = target_from_behavior({"kind": "lowpass", "cutoff_hz": 1000}, freqs)
    log_params = torch.tensor([np.log10(params["X1"]), np.log10(params["X2"])], dtype=torch.float64)

    response = TorchGraphMNA(template, freqs, target.values).response(log_params).detach().numpy()

    assert np.max(np.abs(response - template.response(params, freqs))) < 1e-9


def test_torch_linear_circuit_mna_supports_vcvs_template() -> None:
    template = GainRCLowPass()
    freqs = frequency_grid(10, 100000, 48)
    params = {"R": 10_000.0, "C": 15.915494309e-9, "Rg": 10_000.0, "Rf": 20_000.0}
    target = target_from_behavior({"kind": "lowpass", "cutoff_hz": 1000, "gain": 3.0}, freqs)
    log_params = torch.tensor(
        [np.log10(params[param.name]) for param in template.params],
        dtype=torch.float64,
    )

    response = TorchLinearCircuitMNA(template, freqs, target.values, target.analysis).response(log_params)

    assert np.max(np.abs(response.detach().numpy() - template.mna_response(params, freqs))) < 1e-9


def test_torch_graph_mna_gradcheck() -> None:
    template = GraphCircuitTemplate(
        (
            ElementSlot("X1", "R", "in", "out"),
            ElementSlot("X2", "C", "out", "0"),
        )
    )
    freqs = frequency_grid(100, 10000, 6)
    target = target_from_behavior({"kind": "lowpass", "cutoff_hz": 1000}, freqs)
    model = TorchGraphMNA(template, freqs, target.values)
    log_params = torch.tensor(
        [np.log10(8200.0), np.log10(22e-9)],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        lambda values: model.loss(values, phase_weight=0.02),
        (log_params,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_torch_linear_circuit_mna_vcvs_gradcheck() -> None:
    template = GainRCLowPass()
    freqs = frequency_grid(100, 10000, 5)
    target = target_from_behavior({"kind": "lowpass", "cutoff_hz": 1200, "gain": 2.5}, freqs)
    log_params = torch.tensor(
        [np.log10(8200.0), np.log10(18e-9), np.log10(12_000.0), np.log10(18_000.0)],
        dtype=torch.float64,
        requires_grad=True,
    )
    model = TorchLinearCircuitMNA(template, freqs, target.values, target.analysis)

    assert torch.autograd.gradcheck(
        lambda values: model.loss(values, phase_weight=0.02),
        (log_params,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_differentiable_refinement_reduces_graph_loss() -> None:
    template = GraphCircuitTemplate(
        (
            ElementSlot("X1", "R", "in", "out"),
            ElementSlot("X2", "C", "out", "0"),
        )
    )
    freqs = frequency_grid(10, 100000, 64)
    target = target_from_behavior({"kind": "lowpass", "cutoff_hz": 1000}, freqs)
    params = {"X1": 10_000.0, "X2": 1e-9}

    refined, result = refine_graph_parameters(
        template,
        params,
        bounds=[(100.0, 1_000_000.0), (1e-10, 1e-4)],
        target=target,
        options={"enabled": True, "steps": 160, "learning_rate": 0.04},
    )

    assert result is not None
    assert result.status == "completed"
    assert result.initial_loss is not None
    assert result.final_loss is not None
    assert result.final_loss < result.initial_loss
    assert refined != params


def test_differentiable_refinement_supports_handwritten_vcvs_template() -> None:
    template = GainRCLowPass()
    freqs = frequency_grid(10, 100000, 64)
    target = target_from_behavior({"kind": "lowpass", "cutoff_hz": 1000, "gain": 3.0}, freqs)
    params = {"R": 10_000.0, "C": 1e-9, "Rg": 10_000.0, "Rf": 5_000.0}

    refined, result = refine_graph_parameters(
        template,
        params,
        bounds=[(100.0, 1_000_000.0), (1e-10, 1e-4), (100.0, 1_000_000.0), (100.0, 1_000_000.0)],
        target=target,
        options={"enabled": True, "steps": 180, "learning_rate": 0.03},
    )

    assert result is not None
    assert result.status == "completed"
    assert result.initial_loss is not None
    assert result.final_loss is not None
    assert result.final_loss < result.initial_loss
    assert refined != params


def test_synthesizer_reports_differentiable_refinement_for_graphs() -> None:
    spec = SynthesisSpec.from_dict(
        {
            "name": "diff_graph_lowpass_test",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {
                "points": 40,
                "max_components": 2,
                "max_iterations": 2,
                "top_k": 1,
                "seed": 14,
                "differentiable": {"enabled": True, "steps": 40, "learning_rate": 0.03},
            },
        }
    )

    result = CircuitSynthesizer(
        proposer=GraphSearchProposer(fallback=None, max_candidates=6, include_fallback=False)
    ).synthesize(spec)[0]

    assert isinstance(result.template, GraphCircuitTemplate)
    assert result.metrics.differentiable is not None
    assert result.as_dict()["metrics"]["differentiable"]["backend"] == "torch"
