from __future__ import annotations

import numpy as np
import pytest

from circuit_ai.dataset import generate_rows
from circuit_ai.spec import LibrarySpec, SynthesisSpec
from circuit_ai.surrogate import (
    export_surrogate_rows,
    load_surrogate_rows,
    predict_response,
    result_to_surrogate_row,
    save_response_surrogate,
    train_response_surrogate,
)
from circuit_ai.templates import RCLowPass
from circuit_ai.synthesis import CircuitSynthesizer


def _validated_rows() -> list[dict]:
    library = LibrarySpec.from_dict(
        {
            "allowed": ["R", "C"],
            "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
        }
    )
    rows = generate_rows(
        library,
        samples_per_template=20,
        f_min_hz=10,
        f_max_hz=100000,
        points=32,
        seed=12,
        templates=[RCLowPass()],
    )
    for row in rows:
        row["validation"] = {"status": "passed", "backend": "mna_truth"}
    return rows


def test_surrogate_requires_explicit_validation_provenance() -> None:
    rows = _validated_rows()
    for row in rows:
        row.pop("validation")

    with pytest.raises(ValueError, match="validated"):
        train_response_surrogate(rows)


def test_validated_surrogate_predicts_with_domain_and_uncertainty_gate(tmp_path) -> None:
    rows = _validated_rows()
    bundle = train_response_surrogate(
        rows,
        n_estimators=12,
        min_samples_leaf=1,
        seed=8,
        trust_uncertainty_db=100.0,
    )
    first = rows[0]
    prediction = predict_response(
        bundle,
        "rc_lowpass",
        first["parameters"],
        [10.0, 1000.0, 100000.0],
    )

    assert prediction.response.shape == (3,)
    assert np.all(np.isfinite(prediction.response))
    assert prediction.in_domain is True
    assert prediction.trusted is True
    assert prediction.uncertainty_db >= 0.0
    assert prediction.reason.startswith("within")
    assert bundle["training_rows"] == len(rows)

    path = tmp_path / "response_surrogate.joblib"
    save_response_surrogate(bundle, path)
    assert path.exists()


def test_surrogate_row_export_preserves_validation_status(tmp_path) -> None:
    class Result:
        template = RCLowPass()
        parameters = {"R": 1000.0, "C": 1e-6}
        target = type("Target", (), {"frequencies_hz": np.asarray([10.0, 1000.0])})()
        response = np.asarray([1.0 + 0j, 0.5 - 0.5j])

    row = result_to_surrogate_row(Result(), {"status": "passed"}, source="ngspice")
    assert row["provenance"]["validated"] is True

    path = tmp_path / "surrogate_rows.jsonl"
    summary = export_surrogate_rows([Result()], path, [{"status": "passed"}], append=False)
    assert summary["validated_rows"] == 1
    assert load_surrogate_rows(path)[0]["validation"]["status"] == "passed"


def test_synthesis_can_use_surrogate_for_screening_then_truth_evaluate(tmp_path) -> None:
    rows = _validated_rows()
    bundle = train_response_surrogate(
        rows,
        n_estimators=12,
        min_samples_leaf=1,
        seed=13,
        trust_uncertainty_db=100.0,
    )
    model_path = tmp_path / "response_surrogate.joblib"
    save_response_surrogate(bundle, model_path)
    spec = SynthesisSpec.from_dict(
        {
            "name": "surrogate_screening",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {
                "points": 32,
                "max_iterations": 3,
                "top_k": 1,
                "seed": 3,
                "surrogate": {"enabled": True, "model_path": str(model_path), "trust_uncertainty_db": 100.0},
            },
        }
    )

    result = CircuitSynthesizer().synthesize(spec)[0]

    assert result.metrics.surrogate is not None
    assert result.metrics.surrogate["used"] is True
    assert np.all(np.isfinite(result.response))


def test_surrogate_out_of_domain_prediction_is_untrusted() -> None:
    rows = _validated_rows()
    bundle = train_response_surrogate(
        rows,
        n_estimators=12,
        min_samples_leaf=1,
        seed=13,
        trust_uncertainty_db=100.0,
    )

    outside = predict_response(bundle, "rc_lowpass", {"R": 1e9, "C": 1e-12}, [10.0, 1000.0])

    assert outside.in_domain is False
    assert outside.trusted is False
    assert outside.reason == "parameter point is outside validated training domain"


def _synthesize_with_surrogate(rows: list[dict], model_path, **bundle_updates) -> object:
    bundle = train_response_surrogate(
        rows,
        n_estimators=12,
        min_samples_leaf=1,
        seed=13,
        trust_uncertainty_db=100.0,
    )
    bundle["models"]["rc_lowpass"].update(bundle_updates)
    save_response_surrogate(bundle, model_path)
    spec = SynthesisSpec.from_dict(
        {
            "name": "surrogate_domain_gate",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "frequency_range_hz": [10, 100000],
            },
            "library": {
                "allowed": ["R", "C"],
                "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]},
            },
            "optimization": {
                "points": 32,
                "max_iterations": 3,
                "top_k": 1,
                "seed": 3,
                "surrogate": {
                    "enabled": True,
                    "model_path": str(model_path),
                    "trust_uncertainty_db": 100.0,
                },
            },
        }
    )
    return CircuitSynthesizer().synthesize(spec)[0]


def test_synthesis_falls_back_to_truth_when_surrogate_leaves_validated_domain(tmp_path) -> None:
    rows = _validated_rows()
    record = train_response_surrogate(
        rows,
        n_estimators=12,
        min_samples_leaf=1,
        seed=13,
        trust_uncertainty_db=100.0,
    )["models"]["rc_lowpass"]
    # Shrink the validated domain so every proposal is out of domain.
    shrunk_min = [value + 0.5 for value in record["input_min"]]
    shrunk_max = [value - 0.5 for value in record["input_max"]]

    result = _synthesize_with_surrogate(
        rows,
        tmp_path / "ood_surrogate.joblib",
        input_min=shrunk_min,
        input_max=shrunk_max,
    )

    assert result.metrics.surrogate is not None
    assert result.metrics.surrogate["used"] is True
    assert result.metrics.surrogate["fallback_to_truth"] is True
    assert result.metrics.surrogate["rejected_evaluations"] > 0
    # The reported response must be the physical re-evaluation, not a prediction.
    truth = result.template.analyze(
        result.parameters,
        result.target.frequencies_hz,
        result.target.analysis,
    )
    assert np.all(np.isfinite(result.response))
    assert np.allclose(result.response, truth)
