from __future__ import annotations

import json

import numpy as np
import pytest

from circuit_ai import render_circuit_svg, render_result_kicad_schematic, render_result_svg
from circuit_ai.targets import TargetResponse
from circuit_ai.templates import GainRCLowPass, RCLowPass
from circuit_ai.synthesis import SynthesisMetrics, SynthesisResult, write_results


def _result(template, parameters: dict[str, float]) -> SynthesisResult:
    frequencies = np.asarray([10.0, 1000.0, 100000.0])
    response = template.response(parameters, frequencies)
    return SynthesisResult(
        template=template,
        parameters=parameters,
        metrics=SynthesisMetrics(
            score=0.12,
            rmse_db=0.05,
            max_abs_db=0.08,
            phase_rmse_deg=1.5,
            component_count=template.component_count,
            optimizer_success=True,
            optimizer_message="test optimizer",
            pareto_rank=0,
        ),
        target=TargetResponse(frequencies, response),
        response=response,
    )


def test_render_circuit_svg_draws_basic_rc_components() -> None:
    template = RCLowPass()
    svg = render_circuit_svg(
        template.to_circuit({"R": 1000.0, "C": 1e-6}),
        title="RC low-pass",
    )

    assert svg.startswith("<svg")
    assert "RC low-pass" in svg
    assert "R1 1kOhm" in svg
    assert "C1 1uF" in svg
    assert "Vin 1" in svg
    assert "OUT" in svg
    assert "class=\"ground\"" in svg


def test_render_result_svg_draws_vcvs_controlled_source() -> None:
    result = _result(
        GainRCLowPass(),
        {"R": 1000.0, "C": 1e-9, "Rg": 1000.0, "Rf": 3000.0},
    )

    svg = render_result_svg(result)

    assert "gain_rc_lowpass" in svg
    assert "Eop gain=4" in svg
    assert "class=\"vcvs\"" in svg
    assert "control" in svg
    assert "Pareto rank: 0" in svg


def test_render_result_kicad_schematic_uses_kicad_sexpr(tmp_path) -> None:
    result = _result(RCLowPass(), {"R": 1000.0, "C": 1e-6})
    result = result.__class__(**{**result.__dict__, "model_bindings": ({"component": "R1", "library": str(tmp_path / "r.lib"), "name": "R_REAL", "pins": "1=1 2=2", "device": "X"},)})

    schematic = render_result_kicad_schematic(result, project_name="unit_test")

    assert schematic.startswith("(kicad_sch")
    assert '(symbol "CircuitAI:R"' in schematic
    assert '(symbol "CircuitAI:C"' in schematic
    assert '(lib_id "CircuitAI:R")' in schematic
    assert '(lib_id "CircuitAI:C")' in schematic
    assert '(lib_id "CircuitAI:VDC")' in schematic
    assert '(label "in"' in schematic
    assert '(label "out"' in schematic
    assert '(label "0"' in schematic
    assert '(property "Sim.Library"' in schematic
    assert '(property "Sim.Name" "R_REAL"' in schematic
    assert '(property "Sim.Pins" "1=1 2=2"' in schematic


def test_render_result_kicad_schematic_binds_four_pin_controlled_source(tmp_path) -> None:
    result = _result(GainRCLowPass(), {"R": 1000.0, "C": 1e-9, "Rg": 1000.0, "Rf": 3000.0})
    result = result.__class__(
        **{
            **result.__dict__,
            "model_bindings": (
                {
                    "component": "Eop",
                    "library": str(tmp_path / "opamp.lib"),
                    "name": "OPAMP_TEST",
                    "pins": "1=OUT 2=0 3=INP 4=INM",
                    "device": "X",
                },
            ),
        }
    )

    schematic = render_result_kicad_schematic(result, project_name="four_pin_model")

    assert '(lib_id "CircuitAI:ESOURCE")' in schematic
    assert '(property "Sim.Name" "OPAMP_TEST"' in schematic
    assert '(property "Sim.Pins" "1=OUT 2=0 3=INP 4=INM"' in schematic


def test_render_result_kicad_schematic_rejects_incomplete_pin_mapping(tmp_path) -> None:
    result = _result(GainRCLowPass(), {"R": 1000.0, "C": 1e-9, "Rg": 1000.0, "Rf": 3000.0})
    result = result.__class__(
        **{
            **result.__dict__,
            "model_bindings": (
                {
                    "component": "Eop",
                    "library": str(tmp_path / "opamp.lib"),
                    "name": "OPAMP_TEST",
                    "pins": "1=OUT 2=0 3=INP",
                },
            ),
        }
    )

    with pytest.raises(ValueError, match="Sim.Pins"):
        render_result_kicad_schematic(result, project_name="invalid_four_pin_model")


def test_write_results_exports_schematic_artifacts(tmp_path) -> None:
    result = _result(RCLowPass(), {"R": 1000.0, "C": 1e-6})

    write_results([result], tmp_path)

    assert (tmp_path / "candidate_1.svg").exists()
    assert (tmp_path / "best.svg").exists()
    assert (tmp_path / "candidate_1.kicad_sch").exists()
    assert (tmp_path / "best.kicad_sch").exists()
    assert (tmp_path / "best.kicad_pro").exists()
    assert (tmp_path / "schematics.html").exists()
    assert (tmp_path / "best.spice").exists()
    assert (tmp_path / "export_manifest.json").exists()

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report[0]["schematic"] == "candidate_1.svg"
    assert report[0]["best_schematic"] == "best.svg"
    assert report[0]["kicad_schematic"] == "candidate_1.kicad_sch"
    assert report[0]["best_kicad_schematic"] == "best.kicad_sch"
    assert report[0]["best_kicad_project"] == "best.kicad_pro"
    export_manifest = json.loads((tmp_path / "export_manifest.json").read_text(encoding="utf-8"))
    assert export_manifest[0]["graph_hash"] == result.circuit_graph().graph_hash
    assert export_manifest[0]["topology_hash"] == result.circuit_graph().topology_hash
    assert "R1 1kOhm" in (tmp_path / "best.svg").read_text(encoding="utf-8")
    assert '(kicad_sch' in (tmp_path / "best.kicad_sch").read_text(encoding="utf-8")


def test_write_results_materializes_model_files(tmp_path) -> None:
    model = tmp_path / "source.lib"
    model.write_text(".MODEL R_REAL R(R=1)", encoding="utf-8")
    result = _result(RCLowPass(), {"R": 1000.0, "C": 1e-6})
    result = result.__class__(**{**result.__dict__, "model_bindings": ({"component": "R1", "library": str(model), "name": "R_REAL", "pins": "1=1 2=2", "device": "X"},)})
    output = tmp_path / "out"
    write_results([result], output)
    assert (output / "models" / "source.lib").exists()
    schematic = (output / "candidate_1.kicad_sch").read_text(encoding="utf-8")
    assert '(property "Sim.Library" "models/source.lib"' in schematic
    best_spice = (output / "best.spice").read_text(encoding="utf-8")
    assert '.include "models/source.lib"' in best_spice
    assert "XR1 in out R_REAL" in best_spice
