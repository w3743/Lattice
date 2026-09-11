from pathlib import Path

from circuit_ai.error_reporting import ERROR_SCHEMA, ERROR_SCHEMA_VERSION, error_report


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "circuit_ai" / "web"


def test_frontend_has_one_rendered_section_per_wizard_step():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert [html.count(f'data-wizard-step="{step}"') for step in range(5)] == [1, 1, 1, 1, 1]


def test_frontend_exposes_global_and_stage_error_surfaces():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "error-panel",
        "error-phase",
        "error-code",
        "error-message",
        "error-detail",
        "error-suggestion-text",
    ):
        assert f'id="{element_id}"' in html
    assert "function requestJson" in javascript
    assert "function showError" in javascript
    assert "function renderStageError" in javascript
    assert "stage.error" in javascript


def test_frontend_submit_paths_use_structured_stage_context():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="compile-button"' in html
    assert 'id="compile-preview"' in html
    assert 'id="candidate-evidence"' in html
    assert 'id="edit-spec-button"' in html
    assert 'requestJson("/api/spec/compile"' in javascript
    assert "function renderCompileResult" in javascript
    assert 'requestJson("/api/pbdl/synthesize"' in javascript
    assert 'top_k: topK' in javascript
    assert 'const isMultiStage = (spec.analyses || []).length > 1' in javascript
    assert 'const usePbdlRunner = isPowerDesign || isMultiStage' in javascript
    assert 'requestJson(usePbdlRunner ? "/api/pbdl/synthesize" : "/api/synthesize"' in javascript
    assert 'function primaryRelationKind' in javascript
    assert 'control.disabled = true' in javascript
    assert 'showError(gap, "candidate_search", "capability_gap")' in javascript
    assert 'phase: "pbdl"' in javascript
    assert 'code: "invalid_pbdl"' in javascript


def test_error_report_contract_is_machine_readable():
    report = error_report(
        "simulation",
        "simulation_timeout",
        "simulation timed out",
        detail="solver did not finish",
        suggestion="reduce the search budget",
        retryable=True,
        context={"candidate": "test"},
    )

    assert report["schema"] == ERROR_SCHEMA
    assert report["schema_version"] == ERROR_SCHEMA_VERSION
    assert report["phase"] == "simulation"
    assert report["code"] == "simulation_timeout"
    assert report["retryable"] is True
    assert report["context"] == {"candidate": "test"}
