import json

from circuit_ai.webapp import SpecCompileRequest, _candidate_payload, _power_stage_payload, compile_spec, topology_suggestions, workbench_architecture
from circuit_ai.spec import SynthesisSpec
from circuit_ai.synthesis import CircuitSynthesizer


def test_workbench_payload_has_renderable_candidate_data():
    spec = SynthesisSpec.from_dict(
        {
            "name": "web_test",
            "ports": 2,
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000, "frequency_range_hz": [10, 100000]},
            "library": {"allowed": ["R", "C"]},
            "optimization": {"points": 16, "max_iterations": 4, "top_k": 1, "seed": 3},
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]
    payload = _candidate_payload(result, "run", 1)
    assert payload["artifacts"]["kicad_schematic"].endswith("candidate_1.kicad_sch")
    assert len(payload["plot"]["frequency_hz"]) == 16
    assert payload["metrics"]["component_count"] >= 1
    assert "verification" in payload
    assert "optimization_run" in payload["verification"]


def test_workbench_exposes_the_executable_architecture_contract():
    payload = workbench_architecture()

    assert [step["id"] for step in payload["pipeline"]] == [
        "pbdl",
        "ir",
        "candidates",
        "simulation",
        "verification",
        "export",
    ]
    assert any("ideal averaged power" in item for item in payload["limits"])
    assert payload["capabilities"]


def test_spec_compile_returns_canonical_pbdl_ir_and_ready_execution():
    payload = compile_spec(SpecCompileRequest(spec={
        "name": "compile_test",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        ],
        "analyses": [{"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [10, 100000, 16]}],
        "targets": [{"target_kind": "filter", "filter_kind": "lowpass", "cutoff_hz": 1000}],
        "constraints": {"element_types": ["R", "C"]},
    }))

    assert payload["schema"] == "circuit_ai.spec_compile"
    assert payload["execution"]["status"] == "ready"
    assert len(payload["ir"]["ports"]) == 2
    assert payload["plan"]["stages"][0]["status"] == "supported"
    assert payload["diagnostics"]["lossless_ir"] is True


def test_spec_compile_blocks_multiport_execution_instead_of_downgrading():
    payload = compile_spec(SpecCompileRequest(spec={
        "name": "multiport_compile_test",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "supply", "terminals": [{"name": "vcc", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        ],
        "analyses": [{"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [10, 100000, 16]}],
        "targets": [{"target_kind": "filter", "filter_kind": "lowpass", "cutoff_hz": 1000}],
        "constraints": {"element_types": ["R", "C"]},
    }))

    assert payload["execution"]["status"] == "blocked"
    assert payload["execution"]["port_count"] == 3
    assert payload["plan"]["stages"][0]["status"] == "unsupported"
    assert payload["errors"][0]["code"] == "capability_gap"


def test_topology_suggestions_respect_required_elements():
    payload = topology_suggestions(
        type("Request", (), {"spec": {"name": "ui", "ports": 2, "behavior": {"kind": "lowpass", "cutoff_hz": 1_000}, "library": {"allowed": ["R", "C", "opamp"], "required": ["opamp"]}, "optimization": {"max_components": 8}}})()
    )
    names = {item["name"] for item in payload["catalog"]}
    assert "rc_lowpass" not in names
    assert "gain_rc_lowpass" in names


def test_workbench_contains_visual_waveform_editor():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="waveform-preview"' in html
    assert 'value="periodic_samples"' in html
    assert "behavior.excitation" in javascript
    assert "function updateWaveform()" in javascript


def test_workbench_contains_independent_multiport_editor():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="port-editor"' in html
    assert 'id="add-port"' in html
    assert "function portSpec(port)" in javascript
    assert "ports: portDefinitions.map(portSpec)" in javascript
    assert "analyses," in javascript
    assert "const targets = dcIntent ? [target]" in javascript
    assert "behavior.excitations = excitations" in javascript
    assert 'id="measurement-editor"' in html
    assert "measurementDefinitions.map" in javascript
    assert 'value="s_parameter"' in javascript


def test_workbench_guides_the_user_through_a_validated_design_flow():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert html.count("data-wizard-step=") == 5
    assert html.count('data-wizard-step="1"') == 1
    assert "接口定义" in html
    assert "行为与激励" in html
    assert 'id="wizard-next"' in html
    assert 'id="design-summary"' in html
    assert "function validateWizardStep(step)" in javascript
    assert "function validateAllSteps()" in javascript
    assert "function deriveDcIntent()" in javascript
    assert 'value="galvanic_isolation"' in javascript
    assert "output_current_a" in javascript
    assert 'value="ideal_transformer"' in html


def test_workbench_uses_a_canonical_draft_store_and_result_tabs():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    store = (webapp.STATIC_DIR / "store.js").read_text(encoding="utf-8")

    assert 'type="module" src="/app.js?v=workbench-11"' in html
    assert "createWorkbenchStore" in javascript
    assert "replaceDraftCollection" in javascript
    assert "export function createWorkbenchStore" in store
    assert html.count('role="tab"') == 6
    assert 'id="status-announcer"' in html
    assert 'id="error-announcer"' in html


def test_workbench_has_no_examples_or_history_ui():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "example-select" not in html
    assert "pbdl-example-select" not in html
    assert "run-history" not in html
    assert "loadExamples" not in javascript
    assert "loadHistory" not in javascript


def test_port_editor_uses_physical_variables_and_scoped_constraints():
    from circuit_ai import webapp

    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'variable_constraints: port.constraints.map' in javascript
    assert 'role: "potential"' in javascript
    assert 'role: "flow"' in javascript
    assert 'value="dc_operating_point"' in javascript
    assert 'value="periodic_steady_state"' in javascript
    assert "正值表示端口吸收功率" in javascript


def test_component_library_api_uses_local_kicad_index():
    from circuit_ai import webapp

    routes = {route.path for route in webapp.app.routes}
    assert "/store.js" in routes
    assert "/api/components/status" in routes
    assert "/api/components/reindex" in routes
    assert "/api/components/search" in routes
    assert "/api/components/validate" in routes
    assert "real_components" in (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")


def test_workbench_supports_spice_model_selection_and_pin_mapping():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="model-component"' in html
    assert 'id="model-pin-mapping"' in html
    assert 'id="add-model-binding"' in html
    assert 'data-real-model' in javascript
    assert "function renderModelBindingEditor()" in javascript
    assert "real_components: selectedRealComponents" in javascript
    assert "model_bindings: modelBindings" in javascript
    assert '`${pin.symbol}=${pin.model}`' in javascript


def test_dc_workbench_uses_power_pipeline_and_hides_frequency_controls():
    from circuit_ai import webapp

    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="signal-definition-fields"' in html
    assert 'id="frequency-target-fields"' in html
    assert '$("#signal-definition-fields").hidden = Boolean(intent)' in javascript
    assert 'const usePbdlRunner = isPowerDesign || isMultiStage' in javascript
    assert "if (preliminaryDcIntent) portDefinitions.forEach" in javascript
    assert javascript.count("function renderPorts()") == 1
    assert "function renderPowerStage(stage)" in javascript
    assert "端口与目标验收" in javascript
    assert "候选电路比较" in javascript
    assert "搜索证书" in javascript


def test_dc_topology_suggestions_use_isolation_expert():
    payload = topology_suggestions(
        type("Request", (), {"spec": {
            "name": "isolated_ui",
            "ports": [
                {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
                {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "out_0", "quantity": "ground"}]},
            ],
            "relations": [{"kind": "galvanic_isolation", "source_port": "input", "response_port": "output"}],
            "analyses": [{"kind": "dc_transfer", "source_port": "input", "output_port": "output"}],
            "targets": [{"target_kind": "dc", "input_voltage_v": 5, "output_voltage_v": 10, "output_current_a": 10}],
            "constraints": {"element_types": ["R", "C", "ideal_switch", "ideal_transformer", "ideal_diode"], "max_component_count": 5},
        }})()
    )
    assert payload["catalog"][0]["name"] == "ideal_isolated_flyback"


def test_power_stage_payload_exposes_search_and_constraint_evidence(tmp_path):
    report = {
        "design": "dc_power",
        "topology": {
            "name": "ideal_asynchronous_boost",
            "family": "dc_boost",
            "rationale": "step-up",
            "metadata": {
                "origin": "functional_block_knowledge_base",
                "knowledge_id": "test-knowledge",
                "solver": "ideal_boost_averaged",
            },
        },
        "parameters": {"duty_cycle": 0.5},
        "operating_point": {"output_voltage_v": 10.0, "output_current_a": 1.0},
        "validation": {"passed": True},
        "simulation_task_evaluations": [{"passed": True, "observations": []}],
        "topology_search": {"accepted_candidates": 1, "constructed_candidates": 2},
        "candidates": [{
            "candidate": {"name": "ideal_asynchronous_boost", "family": "dc_boost"},
            "objective": 0.1,
            "selection_score": 0.2,
            "validation": {"passed": True},
            "simulation_task_evaluations": [{"passed": True}],
        }],
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    payload = _power_stage_payload(tmp_path)

    assert payload is not None
    assert payload["topology"]["knowledge_id"] == "test-knowledge"
    assert payload["topology_search"]["constructed_candidates"] == 2
    assert payload["candidates"][0]["passed"]
