from __future__ import annotations

from circuit_ai.pbdl_boundary import canonicalize_pbdl_dict, translate_pbdl_dict


def test_legacy_workbench_payload_is_normalized_to_pbdl() -> None:
    canonical = canonicalize_pbdl_dict(
        {
            "name": "boundary_smoke",
            "ports": 2,
            "interface": {
                "ports": [
                    {"name": "input", "terminals": {"positive": "in", "negative": "0"}},
                    {"name": "output", "terminals": {"positive": "out", "negative": "0"}},
                ],
                "relations": [{"kind": "voltage_transfer", "source_port": "input", "response_port": "output"}],
            },
            "analysis": {"kind": "voltage_transfer"},
            "behavior": {"kind": "lowpass", "cutoff_hz": 1000, "gain": 1},
            "library": {"allowed": ["R", "C"], "required": ["C"]},
            "optimization": {"max_components": 4, "top_k": 1, "max_iterations": 3},
        }
    )

    assert set(("ports", "relations", "analyses", "targets", "constraints", "operating_point", "optimization")) <= set(canonical)
    assert len(canonical["ports"]) == 2
    assert canonical["constraints"]["required_elements"] == ["C"]


def test_canonical_pbdl_translates_to_internal_execution_ir() -> None:
    canonical = {
        "name": "canonical_smoke",
        "ports": [
            {"name": "input", "terminals": [{"name": "in", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
            {"name": "output", "terminals": [{"name": "out", "quantity": "voltage"}, {"name": "0", "quantity": "ground"}]},
        ],
        "analyses": [{"kind": "voltage_transfer", "source_port": "input", "output_port": "output", "frequency_hz": [10, 100000, 16]}],
        "targets": [{"target_kind": "filter", "kind": "lowpass", "cutoff_hz": 1000}],
        "constraints": {"element_types": ["R", "C"], "max_component_count": 4},
    }

    internal = translate_pbdl_dict(canonical)
    assert internal["ports"] == 2
    assert internal["behavior"]["kind"] == "lowpass"
    assert internal["optimization"]["points"] == 16
