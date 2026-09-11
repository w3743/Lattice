"""Regression checks for the versioned classical PBDL requirement corpus."""

from __future__ import annotations

import json
from pathlib import Path

from circuit_ai.pbdl_boundary import canonicalize_pbdl_dict, load_pbdl_dict


SUITE_ROOT = Path("测试集")
MANIFEST_PATH = SUITE_ROOT / "清单.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _stages_for(data: dict) -> list[str]:
    from 端口描述语言 import to_circuit_ai_plan

    return [stage.status for stage in to_circuit_ai_plan(load_pbdl_dict(data)).stages]


def test_classic_requirement_suite_is_complete_and_self_contained() -> None:
    manifest = _manifest()

    assert manifest["schema"] == "circuit_ai.classic_requirement_suite"
    assert manifest["counts"]["classic"] == 101
    assert manifest["counts"]["baseline"] == 10
    assert manifest["counts"]["total"] == 111
    assert manifest["counts"]["by_expected_planning"] == {
        "mixed": 1,
        "supported": 71,
        "unsupported": 39,
    }
    assert len(manifest["cases"]) == manifest["counts"]["total"]
    assert {item["category"] for item in manifest["cases"]} >= {
        "filter", "amplifier", "transimpedance", "dc_power", "isolated_power", "oscillator", "rf_matching"
    }
    assert manifest["counts"]["by_category"]["multi_stage"] == 3
    assert manifest["counts"]["by_category"]["multi_port"] == 2
    assert sum(item.get("expected_error_code") == "capability_gap" for item in manifest["cases"]) >= 35
    assert all("validation_scope" in item for item in manifest["cases"])
    assert all((SUITE_ROOT / item["path"]).is_file() for item in manifest["cases"])


def test_all_requirement_cases_are_valid_canonical_pbdl() -> None:
    for item in _manifest()["cases"]:
        data = json.loads((SUITE_ROOT / item["path"]).read_text(encoding="utf-8"))
        canonical = canonicalize_pbdl_dict(data)
        if item["path"].startswith("pbdl/classic/"):
            assert canonical["name"] == item["id"]
        else:
            assert canonical["name"]
        assert canonical["ports"]
        assert canonical["analyses"]
        assert canonical["targets"]
        assert [analysis.get("kind") for analysis in canonical["analyses"]] == [
            analysis.get("kind") for analysis in data["analyses"]
        ]


def test_planner_statuses_match_the_declared_capability_expectation() -> None:
    for item in _manifest()["cases"]:
        data = json.loads((SUITE_ROOT / item["path"]).read_text(encoding="utf-8"))
        stages = _stages_for(data)
        expected = item["expected_planning"]
        if expected == "supported":
            assert stages and all(status == "supported" for status in stages), item["id"]
        elif expected == "unsupported":
            assert stages and all(status == "unsupported" for status in stages), item["id"]
        else:
            assert "supported" in stages and "unsupported" in stages, item["id"]


def test_smoke_requirements_execute_through_the_supported_pipeline(tmp_path) -> None:
    from circuit_ai.pbdl_runner import run_pbdl_file

    smoke_cases = [
        item
        for item in _manifest()["cases"]
        if item["execution_tier"] == "smoke"
    ]
    assert {item["id"] for item in smoke_cases} == {
        "ac_rc_lowpass_1khz",
        "ac_sensor_filter_match_10khz",
        "dc_boost_5v_to_12v",
        "dc_flyback_5v_to_12v",
    }
    for item in smoke_cases:
        report = run_pbdl_file(
            SUITE_ROOT / item["path"],
            tmp_path / item["id"],
            top_k=1,
        )
        assert report.succeeded, item["id"]
        assert report.executed_count == len(report.stages), item["id"]
        for stage in report.stages:
            assert stage.output_dir is not None
            assert list(Path(stage.output_dir).glob("*.kicad_sch")), item["id"]
