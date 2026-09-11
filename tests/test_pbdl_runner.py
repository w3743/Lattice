from __future__ import annotations

import json
from pathlib import Path

from circuit_ai.pbdl_runner import run_pbdl_file
from circuit_ai.pbdl_cli import main as pbdl_cli_main
from circuit_ai.learned import save_model
from circuit_ai.ranking import train_replay_pairwise_ranker


def test_pbdl_runner_executes_supported_sensor_interface_stages(tmp_path) -> None:
    report = run_pbdl_file(
        "端口描述语言/examples/sensor_interface.json",
        tmp_path,
        top_k=1,
    )

    assert report.succeeded
    assert report.executed_count == 2
    assert report.unsupported_count == 0
    assert [stage.best_template for stage in report.stages] == [
        "sallen_key_gain_lowpass",
        "output_resistor_impedance",
    ]

    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    pbdl_report = json.loads((tmp_path / "pbdl_report.json").read_text(encoding="utf-8"))
    assert plan["fully_supported"] is True
    assert pbdl_report["executed_count"] == 2
    assert (tmp_path / "stage_1_voltage_transfer" / "best.svg").exists()
    assert (tmp_path / "stage_2_output_impedance" / "report.json").exists()


def test_pbdl_runner_blocks_multiport_downgrade(tmp_path) -> None:
    report = run_pbdl_file(
        "端口描述语言/examples/photodiode_tia.json",
        tmp_path,
        top_k=1,
    )

    assert report.succeeded
    assert report.executed_count == 0
    assert report.unsupported_count == 2
    assert all(stage.status == "unsupported" for stage in report.stages)
    assert all(stage.error and stage.error["code"] == "capability_gap" for stage in report.stages)
    assert all(stage.output_dir is None for stage in report.stages)


def test_pbdl_runner_plan_only_writes_plan_without_stage_outputs(tmp_path) -> None:
    report = run_pbdl_file(
        "端口描述语言/examples/sensor_interface.json",
        tmp_path,
        execute=False,
    )

    assert report.succeeded
    assert report.executed_count == 0
    assert [stage.status for stage in report.stages] == ["planned", "planned"]
    assert (tmp_path / "plan.json").exists()
    assert (tmp_path / "pbdl_report.json").exists()
    assert not (tmp_path / "stage_1_voltage_transfer").exists()


def test_pbdl_cli_prints_power_result_without_ac_rmse(tmp_path, capsys) -> None:
    exit_code = pbdl_cli_main(
        [
            "端口描述语言/examples/5v_to_10v_dc_boost.json",
            "--out",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "ideal_asynchronous_boost" in output
    assert "rmse=" not in output


def test_pbdl_runner_can_load_optional_replay_ranker_bundle(tmp_path) -> None:
    rows = [
        {
            "problem_id": "ranker_problem",
            "candidate_id": "a",
            "graph": {},
            "label": {"spec_name": "ranker_problem", "score": 0.1},
        },
        {
            "problem_id": "ranker_problem",
            "candidate_id": "b",
            "graph": {},
            "label": {"spec_name": "ranker_problem", "score": 1.0},
        },
    ]
    ranker = train_replay_pairwise_ranker(rows)
    ranker_path = tmp_path / "ranker.joblib"
    save_model({"ranking_model": ranker}, ranker_path)

    output = tmp_path / "run"
    report = run_pbdl_file(
        next(Path(".").glob("*/examples/sensor_interface.json")),
        output,
        top_k=1,
        replay_ranker_path=ranker_path,
    )

    assert report.succeeded
    manifest = json.loads((output / "replay_ranker_manifest.json").read_text(encoding="utf-8"))
    assert manifest["enabled"] is True
    assert manifest["training_pairs"] > 0
