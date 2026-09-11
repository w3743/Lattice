from __future__ import annotations

import json
from pathlib import Path

from circuit_ai import (
    CandidateEvaluation,
    DesignOrchestrator,
    compile_design_problem,
)


SENSOR_SPEC = Path("端口描述语言/examples/sensor_interface.json")
BOOST_SPEC = Path("端口描述语言/examples/5v_to_10v_dc_boost.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_design_problem_compiles_all_analyses_into_one_contract() -> None:
    problem = compile_design_problem(_load(SENSOR_SPEC))

    assert problem.problem_id == "sensor_interface.design"
    assert [task.analysis_kind for task in problem.tasks] == [
        "voltage_transfer",
        "impedance",
    ]
    assert {item.constraint_id for item in problem.constraints} >= {
        "structure.allowed_elements",
        "target.0.minimize_rmse_db",
        "target.1.minimize_rmse_db",
    }
    assert set(problem.optimization_problem.constraint_ids) == {
        item.constraint_id for item in problem.constraints
    }


def test_orchestrator_returns_common_ac_candidate_evaluations(tmp_path) -> None:
    problem = compile_design_problem(_load(SENSOR_SPEC))
    result = DesignOrchestrator().run(
        problem,
        stage_index=0,
        output_dir=tmp_path,
        top_k=1,
    )

    assert result.succeeded
    assert isinstance(result.selected, CandidateEvaluation)
    assert result.selected.domain == "ac"
    assert result.selected.validation.passed
    assert result.selected.simulation_results[0].status.value == "passed"
    assert (tmp_path / "design_problem.json").exists()
    assert (tmp_path / "candidate_evaluations.json").exists()
    replay_rows = [
        json.loads(line)
        for line in (tmp_path / "design_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert replay_rows and all(row["schema"] == "circuit_ai.design_replay" for row in replay_rows)
    assert all(row["domain"] == "ac" for row in replay_rows)
    assert replay_rows[0]["label"]["evidence_status"] == "verified"
    assert replay_rows[0]["frequency_hz"]
    assert replay_rows[0]["graph"]["components"]


def test_orchestrator_returns_common_dc_candidate_evaluations(tmp_path) -> None:
    problem = compile_design_problem(_load(BOOST_SPEC))
    result = DesignOrchestrator().run(problem, output_dir=tmp_path)

    assert result.succeeded
    assert result.selected.domain == "dc"
    assert result.selected.name == "ideal_asynchronous_boost"
    assert result.selected.validation.passed
    assert result.selected.simulation_results[0].status.value == "passed"
    assert (tmp_path / "candidate_evaluations.json").exists()
    replay_rows = [
        json.loads(line)
        for line in (tmp_path / "design_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert replay_rows and all(row["domain"] == "dc" for row in replay_rows)
    assert all("graph_hash" in row and "validation" in row for row in replay_rows)


def test_orchestrator_replay_ranker_is_explicit_and_evidence_first() -> None:
    class _ReplayRanker:
        def decision_function(self, features):
            return [float(features[0, 0])]

    problem = compile_design_problem(_load(SENSOR_SPEC))
    result = DesignOrchestrator().run(
        problem,
        stage_index=0,
        top_k=1,
        replay_ranker={"enabled": True, "model": _ReplayRanker()},
    )

    assert result.selected.validation.passed
    assert result.selected.metadata["replay_ranker_enabled"] is True
    assert "replay_rank_score" in result.selected.metadata
