from __future__ import annotations

import json

from circuit_ai.benchmark import (
    build_benchmark_proposer,
    generate_benchmark_cases,
    run_benchmark,
    summarize_results,
    write_benchmark_report,
)


def test_generated_benchmark_cases_cover_filter_kinds() -> None:
    cases = generate_benchmark_cases(count_per_kind=1, points=24, max_iterations=3)
    kinds = {case.spec.behavior["kind"] for case in cases}

    assert kinds == {"lowpass", "highpass", "bandpass"}
    assert all(case.spec.optimization.points == 24 for case in cases)


def test_benchmark_runs_and_writes_report(tmp_path) -> None:
    cases = generate_benchmark_cases(count_per_kind=1, points=28, max_iterations=8, max_components=5)
    report = run_benchmark(
        cases,
        build_benchmark_proposer("heuristic"),
        accept_rmse_db=2.0,
        accept_max_abs_db=8.0,
        max_iterations=8,
    )
    output = tmp_path / "benchmark.json"
    write_benchmark_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["summary"]["total"] == 3
    assert loaded["summary"]["success_rate"] >= 2 / 3
    assert len(loaded["results"]) == 3
    assert set(loaded["summary"]["by_kind"]) == {"bandpass", "highpass", "lowpass"}


def test_empty_benchmark_summary_is_well_defined() -> None:
    summary = summarize_results([])

    assert summary["total"] == 0
    assert summary["success_rate"] == 0.0
