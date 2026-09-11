from __future__ import annotations

import argparse
from pathlib import Path

from .pbdl_runner import run_pbdl_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and execute a PBDL multi-stage circuit synthesis spec.")
    parser.add_argument("spec", type=Path, help="PBDL JSON specification")
    parser.add_argument("--out", type=Path, default=None, help="output directory for plan and stage results")
    parser.add_argument("--top", type=int, default=None, help="override top-k candidates for each executed stage")
    parser.add_argument(
        "--replay-ranker",
        type=Path,
        default=None,
        help="optional .joblib replay bundle; applies only experimental within-tier ranking",
    )
    parser.add_argument("--plan-only", action="store_true", help="write plan.json without executing supported stages")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.out or Path("outputs") / f"{args.spec.stem}_pbdl"
    report = run_pbdl_file(
        args.spec,
        output_dir,
        execute=not args.plan_only,
        top_k=args.top,
        replay_ranker_path=args.replay_ranker,
    )

    print(f"PBDL synthesis: {report.spec_name}")
    print(f"Plan: {report.plan_path}")
    for stage in report.stages:
        line = f"{stage.stage_index}. {stage.stage_name}: {stage.status} {stage.analysis_kind}/{stage.target_kind}"
        if stage.best_template:
            line += f" -> {stage.best_template}"
            if stage.best_rmse_db is not None:
                line += f" rmse={stage.best_rmse_db:.4g}dB"
        elif stage.message:
            line += f" ({stage.message})"
        print(line)
    print(f"Wrote {output_dir / 'pbdl_report.json'}")
    return 0 if report.error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
