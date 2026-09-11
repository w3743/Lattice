"""Inspect or execute the classic PBDL requirement suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_ai.pbdl_boundary import load_pbdl_dict


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "清单.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _plan_status(path: Path) -> list[str]:
    from 端口描述语言 import to_circuit_ai_plan

    spec = load_pbdl_dict(json.loads(path.read_text(encoding="utf-8")))
    return [stage.status for stage in to_circuit_ai_plan(spec).stages]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or execute the classic PBDL requirement suite.")
    parser.add_argument("--mode", choices=("plan", "smoke"), default="plan")
    parser.add_argument("--case", action="append", help="case id; may be repeated")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "test_suite")
    args = parser.parse_args()

    manifest = _load_manifest()
    selected = [item for item in manifest["cases"] if not args.case or item["id"] in set(args.case)]
    if args.case:
        missing = set(args.case) - {item["id"] for item in selected}
        if missing:
            parser.error(f"unknown case id: {', '.join(sorted(missing))}")

    if args.mode == "plan":
        status_counts: dict[str, int] = {}
        for item in selected:
            statuses = _plan_status(ROOT / item["path"])
            key = "+".join(statuses)
            status_counts[key] = status_counts.get(key, 0) + 1
            print(f"{item['id']}: {key}")
        print(f"summary: {len(selected)} cases; " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
        return 0

    from circuit_ai.pbdl_runner import run_pbdl_file

    smoke = [item for item in selected if item["execution_tier"] == "smoke"]
    if not smoke:
        parser.error("--mode smoke requires at least one smoke case; use --case with a smoke id")
    failed = 0
    for item in smoke:
        report = run_pbdl_file(ROOT / item["path"], args.out / item["id"], top_k=1)
        print(f"{item['id']}: succeeded={report.succeeded}, executed={report.executed_count}, errors={report.error_count}")
        failed += int(not report.succeeded)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
