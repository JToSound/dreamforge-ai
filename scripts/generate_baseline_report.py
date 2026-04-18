#!/usr/bin/env python3
"""Generate JSON and markdown benchmark comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from benchmark_pack import run_benchmark


def _build_markdown(payload: Dict[str, Any]) -> str:
    rows = payload.get("runs", [])
    if not isinstance(rows, list):
        rows = []
    lines: List[str] = []
    lines.append("# DreamForge baseline benchmark report")
    lines.append("")
    lines.append(f"- seed: `{payload.get('seed')}`")
    lines.append(f"- api_contract: `{payload.get('api_contract')}`")
    lines.append("")
    lines.append(
        "| profile | status | runtime_seconds | narrative_quality_mean | llm_fallback_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + f"{row.get('profile', '')} | {row.get('status', '')} | "
            + f"{row.get('runtime_seconds', '')} | {row.get('narrative_quality_mean', '')} | "
            + f"{row.get('llm_fallback_rate', '')} |"
        )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark JSON + markdown report."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON path for delta comparison.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports") / "benchmark_pack" / "latest.json",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("reports") / "benchmark_pack" / "latest.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_benchmark(seed=int(args.seed), baseline_path=args.baseline)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)

    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.md_output.write_text(_build_markdown(payload), encoding="utf-8")
    print(str(args.json_output))
    print(str(args.md_output))


if __name__ == "__main__":
    main()
