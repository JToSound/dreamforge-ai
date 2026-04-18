#!/usr/bin/env python3
"""Run a reproducible DreamForge benchmark pack and emit machine-readable scores."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi.testclient import TestClient

import api.main as api_main


def _profile_payloads() -> List[Dict[str, Any]]:
    return [
        {
            "name": "no-llm-fast",
            "payload": {
                "duration_hours": 1.0,
                "dt_minutes": 0.5,
                "ssri_strength": 1.0,
                "stress_level": 0.2,
                "sleep_start_hour": 23.0,
                "melatonin": False,
                "cannabis": False,
                "prior_day_events": ["benchmark-pack control run"],
                "emotional_state": "neutral",
                "style_preset": "scientific",
                "prompt_profile": "A",
                "use_llm": False,
                "llm_segments_only": False,
            },
        },
        {
            "name": "llm-short",
            "payload": {
                "duration_hours": 1.5,
                "dt_minutes": 0.5,
                "ssri_strength": 1.0,
                "stress_level": 0.25,
                "sleep_start_hour": 22.5,
                "melatonin": False,
                "cannabis": False,
                "prior_day_events": ["benchmark-pack short llm run"],
                "emotional_state": "curious",
                "style_preset": "scientific",
                "prompt_profile": "A",
                "use_llm": True,
                "llm_segments_only": True,
            },
        },
        {
            "name": "llm-long",
            "payload": {
                "duration_hours": 3.0,
                "dt_minutes": 0.5,
                "ssri_strength": 1.0,
                "stress_level": 0.3,
                "sleep_start_hour": 23.0,
                "melatonin": True,
                "cannabis": False,
                "prior_day_events": ["benchmark-pack long llm run"],
                "emotional_state": "neutral",
                "style_preset": "scientific",
                "prompt_profile": "A",
                "use_llm": True,
                "llm_segments_only": False,
            },
        },
    ]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _derive_score_row(
    name: str, payload: Dict[str, Any], runtime_seconds: float
) -> Dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    llm_total = int(summary.get("llm_total_invocations", 0) or 0)
    llm_fallback = int(summary.get("llm_fallback_segments", 0) or 0)
    fallback_rate = (float(llm_fallback) / float(llm_total)) if llm_total > 0 else 0.0
    return {
        "profile": name,
        "simulation_id": str(payload.get("id", "")),
        "runtime_seconds": round(runtime_seconds, 3),
        "segment_count": len(payload.get("segments") or []),
        "mean_bizarreness": float(summary.get("mean_bizarreness", 0.0) or 0.0),
        "rem_fraction": float(summary.get("rem_fraction", 0.0) or 0.0),
        "lucid_event_count": int(summary.get("lucid_event_count", 0) or 0),
        "narrative_quality_mean": float(
            summary.get("narrative_quality_mean", 0.0) or 0.0
        ),
        "narrative_memory_grounding_mean": float(
            summary.get("narrative_memory_grounding_mean", 0.0) or 0.0
        ),
        "llm_total_invocations": llm_total,
        "llm_fallback_segments": llm_fallback,
        "llm_fallback_rate": round(fallback_rate, 6),
        "status": "ok",
    }


def _load_baseline(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    rows = payload.get("runs", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile", "")).strip()
        if profile:
            out[profile] = row
    return out


def _delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    numeric_keys = [
        "runtime_seconds",
        "mean_bizarreness",
        "rem_fraction",
        "lucid_event_count",
        "narrative_quality_mean",
        "narrative_memory_grounding_mean",
        "llm_fallback_rate",
    ]
    diff: Dict[str, float] = {}
    for key in numeric_keys:
        cur = float(current.get(key, 0.0) or 0.0)
        base = float(baseline.get(key, 0.0) or 0.0)
        diff[key] = round(cur - base, 6)
    return diff


def run_benchmark(seed: int, baseline_path: Optional[Path]) -> Dict[str, Any]:
    baseline_rows = _load_baseline(baseline_path)
    runs: List[Dict[str, Any]] = []
    with TestClient(api_main.app) as client:
        for idx, profile in enumerate(_profile_payloads()):
            profile_name = str(profile["name"])
            _seed_everything(seed + idx)
            started_at = time.perf_counter()
            response = client.post("/api/simulation/night", json=profile["payload"])
            runtime_seconds = time.perf_counter() - started_at
            if response.status_code != 201:
                runs.append(
                    {
                        "profile": profile_name,
                        "status": "error",
                        "http_status": int(response.status_code),
                        "detail": response.text,
                    }
                )
                continue
            row = _derive_score_row(profile_name, response.json(), runtime_seconds)
            if profile_name in baseline_rows:
                row["delta_vs_baseline"] = _delta(row, baseline_rows[profile_name])
            runs.append(row)
    return {
        "benchmark_pack_version": 1,
        "api_contract": api_main.API_CONTRACT_VERSION,
        "prompt_profile_version": api_main.PROMPT_PROFILE_VERSION,
        "seed": int(seed),
        "created_at_unix": time.time(),
        "runs": runs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DreamForge benchmark pack with fixed profiles."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for deterministic benchmark execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "benchmark_pack" / "latest.json",
        help="Path to write benchmark JSON results.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional prior benchmark JSON for delta comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_benchmark(seed=int(args.seed), baseline_path=args.baseline)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
