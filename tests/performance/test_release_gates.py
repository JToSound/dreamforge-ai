from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

import api.main as api_main


def _simulation_payload(seed: int) -> dict[str, Any]:
    return {
        "duration_hours": 1.0,
        "dt_minutes": 5.0,
        "sleep_start_hour": 23.0,
        "stress_level": 0.45 + (0.01 * float(seed % 5)),
        "ssri_strength": 1.0,
        "melatonin": False,
        "cannabis": False,
        "emotional_state": "neutral",
        "style_preset": "scientific",
        "prompt_profile": "A",
        "use_llm": False,
        "prior_day_events": ["code review", "design discussion"],
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, int(round(0.95 * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def test_release_gate_load_profile() -> None:
    runs = 8
    latencies: list[float] = []
    successes = 0

    with TestClient(api_main.app) as client:
        for i in range(runs):
            t0 = perf_counter()
            response = client.post("/api/simulation/night", json=_simulation_payload(i))
            latencies.append(perf_counter() - t0)
            if response.status_code == 201:
                body = response.json()
                if isinstance(body, dict) and body.get("id"):
                    successes += 1

    success_ratio = float(successes) / float(runs)
    assert success_ratio >= 1.0
    assert _p95(latencies) <= 8.0


def test_release_gate_soak_profile() -> None:
    runs = 12
    latencies: list[float] = []

    with TestClient(api_main.app) as client:
        for i in range(runs):
            t0 = perf_counter()
            response = client.post(
                "/api/simulation/night",
                json=_simulation_payload(seed=100 + i),
            )
            latencies.append(perf_counter() - t0)
            assert response.status_code == 201

        release_gate = client.get("/api/release-gate")
        assert release_gate.status_code == 200
        payload = release_gate.json()
        assert "pass" in payload
        assert "checks" in payload

    assert float(mean(latencies)) <= 6.0
    assert max(latencies) <= 12.0
