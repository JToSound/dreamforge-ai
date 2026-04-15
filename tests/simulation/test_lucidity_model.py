from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main
from core.simulation.lucidity_model import LucidityModel, LucidityTickState


def test_peak_lucidity_gte_0_60_in_late_rem() -> None:
    model = LucidityModel(threshold=0.60, steepness=8.0)
    values = [
        model.compute_lucidity(
            LucidityTickState(
                stage="REM",
                rem_depth=depth,
                t_rem_fraction=frac,
                cycle_index=4,
            )
        )
        for depth in (0.62, 0.68, 0.74, 0.80)
        for frac in (0.1, 0.25, 0.6, 0.9)
    ]
    assert max(values) >= 0.60


def test_lucid_event_recorded_when_threshold_exceeded() -> None:
    segments = [
        {"start_time_hours": 1.0, "lucidity_probability": 0.1, "is_lucid": False},
        {"start_time_hours": 1.1, "lucidity_probability": 0.62, "is_lucid": False},
        {"start_time_hours": 1.2, "lucidity_probability": 0.65, "is_lucid": False},
        {"start_time_hours": 1.3, "lucidity_probability": 0.67, "is_lucid": False},
        {"start_time_hours": 1.4, "lucidity_probability": 0.2, "is_lucid": False},
    ]
    events = api_main._annotate_lucid_events(segments, 0.60)
    assert len(events) == 1
    assert events[0]["duration_ticks"] == 3
    assert all(segments[i]["is_lucid"] for i in (1, 2, 3))


def test_lucidity_result_json_contains_lucid_events_key() -> None:
    payload = {
        "duration_hours": 8.0,
        "dt_minutes": 0.5,
        "ssri_strength": 1.0,
        "stress_level": 0.8,
        "sleep_start_hour": 23.0,
        "prior_day_events": ["deadline", "crowded train"],
        "emotional_state": "anxious",
        "use_llm": False,
        "llm_segments_only": False,
    }
    with TestClient(api_main.app) as client:
        response = client.post("/api/simulation/night", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert "lucid_events" in body
    assert isinstance(body["lucid_events"], list)
