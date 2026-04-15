from __future__ import annotations

import random
from pathlib import Path

import numpy as np

import api.main as api_main
from core.agents.metacognitive_agent import MetacognitiveAgent
from core.models.neurochemistry import cortisol_profile
from core.models.sleep_cycle import CYCLE_TEMPLATES, SleepStage, TwoProcessParameters
from core.utils.bizarreness_scorer import compute_bizarreness


def test_tau_sleep_and_cycle_stage_proportions() -> None:
    params = TwoProcessParameters()
    assert params.tau_sleep == 4.2

    total_minutes = 0.0
    rem_minutes = 0.0
    n3_minutes = 0.0
    for cycle_id, template in CYCLE_TEMPLATES.items():
        if cycle_id > 5:
            continue
        for stage, minutes in template:
            total_minutes += float(minutes)
            if stage == SleepStage.REM:
                rem_minutes += float(minutes)
            if stage == SleepStage.N3:
                n3_minutes += float(minutes)

    rem_ratio = rem_minutes / total_minutes
    n3_ratio = n3_minutes / total_minutes
    assert 0.16 <= rem_ratio <= 0.26
    assert 0.13 <= n3_ratio <= 0.23


def test_cortisol_profile_matches_target_shape() -> None:
    assert cortisol_profile(2.5) <= 0.20
    assert cortisol_profile(7.5) >= 0.85

    values = [cortisol_profile(t) for t in (3.0, 4.5, 6.0, 7.5)]
    assert values[0] <= values[1] + 1e-9
    assert values[1] <= values[2]
    assert values[2] <= values[3]


def test_bizarreness_rem_is_high_but_bounded() -> None:
    np.random.seed(0)
    rem_score = compute_bizarreness(
        stage=SleepStage.REM,
        ach_level=0.9,
        ne_level=0.1,
        memory_arousal=0.8,
        cycle_index=4,
    ).total_score
    n2_score = compute_bizarreness(
        stage=SleepStage.N2,
        ach_level=0.45,
        ne_level=0.3,
        memory_arousal=0.2,
        cycle_index=1,
    ).total_score

    assert 0.75 <= rem_score <= 0.98
    assert 0.05 <= n2_score <= 0.45
    assert rem_score > n2_score


def test_lucidity_model_is_rem_only_and_ach_gated() -> None:
    agent = MetacognitiveAgent()
    non_rem = agent.compute_lucidity_probability(
        stage=SleepStage.N2,
        ach_level=0.7,
        cortisol_level=0.4,
        bizarreness_score=0.6,
        time_in_night_hours=4.0,
        reality_check_failures_recent=0,
    )
    rem_low_ach = agent.compute_lucidity_probability(
        stage=SleepStage.REM,
        ach_level=0.4,
        cortisol_level=0.4,
        bizarreness_score=0.8,
        time_in_night_hours=6.0,
        reality_check_failures_recent=0,
    )
    rem_high_ach = agent.compute_lucidity_probability(
        stage=SleepStage.REM,
        ach_level=0.9,
        cortisol_level=0.4,
        bizarreness_score=0.8,
        time_in_night_hours=6.0,
        reality_check_failures_recent=0,
    )

    assert non_rem == 0.0
    assert rem_high_ach > rem_low_ach


def test_api_physics_produces_nontrivial_rem_lucidity_and_bizarreness() -> None:
    random.seed(0)
    np.random.seed(0)

    cfg = api_main.SimulationConfig(
        duration_hours=8.0,
        dt_minutes=0.5,
        stress_level=0.6,
        use_llm=False,
    )
    segments = api_main._simulate_night_physics(cfg)

    non_rem_lucidity = [
        s["lucidity_probability"] for s in segments if s["stage"] != "REM"
    ]
    rem_lucidity = [s["lucidity_probability"] for s in segments if s["stage"] == "REM"]
    rem_biz = [s["bizarreness_score"] for s in segments if s["stage"] == "REM"]

    assert all(v == 0.0 for v in non_rem_lucidity)
    assert len(rem_lucidity) > 0
    assert max(rem_lucidity) > 0.1
    assert 0.7 <= float(np.mean(rem_biz)) <= 0.98


def test_dashboard_uses_fallback_neuro_and_narrative_keys() -> None:
    src = Path("visualization/dashboard/app.py").read_text(encoding="utf-8")

    assert 'result.get("neurochemistry_series")' in src
    assert 'result.get("neurochemistry_ticks")' in src
    assert 'seg.get("narrative")' in src
    assert 'seg.get("scene_description")' in src
    assert "No narrative segments found in simulation result." in src
