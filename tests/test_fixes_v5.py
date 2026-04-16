from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

import api.main as api_main
from core.agents.metacognitive_agent import MetacognitiveAgent
from core.models.memory_graph import (
    EmotionLabel,
    MemoryGraph,
    MemoryNodeModel,
    MemoryType,
)
from core.models.neurochemistry import cortisol_profile
from core.models.sleep_cycle import (
    CYCLE_TEMPLATES,
    SleepCycleModel,
    SleepStage,
    TwoProcessParameters,
)
from core.utils.bizarreness_scorer import compute_bizarreness
from core.utils.neurochemistry_descriptors import nchem_to_descriptors


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


def test_cycle_rem_durations_increase_across_night() -> None:
    rem_by_cycle = []
    for cycle_idx in [1, 2, 3, 4, 5]:
        template = CYCLE_TEMPLATES[cycle_idx]
        rem_by_cycle.append(
            float(
                next(minutes for stage, minutes in template if stage == SleepStage.REM)
            )
        )

    assert 8.0 <= rem_by_cycle[0] <= 12.0
    assert rem_by_cycle[1] > rem_by_cycle[0]
    assert rem_by_cycle[2] > rem_by_cycle[1]
    assert rem_by_cycle[3] >= rem_by_cycle[2]
    assert rem_by_cycle[4] >= rem_by_cycle[3]
    assert rem_by_cycle[-1] <= 60.0


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


def test_lucidity_model_is_rem_only_and_ach_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.agents.metacognitive_agent.random.random", lambda: 1.0)
    monkeypatch.setattr(
        "core.agents.metacognitive_agent.random.gauss", lambda _mu, _sigma: 0.0
    )
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


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_sleep_architecture_hits_n2_and_rem_ranges(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    model = SleepCycleModel()
    _, stages = model.simulate_night(duration_hours=8.0, dt_minutes=0.5)
    total = len(stages)
    n2_fraction = sum(1 for s in stages if s == SleepStage.N2) / total
    rem_fraction = sum(1 for s in stages if s == SleepStage.REM) / total
    assert 0.45 <= n2_fraction <= 0.59
    assert 0.18 <= rem_fraction <= 0.28


def test_dashboard_uses_fallback_neuro_and_narrative_keys() -> None:
    src = Path("visualization/dashboard/app.py").read_text(encoding="utf-8")

    assert 'result.get("neurochemistry_series")' in src
    assert 'result.get("neurochemistry_ticks")' in src
    assert 'seg.get("narrative")' in src
    assert 'seg.get("scene_description")' in src
    assert '"llm_trigger_type"' in src
    assert '"llm_latency_ms"' in src
    assert '"template_bank"' in src
    assert "No narrative segments found in simulation result." in src


def test_dashboard_segments_csv_preserves_zero_values() -> None:
    src = Path("visualization/dashboard/app.py").read_text(encoding="utf-8")
    assert "def _first_non_none" in src
    assert "if value is not None" in src
    assert '"llm_fallback_reason"' in src
    assert '"is_lucid"' in src
    assert '"memory_activations.csv"' in src
    assert "html.escape(" in src
    assert "scene_prefix_pattern" in src
    assert "pasted_content_pattern" in src
    assert "Export HTML" in src
    assert '"/api/charts/export"' in src
    assert '"/api/v1/charts/export"' in src
    assert "def _export_image_bytes" in src
    assert "Static image export is unavailable in this runtime." not in src
    assert 'artifact_prefix = f"dreamforge-sim-{sim_id}"' in src


def test_dashboard_memory_controls_and_compare_guards_are_stable() -> None:
    src = Path("visualization/dashboard/app.py").read_text(encoding="utf-8")
    assert "sim_key = sim_id_val" in src
    assert "sim_ts = int(time.time())" not in src
    assert 'key=f"mem_nodes_limit_{sim_key}"' in src
    assert 'key=f"mem_heat_top_nodes_{sim_key}"' in src
    assert "f\"⏹  {tr(_locale, 'stop_simulation')}\"" in src
    assert '"/api/simulation/night/async"' in src
    assert 'f"/api/simulation/jobs/{active_job_id}/cancel"' in src
    assert "def _format_eta_mmss" in src
    assert "def _api_get_bytes" in src
    assert "def _format_eta_with_margin" in src
    assert "def _phase_label" in src
    assert "progress_percent" in src
    assert "ETA" in src
    assert "Finalizing report..." in src
    assert "Baseline and candidate are the same run." in src
    assert "Delta formula: candidate - baseline" in src
    assert "anomaly_explanations = {" in src
    assert "llm_fallback_spike" in src
    assert "memory_grounding_drop" in src
    assert "Comparison methodology" in src
    assert "Methodology details" in src
    assert "/api/simulation/{sim_id_str}/report/bundle" in src
    assert "Download product report bundle (ZIP)" in src


def test_memory_graph_exports_activation_snapshots() -> None:
    graph = MemoryGraph()
    node_id = graph.add_memory(
        MemoryNodeModel(
            label="rainy street",
            memory_type=MemoryType.EPISODIC,
            emotion=EmotionLabel.NEUTRAL,
        )
    )
    graph.to_networkx().nodes[node_id]["activation"] = 0.75
    graph.capture_memory_snapshot(time_hours=1.0, stage="REM")
    exported = graph.to_json_serializable()

    assert "activation_snapshots" in exported
    assert len(exported["activation_snapshots"]) == 1


@pytest.mark.parametrize(
    ("ach", "serotonin", "ne", "cortisol"),
    [
        (0.8, 0.7, 0.8, 0.8),
        (0.6, 0.5, 0.5, 0.5),
        (0.3, 0.2, 0.3, 0.3),
        (0.1, 0.1, 0.1, 0.1),
    ],
)
def test_nchem_descriptor_mapping_has_all_keys(
    ach: float, serotonin: float, ne: float, cortisol: float
) -> None:
    desc = nchem_to_descriptors(ach=ach, serotonin=serotonin, ne=ne, cortisol=cortisol)
    assert set(desc.keys()) == {
        "ach_state",
        "mood_tone",
        "arousal_level",
        "stress_signature",
    }
    assert all(isinstance(v, str) and v for v in desc.values())
