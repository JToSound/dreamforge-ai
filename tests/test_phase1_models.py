import random

import numpy as np
from scipy.stats import pearsonr

from core.agents.orchestrator import OrchestratorAgent
from core.models.memory_graph import (
    EmotionLabel,
    MemoryGraph,
    MemoryNodeModel,
    MemoryType,
    ReplaySequence,
)
from core.models.neurochemistry import NeurochemistryModel, NeurochemistryParameters
from core.models.sleep_cycle import CYCLE_TEMPLATES, SleepCycleModel, SleepStage


def test_neurochemistry_integrate_staged_transitions():
    model = NeurochemistryModel(params=NeurochemistryParameters(noise_std=0.0))
    state = model.initial_state(
        time_hours=0.0, ach=0.4, serotonin=0.6, ne=0.6, cortisol=0.5
    )

    trajectory = model.integrate_staged(
        state,
        [
            (SleepStage.N2, 0.5),
            (SleepStage.REM, 0.5),
        ],
    )

    at_boundary = next(s for s in trajectory if abs(s.time_hours - 0.5) < 1e-6)
    final = trajectory[-1]
    assert final.time_hours == 1.0
    assert final.ach > at_boundary.ach
    assert final.serotonin < at_boundary.serotonin
    assert final.ne < at_boundary.ne


def test_cortisol_drive_peaks_at_configured_hour():
    params = NeurochemistryParameters(
        noise_std=0.0, cortisol_rise_time=5.5, cortisol_k_rise=6.0, cortisol_k_fall=1.0
    )
    model = NeurochemistryModel(params=params)

    left = model._cortisol_drive(5.0)
    peak = model._cortisol_drive(5.5)
    right = model._cortisol_drive(6.0)

    assert peak > left
    assert peak > right


def test_sleep_cycle_templates_and_n3_fraction():
    assert 1 in CYCLE_TEMPLATES
    assert any(stage == SleepStage.N3 for stage, _ in CYCLE_TEMPLATES[1])

    np.random.seed(0)
    model = SleepCycleModel()
    _, stages = model.simulate_night(duration_hours=8.0, dt_minutes=0.5)
    n3_fraction = sum(1 for stage in stages if stage == SleepStage.N3) / len(stages)

    assert 0.12 <= n3_fraction <= 0.25


def test_memory_graph_replay_pulse_then_decay():
    graph = MemoryGraph()
    n1 = graph.add_memory(
        MemoryNodeModel(
            label="airport",
            memory_type=MemoryType.EPISODIC,
            emotion=EmotionLabel.NEUTRAL,
            activation=0.2,
            salience=0.4,
        )
    )
    n2 = graph.add_memory(
        MemoryNodeModel(
            label="train station",
            memory_type=MemoryType.EPISODIC,
            emotion=EmotionLabel.NEUTRAL,
            activation=0.2,
            salience=0.4,
        )
    )

    sequence = ReplaySequence(
        id="replay-1",
        node_ids=[n1, n2],
        total_weight=1.0,
        dominant_emotion=EmotionLabel.NEUTRAL,
    )
    graph.apply_replay_pulse(sequence, pulse_height=0.3, current_time_hours=1.0)

    after_pulse = graph.to_networkx().nodes[n1]["activation"]
    graph.decay_activations(dt_hours=0.05, decay_tau_hours=0.05)
    after_decay = graph.to_networkx().nodes[n1]["activation"]

    assert after_decay < after_pulse


def test_lucidity_bizarreness_correlation_bounded():
    random.seed(0)
    np.random.seed(0)
    orchestrator = OrchestratorAgent()
    result = orchestrator.run_night(
        duration_hours=8.0, dt_minutes=0.5, llm_every_n_segments=6
    )
    segments = result.get("dream_segments") or result.get("segments") or []

    biz = [
        float(s.get("bizarreness_score", s.get("bizarreness", 0.0))) for s in segments
    ]
    lucidity = [float(s.get("lucidity_probability", 0.0)) for s in segments]

    assert len(biz) > 2
    r, _ = pearsonr(biz, lucidity)
    assert abs(float(r)) < 0.55
