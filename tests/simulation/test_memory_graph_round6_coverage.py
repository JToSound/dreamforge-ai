from __future__ import annotations

import numpy as np
import pytest

from core.models.memory_graph import (
    EmotionLabel,
    MemoryEdgeModel,
    MemoryGraph,
    MemoryNodeModel,
    MemoryType,
    ReplaySequence,
)


def _node(label: str, emotion: EmotionLabel = EmotionLabel.NEUTRAL) -> MemoryNodeModel:
    return MemoryNodeModel(
        label=label,
        memory_type=MemoryType.EPISODIC,
        emotion=emotion,
        salience=0.8,
        activation=0.7,
        tags=["city", "night"],
    )


def test_memory_graph_decay_prune_and_label() -> None:
    graph = MemoryGraph()
    n1 = graph.add_memory(_node("station"))
    n2 = graph.add_memory(_node("platform", EmotionLabel.JOY))
    graph.add_association(
        MemoryEdgeModel(source_id=n1, target_id=n2, weight=0.7, context_overlap=0.8)
    )

    assert graph.label(n1) == "station"
    graph.decay_salience(dt_hours=0.5)
    graph.prune_low_salience(threshold=0.01)
    assert n1 in graph.to_networkx().nodes


def test_memory_graph_replay_sequence_snapshot_and_export() -> None:
    np.random.seed(0)
    graph = MemoryGraph()
    n1 = graph.add_memory(_node("hallway", EmotionLabel.FEAR))
    n2 = graph.add_memory(_node("door", EmotionLabel.FEAR))
    n3 = graph.add_memory(_node("window", EmotionLabel.SURPRISE))
    graph.add_association(MemoryEdgeModel(source_id=n1, target_id=n2, weight=0.9))
    graph.add_association(MemoryEdgeModel(source_id=n2, target_id=n3, weight=0.8))

    seq = graph.sample_replay_sequence(
        max_length=4,
        start_bias_tags=["city"],
        current_time_hours=2.0,
    )
    assert seq is not None
    assert graph.replay_event_log

    graph.apply_replay_effect(seq, spike=0.2)
    active_ids = graph.apply_replay_pulse(
        seq, pulse_height=0.2, current_time_hours=2.1, activation_threshold=0.4
    )
    assert isinstance(active_ids, list)

    graph.decay_activations(dt_hours=0.05)
    graph.capture_memory_snapshot(time_hours=2.2, stage="REM")
    exported = graph.to_json_serializable()
    assert exported["replay_events"]
    assert exported["activation_snapshots"]


def test_memory_graph_empty_replay_and_empty_sampling() -> None:
    graph = MemoryGraph()
    assert graph.sample_replay_sequence() is None
    seq = ReplaySequence(
        id="empty",
        node_ids=[],
        total_weight=0.0,
        dominant_emotion=EmotionLabel.NEUTRAL,
    )
    assert graph.apply_replay_pulse(seq) == []


def test_add_association_requires_existing_nodes() -> None:
    graph = MemoryGraph()
    with pytest.raises(ValueError):
        graph.add_association(
            MemoryEdgeModel(source_id="missing-a", target_id="missing-b")
        )
