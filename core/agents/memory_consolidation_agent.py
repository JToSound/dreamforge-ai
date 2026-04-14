from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models.memory_graph import MemoryGraph, ReplaySequence
from core.simulation.event_bus import EventBus, Event, EventType


@dataclass
class MemoryConsolidationConfig:
    swr_probability_per_step: float = 0.2


class MemoryConsolidationAgent:
    """Agent responsible for hippocampal replay and memory maintenance."""

    def __init__(
        self,
        graph: Optional[MemoryGraph] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[MemoryConsolidationConfig] = None,
    ) -> None:
        self.graph = graph or MemoryGraph()
        self.event_bus = event_bus or EventBus()
        self.config = config or MemoryConsolidationConfig()

    def maybe_replay(self, current_time_hours: float) -> Optional[ReplaySequence]:
        import random

        if random.random() > self.config.swr_probability_per_step:
            return None

        seq = self.graph.sample_replay_sequence()
        if seq is None:
            return None

        self._emit_replay_event(seq, current_time_hours)
        return seq

    def decay_and_prune(self, dt_hours: float) -> None:
        self.graph.decay_salience(dt_hours=dt_hours)
        self.graph.prune_low_salience()

    def _emit_replay_event(
        self, seq: ReplaySequence, current_time_hours: float
    ) -> None:
        event = Event(
            type=EventType.MEMORY_REPLAY_EVENT,
            payload={
                "replay_id": seq.id,
                "node_ids": seq.node_ids,
                "total_weight": seq.total_weight,
                "dominant_emotion": seq.dominant_emotion.value,
            },
            timestamp_hours=current_time_hours,
        )
        self.event_bus.publish(event)
